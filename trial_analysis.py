#!/usr/bin/env python3
"""
trial_analysis.py
=================
Trial inventory, manifest generation, demographics, and QC pipeline
for the Apple Watch + Force Plate multi-modal GRF dataset.

Described in:
  Ghaffarzadeh et al. (2025). A Multi-Modal Dataset for Ground Reaction
  Force Estimation Using Consumer Wearable Sensors.
  Scientific Data (under review). https://doi.org/10.5281/zenodo.17376717

Usage
-----
  python trial_analysis.py --dataset_dir /path/to/Dataset --output_dir ./outputs

  # Optional flags:
  python trial_analysis.py \\
      --dataset_dir /path/to/Dataset \\
      --output_dir  ./outputs \\
      --ref_year    2024

Outputs
-------
Core (output_dir/):
  file_inventory.csv
  unmatched_files.csv          (only if unmatched files exist)
  trial_manifest.csv
  participant_demographics.csv
  trial_counts_all.csv
  trial_counts_triad_complete.csv
  methods_participants_section.tex

LaTeX (output_dir/latex_tables/):
  table_demographics.tex
  table_trial_counts_triad_complete.tex

QC (output_dir/QC/)  — only written when issues are detected:
  qc_unmatched_files.csv
  qc_unknown_participants.csv
  qc_duplicates_same_trial_key.csv
  qc_noncanonical_activities.csv
  qc_incomplete_trials.csv
  qc_completeness_by_activity.csv

Requirements
------------
  pip install pandas numpy
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# ============================================================================
# CONSTANTS
# ============================================================================

CANONICAL_ACTIVITY_ORDER: List[str] = ["walking", "jogging", "running", "heel", "drop"]
MAX_ROWCOUNT_CAP: int = 2_000_000

# ============================================================================
# HELPERS
# ============================================================================

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def normalize_activity(raw: str) -> str:
    """Normalise raw activity string to canonical name."""
    s = (raw or "").strip().lower()
    s = s.replace("jooging", "jogging").replace("joggin", "jogging")
    mapping = {
        "walk":      "walking",
        "walking":   "walking",
        "jog":       "jogging",
        "jogging":   "jogging",
        "joog":      "jogging",
        "run":       "running",
        "running":   "running",
        "heel":      "heel",
        "heeldrop":  "heel",
        "heel-drop": "heel",
        "drop":      "drop",
        "stepdrop":  "drop",
        "step-drop": "drop",
    }
    for k, v in mapping.items():
        if s.startswith(k):
            return v
    return s


def infer_participant_from_path(path: Path) -> str:
    """Extract participant ID (e.g. P8) from folder structure."""
    for part in reversed(path.parts):
        m = re.match(r"^P(\d+)[-_]", part)
        if m:
            return f"P{int(m.group(1))}"
    return "UNKNOWN"


def extract_participant_info(
    folder_name: str,
) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[float]]:
    """
    Parse participant folder name into (pid, birth_year, height_cm, weight_kg).
    Expects format: P<N>-<year>-<height>-<weight>  or  P<N>_<year>_<height>_<weight>
    Returns (None, None, None, None) if pattern does not match.
    """
    patterns = [
        r"^P(\d+)-(\d+)-([\d.]+)-([\d.]+)$",
        r"^P(\d+)_(\d+)_([\d.]+)_([\d.]+)$",
    ]
    for pat in patterns:
        m = re.match(pat, folder_name)
        if m:
            return int(m[1]), int(m[2]), float(m[3]), float(m[4])
    return None, None, None, None


def count_rows_fast(path: Path, cap: int = MAX_ROWCOUNT_CAP) -> int:
    """Count lines in a file quickly (used to pick best duplicate file)."""
    try:
        n = 0
        with path.open("r", errors="ignore") as f:
            for _ in f:
                n += 1
                if n >= cap:
                    break
        return n
    except Exception:
        return -1


def pick_best_file(filepaths: pd.Series) -> str:
    """
    Among duplicate file paths, return the one with the most rows
    (tie-break: largest file size). Returns empty string if input is empty.
    """
    if filepaths is None or len(filepaths) == 0:
        return ""
    best_fp, best_rows, best_size = "", -1, -1
    for fp in filepaths.tolist():
        p = Path(fp)
        rows = count_rows_fast(p)
        try:
            size = p.stat().st_size
        except Exception:
            size = -1
        if (rows > best_rows) or (rows == best_rows and size > best_size):
            best_fp, best_rows, best_size = fp, rows, size
    return best_fp

# ============================================================================
# FILENAME PARSING
# ============================================================================

def extract_trial_info(
    filename: str,
) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[int]]:
    """
    Parse a CSV filename into (activity, placement, trial_num, id_token).

    Apple Watch files:
        Activity_(Wrist|Waist)_Trial_<UUID>_ID_<N>_<timestamp>.csv
        -> placement = 'wrist' or 'waist'
        -> id_token  = N   (join key to force plate trial_num)
        -> trial_num = None

    Force plate files:
        <activity><NN>_f_<plate>.csv   e.g. walk01_f_6.csv
        -> placement  = 'force_plate'
        -> trial_num  = NN  (join key to watch id_token)
        -> id_token   = None
    """
    # Watch pattern
    m = re.match(r"^([A-Za-z]+)_(Waist|Wrist)_Trial_.*?_ID_(\d+)_", filename)
    if m:
        activity  = normalize_activity(m.group(1))
        placement = m.group(2).lower()
        id_token  = int(m.group(3))
        return activity, placement, None, id_token

    # Force plate pattern
    m = re.match(r"^([a-z]+)(\d+)_f_(\d+)\.csv$", filename.lower())
    if m:
        activity  = normalize_activity(m.group(1))
        trial_num = int(m.group(2))
        return activity, "force_plate", trial_num, None

    return None, None, None, None

# ============================================================================
# STEP 1: FILE INVENTORY
# ============================================================================

def build_file_inventory(
    base_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recursively scan base_path for CSV files and parse each filename.
    Returns (matched_df, unmatched_df).
    """
    files = sorted(base_path.rglob("*.csv"))
    matched: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, Any]] = []

    for f in files:
        participant = infer_participant_from_path(f)
        activity, placement, trial_num, id_token = extract_trial_info(f.name)

        if activity is None:
            unmatched.append({
                "filepath":              str(f),
                "filename":              f.name,
                "participant_from_path": participant,
                "reason":                "pattern_not_matched",
            })
            continue

        matched.append({
            "filepath":    str(f),
            "filename":    f.name,
            "participant": participant,
            "activity":    activity,
            "placement":   placement,
            "trial_num":   trial_num,
            "id_token":    id_token,
        })

    inv_df = pd.DataFrame(matched)
    un_df  = pd.DataFrame(unmatched)

    print(f"[Inventory]  Found {len(files)} CSV files under {base_path}")
    print(f"             Matched: {len(inv_df)} | Unmatched: {len(un_df)}")
    if len(inv_df):
        print(f"             Participants : {inv_df['participant'].nunique()}")
        print(f"             Activities   : {sorted(inv_df['activity'].unique())}")
        print(f"             Placements   : {sorted(inv_df['placement'].unique())}")

    return inv_df, un_df


def qc_inventory(
    inv_df: pd.DataFrame,
    un_df:  pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Generate inventory-level QC tables."""
    qc: Dict[str, pd.DataFrame] = {}

    if len(un_df):
        qc["qc_unmatched_files"] = un_df.sort_values(
            ["participant_from_path", "filename"]
        )

    unknown = inv_df[inv_df["participant"] == "UNKNOWN"].copy()
    if len(unknown):
        qc["qc_unknown_participants"] = unknown.sort_values(
            ["activity", "placement", "filename"]
        )

    key  = ["participant", "activity", "placement", "trial_num", "id_token"]
    dups = inv_df[inv_df.duplicated(key, keep=False)].copy()
    if len(dups):
        qc["qc_duplicates_same_trial_key"] = dups.sort_values(key + ["filename"])

    non_canon = sorted(
        [a for a in inv_df["activity"].unique() if a not in CANONICAL_ACTIVITY_ORDER]
    )
    if non_canon:
        qc["qc_noncanonical_activities"] = pd.DataFrame({"activity": non_canon})

    return qc

# ============================================================================
# STEP 2: TRIAL MANIFEST
# ============================================================================

def build_manifest(inv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build trial manifest by joining watch files (id_token) with force plate
    files (trial_num) on the common integer key.

    Each row = one trial = one (participant, activity, join_id) triple,
    with paths to wrist, waist, and force plate files (empty string if absent).
    """
    df = inv_df.copy()

    def _join_id(row: pd.Series) -> Optional[int]:
        if row["placement"] == "force_plate":
            return int(row["trial_num"]) if pd.notna(row["trial_num"]) else None
        return int(row["id_token"]) if pd.notna(row["id_token"]) else None

    df["join_id"] = df.apply(_join_id, axis=1)
    df = df.dropna(subset=["join_id"]).copy()
    df["join_id"] = df["join_id"].astype(int)

    trials: List[Dict[str, Any]] = []

    for (participant, activity, join_id), g in df.groupby(
        ["participant", "activity", "join_id"]
    ):
        wrist = g[g["placement"] == "wrist"]["filepath"]
        waist = g[g["placement"] == "waist"]["filepath"]
        fp    = g[g["placement"] == "force_plate"]["filepath"]

        trial: Dict[str, Any] = {
            "participant":    participant,
            "activity":       activity,
            "trial_key":      int(join_id),
            "wrist_file":     pick_best_file(wrist),
            "waist_file":     pick_best_file(waist),
            "fp_file":        pick_best_file(fp),
            "pairing_method": "id_token",
        }
        trial["has_wrist"]       = bool(trial["wrist_file"])
        trial["has_waist"]       = bool(trial["waist_file"])
        trial["has_force_plate"] = bool(trial["fp_file"])
        trial["triad_complete"]  = (
            trial["has_wrist"] and trial["has_waist"] and trial["has_force_plate"]
        )
        trials.append(trial)

    out = (
        pd.DataFrame(trials)
        .sort_values(["participant", "activity", "trial_key"])
        .reset_index(drop=True)
    )

    triad_n   = int(out["triad_complete"].sum())
    triad_pct = 100.0 * out["triad_complete"].mean()
    print(f"[Manifest]   {len(out)} trials | Triad-complete: {triad_n} ({triad_pct:.1f}%)")
    return out


def qc_manifest(manifest_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Generate manifest-level QC tables."""
    qc: Dict[str, pd.DataFrame] = {}

    summary = (
        manifest_df
        .groupby("activity")
        .agg(
            n_trials         = ("trial_key",       "count"),
            n_triad_complete = ("triad_complete",   "sum"),
            pct_complete     = ("triad_complete",   lambda x: 100.0 * float(np.mean(x))),
            missing_wrist    = ("has_wrist",        lambda x: int((~x).sum())),
            missing_waist    = ("has_waist",        lambda x: int((~x).sum())),
            missing_fp       = ("has_force_plate",  lambda x: int((~x).sum())),
        )
        .reset_index()
        .sort_values("activity")
    )
    qc["qc_completeness_by_activity"] = summary

    incomplete = manifest_df[~manifest_df["triad_complete"]].copy()
    if len(incomplete):
        qc["qc_incomplete_trials"] = incomplete.sort_values(
            ["participant", "activity", "trial_key"]
        )

    return qc

# ============================================================================
# STEP 3: DEMOGRAPHICS
# ============================================================================

def extract_demographics(base_path: Path, ref_year: int) -> pd.DataFrame:
    """
    Parse participant folder names to extract age, height, weight, BMI.
    Folder format: P<N>-<birth_year>-<height_cm>-<weight_kg>
    """
    participants: List[Dict[str, Any]] = []

    for folder in sorted(base_path.iterdir()):
        if not folder.is_dir():
            continue
        pid, year, height, weight = extract_participant_info(folder.name)
        if pid is None:
            continue

        age = ref_year - year
        bmi = weight / ((height / 100.0) ** 2)
        participants.append({
            "Participant": f"P{pid}",
            "Age":         float(age),
            "Height_cm":   float(height),
            "Weight_kg":   float(weight),
            "BMI":         float(bmi),
        })

    df = (
        pd.DataFrame(participants)
        .sort_values("Participant")
        .reset_index(drop=True)
    )
    print(
        f"[Demographics] {len(df)} participants | "
        f"Age {df['Age'].mean():.1f}±{df['Age'].std(ddof=1):.1f} yr | "
        f"BMI {df['BMI'].mean():.1f}±{df['BMI'].std(ddof=1):.1f}"
    )
    return df

# ============================================================================
# STEP 4: TRIAL COUNTS
# ============================================================================

def create_trial_counts(
    manifest_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (counts_all, counts_triad) pivot tables with TOTAL row."""

    def _pivot(df: pd.DataFrame) -> pd.DataFrame:
        counts = (
            df.groupby(["participant", "activity"])
            .size()
            .unstack(fill_value=0)
        )
        for act in CANONICAL_ACTIVITY_ORDER:
            if act not in counts.columns:
                counts[act] = 0
        counts = counts[CANONICAL_ACTIVITY_ORDER].copy()
        counts["Total"] = counts.sum(axis=1)
        totals      = counts.sum(numeric_only=True)
        totals.name = "TOTAL"
        return pd.concat([counts, pd.DataFrame([totals])]).astype(int)

    counts_all   = _pivot(manifest_df)
    counts_triad = _pivot(manifest_df[manifest_df["triad_complete"]])
    return counts_all, counts_triad

# ============================================================================
# STEP 5: LaTeX TABLES
# ============================================================================

def write_latex_tables(
    demo_df:      pd.DataFrame,
    counts_triad: pd.DataFrame,
    latex_dir:    Path,
) -> None:
    """Write table_demographics.tex and table_trial_counts_triad_complete.tex."""
    safe_mkdir(latex_dir)

    # --- Demographics ---
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Participant demographics (N=" + str(len(demo_df)) + r").}",
        r"\label{tab:demographics}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Participant} & \textbf{Age (yr)} & \textbf{Height (cm)} "
        r"& \textbf{Weight (kg)} & \textbf{BMI (kg/m$^2$)} \\",
        r"\midrule",
    ]
    for _, row in demo_df.iterrows():
        lines.append(
            f"{row['Participant']} & {row['Age']:.0f} & {row['Height_cm']:.1f} "
            f"& {row['Weight_kg']:.1f} & {row['BMI']:.1f} \\\\"
        )
    lines += [
        r"\midrule",
        r"\textbf{Mean $\pm$ SD} & "
        + f"{demo_df['Age'].mean():.1f} $\\pm$ {demo_df['Age'].std(ddof=1):.1f} & "
        + f"{demo_df['Height_cm'].mean():.1f} $\\pm$ {demo_df['Height_cm'].std(ddof=1):.1f} & "
        + f"{demo_df['Weight_kg'].mean():.1f} $\\pm$ {demo_df['Weight_kg'].std(ddof=1):.1f} & "
        + f"{demo_df['BMI'].mean():.1f} $\\pm$ {demo_df['BMI'].std(ddof=1):.1f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (latex_dir / "table_demographics.tex").write_text("\n".join(lines))

    # --- Triad-complete trial counts ---
    cols = [c for c in CANONICAL_ACTIVITY_ORDER if c in counts_triad.columns]
    col_spec = "l" + "c" * len(cols) + "c"
    header = (
        r"\textbf{Participant} "
        + " ".join(f"& \\textbf{{{c.capitalize()}}}" for c in cols)
        + r" & \textbf{Total} \\"
    )
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Number of triad-complete trials per participant by activity.}",
        r"\label{tab:trial_counts_triad}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for idx in [i for i in counts_triad.index if i != "TOTAL"]:
        row_vals = " ".join(f"& {int(counts_triad.loc[idx, c])}" for c in cols)
        lines.append(
            f"{idx} {row_vals} & {int(counts_triad.loc[idx, 'Total'])} \\\\"
        )
    total_vals = " ".join(f"& {int(counts_triad.loc['TOTAL', c])}" for c in cols)
    lines += [
        r"\midrule",
        r"\textbf{TOTAL} " + total_vals + f" & {int(counts_triad.loc['TOTAL', 'Total'])} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (latex_dir / "table_trial_counts_triad_complete.tex").write_text("\n".join(lines))

# ============================================================================
# STEP 6: METHODS SNIPPET
# ============================================================================

def write_methods_snippet(
    demo_df:      pd.DataFrame,
    counts_all:   pd.DataFrame,
    counts_triad: pd.DataFrame,
    output_dir:   Path,
) -> None:
    """Write a LaTeX methods snippet: methods_participants_section.tex."""
    per_part = counts_all.loc[
        [i for i in counts_all.index if i != "TOTAL"], "Total"
    ]
    n_all    = int(counts_all.loc["TOTAL", "Total"])
    n_triad  = int(counts_triad.loc["TOTAL", "Total"])
    pct      = 100.0 * n_triad / n_all if n_all else 0.0

    text = (
        r"\subsection*{Participants}" + "\n"
        f"Ten healthy adults participated "
        f"(age {demo_df['Age'].mean():.1f} $\\pm$ {demo_df['Age'].std(ddof=1):.1f}\\,yr; "
        f"height {demo_df['Height_cm'].mean():.1f} $\\pm$ {demo_df['Height_cm'].std(ddof=1):.1f}\\,cm; "
        f"weight {demo_df['Weight_kg'].mean():.1f} $\\pm$ {demo_df['Weight_kg'].std(ddof=1):.1f}\\,kg; "
        f"BMI {demo_df['BMI'].mean():.1f} $\\pm$ {demo_df['BMI'].std(ddof=1):.1f}\\,kg/m$^2$). "
        r"Participant demographics are summarised in Table~\ref{tab:demographics}." + "\n\n"
        r"\subsection*{Trial counts and completeness}" + "\n"
        f"The released dataset contains {n_all} validated trials across 10 participants "
        f"and 5 activities. Of these, {n_triad} trials are triad-complete "
        f"(wrist + waist + force plate), corresponding to {pct:.1f}\\% of all trials. "
        f"Per-participant trial totals range from {int(per_part.min())} to {int(per_part.max())} "
        f"(mean {per_part.mean():.1f} $\\pm$ {per_part.std(ddof=1):.1f}). "
        r"Counts are reported in Table~\ref{tab:trial_counts_triad}." + "\n"
    )
    (output_dir / "methods_participants_section.tex").write_text(text)

# ============================================================================
# SAVE QC TABLES
# ============================================================================

def save_qc_tables(
    qc_tables: Dict[str, pd.DataFrame],
    qc_dir:    Path,
) -> None:
    safe_mkdir(qc_dir)
    for name, df in qc_tables.items():
        if df is not None and len(df) > 0:
            df.to_csv(qc_dir / f"{name}.csv", index=False)

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trial inventory, manifest generation, and QC pipeline "
                    "for the Apple Watch + Force Plate GRF dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        type=Path,
        help="Root directory of the dataset (contains participant sub-folders).",
    )
    parser.add_argument(
        "--output_dir",
        default=Path("./outputs"),
        type=Path,
        help="Directory where all output files will be written.",
    )
    parser.add_argument(
        "--ref_year",
        default=2024,
        type=int,
        help="Reference year used to calculate participant ages from birth year.",
    )
    return parser.parse_args()

# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args = parse_args()

    base_path  = args.dataset_dir
    output_dir = args.output_dir
    ref_year   = args.ref_year
    qc_dir     = output_dir / "QC"
    latex_dir  = output_dir / "latex_tables"

    if not base_path.exists():
        print(f"ERROR: dataset_dir does not exist: {base_path}", file=sys.stderr)
        sys.exit(1)

    safe_mkdir(output_dir)
    safe_mkdir(qc_dir)
    safe_mkdir(latex_dir)

    print("=" * 60)
    print("TRIAL ANALYSIS PIPELINE")
    print(f"Dataset : {base_path}")
    print(f"Output  : {output_dir}")
    print("=" * 60)

    # Step 1 — inventory
    print("\n[1/6] Building file inventory ...")
    inv_df, un_df = build_file_inventory(base_path)
    qc_inv = qc_inventory(inv_df, un_df)

    # Step 2 — manifest
    print("\n[2/6] Building trial manifest ...")
    manifest_df = build_manifest(inv_df)
    qc_man = qc_manifest(manifest_df)

    # Step 3 — demographics
    print("\n[3/6] Extracting participant demographics ...")
    demo_df = extract_demographics(base_path, ref_year)

    # Step 4 — trial counts
    print("\n[4/6] Creating trial count tables ...")
    counts_all, counts_triad = create_trial_counts(manifest_df)

    # Step 5 — write outputs
    print("\n[5/6] Writing output files ...")
    inv_df.to_csv(output_dir / "file_inventory.csv", index=False)
    if len(un_df):
        un_df.to_csv(output_dir / "unmatched_files.csv", index=False)
    manifest_df.to_csv(output_dir / "trial_manifest.csv", index=False)
    demo_df.to_csv(output_dir / "participant_demographics.csv", index=False)
    counts_all.to_csv(output_dir / "trial_counts_all.csv")
    counts_triad.to_csv(output_dir / "trial_counts_triad_complete.csv")

    write_latex_tables(demo_df, counts_triad, latex_dir)
    write_methods_snippet(demo_df, counts_all, counts_triad, output_dir)
    save_qc_tables({**qc_inv, **qc_man}, qc_dir)

    # Step 6 — summary
    print("\n[6/6] Done.")
    print("=" * 60)
    triad_n = int(manifest_df["triad_complete"].sum())
    print(f"  Total validated trials : {len(manifest_df)}")
    print(f"  Triad-complete trials  : {triad_n} ({100*manifest_df['triad_complete'].mean():.1f}%)")
    print(f"  Participants           : {len(demo_df)}")
    print(f"  Activities             : {len(CANONICAL_ACTIVITY_ORDER)}")
    print()
    print("  Core outputs:")
    for fname in [
        "file_inventory.csv",
        "trial_manifest.csv",
        "participant_demographics.csv",
        "trial_counts_all.csv",
        "trial_counts_triad_complete.csv",
        "methods_participants_section.tex",
    ]:
        print(f"    {output_dir / fname}")
    print(f"  LaTeX tables : {latex_dir}")
    print(f"  QC tables    : {qc_dir}  (only written if issues detected)")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
