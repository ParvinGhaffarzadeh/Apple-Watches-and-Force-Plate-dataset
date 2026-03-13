#!/usr/bin/env python3
"""
monte_carlo_sensitivity.py
==========================
Monte Carlo synchronisation sensitivity analysis for the Apple Watch +
Force Plate multi-modal GRF dataset.

Assesses robustness of Pearson correlation metrics to small residual timing
uncertainty (±max_lag_ms) around the established post-hoc alignment.

Described in:
  Ghaffarzadeh et al. (2025). A Multi-Modal Dataset for Ground Reaction
  Force Estimation Using Consumer Wearable Sensors.
  Scientific Data (under review). https://doi.org/10.5281/zenodo.17376717

Three phases analysed per aligned trial file:
  Phase 1 — Waist → Force Plate : waist_acc_mag  vs force_z_N
  Phase 2 — Wrist → Waist       : wrist_acc_mag  vs waist_acc_mag
  Phase 3 — Wrist → Force Plate : wrist_acc_mag  vs force_z_N

Usage
-----
  python monte_carlo_sensitivity.py \\
      --aligned_dir /path/to/aligned_csvs \\
      --out_dir     ./sensitivity_outputs

  # Full options:
  python monte_carlo_sensitivity.py \\
      --aligned_dir  /path/to/aligned_csvs \\
      --out_dir      ./sensitivity_outputs \\
      --max_lag_ms   10.0 \\
      --target_fs    100.0 \\
      --n_iterations 100 \\
      --seed         42

Input
-----
  Directory of *_aligned.csv files, each containing columns:
    force_z_N, waist_acc_mag, wrist_acc_mag

Outputs (out_dir/)
------------------
  sync_sensitivity_results.csv          — per-trial-phase results
  sync_sensitivity_summary_by_phase.csv — grouped summary by phase
  sync_sensitivity_global_stats.csv     — global stats (all + subset)
  sync_sensitivity_manuscript_text.txt  — ready-to-paste manuscript paragraph
  sync_sensitivity.png                  — summary figure

Requirements
------------
  pip install pandas numpy matplotlib tqdm
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# =============================================================================
# Core utilities
# =============================================================================

def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient with basic safeguards."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n <= 2:
        return np.nan
    a = a[:n]
    b = b[:n]
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def apply_lag(
    sig1: np.ndarray,
    sig2: np.ndarray,
    lag_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a sample lag to sig2 relative to sig1, trimming to equal length.

    Positive lag: sig2 delayed — drop tail of sig1 and head of sig2.
    Negative lag: sig2 advanced — drop head of sig1 and tail of sig2.
    """
    s1 = np.asarray(sig1, dtype=float)
    s2 = np.asarray(sig2, dtype=float)

    if abs(lag_samples) >= min(len(s1), len(s2)):
        return np.array([]), np.array([])

    if lag_samples > 0:
        s1 = s1[:-lag_samples]
        s2 = s2[lag_samples:]
    elif lag_samples < 0:
        k = -lag_samples
        s1 = s1[k:]
        s2 = s2[:-k]

    n = min(len(s1), len(s2))
    return s1[:n], s2[:n]


def infer_metadata_from_filename(filename: str) -> Dict[str, object]:
    """
    Extract participant, activity, and trial_id from an aligned filename.
    Expected format: P01_Walk_ID9_walk0009_f_8_aligned.csv
    """
    participant = "unknown"
    m = re.search(r"^(P\d+)", filename)
    if m:
        participant = m.group(1)

    activity = "unknown"
    m = re.search(r"_(Walk|Jog|Run|Heel|Drop)_", filename, flags=re.IGNORECASE)
    if m:
        activity = m.group(1).lower()
        if activity == "jog":
            activity = "jogging"

    trial_id = None
    m = re.search(r"_ID(\d+)_", filename, flags=re.IGNORECASE)
    if m:
        trial_id = int(m.group(1))

    return {"participant": participant, "activity": activity, "trial_id": trial_id}


def analyze_signal_pair(
    sig1: np.ndarray,
    sig2: np.ndarray,
    max_lag_samples: int,
    n_iterations: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Monte Carlo sensitivity analysis for one signal pair (one phase, one trial).

    Computes baseline Pearson r at lag=0, then applies n_iterations random
    lag perturbations uniformly sampled from [-max_lag_samples, +max_lag_samples].
    Returns mean, 95th percentile, and maximum of |Δr| across perturbations.
    """
    baseline_r = pearson_r(sig1, sig2)

    perturbed_rs = []
    for _ in range(n_iterations):
        lag = int(rng.integers(-max_lag_samples, max_lag_samples + 1))
        s1, s2 = apply_lag(sig1, sig2, lag)
        r = pearson_r(s1, s2)
        if not np.isnan(r):
            perturbed_rs.append(r)

    perturbed_rs = np.asarray(perturbed_rs, dtype=float)

    if perturbed_rs.size == 0 or np.isnan(baseline_r):
        return {
            "baseline_r":       float(baseline_r) if not np.isnan(baseline_r) else np.nan,
            "n_valid":          0,
            "mean_abs_delta_r": np.nan,
            "p95_abs_delta_r":  np.nan,
            "max_abs_delta_r":  np.nan,
        }

    abs_deltas = np.abs(perturbed_rs - baseline_r)
    return {
        "baseline_r":       float(baseline_r),
        "n_valid":          int(perturbed_rs.size),
        "mean_abs_delta_r": float(np.mean(abs_deltas)),
        "p95_abs_delta_r":  float(np.quantile(abs_deltas, 0.95)),
        "max_abs_delta_r":  float(np.max(abs_deltas)),
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_sensitivity_results(results: pd.DataFrame, output_path: Path) -> None:
    """
    Two-panel summary figure:
      Left  — boxplot of mean|Δr| by phase
      Right — scatter of baseline r vs mean|Δr|
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    phases = ["Phase1_Waist_to_FP", "Phase2_Wrist_to_Waist", "Phase3_Wrist_to_FP"]
    data = [
        results.loc[results["phase"] == p, "mean_abs_delta_r"].dropna().to_numpy()
        for p in phases
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].boxplot(data, tick_labels=phases)
    axes[0].set_ylabel("Mean |Δr| (per comparison)")
    axes[0].set_title("Sensitivity distribution by phase")
    axes[0].grid(True, alpha=0.3, linestyle="--")

    axes[1].scatter(results["baseline_r"], results["mean_abs_delta_r"], alpha=0.35, s=18)
    axes[1].set_xlabel("Baseline r (aligned)")
    axes[1].set_ylabel("Mean |Δr| (per comparison)")
    axes[1].set_title("Sensitivity vs baseline correlation")
    axes[1].grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


# =============================================================================
# Global summary
# =============================================================================

def compute_global_stats(
    results: pd.DataFrame,
    subset_threshold: float,
) -> pd.DataFrame:
    """
    Compute global sensitivity statistics across all comparisons (rows),
    and for the subset with |baseline r| >= subset_threshold.
    """
    def summarize(df: pd.DataFrame, label: str) -> Dict[str, object]:
        return {
            "label":                          label,
            "n":                              int(len(df)),
            "mean_mean_abs_delta_r":          float(df["mean_abs_delta_r"].mean()),
            "p95_of_trial_mean_abs_delta_r":  float(df["mean_abs_delta_r"].quantile(0.95)),
            "p95_of_trial_max_abs_delta_r":   float(df["max_abs_delta_r"].quantile(0.95)),
            "overall_max_abs_delta_r":        float(df["max_abs_delta_r"].max()),
            "pct_max_gt_0_03":                float((df["max_abs_delta_r"] > 0.03).mean() * 100.0),
            "pct_max_gt_0_05":                float((df["max_abs_delta_r"] > 0.05).mean() * 100.0),
        }

    subset = results[results["baseline_r"].abs() >= subset_threshold]
    return pd.DataFrame([
        summarize(results, "ALL"),
        summarize(subset, f"|baseline r| >= {subset_threshold:g}"),
    ])


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte Carlo synchronisation sensitivity analysis on aligned trial CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--aligned_dir",
        required=True,
        type=Path,
        help="Directory containing *_aligned.csv files.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        type=Path,
        help="Output directory for results, figures, and manuscript text.",
    )
    parser.add_argument(
        "--max_lag_ms",
        type=float,
        default=10.0,
        help="Maximum lag perturbation in milliseconds (±).",
    )
    parser.add_argument(
        "--target_fs",
        type=float,
        default=100.0,
        help="Sampling rate of aligned files in Hz.",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=100,
        help="Monte Carlo iterations per trial-phase comparison.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--warn_sensitivity",
        type=float,
        default=0.05,
        help="Print a warning for trial-phase entries where max|Δr| exceeds this value.",
    )
    parser.add_argument(
        "--subset_threshold",
        type=float,
        default=0.2,
        help="Threshold on |baseline r| for the high-correlation subset statistics.",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    aligned_dir = args.aligned_dir
    out_dir     = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    aligned_files = sorted(aligned_dir.glob("*_aligned.csv"))
    if not aligned_files:
        raise SystemExit(f"No *_aligned.csv files found in {aligned_dir}")

    max_lag_samples = int(round(args.max_lag_ms * args.target_fs / 1000.0))
    rng = np.random.default_rng(args.seed)

    print("\n" + "=" * 70)
    print("MONTE CARLO SYNCHRONISATION SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"Aligned files : {aligned_dir}  ({len(aligned_files)} files)")
    print(f"Output        : {out_dir}")
    print(f"Max lag       : ±{args.max_lag_ms} ms  ({max_lag_samples} samples at {args.target_fs} Hz)")
    print(f"Iterations    : {args.n_iterations} per comparison")
    print(f"Seed          : {args.seed}")
    print(f"Subset cutoff : |baseline r| ≥ {args.subset_threshold}")
    print("=" * 70 + "\n")

    required_cols = {"force_z_N", "waist_acc_mag", "wrist_acc_mag"}
    rows = []
    skipped = 0

    for fp in tqdm(aligned_files, desc="Analysing trials"):
        try:
            df = pd.read_csv(fp)
            if not required_cols.issubset(df.columns):
                skipped += 1
                continue

            meta  = infer_metadata_from_filename(fp.name)
            force = df["force_z_N"].to_numpy(float)
            waist = df["waist_acc_mag"].to_numpy(float)
            wrist = df["wrist_acc_mag"].to_numpy(float)

            for phase, s1, s2 in [
                ("Phase1_Waist_to_FP",    waist, force),
                ("Phase2_Wrist_to_Waist", wrist, waist),
                ("Phase3_Wrist_to_FP",    wrist, force),
            ]:
                r = analyze_signal_pair(s1, s2, max_lag_samples, args.n_iterations, rng)
                rows.append({
                    "file":        fp.name,
                    "participant": meta["participant"],
                    "activity":    meta["activity"],
                    "trial_id":    meta["trial_id"],
                    "phase":       phase,
                    **r,
                })

        except Exception:
            skipped += 1
            continue

    results = pd.DataFrame(rows)
    if results.empty:
        raise SystemExit("No results produced — check input files and required columns.")

    n_ok = len(aligned_files) - skipped
    print(f"\nSuccessfully analysed {n_ok}/{len(aligned_files)} files"
          + (f"  ({skipped} skipped)" if skipped else ""))

    # Save per-trial-phase results
    out_csv = out_dir / "sync_sensitivity_results.csv"
    results.to_csv(out_csv, index=False)
    print(f"  Results   : {out_csv}")

    # Save per-phase summary
    summary = results.groupby("phase").agg({
        "mean_abs_delta_r": ["mean", "std", "min", "max"],
        "p95_abs_delta_r":  ["mean", "min", "max"],
        "max_abs_delta_r":  ["mean", "min", "max"],
        "baseline_r":       ["mean", "std", "min", "max"],
        "n_valid":          ["mean", "min", "max"],
    }).round(4)
    out_summary = out_dir / "sync_sensitivity_summary_by_phase.csv"
    summary.to_csv(out_summary)
    print(f"  Summary   : {out_summary}")

    # Plot
    plot_sensitivity_results(results, out_dir / "sync_sensitivity.png")

    # Global stats
    global_stats = compute_global_stats(results, subset_threshold=args.subset_threshold)
    out_global = out_dir / "sync_sensitivity_global_stats.csv"
    global_stats.to_csv(out_global, index=False)
    print(f"  Global    : {out_global}")

    # Console summary
    all_row = global_stats.iloc[0]
    sub_row = global_stats.iloc[1]
    n_total  = int(all_row["n"])
    n_subset = int(sub_row["n"])
    pct_subset = 100.0 * n_subset / n_total if n_total > 0 else np.nan

    print("\n" + "=" * 70)
    print("GLOBAL SUMMARY")
    print("=" * 70)
    print(f"Trials: {len(aligned_files)} | Comparisons: {n_total}")
    print(
        f"ALL  : mean|Δr| = {all_row['mean_mean_abs_delta_r']:.4f} | "
        f"P95(mean|Δr|) = {all_row['p95_of_trial_mean_abs_delta_r']:.4f} | "
        f"max|Δr| = {all_row['overall_max_abs_delta_r']:.4f}"
    )
    print(
        f"ALL  : % max|Δr|>0.03 = {all_row['pct_max_gt_0_03']:.1f}% | "
        f">0.05 = {all_row['pct_max_gt_0_05']:.1f}%"
    )
    print(
        f"SUBSET (|r|≥{args.subset_threshold}): N={n_subset} ({pct_subset:.1f}%) | "
        f"mean|Δr| = {sub_row['mean_mean_abs_delta_r']:.4f} | "
        f"P95 = {sub_row['p95_of_trial_mean_abs_delta_r']:.4f} | "
        f"max = {sub_row['overall_max_abs_delta_r']:.4f}"
    )

    # Phase breakdown
    print("\nBy phase (mean):")
    phase_summary = results.groupby("phase").agg(
        mean_abs_delta_r=("mean_abs_delta_r", "mean"),
        max_abs_delta_r =("max_abs_delta_r",  "max"),
        baseline_r      =("baseline_r",        "mean"),
    ).round(4)
    print(phase_summary.to_string())

    # Sensitivity warnings
    problematic = results[results["max_abs_delta_r"] > args.warn_sensitivity]
    if not problematic.empty:
        print(f"\nWARNING: {len(problematic)} trial-phase entries have max|Δr| > {args.warn_sensitivity}")
        low_baseline = problematic[problematic["baseline_r"].abs() < args.subset_threshold]
        if len(low_baseline):
            print(
                f"  Note: {len(low_baseline)}/{len(problematic)} of these have "
                f"|baseline r| < {args.subset_threshold} — elevated sensitivity "
                f"near-zero baseline correlation is expected."
            )

    # Manuscript paragraph
    manuscript_text = (
        f"To assess the robustness of the synchronisation approach, we performed Monte Carlo "
        f"simulations (n={args.n_iterations} iterations per comparison) applying random lag "
        f"perturbations of ±{args.max_lag_ms:g} ms to the aligned signals (sampled at "
        f"{args.target_fs:g} Hz). Across {len(aligned_files)} trials ({n_total} trial-phase "
        f"comparisons), the mean of the per-comparison mean absolute change in Pearson "
        f"correlation was {all_row['mean_mean_abs_delta_r']:.3f}; the 95th percentile of "
        f"mean |Δr| across comparisons was {all_row['p95_of_trial_mean_abs_delta_r']:.3f}, "
        f"and the overall maximum |Δr| was {all_row['overall_max_abs_delta_r']:.3f}. "
        f"Sensitivity was higher primarily when baseline correlation was near zero; restricting "
        f"to comparisons with |baseline r| ≥ {args.subset_threshold:g} ({n_subset}/{n_total}, "
        f"{pct_subset:.1f}%), the mean of mean |Δr| was {sub_row['mean_mean_abs_delta_r']:.3f} "
        f"and the 95th percentile was {sub_row['p95_of_trial_mean_abs_delta_r']:.3f} "
        f"(maximum {sub_row['overall_max_abs_delta_r']:.3f})."
    )

    out_txt = out_dir / "sync_sensitivity_manuscript_text.txt"
    out_txt.write_text(manuscript_text, encoding="utf-8")
    print("\n" + "=" * 70)
    print("MANUSCRIPT PARAGRAPH")
    print("=" * 70)
    print(manuscript_text)
    print("=" * 70)
    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
