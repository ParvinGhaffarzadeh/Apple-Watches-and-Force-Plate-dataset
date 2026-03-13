#!/usr/bin/env python3
"""
alignment_pipeline.py
=====================
Robust hybrid temporal alignment pipeline for the Apple Watch + Force Plate
multi-modal GRF dataset.

Aligns force plate signals (resampled to 100 Hz) with wrist and waist Apple
Watch IMU signals (100 Hz) using a three-strategy hybrid approach:
  A) Stance × wrist impact peak matching  (used in 98.3% of trials)
  B) Cross-correlation fallback using wrist proxy
  C) Cross-correlation fallback using waist proxy

No synthetic padding is applied. Only the true overlap region between force
and IMU signals is retained after applying the estimated lag.

Described in:
  Ghaffarzadeh et al. (2025). A Multi-Modal Dataset for Ground Reaction
  Force Estimation Using Consumer Wearable Sensors.
  Scientific Data (under review). https://doi.org/10.5281/zenodo.17376717

Usage
-----
  python alignment_pipeline.py \\
      --dataset_dir /path/to/Dataset \\
      --output_dir  ./Dataset_Aligned

  # All options:
  python alignment_pipeline.py \\
      --dataset_dir   /path/to/Dataset \\
      --output_dir    ./Dataset_Aligned \\
      --target_fs     100.0 \\
      --xcorr_max_lag 1.25 \\
      --min_coverage  0.80

Outputs (output_dir/)
---------------------
  *_aligned.csv       — one aligned CSV per successful trial, containing:
                        time_s, force_z_N,
                        waist_acc[XYZ], waist_gyro[XYZ], waist_acc_mag,
                        wrist_acc[XYZ], wrist_gyro[XYZ], wrist_acc_mag

  alignment_log.csv   — per-trial log with columns:
                        participant, activity, trial_num,
                        force_file, waist_file, wrist_file,
                        status, estimated_lag_ms, peak_time_difference_ms,
                        alignment_method, overlap_coverage,
                        cross_correlation_quality,
                        peak_orig, peak_win, n, out

Requirements
------------
  pip install pandas numpy scipy tqdm
"""

from __future__ import annotations

import argparse
import io
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags, find_peaks
from tqdm import tqdm


# =============================================================================
# GLOBAL CONFIG
# =============================================================================

TARGET_FS = 100.0  # Hz — overridden by --target_fs

# Force stance detection defaults
DEFAULT_MIN_PEAK_GRF = 400.0   # N
DEFAULT_CONTACT_THR  = 80.0    # N
DEFAULT_MIN_STANCE_S = 0.15    # s
DEFAULT_MAX_STANCE_S = 2.0     # s

# Window around stance saved to aligned CSV (seconds)
PRE_STANCE_S  = 0.50
POST_STANCE_S = 0.50

# Wrist impact detection
IMPACT_MIN_DISTANCE_S = 0.30
IMPACT_PERCENTILE     = 70
IMPACT_PROMINENCE     = 0.30

# Minimum fraction of stance window that must be covered by IMU overlap
MIN_OVERLAP_COVERAGE = 0.80  # overridden by --min_coverage

# Watch file column names
WATCH_TIME_COL = "timestamp"   # ms
WATCH_IMU_COLS = [
    "userAccelerationX", "userAccelerationY", "userAccelerationZ",
    "rotationRateX",     "rotationRateY",     "rotationRateZ",
]


# =============================================================================
# ACTIVITY-AWARE THRESHOLDS
# =============================================================================

def activity_params(activity: str) -> Tuple[float, float, float, float]:
    """
    Return (min_peak_N, contact_thr_N, min_stance_s, max_stance_s) per activity.
    Heel drop uses looser thresholds due to shorter, sharper contact events.
    """
    if activity.lower() == "heel":
        return DEFAULT_MIN_PEAK_GRF, 40.0, 0.05, DEFAULT_MAX_STANCE_S
    return DEFAULT_MIN_PEAK_GRF, DEFAULT_CONTACT_THR, DEFAULT_MIN_STANCE_S, DEFAULT_MAX_STANCE_S


# =============================================================================
# FILE LOADING
# =============================================================================

def load_force_plate(fp_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load AMTI force plate CSV (with preamble) and return (time_s, force_z_N).

    Applies:
    - Automatic header detection (looks for row containing 'time' and 'force z')
    - Median baseline correction on first 10% of signal
    - Sign flip if signal is predominantly negative
    """
    if "_aligned" in fp_path.name.lower():
        raise ValueError(f"Refusing to load an aligned file as raw force: {fp_path.name}")

    raw = fp_path.read_bytes().decode("utf-8-sig", errors="replace")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    header_idx = 26  # AMTI default preamble length
    for i, ln in enumerate(lines[:200]):
        low = ln.lower()
        if "time" in low and "force" in low and "z" in low:
            header_idx = i
            break

    data_block = "\n".join(lines[header_idx + 1:])
    df = pd.read_csv(io.StringIO(lines[header_idx] + "\n" + data_block))
    df.columns = [c.strip() for c in df.columns]

    time_col = next((c for c in df.columns if c.lower() == "time"), df.columns[0])
    fz_col   = next(
        (c for c in df.columns if "force" in c.lower() and "z" in c.lower()), None
    )
    if fz_col is None:
        raise ValueError(
            f"Could not find Force_Z column in {fp_path.name}. "
            f"Columns: {df.columns.tolist()}"
        )

    t  = df[time_col].astype(float).to_numpy()
    fz = np.nan_to_num(df[fz_col].astype(float).to_numpy(), nan=0.0)

    # Baseline correction
    bN = int(0.10 * len(fz))
    if bN >= 10:
        baseline = float(np.nanmedian(fz[:bN]))
        if np.isfinite(baseline) and abs(baseline) > 5:
            fz = fz - baseline

    # Sign correction
    if len(fz) > 50 and np.abs(np.percentile(fz, 1)) > np.percentile(fz, 99):
        fz = -fz

    return t, fz


def load_watch_imu(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Apple Watch IMU CSV and return (time_s_relative, imu_6xN).
    time is converted from ms timestamps to seconds relative to first sample.
    """
    df = pd.read_csv(csv_path)
    for c in [WATCH_TIME_COL, *WATCH_IMU_COLS]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {csv_path.name}")

    ts_ms = df[WATCH_TIME_COL].astype("int64").to_numpy()
    ts    = (ts_ms - ts_ms[0]) / 1000.0
    imu   = df[WATCH_IMU_COLS].astype(float).to_numpy()
    return ts, imu


# =============================================================================
# RESAMPLING
# =============================================================================

def resample_to_fs(
    t_in: np.ndarray,
    x_in: np.ndarray,
    target_fs: float = TARGET_FS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample signal to uniform grid at target_fs Hz using linear interpolation."""
    t_in = np.asarray(t_in, float)
    x_in = np.asarray(x_in, float)
    if x_in.ndim == 1:
        x_in = x_in[:, None]

    if len(t_in) < 2:
        return np.array([]), np.empty((0, x_in.shape[1]))

    order = np.argsort(t_in)
    t_in, x_in = t_in[order], x_in[order]
    keep = np.concatenate([[True], np.diff(t_in) > 1e-9])
    t_in, x_in = t_in[keep], x_in[keep]

    dur = float(t_in[-1] - t_in[0])
    if dur <= 0:
        return np.array([]), np.empty((0, x_in.shape[1]))

    n_out = int(dur * target_fs) + 1
    t_out = np.arange(n_out, dtype=float) / target_fs
    t_rel = t_in - t_in[0]

    x_out = np.column_stack([
        np.interp(t_out, t_rel, x_in[:, k])
        for k in range(x_in.shape[1])
    ])
    return t_out, x_out[:, 0] if x_out.shape[1] == 1 else x_out


# =============================================================================
# SIGNAL PROXIES FOR CROSS-CORRELATION
# =============================================================================

def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    m, s = np.nanmean(x), np.nanstd(x)
    if not np.isfinite(m): m = 0.0
    if not np.isfinite(s) or s < 1e-12: s = 1.0
    return (x - m) / (s + 1e-8)


def force_proxy(fz: np.ndarray) -> np.ndarray:
    """Gradient of force, z-scored — used for cross-correlation alignment."""
    fz = np.asarray(fz, float)
    return zscore(np.gradient(fz) if len(fz) >= 3 else fz)


def imu_proxy(imu_6: np.ndarray) -> np.ndarray:
    """Weighted acceleration + gyroscope magnitude gradient, z-scored."""
    imu_6 = np.asarray(imu_6, float)
    if len(imu_6) < 3:
        return zscore(np.zeros(len(imu_6)))
    p = np.linalg.norm(imu_6[:, :3], axis=1) + 0.2 * np.linalg.norm(imu_6[:, 3:], axis=1)
    p = p - float(np.nanmean(p))
    return zscore(np.gradient(p))


# =============================================================================
# STANCE DETECTION
# =============================================================================

def detect_stances(
    fz_100: np.ndarray,
    activity: str,
    fs: float = TARGET_FS,
) -> List[Dict]:
    """
    Detect valid stance phases in the force signal.
    Returns list of dicts: onset, offset, peak_idx, peak_value, duration.
    """
    min_peak, contact_thr, min_stance_s, max_stance_s = activity_params(activity)
    fz = np.asarray(fz_100, float)
    contact = fz > contact_thr
    if not contact.any():
        return []

    idx = np.where(contact)[0]
    cuts = np.where(np.diff(idx) > 5)[0]
    segments = []
    s = 0
    for g in cuts:
        segments.append(idx[s:g + 1])
        s = g + 1
    segments.append(idx[s:])

    stances = []
    for seg in segments:
        onset, offset = int(seg[0]), int(seg[-1])
        dur = (offset - onset) / fs
        if not (min_stance_s <= dur <= max_stance_s):
            continue
        pk = onset + int(np.argmax(fz[onset:offset + 1]))
        pkval = float(fz[pk])
        if pkval >= min_peak:
            stances.append({"onset": onset, "offset": offset,
                            "peak_idx": pk, "peak_value": pkval, "duration": dur})
    return stances


# =============================================================================
# WRIST IMPACT DETECTION
# =============================================================================

def detect_impacts(wrist_imu_6: np.ndarray, fs: float = TARGET_FS) -> List[int]:
    """Detect wrist acceleration impact peaks used for stance × impact alignment."""
    acc_mag = np.linalg.norm(np.asarray(wrist_imu_6, float)[:, :3], axis=1)
    if len(acc_mag) < 10:
        return []
    thr = np.percentile(acc_mag, IMPACT_PERCENTILE)
    peaks, _ = find_peaks(
        acc_mag,
        height=thr,
        distance=int(IMPACT_MIN_DISTANCE_S * fs),
        prominence=IMPACT_PROMINENCE,
    )
    return peaks.tolist()


# =============================================================================
# OVERLAP (NO SYNTHETIC PADDING)
# =============================================================================

def overlap_slices(
    lag: int,
    n_force: int,
    n_imu: int,
) -> Tuple[int, int, int]:
    """
    Compute overlap indices between force and lag-shifted IMU arrays.

    lag > 0 : IMU delayed — force[lag : lag+L] aligns with imu[0 : L]
    lag < 0 : IMU advanced — force[0 : L] aligns with imu[-lag : -lag+L]

    Returns (force_start, imu_start, overlap_length).
    """
    if n_force <= 0 or n_imu <= 0:
        return 0, 0, 0
    if lag >= 0:
        f0, i0 = lag, 0
    else:
        f0, i0 = 0, -lag
    L = min(n_force - f0, n_imu - i0)
    return int(f0), int(i0), max(0, int(L))


# =============================================================================
# CROSS-CORRELATION LAG
# =============================================================================

def best_lag_xcorr(
    ref: np.ndarray,
    sig: np.ndarray,
    max_lag_s: float,
    fs: float = TARGET_FS,
) -> Tuple[int, float]:
    """Return (best_lag_samples, normalised_correlation_strength)."""
    ref, sig = zscore(ref), zscore(sig)
    n = min(len(ref), len(sig))
    if n < 20:
        return 0, 0.0
    ref, sig = ref[:n], sig[:n]

    corr = correlate(sig, ref, mode="full")
    lags = correlation_lags(len(sig), len(ref), mode="full")
    max_lag = int(max_lag_s * fs)
    mask = (lags >= -max_lag) & (lags <= max_lag)
    corr_m, lags_m = corr[mask], lags[mask]
    if len(corr_m) == 0:
        return 0, 0.0

    k = int(np.argmax(np.abs(corr_m)))
    strength = float(np.abs(corr_m[k]) / (np.linalg.norm(ref) * np.linalg.norm(sig) + 1e-8))
    return int(lags_m[k]), strength


# =============================================================================
# TRIAL DISCOVERY
# =============================================================================

def infer_activity(name: str) -> Optional[str]:
    n = name.lower()
    for prefix, label in [("walk","Walk"),("jog","Jog"),("run","Run"),
                           ("heel","Heel"),("drop","Drop")]:
        if n.startswith(prefix):
            return label
    return None


def extract_participant_label(folder_name: str) -> str:
    m = re.match(r"^P(\d+)", folder_name)
    return f"P{int(m.group(1)):02d}" if m else folder_name


@dataclass
class TrialPaths:
    participant: str
    activity:    str
    trial_num:   int
    force_fp:    Path
    waist_fp:    Path
    wrist_fp:    Path


def discover_trials(base_path: Path) -> List[TrialPaths]:
    """
    Match force plate files to Apple Watch files by trial number.

    Force plate: <activity><NN>_f_<plate>.csv  (trial_num = NN)
    Watch files: ..._ID_<N>_...                (matched when N == NN)
    """
    trials = []
    for folder in sorted(base_path.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("P"):
            continue

        participant = extract_participant_label(folder.name)
        files = [p for p in folder.iterdir() if p.suffix.lower() == ".csv"]

        force_files = [
            fp for fp in files
            if re.match(r"^(walk|jog|run|heel|drop)\d+_f_\d+\.csv$",
                        fp.name, re.IGNORECASE)
        ]

        id_to_watch: Dict[int, List[Path]] = defaultdict(list)
        for fp in files:
            if ("Trial" in fp.name) and (("Waist" in fp.name) or ("Wrist" in fp.name)):
                m = re.search(r"ID_(\d+)", fp.name)
                if m:
                    id_to_watch[int(m.group(1))].append(fp)

        for ffp in force_files:
            m = re.match(r"^(walk|jog|run|heel|drop)(\d+)_f_(\d+)\.csv$",
                         ffp.name, re.IGNORECASE)
            if not m:
                continue
            activity  = infer_activity(ffp.name)
            trial_num = int(m.group(2))
            candidates = id_to_watch.get(trial_num, [])
            if not candidates:
                continue

            # Prefer files with matching activity label; fall back to any
            waist = (next((c for c in candidates if "Waist" in c.name and activity in c.name), None)
                     or next((c for c in candidates if "Waist" in c.name), None))
            wrist = (next((c for c in candidates if "Wrist" in c.name and activity in c.name), None)
                     or next((c for c in candidates if "Wrist" in c.name), None))

            if waist and wrist:
                trials.append(TrialPaths(participant, activity, trial_num, ffp, waist, wrist))

    return trials


# =============================================================================
# ALIGN ONE TRIAL
# =============================================================================

def align_one_trial(
    tr: TrialPaths,
    target_fs: float,
    xcorr_max_lag: float,
    min_coverage: float,
) -> Dict:
    """
    Align force plate and IMU signals for one trial.
    Returns a dict with alignment metadata and saves the aligned CSV.
    Raises RuntimeError with a descriptive code on failure.
    """
    # Load and resample
    tF, fz_raw   = load_force_plate(tr.force_fp)
    tW, waist_raw = load_watch_imu(tr.waist_fp)
    tR, wrist_raw = load_watch_imu(tr.wrist_fp)

    _, fz    = resample_to_fs(tF, fz_raw,   target_fs)
    _, waist = resample_to_fs(tW, waist_raw, target_fs)
    _, wrist = resample_to_fs(tR, wrist_raw, target_fs)

    if any(len(x) < 50 for x in [fz, waist, wrist]):
        raise RuntimeError("too_short_after_resample")

    fz    = np.asarray(fz, float)
    waist = np.asarray(waist, float)
    wrist = np.asarray(wrist, float)

    min_peak, _, _, _ = activity_params(tr.activity)
    if float(np.max(fz)) < min_peak:
        raise RuntimeError(f"force_peak<{min_peak} (peak={np.max(fz):.1f})")

    stances = detect_stances(fz, tr.activity, fs=target_fs)
    if not stances:
        raise RuntimeError("no_valid_stance_in_force")

    impacts = detect_impacts(wrist, fs=target_fs)
    pre     = int(PRE_STANCE_S  * target_fs)
    post    = int(POST_STANCE_S * target_fs)
    candidates = []

    def _try_candidate(lag, method, corr_strength, st):
        f0, i0, L = overlap_slices(lag, len(fz), len(wrist))
        if L <= 0:
            return
        w0 = max(0, st["onset"] - pre)
        w1 = min(len(fz), st["offset"] + post + 1)
        if (w1 - w0) < int(0.20 * target_fs):
            return
        ov0 = max(w0, f0)
        ov1 = min(w1, f0 + L)
        if ov1 <= ov0:
            return
        coverage = float((ov1 - ov0) / max(1, w1 - w0))
        if coverage < min_coverage:
            return
        imu0 = i0 + (ov0 - f0)
        imu1 = imu0 + (ov1 - ov0)
        if imu1 > len(wrist) or imu1 > len(waist):
            return
        peak_win     = float(np.max(fz[ov0:ov1]))
        contact_pct  = float((fz[ov0:ov1] > 100.0).mean())
        valid        = (peak_win >= min_peak) and (contact_pct >= 0.03)
        reason       = "OK" if valid else (
            f"peak<{min_peak}" if peak_win < min_peak else "contact<3%"
        )
        candidates.append({
            "method":      method,
            "lag":         lag,
            "st_peak":     st["peak_value"],
            "ov0":         int(ov0), "ov1": int(ov1),
            "imu0":        int(imu0), "imu1": int(imu1),
            "coverage":    coverage,
            "corr":        float(corr_strength),
            "peak_win":    peak_win,
            "contact_pct": contact_pct,
            "valid":       valid,
            "reason":      reason,
        })

    # Strategy A: stance × impact
    if impacts:
        for st in stances:
            for imp in impacts:
                _try_candidate(int(st["peak_idx"] - imp), "impact", 0.0, st)

    # Strategy B: xcorr wrist fallback
    if not candidates:
        lag, strength = best_lag_xcorr(
            force_proxy(fz), imu_proxy(wrist), xcorr_max_lag, target_fs
        )
        _try_candidate(lag, "xcorr-wrist", strength,
                       max(stances, key=lambda s: s["peak_value"]))

    # Strategy C: xcorr waist fallback
    if not candidates:
        lag, strength = best_lag_xcorr(
            force_proxy(fz), imu_proxy(waist), xcorr_max_lag, target_fs
        )
        _try_candidate(lag, "xcorr-waist", strength,
                       max(stances, key=lambda s: s["peak_value"]))

    if not candidates:
        raise RuntimeError("no_candidates")

    valids = [c for c in candidates if c["valid"]]
    if not valids:
        best_bad = sorted(candidates,
                          key=lambda c: (c["coverage"], c["corr"], c["peak_win"]),
                          reverse=True)[0]
        raise RuntimeError(
            f"all_candidates_invalid(best={best_bad['method']} reason={best_bad['reason']})"
        )

    best = sorted(valids,
                  key=lambda c: (c["coverage"], c["corr"], c["peak_win"]),
                  reverse=True)[0]

    ov0, ov1   = best["ov0"],  best["ov1"]
    imu0, imu1 = best["imu0"], best["imu1"]
    fz_win    = fz[ov0:ov1]
    wrist_win = wrist[imu0:imu1]
    waist_win = waist[imu0:imu1]
    n = len(fz_win)

    if n < int(0.20 * target_fs):
        raise RuntimeError("final_window_too_short")

    t_common = np.arange(n, dtype=float) / target_fs

    out_df = pd.DataFrame({
        "time_s":    t_common.astype(np.float32),
        "force_z_N": fz_win.astype(np.float32),

        "waist_accX":  waist_win[:, 0].astype(np.float32),
        "waist_accY":  waist_win[:, 1].astype(np.float32),
        "waist_accZ":  waist_win[:, 2].astype(np.float32),
        "waist_gyroX": waist_win[:, 3].astype(np.float32),
        "waist_gyroY": waist_win[:, 4].astype(np.float32),
        "waist_gyroZ": waist_win[:, 5].astype(np.float32),

        "wrist_accX":  wrist_win[:, 0].astype(np.float32),
        "wrist_accY":  wrist_win[:, 1].astype(np.float32),
        "wrist_accZ":  wrist_win[:, 2].astype(np.float32),
        "wrist_gyroX": wrist_win[:, 3].astype(np.float32),
        "wrist_gyroY": wrist_win[:, 4].astype(np.float32),
        "wrist_gyroZ": wrist_win[:, 5].astype(np.float32),
    })
    out_df["waist_acc_mag"] = (
        out_df[["waist_accX","waist_accY","waist_accZ"]].pow(2).sum(axis=1).pow(0.5)
        .astype(np.float32)
    )
    out_df["wrist_acc_mag"] = (
        out_df[["wrist_accX","wrist_accY","wrist_accZ"]].pow(2).sum(axis=1).pow(0.5)
        .astype(np.float32)
    )

    return {
        "method":   best["method"],
        "lag":      best["lag"],
        "coverage": best["coverage"],
        "corr":     best["corr"],
        "peak_orig": float(np.max(fz)),
        "peak_win":  float(np.max(fz_win)),
        "n":         int(n),
        "df":        out_df,
    }


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid temporal alignment pipeline for Apple Watch + Force Plate GRF dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir",   required=True, type=Path,
                        help="Root dataset directory (contains P01-, P02- ... sub-folders).")
    parser.add_argument("--output_dir",    required=True, type=Path,
                        help="Directory to write aligned CSVs and alignment_log.csv.")
    parser.add_argument("--target_fs",     type=float, default=100.0,
                        help="Target sampling rate in Hz.")
    parser.add_argument("--xcorr_max_lag", type=float, default=1.25,
                        help="Maximum cross-correlation lag search window in seconds.")
    parser.add_argument("--min_coverage",  type=float, default=0.80,
                        help="Minimum IMU overlap fraction of the stance window.")
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("ALIGNMENT PIPELINE v3.6 — HYBRID STANCE×IMPACT + XCORR FALLBACK")
    print("=" * 70)
    print(f"Dataset : {args.dataset_dir}")
    print(f"Output  : {output_dir}")
    print(f"Fs      : {args.target_fs} Hz")
    print("=" * 70)

    trials = discover_trials(args.dataset_dir)
    print(f"Discovered {len(trials)} matched trials (force + waist + wrist).\n")

    if not trials:
        print("No matched trials found. Check dataset directory structure.")
        return

    stats   = Counter()
    methods = Counter()
    log_rows = []

    for tr in tqdm(trials, desc="Aligning"):
        try:
            r = align_one_trial(
                tr,
                target_fs=args.target_fs,
                xcorr_max_lag=args.xcorr_max_lag,
                min_coverage=args.min_coverage,
            )
            out_name = (
                f"{tr.participant}_{tr.activity}_ID{tr.trial_num}"
                f"_{tr.force_fp.stem}_aligned.csv"
            )
            out_path = output_dir / out_name
            r["df"].to_csv(out_path, index=False)

            lag_ms = round(r["lag"] / args.target_fs * 1000.0, 1)
            stats["ok"] += 1
            methods[r["method"]] += 1
            log_rows.append({
                "participant":              tr.participant,
                "activity":                 tr.activity,
                "trial_num":                tr.trial_num,
                "force_file":               tr.force_fp.name,
                "waist_file":               tr.waist_fp.name,
                "wrist_file":               tr.wrist_fp.name,
                "status":                   "ok",
                "estimated_lag_ms":         lag_ms,
                "peak_time_difference_ms":  lag_ms if r["method"] == "impact" else np.nan,
                "alignment_method":         r["method"],
                "overlap_coverage":         round(r["coverage"], 4),
                "cross_correlation_quality": round(r["corr"], 4),
                "peak_orig":                round(r["peak_orig"], 1),
                "peak_win":                 round(r["peak_win"], 1),
                "n":                        r["n"],
                "out":                      out_name,
            })

        except Exception as e:
            stats["fail"] += 1
            log_rows.append({
                "participant": tr.participant,
                "activity":    tr.activity,
                "trial_num":   tr.trial_num,
                "force_file":  tr.force_fp.name,
                "waist_file":  tr.waist_fp.name,
                "wrist_file":  tr.wrist_fp.name,
                "status":      "fail",
                "error":       str(e),
            })

    # Save alignment log with manuscript-specified column names
    log_df  = pd.DataFrame(log_rows)
    log_csv = output_dir / "alignment_log.csv"
    log_df.to_csv(log_csv, index=False)

    total = stats["ok"] + stats["fail"]
    print("\n" + "=" * 70)
    print("ALIGNMENT SUMMARY")
    print("=" * 70)
    print(f"Total: {total} | OK: {stats['ok']} | "
          f"Fail: {stats['fail']} | "
          f"Success: {100*stats['ok']/max(1,total):.1f}%")
    print("\nMethods used:")
    for k, v in methods.most_common():
        print(f"  {k}: {v} ({100*v/max(1,stats['ok']):.1f}%)")
    print(f"\nAlignment log : {log_csv}")
    print(f"Aligned CSVs  : {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
