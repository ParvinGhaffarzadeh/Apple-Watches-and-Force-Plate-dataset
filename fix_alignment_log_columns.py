#!/usr/bin/env python3
"""
fix_alignment_log_columns.py
============================
One-off script to rename alignment_log.csv columns to match the
column names promised in the manuscript.

Run once before uploading alignment_log.csv to Zenodo.

Usage
-----
  python fix_alignment_log_columns.py \
      --input  /path/to/alignment_log.csv \
      --output ./alignment_log_final.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


COLUMN_RENAMES = {
    "method":   "alignment_method",
    "coverage": "overlap_coverage",
    "corr":     "cross_correlation_quality",
}

# Sampling rate used during alignment (Hz)
TARGET_FS = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename alignment log columns to match manuscript specification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",  required=True, type=Path,
                        help="Path to raw alignment_log.csv from alignment_pipeline.py")
    parser.add_argument("--output", required=True, type=Path,
                        help="Path for the corrected output CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Reading : {args.input}")
    df = pd.read_csv(args.input)
    print(f"Rows    : {len(df)}  |  Columns: {df.columns.tolist()}")

    # 1. Convert lag (samples) -> estimated_lag_ms
    if "lag" in df.columns:
        df["estimated_lag_ms"] = (df["lag"] / TARGET_FS * 1000.0).round(1)
        df = df.drop(columns=["lag"])
        print("  lag (samples) -> estimated_lag_ms (ms)  [÷100 × 1000]")

    # 2. Rename remaining columns
    df = df.rename(columns=COLUMN_RENAMES)
    for old, new in COLUMN_RENAMES.items():
        if new in df.columns:
            print(f"  {old} -> {new}")

    # 3. Add peak_time_difference_ms
    #    Defined as: time of force peak within window minus time of wrist
    #    impact peak. For trials aligned by impact method this equals
    #    estimated_lag_ms by construction; for xcorr fallback trials it is
    #    not directly available, so we set NaN and flag the method.
    if "peak_time_difference_ms" not in df.columns:
        if "alignment_method" in df.columns:
            df["peak_time_difference_ms"] = np.where(
                df["alignment_method"] == "impact",
                df["estimated_lag_ms"],   # lag == peak-time difference for impact method
                np.nan                    # not available for xcorr fallback
            )
        else:
            df["peak_time_difference_ms"] = np.nan
        print("  peak_time_difference_ms added")

    # 4. Reorder columns: metadata first, then the four promised columns, then rest
    priority = [
        "participant", "activity", "trial_num",
        "force_file", "waist_file", "wrist_file",
        "status",
        "estimated_lag_ms",
        "peak_time_difference_ms",
        "alignment_method",
        "overlap_coverage",
        "cross_correlation_quality",
    ]
    rest = [c for c in df.columns if c not in priority]
    final_cols = [c for c in priority if c in df.columns] + rest
    df = df[final_cols]

    # 5. Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved   : {args.output}")
    print(f"Columns : {df.columns.tolist()}")

    # 6. Quick summary
    if "status" in df.columns:
        print(f"\nStatus counts:\n{df['status'].value_counts().to_string()}")
    if "alignment_method" in df.columns:
        ok = df[df["status"] == "ok"] if "status" in df.columns else df
        print(f"\nAlignment method counts (OK trials):\n{ok['alignment_method'].value_counts().to_string()}")
    if "estimated_lag_ms" in df.columns:
        ok = df[df["status"] == "ok"] if "status" in df.columns else df
        print(f"\nestimated_lag_ms (OK trials):")
        print(f"  mean={ok['estimated_lag_ms'].mean():.1f} ms  "
              f"std={ok['estimated_lag_ms'].std():.1f} ms  "
              f"min={ok['estimated_lag_ms'].min():.1f}  "
              f"max={ok['estimated_lag_ms'].max():.1f}")


if __name__ == "__main__":
    main()
