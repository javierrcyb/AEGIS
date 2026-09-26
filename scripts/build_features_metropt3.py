import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
Usage:
    python scripts/build_features_metropt3.py --input data/processed/metropt3_labeled.csv
"""

import argparse

import pandas as pd

from aegis_core.time_series_engine import (
    build_features,
    detect_regime_rule_based,
    METROPT3_CURRENT_THRESHOLDS,
    compute_regime_aware_zscore,
    compute_isolation_forest_score,
    combine_anomaly_score,
    detect_change_points,
)

# Relevant columns in MetroPT-3 for features and anomaly detection.

VALUE_COLUMNS = [
    "current",
    "temperature",
    "pressure_main",
    "pressure_panel",
    "pressure_reservoir",
    "dv_pressure",
    "h1",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/metropt3_labeled.csv")
    parser.add_argument("--output", default="data/processed/metropt3_with_features.csv")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input, index_col="timestamp", parse_dates=True)
    df = df.drop(columns=[c for c in df.columns if "unnamed" in c.lower()], errors="ignore")
    print(f"  -> {len(df):,} rows")

    print("\n1. Regime detection (based on motor current) ...")
    df["regime"] = detect_regime_rule_based(df, "current", METROPT3_CURRENT_THRESHOLDS)
    print(df["regime"].value_counts())

    print("\n2. Feature engineering ...")
    df_feat = build_features(df[VALUE_COLUMNS], VALUE_COLUMNS)
    df_feat["regime"] = df["regime"]
    df_feat["health_label"] = df["health_label"]
    print(f"  -> {len(df_feat.columns)} total columns (features + originals)")

    print("\n3. Anomaly detection (z-score by regime + Isolation Forest) ...")
    df_z = compute_regime_aware_zscore(df_feat, VALUE_COLUMNS, "regime")
    df_z["iforest_score"] = compute_isolation_forest_score(df_z, VALUE_COLUMNS, contamination=0.02)
    zscore_cols = [f"{c}_zscore_regime" for c in VALUE_COLUMNS]
    df_z["anomaly_score"] = combine_anomaly_score(df_z, zscore_cols, "iforest_score")

    print("\n   Anomaly score by actual health_label (cross-validation):")
    print(df_z.groupby("health_label")["anomaly_score"].mean().sort_values(ascending=False))

    print("\n4. Change-point detection (regime-dependent, to avoid confusion")
    print("   normal on/off cycles with actual degradation) ...")
    baseline_dist = df["health_label"].value_counts(normalize=True) * 100
    print("\n   Base distribution of labels (for comparison with change points):")
    print(baseline_dist.round(2))

    TARGET_REGIME = "LOADED"  # el estado operativo más comparable entre sí
    regime_mask = df["regime"] == TARGET_REGIME
    print(f"\n   Rows in regime '{TARGET_REGIME}': {regime_mask.sum():,} of {len(df):,}")

    for col in ["current", "temperature"]:
        # Only values while the machine is in the same operating mode.
        # This ignores power-on/power-off transitions, which do not constitute degradation.
        series_in_regime = df.loc[regime_mask, col].resample("15min").mean().dropna()

        cps = detect_change_points(series_in_regime, method="auto", penalty=8)
        print(f"\n  {col} ({TARGET_REGIME} regime only): {len(cps)} change points detected")
        if cps:
            cp_timestamps = [series_in_regime.index[i] for i in cps]
            print(f"    first timestamps: {[str(t) for t in cp_timestamps[:5]]}")

            cp_labels = df["health_label"].reindex(cp_timestamps, method="nearest")
            cp_dist = cp_labels.value_counts(normalize=True) * 100
            print(f"    Distribution of labels at the change points in {col}:")
            print(cp_dist.round(2))

    print(f"\nSaving dataset with features and anomaly scores to {args.output} ...")
    df_z.to_csv(args.output)
    print("Ready.")


if __name__ == "__main__":
    main()
