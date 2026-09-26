import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
Instead of using a single train/test split, split the timeline into 4 "eras",
one for each real failure event. Train on 3 eras and test on the remaining
one, rotating through all 4.
This helps check whether the results are mainly caused by the July split,
or if the problem is more structural, especially since there are very few
real failure events to learn from.
Also compare two feature sets:
A) Full feature set: 91 columns, including all rolling and lag features.
B) Reduced feature set: only the core signals, anomaly_score, and regime.
The goal is to see if the full feature set is overfitting because there are
too many correlated features for such a small number of positive events.
"""

import pandas as pd
import numpy as np

from aegis_core.ml_engine import (
    build_binary_target,
    build_feature_matrix,
    train_and_evaluate_all,
)

FAILURE_STARTS = [
    "2020-04-18 00:00",
    "2020-05-29 23:30",
    "2020-06-05 10:00",
    "2020-07-15 14:30",
]

CORE_FEATURES = [
    "current",
    "temperature",
    "pressure_main",
    "pressure_panel",
    "pressure_reservoir",
    "dv_pressure",
    "h1",
    "regime",
]
if True:
    CORE_FEATURES_WITH_ANOMALY = CORE_FEATURES + ["anomaly_score"]


def build_eras(index: pd.DatetimeIndex) -> pd.Series:
    starts = [pd.Timestamp(s) for s in FAILURE_STARTS]
    boundaries = [starts[i] + (starts[i + 1] - starts[i]) / 2 for i in range(len(starts) - 1)]

    era = pd.Series(0, index=index)
    for i, b in enumerate(boundaries):
        era[index >= b] = i + 1
    return era


def run_leave_one_out(df: pd.DataFrame, feature_cols: list, label: str) -> pd.DataFrame:
    era = build_eras(df.index)
    rows = []

    for held_out_era in sorted(era.unique()):
        train_mask = era != held_out_era
        test_mask = era == held_out_era

        train, test = df[train_mask], df[test_mask]
        y_train = build_binary_target(train, "health_label", ["FAULT", "DEGRADING"])
        y_test = build_binary_target(test, "health_label", ["FAULT", "DEGRADING"])

        if y_train.nunique() < 2 or y_test.sum() == 0:
            print(f"The {held_out_era} era was skipped (not enough positive cases)")
            continue

        X_train = build_feature_matrix(train[feature_cols + ["health_label"]], exclude_cols=["health_label"])
        X_test = build_feature_matrix(test[feature_cols + ["health_label"]], exclude_cols=["health_label"])
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        try:
            results = train_and_evaluate_all(X_train, y_train, X_test, y_test)
        except ValueError as e:
            print(f"It was a {held_out_era} error: {e}")
            continue

        for r in results:
            rows.append(
                {
                    "feature_set": label,
                    "held_out_era": held_out_era,
                    "model": r.name,
                    "f1": r.f1,
                    "precision": r.precision,
                    "recall": r.recall,
                    "roc_auc": r.roc_auc,
                    "pr_auc": r.pr_auc,
                }
            )
    return pd.DataFrame(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/metropt3_with_features.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input, index_col="timestamp", parse_dates=True)
    df = df.dropna()

    era = build_eras(df.index)
    print("Distribution of rows by era (0=event1, 1=event2, 2=event3, 3=event4):")
    print(era.value_counts().sort_index())

    full_features = [c for c in df.columns if c not in ("health_label",)]

    print("\n" + "=" * 70)
    print("EXPERIMENT A: Complete Feature Set")
    print("=" * 70)
    results_full = run_leave_one_out(df, full_features, "full")
    if not results_full.empty:
        print(results_full.groupby("model")[["f1", "roc_auc", "pr_auc"]].mean().round(3))

    print("\n" + "=" * 70)
    print("EXPERIMENT B: Reduced feature set (core + anomaly_score + regime)")
    print("=" * 70)
    core_cols = [c for c in CORE_FEATURES_WITH_ANOMALY if c in df.columns]
    print(f"Used Columns: {core_cols}")
    results_core = run_leave_one_out(df, core_cols, "core")
    if not results_core.empty:
        print(results_core.groupby("model")[["f1", "roc_auc", "pr_auc"]].mean().round(3))

    print("\n" + "=" * 70)
    print("BREAKDOWN BY ERA")
    print("=" * 70)
    all_results = pd.concat([results_full, results_core], ignore_index=True)
    if not all_results.empty:
        print(
            all_results.pivot_table(
                index=["held_out_era", "model"], columns="feature_set", values="pr_auc"
            ).round(3)
        )


if __name__ == "__main__":
    main()
