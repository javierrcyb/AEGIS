import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
Usage:
    python scripts/compute_shap_and_health.py --input data/processed/metropt3_with_features.csv
"""

import argparse
import json

import joblib
import pandas as pd

from aegis_core.ml_engine import build_feature_matrix
from aegis_core.explainability import (
    compute_shap_values,
    global_feature_importance,
    top_features_for_row,
    compute_health_score,
    compute_confidence,
    confidence_label,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/metropt3_with_features.csv")
    parser.add_argument("--model", default="models/aegis_classifier.joblib")
    parser.add_argument("--columns", default="models/aegis_feature_columns.json")
    parser.add_argument("--sample-size", type=int, default=20000, help="SHAP sobre una muestra, no todo el dataset")
    parser.add_argument("--output", default="data/processed/metropt3_final.csv")
    args = parser.parse_args()

    print("Loading model and dataset ...")
    model = joblib.load(args.model)
    with open(args.columns) as f:
        feature_columns_raw = json.load(f)

    df = pd.read_csv(args.input, index_col="timestamp", parse_dates=True).dropna()

    core_cols = ["current", "temperature", "pressure_main", "pressure_panel",
                 "pressure_reservoir", "dv_pressure", "h1", "regime", "anomaly_score"]
    core_cols = [c for c in core_cols if c in df.columns]
    X_full = build_feature_matrix(df[core_cols], exclude_cols=[])
    X_full = X_full.reindex(columns=feature_columns_raw, fill_value=0)

    print("Calculating the model's probability across the entire dataset ...")
    model_probability = pd.Series(model.predict_proba(X_full)[:, 1], index=df.index)
    df["model_probability"] = model_probability

    print(f"\nCalculating SHAP on a sample of {args.sample_size:,} rows ...")
    sample_idx = X_full.sample(min(args.sample_size, len(X_full)), random_state=42).index
    X_sample = X_full.loc[sample_idx]
    shap_df, base_value = compute_shap_values(model, X_sample)

    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE GLOBAL (SHAP)")
    print("=" * 50)
    importance = global_feature_importance(shap_df)
    print(importance)

    print("\nExample: top features for the row with the highest probability of failure in the sample")
    top_row = model_probability.loc[sample_idx].idxmax()
    print(f"  Timestamp: {top_row}, probability: {model_probability[top_row]:.3f}")
    print(f"  Top features: {top_features_for_row(shap_df, top_row)}")

    print("\n" + "=" * 50)
    print("HEALTH SCORE")
    print("=" * 50)
    health_df = compute_health_score(
        anomaly_score=df["anomaly_score"],
        model_probability=model_probability,
        regime=df["regime"],
    )
    df = df.join(health_df)
    print("\nAverage health score by actual health_label (cross-validation):")
    print(df.groupby("health_label")["health_score"].mean().sort_values())

    print("\n" + "=" * 50)
    print("CONFIDENCE")
    print("=" * 50)
    df["confidence"] = compute_confidence(model_probability)
    df["confidence_label"] = df["confidence"].apply(confidence_label)
    print("\nConfidence label distribution:")
    print(df["confidence_label"].value_counts())
    print("\nAverage confidence by actual health_label:")
    print(df.groupby("health_label")["confidence"].mean().sort_values())

    print(f"\nSaving the final dataset to {args.output} ...")
    df.to_csv(args.output)


if __name__ == "__main__":
    main()
