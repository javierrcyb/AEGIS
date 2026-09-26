import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
Usage:
    python scripts/demo_reasoning_cli.py --timestamp "2020-06-06 12:00:00" --model llama3.2:3b
    python scripts/demo_reasoning_cli.py --auto-fault --model llama3.2:3b   # choose automatically
"""

import argparse
import json

import joblib
import pandas as pd

from aegis_core.time_series_engine import detect_change_points
from aegis_core.ml_engine import build_feature_matrix
from aegis_core.explainability import compute_shap_values, top_features_for_row
from aegis_core.evidence_engine import build_evidence, flag_recent_change_points
from aegis_core.reasoning_agent import aegis_reasoning, DEFAULT_MODEL


CORE_COLS = ["current", "temperature", "pressure_main", "pressure_panel",
             "pressure_reservoir", "dv_pressure", "h1", "regime", "anomaly_score"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/metropt3_final.csv")
    parser.add_argument("--model-path", default="models/aegis_classifier.joblib")
    parser.add_argument("--columns-path", default="models/aegis_feature_columns.json")
    parser.add_argument("--timestamp", default=None, help='Ej: "2020-06-06 12:00:00"')
    parser.add_argument("--auto-fault", action="store_true", help="Elige el timestamp con mayor health_score de riesgo automáticamente")
    parser.add_argument("--llm-model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    df = pd.read_csv(args.input, index_col="timestamp", parse_dates=True).dropna()

    if args.auto_fault:
        target_ts = df[df["health_label"] == "FAULT"]["anomaly_score"].idxmax()
        print(f"Automatically selected timestamp (highest anomaly_score within FAULT): {target_ts}")
    elif args.timestamp:
        target_ts = pd.Timestamp(args.timestamp)
    else:
        target_ts = df["anomaly_score"].idxmax()
        print(f"Without --timestamp or --auto-fault: using the highest anomaly_score from the entire dataset: {target_ts}")

    print(f"\nSelected row: {target_ts}")
    row = df.loc[target_ts]
    print(row[["health_label", "health_score", "anomaly_score", "regime", "model_probability", "confidence_label"]])

    print("\nCalculating SHAP for this row (and a small window around it, for context) ...")
    model = joblib.load(args.model_path)
    with open(args.columns_path) as f:
        feature_columns = json.load(f)

    window = df.loc[target_ts - pd.Timedelta(hours=2) : target_ts + pd.Timedelta(hours=2)]
    X_window = build_feature_matrix(window[CORE_COLS], exclude_cols=[])
    X_window = X_window.reindex(columns=feature_columns, fill_value=0)

    shap_df, _ = compute_shap_values(model, X_window, background_sample_size=200)
    top_feats = top_features_for_row(shap_df, target_ts, n=3)
    print(f"Top features (SHAP) for this row: {top_feats}")

    print("\nDetecting nearby change points (LOADED regime, 30-day window around)...")
    context = df.loc[target_ts - pd.Timedelta(days=15) : target_ts + pd.Timedelta(days=15)]
    loaded_mask = context["regime"] == "LOADED"
    series_loaded = context.loc[loaded_mask, "temperature"].resample("15min").mean().dropna()
    cps = detect_change_points(series_loaded, method="auto", penalty=8)
    cp_timestamps = [series_loaded.index[i] for i in cps]

    cp_flags = flag_recent_change_points(df.index, cp_timestamps, window_hours=6)
    change_point_nearby = bool(cp_flags.get(target_ts, False))
    print(f"Nearest change point: {change_point_nearby}")

    print("\nBuilding Structured Evidence ...")
    evidence = build_evidence(row, top_feats, change_point_nearby)
    print(json.dumps(evidence.to_dict(), indent=2, ensure_ascii=False))

    print(f"\nCalling the local LLM ({args.llm_model}) to generate the diagnosis ...")
    diagnosis = aegis_reasoning(evidence, model=args.llm_model)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC AEGIS")
    print("=" * 60)
    print(json.dumps(diagnosis, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
