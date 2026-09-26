import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
Unlike scripts/evaluate_leave_one_out.py (which measures generalization using
the “leave-one-event-out” method), this script trains the model THAT WILL BE USED
in the dashboard, using all available events and the
“core” feature set (smaller, generalizes better, and is more interpretable for SHAP).

Uso:
    python scripts/train_classifier.py --input data/processed/metropt3_with_features.csv
"""

import argparse
import json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aegis_core.ml_engine import build_binary_target, build_feature_matrix

CORE_FEATURES = [
    "current",
    "temperature",
    "pressure_main",
    "pressure_panel",
    "pressure_reservoir",
    "dv_pressure",
    "h1",
    "regime",
    "anomaly_score",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/metropt3_with_features.csv")
    parser.add_argument("--output-model", default="models/aegis_classifier.joblib")
    parser.add_argument("--output-columns", default="models/aegis_feature_columns.json")
    args = parser.parse_args()

    df = pd.read_csv(args.input, index_col="timestamp", parse_dates=True).dropna()
    core_cols = [c for c in CORE_FEATURES if c in df.columns]
    print(f"Using columns: {core_cols}")

    y = build_binary_target(df, "health_label", ["FAULT", "DEGRADING"])
    X = build_feature_matrix(df[core_cols + ["health_label"]], exclude_cols=["health_label"])

    print(f"Training with {len(X):,} rows, {y.sum():,} positive (all events) ...")

    # Chosen for having the best average
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, C=0.1)),
    ])
    model.fit(X, y)

    from pathlib import Path

    Path(args.output_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_model)

    with open(args.output_columns, "w") as f:
        json.dump(list(X.columns), f, indent=2)

    print(f"Modelo saved in: {args.output_model}")
    print(f"Columns saved in: {args.output_columns}")
    print(
        "\nMethodological note for the report: This model was trained using ALL "
        "available historical events (it is not the same model that was evaluated in "
        "scripts/evaluate_leave_one_out.py). Its expected performance when faced with a new, short failure "
        "is limited, as documented in the leave-one-event-out evaluation."
    )


if __name__ == "__main__":
    main()
