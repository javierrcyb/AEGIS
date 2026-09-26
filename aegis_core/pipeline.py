"""
Runs the unsupervised analysis (Layer 1) on any dataset with
a timestamp and numerical columns. If a trained model and its
feature columns are provided, it also adds the supervised probability (Layer 2).
If there is no model, the health score is calculated only from
unsupervised detection.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from aegis_core.data_profiling import profile_dataset, DatasetProfile
from aegis_core.time_series_engine import (
    build_features,
    detect_regime_rule_based,
    detect_regime_kmeans,
    METROPT3_CURRENT_THRESHOLDS,
    compute_regime_aware_zscore,
    compute_isolation_forest_score,
    combine_anomaly_score,
)
from aegis_core.ml_engine import build_feature_matrix
from aegis_core.explainability import REGIME_STRESS, compute_degradation_trend, confidence_label


@dataclass
class AnalysisResult:
    df: pd.DataFrame
    profile: DatasetProfile
    has_supervised_model: bool
    value_columns: list
    warnings: list = field(default_factory=list)


def auto_detect_timestamp_column(df: pd.DataFrame) -> str | None:
    """
    First looks at the column name and, if there is no clear match,
    tries to convert the columns to dates and finds the one that works best.
    """
    keywords = ["time", "date", "fecha", "hora", "timestamp", "datetime"]
    keyword_hits = [c for c in df.columns if any(k in c.lower() for k in keywords)]

    candidates = keyword_hits + [c for c in df.columns if c not in keyword_hits]
    for c in candidates:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.95:
                return c
        except Exception:
            continue
    return None


def analyze_dataset(
    raw_df: pd.DataFrame,
    timestamp_col: str = None,
    value_columns: list = None,
    regime_basis_column: str = None,
    model=None,
    model_feature_columns: list = None,
) -> AnalysisResult:
    """
    Entry point for uploading a CSV and analyzing it.

    - timestamp_col: if None, it is detected automatically.
    - value_columns: numerical columns used as signals. If None, all numerical
      columns found during profiling are used.
    - regime_basis_column: column used for regime detection.
      If it is 'current', the MetroPT-3 rules are used. Otherwise,
      K-Means is used.
    - model / model_feature_columns: if provided, the supervised probability
      is added. Otherwise, the analysis only uses Layer 1.
    """
    warnings = []
    df = raw_df.copy()

    # --- 1. Timestamp ---
    if timestamp_col is None:
        timestamp_col = auto_detect_timestamp_column(df)
    if timestamp_col is None or timestamp_col not in df.columns:
        raise ValueError(
            "Unable to automatically detect a timestamp column. "
            "Manually specify which column contains the date and time."
        )
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col).sort_index()

    # --- 2. Generic profiling ---
    profile = profile_dataset(df)

    # --- 3. Columns to analyze ---
    if value_columns is None:
        value_columns = profile.numeric_columns
    value_columns = [c for c in value_columns if c in df.columns]
    if not value_columns:
        raise ValueError("There are no usable numeric columns in this dataset.")

    # --- 4. Regime detection ---
    n_unique_rows = df[value_columns].dropna().shape[0]
    if regime_basis_column == "current" and "current" in df.columns:
        df["regime"] = detect_regime_rule_based(df, "current", METROPT3_CURRENT_THRESHOLDS)
    elif n_unique_rows >= 10:
        basis_cols = [regime_basis_column] if regime_basis_column in df.columns else value_columns[: min(3, len(value_columns))]
        try:
            df["regime"] = detect_regime_kmeans(df, basis_cols, n_clusters=min(3, n_unique_rows))
            warnings.append(
                "No known pattern was recognized for regime detection; "
                f"Generic K-Means was used on {basis_cols}. The names 'REGIME_0/1/2'"
                "They are cluster labels, not operational states with their own meaning."
            )
        except Exception as e:
            df["regime"] = "UNKNOWN"
            warnings.append(f"Mode detection failed ({e}); a single ‘UNKNOWN’ mode was used.")
    else:
        df["regime"] = "UNKNOWN"
        warnings.append("Very few rows for regime detection; a single ‘UNKNOWN’ regime was used.")

    # --- 5. Feature engineering ---
    df_feat = build_features(df[value_columns], value_columns)
    df_feat["regime"] = df["regime"]

    # --- 6. Anomaly detection ---
    df_feat = compute_regime_aware_zscore(df_feat, value_columns, "regime")
    df_feat["iforest_score"] = compute_isolation_forest_score(df_feat, value_columns, contamination=0.02)
    zscore_cols = [f"{c}_zscore_regime" for c in value_columns]
    df_feat["anomaly_score"] = combine_anomaly_score(df_feat, zscore_cols, "iforest_score")

    # --- 7 and 8: Optional Layer 2 + Health Score ---
    df_feat, has_supervised_model, layer_warnings = apply_supervised_layer(
        df_feat, model=model, model_feature_columns=model_feature_columns
    )
    warnings.extend(layer_warnings)

    return AnalysisResult(
        df=df_feat,
        profile=profile,
        has_supervised_model=has_supervised_model,
        value_columns=value_columns,
        warnings=warnings,
    )


def apply_supervised_layer(df_feat: pd.DataFrame, model=None, model_feature_columns=None):
    """
    Applies Layer 2 to a dataframe that already has the features, regime,
    and anomaly_score calculated.

    It can also be used after training a model, without having to
    run the whole Layer 1 analysis again.

    Returns (updated_df_feat, has_supervised_model, warnings).
    """
    df_feat = df_feat.copy()

    # Remove old results first if this was already analyzed before.
    stale_output_cols = [
        "model_probability", "confidence", "confidence_label",
        "health_score", "degradation_trend", "regime_stress",
    ]
    df_feat = df_feat.drop(columns=[c for c in stale_output_cols if c in df_feat.columns])

    warnings = []
    has_supervised_model = model is not None and model_feature_columns is not None

    if has_supervised_model:
        try:
            X_raw = build_feature_matrix(df_feat, exclude_cols=[])
            incomplete_rows = X_raw.isna().any(axis=1)
            X_filled = X_raw.fillna(0).reindex(columns=model_feature_columns, fill_value=0)
            probs = model.predict_proba(X_filled)[:, 1]
            probs[incomplete_rows.values] = np.nan
            df_feat["model_probability"] = probs

            if incomplete_rows.any():
                warnings.append(
                    f"{incomplete_rows.sum()} rows (usually at the beginning of the series, due to "
                    "rolling/lag) do not have enough history to calculate model_probability."
                )

        except Exception as e:
            has_supervised_model = False
            df_feat["model_probability"] = np.nan
            warnings.append(f"The supervised model could not be applied to this dataset: {e}")

    if not has_supervised_model:
        if "model_probability" not in df_feat.columns:
            df_feat["model_probability"] = np.nan

        warnings.append(
            "There is no trained supervised model for this dataset — the health score "
            "is calculated ONLY using unsupervised detection (anomaly score + regime). "
            "This is expected for datasets without labeled failure history."
        )

    degradation_trend = compute_degradation_trend(df_feat["anomaly_score"])
    regime_stress = df_feat["regime"].map(REGIME_STRESS).fillna(0.3)

    if has_supervised_model:
        composite = (
            0.35 * df_feat["anomaly_score"].clip(0, 1)
            + 0.25 * degradation_trend.clip(0, 1)
            + 0.20 * df_feat["model_probability"].fillna(0).clip(0, 1)
            + 0.20 * regime_stress.clip(0, 1)
        )
        df_feat["confidence"] = (2 * (df_feat["model_probability"] - 0.5).abs()).clip(0, 1)
    else:
        composite = (
            0.55 * df_feat["anomaly_score"].clip(0, 1)
            + 0.30 * degradation_trend.clip(0, 1)
            + 0.15 * regime_stress.clip(0, 1)
        )
        df_feat["confidence"] = np.nan

    df_feat["degradation_trend"] = degradation_trend
    df_feat["regime_stress"] = regime_stress
    df_feat["health_score"] = (100 * (1 - composite)).clip(0, 100)
    df_feat["confidence_label"] = df_feat["confidence"].apply(
        lambda c: confidence_label(c) if pd.notna(c) else "N/A"
    )

    return df_feat, has_supervised_model, warnings