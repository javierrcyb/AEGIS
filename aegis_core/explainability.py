"""
1. SHAP: shows which features influence the model's predictions.
2. Health Score: a score from 0 to 100 that represents the overall condition.
3. Confidence: measures how certain the model is about its prediction.
   This is different from the Health Score, which measures how abnormal
   the machine appears.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# SHAP
def compute_shap_values(model, X: pd.DataFrame, background_sample_size: int = 500):
    """
    Returns (shap_values_df, base_value). Detects the model type inside
    the Pipeline and selects the appropriate explainer. LinearExplainer
    is used for linear models, TreeExplainer for tree-based models,
    and KernelExplainer as a general fallback.
    """
    import shap

    # If the model is a Pipeline with a scaler and classifier, use scaled data.
    if hasattr(model, "named_steps"):
        scaler = model.named_steps.get("scaler")
        clf = model.named_steps["clf"]
        X_transformed = scaler.transform(X) if scaler is not None else X.values
    else:
        clf = model
        X_transformed = X.values

    background = X_transformed[np.random.choice(len(X_transformed), min(background_sample_size, len(X_transformed)), replace=False)]

    if isinstance(clf, LogisticRegression):
        explainer = shap.LinearExplainer(clf, background)
        shap_values = explainer.shap_values(X_transformed)
        base_value = explainer.expected_value
    elif hasattr(clf, "feature_importances_"):  # tree-based models (RF, LightGBM)
        explainer = shap.TreeExplainer(clf)
        raw = explainer.shap_values(X_transformed)
        # Some models return separate values for each class. Use the positive class.
        shap_values = raw[1] if isinstance(raw, list) else raw
        base_value = explainer.expected_value
        base_value = base_value[1] if isinstance(base_value, (list, np.ndarray)) and len(np.atleast_1d(base_value)) > 1 else base_value
    else:
        # Use KernelExplainer for models that do not match the cases above.
        explainer = shap.KernelExplainer(clf.predict_proba, background)
        raw = explainer.shap_values(X_transformed, nsamples=100)
        shap_values = raw[1] if isinstance(raw, list) else raw
        base_value = explainer.expected_value

    shap_df = pd.DataFrame(shap_values, columns=X.columns, index=X.index)
    return shap_df, base_value


def top_features_for_row(shap_df: pd.DataFrame, row_index, n: int = 3) -> list:
    """Returns the N features with the largest impact on a given row."""
    row = shap_df.loc[row_index].abs().sort_values(ascending=False)
    return list(row.head(n).index)


def global_feature_importance(shap_df: pd.DataFrame) -> pd.Series:
    """Calculates global feature importance using the mean absolute SHAP value."""
    return shap_df.abs().mean().sort_values(ascending=False)


# HEALTH SCORE

REGIME_STRESS = {"OFF": 0.0, "UNLOADED": 0.3, "LOADED": 0.6, "STARTING": 0.8, "UNKNOWN": 0.3}


def compute_degradation_trend(anomaly_score: pd.Series, window: int = 60) -> pd.Series:
    """
    Calculates the slope of the anomaly score over a rolling window
    and normalizes it to a 0-1 range.
    """
    def _slope(x):
        if len(x) < 2 or x.isna().all():
            return 0.0
        y = x.values
        t = np.arange(len(y))
        valid = ~np.isnan(y)
        if valid.sum() < 2:
            return 0.0
        coef = np.polyfit(t[valid], y[valid], 1)[0]
        return coef

    raw_slope = anomaly_score.rolling(window, min_periods=2).apply(_slope, raw=False)
    # Convert the slope to a 0-1 range.
    normalized = 1 / (1 + np.exp(-raw_slope * 50))
    return normalized.fillna(0.5)


def compute_health_score(
    anomaly_score: pd.Series,
    model_probability: pd.Series,
    regime: pd.Series,
    degradation_trend: pd.Series = None,
) -> pd.DataFrame:
    """
    Calculates the health score using the anomaly score, degradation trend,
    model probability, and operating regime.

    This is a heuristic score for the prototype and is not an industry-validated metric.
    """
    if degradation_trend is None:
        degradation_trend = compute_degradation_trend(anomaly_score)

    regime_stress = regime.map(REGIME_STRESS).fillna(0.3)

    composite_badness = (
        0.35 * anomaly_score.clip(0, 1)
        + 0.25 * degradation_trend.clip(0, 1)
        + 0.20 * model_probability.clip(0, 1)
        + 0.20 * regime_stress.clip(0, 1)
    )
    health_score = (100 * (1 - composite_badness)).clip(0, 100)

    return pd.DataFrame({
        "health_score": health_score,
        "degradation_trend": degradation_trend,
        "regime_stress": regime_stress,
        "composite_badness": composite_badness,
    })


# CONFIDENCE

def compute_confidence(model_probability: pd.Series) -> pd.Series:
    """
    Confidence is based on how far the model probability is from 0.5.
    A probability of 0.5 means maximum uncertainty, while 0 or 1
    means higher confidence.
    """
    return (2 * (model_probability - 0.5).abs()).clip(0, 1)


def confidence_label(confidence: float) -> str:
    if confidence >= 0.6:
        return "HIGH"
    elif confidence >= 0.3:
        return "MEDIUM"
    return "LOW"


if __name__ == "__main__":
    print("This module is imported from scripts/compute_shap_and_health.py, not run directly.")