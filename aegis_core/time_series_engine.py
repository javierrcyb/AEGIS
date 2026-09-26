"""
Feature engineering, regime detection, regime-aware anomaly detection,
and change-point detection.

The code is organized in layers:
- Feature functions are fully generic and work with any numerical columns.
- Regime detection has a rule-based mode for dataset-specific settings
  (for example, MetroPT-3 uses 'current') and a generic K-Means fallback.
- Anomaly detection is regime-aware: each point is compared against the
  distribution of its own regime instead of the global distribution.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def add_rolling_features(df: pd.DataFrame, columns: list, windows: list = [5, 15, 60]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        for w in windows:
            df[f"{col}_roll_mean_{w}"] = df[col].rolling(w, min_periods=1).mean()
            df[f"{col}_roll_std_{w}"] = df[col].rolling(w, min_periods=1).std()
    return df


def add_rate_of_change(df: pd.DataFrame, columns: list, periods: list = [1, 5]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        for p in periods:
            df[f"{col}_roc_{p}"] = df[col].diff(p)
    return df


def add_ewma(df: pd.DataFrame, columns: list, span: int = 30) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        df[f"{col}_ewma_{span}"] = df[col].ewm(span=span, adjust=False).mean()
    return df


def add_lag_features(df: pd.DataFrame, columns: list, lags: list = [1, 5, 10, 20]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def build_features(df: pd.DataFrame, value_columns: list) -> pd.DataFrame:
    """Runs the full feature engineering pipeline on the selected columns."""
    out = df.copy()
    out = add_rolling_features(out, value_columns, windows=[5, 15, 60])
    out = add_rate_of_change(out, value_columns, periods=[1, 5])
    out = add_ewma(out, value_columns, span=30)
    out = add_lag_features(out, value_columns, lags=[1, 5, 10])
    return out


# ---------------------------------------------------------------------------
# 2. REGIME DETECTION
# ---------------------------------------------------------------------------

# MetroPT-3 settings based on the current sensor documentation:
# 0A off / 4A unloaded / 7A loaded / 9A starting
METROPT3_CURRENT_THRESHOLDS = {
    "OFF": (-np.inf, 1.0),
    "UNLOADED": (1.0, 5.5),
    "LOADED": (5.5, 8.0),
    "STARTING": (8.0, np.inf),
}


def detect_regime_rule_based(df: pd.DataFrame, column: str, thresholds: dict) -> pd.Series:
    """Detects the regime using simple rules on one column."""
    regime = pd.Series("UNKNOWN", index=df.index)
    for label, (low, high) in thresholds.items():
        mask = (df[column] >= low) & (df[column] < high)
        regime[mask] = label
    return regime


def detect_regime_kmeans(df: pd.DataFrame, columns: list, n_clusters: int = 3) -> pd.Series:
    """
    Generic fallback when the machine thresholds are not known.
    Groups similar behavior and assigns generic names like REGIME_0, REGIME_1, ...
    """
    data = df[columns].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled)

    regime = pd.Series("UNKNOWN", index=df.index)
    regime.loc[data.index] = [f"REGIME_{c}" for c in cluster_labels]
    return regime


# ---------------------------------------------------------------------------
# 3. ANOMALY DETECTION
# ---------------------------------------------------------------------------

def compute_regime_aware_zscore(df: pd.DataFrame, columns: list, regime_col: str) -> pd.DataFrame:
    """
    Calculates the z-score for each column using the mean and std of its
    own regime instead of the global mean and std. This prevents a normal
    'LOADED' state from being marked as anomalous just because it differs from 'OFF'.
    """
    df = df.copy()
    for col in columns:
        grouped = df.groupby(regime_col)[col]
        regime_mean = grouped.transform("mean")
        regime_std = grouped.transform("std").replace(0, np.nan)
        df[f"{col}_zscore_regime"] = (df[col] - regime_mean) / regime_std
    return df


def compute_isolation_forest_score(
    df: pd.DataFrame, columns: list, contamination: float = 0.02
) -> pd.Series:
    """
    Global anomaly score using Isolation Forest. It is used as a second
    check against the regime-based z-score.
    """
    data = df[columns].dropna()
    model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(data)
    # Lower decision_function values mean more anomalous, so invert it to get a 0-1 score.
    raw_scores = -model.decision_function(data)
    normalized = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)

    scores = pd.Series(np.nan, index=df.index)
    scores.loc[data.index] = normalized
    return scores


def combine_anomaly_score(df: pd.DataFrame, zscore_cols: list, iforest_col: str = "iforest_score") -> pd.Series:
    """
    Combines the regime-based z-scores with the Isolation Forest score
    into one anomaly_score between 0 and 1.
    """
    zscore_avg = df[zscore_cols].abs().mean(axis=1)
    zscore_norm = (zscore_avg - zscore_avg.min()) / (zscore_avg.max() - zscore_avg.min() + 1e-9)

    combined = 0.5 * zscore_norm.fillna(0) + 0.5 * df[iforest_col].fillna(0)
    return combined.clip(0, 1)


# ---------------------------------------------------------------------------
# 4. CHANGE-POINT DETECTION
# ---------------------------------------------------------------------------

def detect_change_points(series: pd.Series, method: str = "auto", penalty: float = 10.0) -> list:
    """
    Returns the positions where the behavior of the series changes.
    IMPORTANT: use an already downsampled series (for example every
    15-60 minutes), not the original 252k points. Exact change-point
    methods can be very slow on hundreds of thousands of points.
    """
    clean = series.dropna().values

    if method in ("auto", "ruptures"):
        try:
            import ruptures as rpt

            algo = rpt.Pelt(model="rbf").fit(clean)
            change_points = algo.predict(pen=penalty)
            return change_points[:-1]  # ruptures adds the last index as the end of the series, not a real change point
        except ImportError:
            if method == "ruptures":
                raise
            # Use CUSUM if ruptures is not installed.

    return _cusum_change_points(clean)


def _cusum_change_points(values: np.ndarray, threshold: float = 5.0, drift: float = 0.5) -> list:
    """Simple CUSUM implementation without external dependencies."""
    mean = np.mean(values)
    std = np.std(values) or 1.0
    normalized = (values - mean) / std

    pos_cusum, neg_cusum = 0.0, 0.0
    change_points = []
    for i, v in enumerate(normalized):
        pos_cusum = max(0, pos_cusum + v - drift)
        neg_cusum = min(0, neg_cusum + v + drift)
        if pos_cusum > threshold or neg_cusum < -threshold:
            change_points.append(i)
            pos_cusum, neg_cusum = 0.0, 0.0
    return change_points


if __name__ == "__main__":
    print("This module is imported from the Streamlit notebook/app, not run by itself.")
    print("See scripts/build_features_metropt3.py for an end-to-end example.")