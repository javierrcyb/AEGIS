"""
Constructs the structured evidence object that is passed to the
reasoning agent. The LLM NEVER sees the raw CSV
"""

from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd


@dataclass
class Evidence:
    timestamp: str
    health_score: float
    risk_level: str  # LOW / MEDIUM / HIGH / CRITICAL, derivate from health_score
    anomaly_score: float
    change_point_nearby: bool
    operating_regime: str
    top_features: list
    model_probability: Optional[float]  # None if there is no supervised model for this dataset
    confidence: Optional[float]
    confidence_label: str  # LOW / MEDIUM / HIGH / N/A
    degradation_trend: float
    has_supervised_model: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def _risk_level_from_health(health_score: float) -> str:
    if health_score >= 80:
        return "LOW"
    elif health_score >= 60:
        return "MEDIUM"
    elif health_score >= 40:
        return "HIGH"
    return "CRITICAL"


def flag_recent_change_points(index: pd.DatetimeIndex, change_point_timestamps: list, window_hours: float = 6.0) -> pd.Series:
    """
    For each timestamp in the index, set to True if a change point
    was detected within the last `window_hours`.
    """
    flags = pd.Series(False, index=index)
    if not change_point_timestamps:
        return flags

    cps = sorted(pd.Timestamp(t) for t in change_point_timestamps)
    window = pd.Timedelta(hours=window_hours)

    for cp in cps:
        mask = (index >= cp) & (index <= cp + window)
        flags[mask] = True
    return flags


def build_evidence(
    row: pd.Series,
    top_features: list,
    change_point_nearby: bool,
) -> Evidence:
    """Create the evidence object for ONE row (one point in time) of the dataset."""
    has_model = pd.notna(row.get("model_probability"))
    return Evidence(
        timestamp=str(row.name),
        health_score=round(float(row["health_score"]), 1),
        risk_level=_risk_level_from_health(row["health_score"]),
        anomaly_score=round(float(row["anomaly_score"]), 3),
        change_point_nearby=bool(change_point_nearby),
        operating_regime=str(row["regime"]),
        top_features=top_features,
        model_probability=round(float(row["model_probability"]), 3) if has_model else None,
        confidence=round(float(row["confidence"]), 3) if has_model else None,
        confidence_label=str(row["confidence_label"]),
        degradation_trend=round(float(row["degradation_trend"]), 3),
        has_supervised_model=has_model,
    )


if __name__ == "__main__":
    print("This module is imported from aegis_core/reasoning_agent.py; it does not run on its own.")
