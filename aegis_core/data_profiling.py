"""
100% generic: does not assume column names from any specific dataset.
It takes a DataFrame with a datetime index and returns:
  - an overview of the dataset
  - column type detection
  - a data quality score (0–100) with subcomponents
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


IGNORE_COLUMNS_DEFAULT = ["unnamed:_0", "unnamed: 0"] # Common CSV index columns that should be ignored during dataset analysis


@dataclass
class DatasetProfile:
    n_rows: int
    n_cols: int
    time_range: Optional[tuple]
    sampling_freq_seconds: Optional[float]
    numeric_columns: list
    categorical_columns: list
    constant_columns: list
    missing_pct_by_col: dict
    duplicate_rows: int
    outlier_pct_by_col: dict
    expected_rows: Optional[int] = None
    missing_timestamp_pct: Optional[float] = None
    quality_scores: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def _detect_column_types(df: pd.DataFrame) -> tuple[list, list]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def _detect_constant_columns(df: pd.DataFrame, numeric_cols: list) -> list:
    constant = []
    for c in numeric_cols:
        if df[c].nunique(dropna=True) <= 1:
            constant.append(c)
    return constant


def _detect_sampling_frequency(index: pd.DatetimeIndex) -> Optional[float]:
    if len(index) < 2:
        return None
    diffs = index.to_series().diff().dropna().dt.total_seconds()
    if diffs.empty:
        return None
    # Mode is more resilient than mean if there are occasional gaps
    return float(diffs.mode().iloc[0])


def _detect_temporal_gaps(index: pd.DatetimeIndex, freq_seconds: Optional[float]) -> tuple:
    """
    Compare the actual rows with the expected rows if the series were continuous.
    This is what distinguishes “no NaNs” from “no missing intervals.”
    """
    if freq_seconds is None or len(index) < 2:
        return None, None
    span_seconds = (index.max() - index.min()).total_seconds()
    expected_rows = int(span_seconds / freq_seconds) + 1
    missing_pct = max(0.0, (1 - len(index) / expected_rows) * 100)
    return expected_rows, missing_pct


def _detect_outliers_iqr(series: pd.Series) -> float:
    """% of values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]. Simple and standard."""
    clean = series.dropna()
    if len(clean) < 10:
        return 0.0
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_mask = (clean < lower) | (clean > upper)
    return float(outlier_mask.mean() * 100)


def profile_dataset(
    df: pd.DataFrame,
    ignore_columns: Optional[list] = None,
) -> DatasetProfile:
    ignore_columns = ignore_columns or IGNORE_COLUMNS_DEFAULT
    work_df = df.drop(columns=[c for c in ignore_columns if c in df.columns], errors="ignore")

    numeric_cols, categorical_cols = _detect_column_types(work_df)
    constant_cols = _detect_constant_columns(work_df, numeric_cols)

    missing_pct = {c: float(work_df[c].isna().mean() * 100) for c in work_df.columns}
    duplicate_rows = int(work_df.duplicated().sum())

    outlier_pct = {c: _detect_outliers_iqr(work_df[c]) for c in numeric_cols if c not in constant_cols}

    time_range = None
    sampling_freq = None
    expected_rows = None
    missing_timestamp_pct = None
    if isinstance(work_df.index, pd.DatetimeIndex):
        time_range = (work_df.index.min(), work_df.index.max())
        sampling_freq = _detect_sampling_frequency(work_df.index)
        expected_rows, missing_timestamp_pct = _detect_temporal_gaps(work_df.index, sampling_freq)

    profile = DatasetProfile(
        n_rows=len(work_df),
        n_cols=len(work_df.columns),
        time_range=time_range,
        sampling_freq_seconds=sampling_freq,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        constant_columns=constant_cols,
        missing_pct_by_col=missing_pct,
        duplicate_rows=duplicate_rows,
        outlier_pct_by_col=outlier_pct,
        expected_rows=expected_rows,
        missing_timestamp_pct=missing_timestamp_pct,
    )

    profile.quality_scores = compute_quality_score(profile)
    profile.warnings = generate_warnings(profile)
    return profile


def compute_quality_score(profile: DatasetProfile) -> dict:
    """
    4 sub-scores (0-100) + global score.
    """
    # Completeness: based on the average % of missing values
    avg_missing = np.mean(list(profile.missing_pct_by_col.values())) if profile.missing_pct_by_col else 0
    completeness = max(0, 100 - avg_missing * 2)  # imposes severe penalties for missing items

    # Consistency: penalizes constant columns (which provide no information) and duplicates
    dup_pct = (profile.duplicate_rows / profile.n_rows * 100) if profile.n_rows else 0
    constant_penalty = len(profile.constant_columns) * 5
    consistency = max(0, 100 - dup_pct * 3 - constant_penalty)

    # Temporal quality: penalizes actual time gaps (missing rows), not just NaN
    if profile.sampling_freq_seconds is None:
        temporal_quality = 50.0  # There is no reliable information at the time.
    elif profile.missing_timestamp_pct is not None:
        temporal_quality = max(0, 100 - profile.missing_timestamp_pct * 2)
    else:
        temporal_quality = 100.0

    # Signal variance: based on the average number of outliers (neither too few nor too many)
    avg_outlier = np.mean(list(profile.outlier_pct_by_col.values())) if profile.outlier_pct_by_col else 0
    signal_variance = max(0, 100 - avg_outlier * 1.5)

    overall = round(
        0.30 * completeness + 0.25 * consistency + 0.25 * temporal_quality + 0.20 * signal_variance,
        1,
    )

    return {
        "completeness": round(completeness, 1),
        "consistency": round(consistency, 1),
        "temporal_quality": round(temporal_quality, 1),
        "signal_variance": round(signal_variance, 1),
        "overall": overall,
    }


def generate_warnings(profile: DatasetProfile) -> list:
    warnings = []
    if profile.missing_timestamp_pct and profile.missing_timestamp_pct > 1:
        warnings.append(
            f"{profile.missing_timestamp_pct:.1f}% of the expected time intervals are missing "
            f"(temporary gaps, not NaN values)."
        )
    for col, pct in profile.missing_pct_by_col.items():
        if pct > 2:
            warnings.append(f"'{col}' has {pct:.1f}% missing values.")

    for col in profile.constant_columns:
        warnings.append(f"'{col}' is constant and provides no useful information.")

    if profile.duplicate_rows > 0:
        warnings.append(f"{profile.duplicate_rows} duplicate rows were found.")

    for col, pct in profile.outlier_pct_by_col.items():
        if pct > 10:
            warnings.append(f"'{col}' has {pct:.1f}% outlier values.")

    if not warnings:
        warnings.append("No major data quality issues were found.")

    return warnings


def print_profile_report(profile: DatasetProfile) -> None:
    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"Rows:              {profile.n_rows:,}")
    print(f"Columns:           {profile.n_cols}")
    if profile.time_range:
        days = (profile.time_range[1] - profile.time_range[0]).days
        print(f"Time range:        {days} days ({profile.time_range[0]} to {profile.time_range[1]})")
    if profile.sampling_freq_seconds:
        print(f"Sampling freq:     {profile.sampling_freq_seconds:.1f} seconds")
    if profile.expected_rows:
        print(f"Expected rows:     {profile.expected_rows:,} (if there were no gaps)")
        print(f"Missing timestamps:{profile.missing_timestamp_pct:.1f}%")
    print(f"Numeric columns:   {len(profile.numeric_columns)}")
    print(f"Categorical cols:  {len(profile.categorical_columns)}")
    print(f"Constant columns:  {profile.constant_columns}")
    print(f"Duplicate rows:    {profile.duplicate_rows}")

    print("\n" + "=" * 50)
    print("DATA QUALITY SCORE")
    print("=" * 50)
    q = profile.quality_scores
    print(f"Overall:           {q['overall']}/100")
    print(f"  Completeness:      {q['completeness']}")
    print(f"  Consistency:       {q['consistency']}")
    print(f"  Temporal quality:  {q['temporal_quality']}")
    print(f"  Signal variance:   {q['signal_variance']}")

    print("\n" + "=" * 50)
    print("WARNINGS")
    print("=" * 50)
    for w in profile.warnings:
        print(w)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--timestamp-col", default="timestamp")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.timestamp_col in df.columns:
        df[args.timestamp_col] = pd.to_datetime(df[args.timestamp_col])
        df = df.set_index(args.timestamp_col)

    profile = profile_dataset(df)
    print_profile_report(profile)
