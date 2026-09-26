"""
Generic: does not assume specific column names from MetroPT-3, except
for the label column provided as a parameter.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

try:
    from lightgbm import LGBMClassifier

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def temporal_train_test_split(df: pd.DataFrame, split_date: str):
    """Everything before split_date is used for training, everything after is test."""
    split_ts = pd.Timestamp(split_date)
    train = df[df.index < split_ts]
    test = df[df.index >= split_ts]
    return train, test


def build_binary_target(df: pd.DataFrame, label_col: str, positive_labels: list) -> pd.Series:
    """Converts multi-class labels into a binary 0/1 target."""
    return df[label_col].isin(positive_labels).astype(int)


def build_feature_matrix(df: pd.DataFrame, exclude_cols: list) -> pd.DataFrame:
    """
    Prepares X by removing excluded columns and one-hot encoding categorical columns.
    NaN values are not removed here. This is handled by the caller
    (see dropna_for_training vs. fillna_for_inference below), since
    removing rows makes sense for training but not for live inference
    on a dataset uploaded by the user.
    """
    work = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore")

    categorical_cols = work.select_dtypes(exclude=[np.number]).columns.tolist()
    if categorical_cols:
        work = pd.get_dummies(work, columns=categorical_cols, dummy_na=False)

    return work


@dataclass
class ModelResult:
    name: str
    model: object
    threshold: float
    f1: float
    precision: float
    recall: float
    roc_auc: float
    pr_auc: float
    confusion: np.ndarray


CANDIDATE_MODELS = {
    # Logistic Regression needs scaled features. Tree-based models do not.
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, C=0.1)),
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    ),
}

if HAS_LIGHTGBM:
    CANDIDATE_MODELS["LightGBM"] = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, class_weight="balanced", random_state=42, verbosity=-1
    )


def build_candidate_models(fast: bool = False) -> dict:
    """
    fast=False (default): full configuration used for the
    curated MetroPT-3 dataset in scripts/.

    fast=True: fewer estimators for training on a user-uploaded
    dataset within a reasonable amount of time.
    """
    if not fast:
        return dict(CANDIDATE_MODELS)

    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=500, C=0.1)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=50, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }

    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=100, learning_rate=0.1, class_weight="balanced", random_state=42, verbosity=-1
        )

    return models


def _find_best_threshold(y_true, y_proba) -> float:
    """
    Tests different thresholds and returns the one with the highest F1.
    The threshold is selected on the validation set, never on the final test set,
    to avoid data leakage.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh, best_f1 = 0.5, -1

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)

        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    return best_thresh


def train_and_evaluate_all(
    X_train, y_train, X_test, y_test, validation_frac: float = 0.2
) -> list:
    if y_train.nunique() < 2:
        raise ValueError(
            "The training set does not contain examples from both classes (normal/abnormal). "
            "This can happen if the temporal split leaves all failure events in the test set. "
            "Choose a split date that leaves at least 2-3 failure events in the training set."
        )

    if y_test.nunique() < 2:
        print(
            "The test set does not contain both classes — ROC-AUC cannot be calculated "
            "reliably. Check the split date."
        )

    # Split TRAIN into fit and validation sets chronologically.
    # The final test set is not used when selecting the decision threshold.
    n_val = int(len(X_train) * validation_frac)
    X_fit, X_val = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
    y_fit, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]

    results = []

    for name, model in CANDIDATE_MODELS.items():
        if y_fit.nunique() < 2:
            print(f"  [{name}] skipped: the fit set does not contain both classes.")
            continue

        model.fit(X_fit, y_fit)

        # Select the best threshold on the validation set.
        val_proba = model.predict_proba(X_val)[:, 1]
        threshold = _find_best_threshold(y_val, val_proba) if y_val.nunique() > 1 else 0.5

        # Retrain using the full training set after fixing the threshold.
        model.fit(X_train, y_train)
        test_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (test_proba >= threshold).astype(int)

        results.append(
            ModelResult(
                name=name,
                model=model,
                threshold=threshold,
                f1=f1_score(y_test, y_pred, zero_division=0),
                precision=precision_score(y_test, y_pred, zero_division=0),
                recall=recall_score(y_test, y_pred, zero_division=0),
                roc_auc=roc_auc_score(y_test, test_proba) if len(set(y_test)) > 1 else float("nan"),
                pr_auc=average_precision_score(y_test, test_proba) if len(set(y_test)) > 1 else float("nan"),
                confusion=confusion_matrix(y_test, y_pred),
            )
        )

    return results


def select_best_model(results: list, by: str = "pr_auc") -> ModelResult:
    return max(results, key=lambda r: getattr(r, by))


def print_comparison(results: list) -> None:
    print(f"{'Model':<20}{'F1':>8}{'Precision':>12}{'Recall':>10}{'ROC-AUC':>10}{'PR-AUC':>10}{'Thresh':>9}")

    for r in sorted(results, key=lambda x: x.pr_auc, reverse=True):
        print(
            f"{r.name:<20}{r.f1:>8.3f}{r.precision:>12.3f}{r.recall:>10.3f}"
            f"{r.roc_auc:>10.3f}{r.pr_auc:>10.3f}{r.threshold:>9.2f}"
        )


# ---------------------------------------------------------------------------
# Training on user-uploaded datasets is never automatic.
# Run basic checks and estimate training time before starting.
# ---------------------------------------------------------------------------

import time


def detect_label_column_candidates(df: pd.DataFrame, exclude_cols: list = None) -> list:
    """
    Finds binary columns (0/1, True/False) that could be failure labels.
    Columns with suggestive names are listed first, but the others are
    kept as candidates. The user makes the final choice.
    """
    exclude_cols = exclude_cols or []
    keywords = ["fail", "fault", "label", "target", "anomaly", "defect", "breakdown", "alarm", "falla"]

    candidates = []

    for c in df.columns:
        if c in exclude_cols:
            continue

        values = df[c].dropna().unique()

        if len(values) == 0 or len(values) > 2:
            continue

        value_set = set(values.tolist())

        if value_set <= {0, 1} or value_set <= {0.0, 1.0} or value_set <= {True, False}:
            candidates.append(c)

    candidates.sort(key=lambda c: 0 if any(k in c.lower() for k in keywords) else 1)

    return candidates


def check_trainability(y: pd.Series, min_rows: int = 500, min_positive: int = 30) -> list:
    """
    Runs basic checks before offering to train a model.
    If this returns any issues, training should not start.
    """
    issues = []

    if len(y) < min_rows:
        issues.append(f"Too few rows ({len(y):,}) — at least {min_rows:,} are recommended.")

    if y.nunique() < 2:
        issues.append("The selected column does not contain both classes (0 and 1).")
        return issues  # No point continuing with the remaining checks.

    positive_count = int(y.sum())

    if positive_count < min_positive:
        issues.append(
            f"Too few positive examples ({positive_count}) — at least "
            f"{min_positive} are recommended so the model has enough examples to learn from, "
            "rather than just memorizing a few cases."
        )

    return issues


def estimate_training_seconds(X: pd.DataFrame, y: pd.Series, sample_size: int = 2000) -> float:
    """
    Measures the time needed for a small Logistic Regression fit and
    uses it to estimate the training time for the full dataset.
    This is only an estimate, but it gives a rough idea of how long
    the full training process may take.
    """
    n = len(X)
    sample_n = min(sample_size, n)
    rng = np.random.RandomState(42)
    idx = rng.choice(n, sample_n, replace=False)
    X_sample, y_sample = X.iloc[idx], y.iloc[idx]

    start = time.time()

    try:
        LogisticRegression(max_iter=200).fit(X_sample, y_sample)
    except Exception:
        pass

    elapsed_sample = time.time() - start

    scale_factor = n / sample_n

    # Empirical factor: training all 3 candidate models (LR+RF+LightGBM)
    # takes roughly 4-6x the time of a small Logistic Regression fit.
    estimated_seconds = elapsed_sample * scale_factor * 5

    return max(estimated_seconds, 1.0)


MAX_TRAINING_ROWS = 100_000  # hard limit: datasets above this are subsampled


PIPELINE_OUTPUT_COLS = [
    "confidence", "confidence_label", "model_probability",
    "health_score", "degradation_trend", "regime_stress",
]


def prepare_training_data(df: pd.DataFrame, label_col: str, exclude_cols: list = None):
    """
    Prepares X and y for training on an uploaded dataset.
    Excludes non-feature columns, including pipeline output columns such as
    'confidence'. These columns may contain mostly NaN values and would
    otherwise cause unexpected rows to be removed by dropna.

    If the dataset exceeds MAX_TRAINING_ROWS, it is subsampled to keep
    training time within a predictable range.
    """
    exclude_cols = (exclude_cols or []) + [label_col] + PIPELINE_OUTPUT_COLS
    y_full = df[label_col].astype(int)
    X_full = build_feature_matrix(df, exclude_cols=exclude_cols)

    combined = X_full.join(y_full.rename("__label__")).dropna()
    X = combined.drop(columns="__label__")
    y = combined["__label__"]

    warnings = []

    if len(X) > MAX_TRAINING_ROWS:
        frac = MAX_TRAINING_ROWS / len(X)
        sampled_idx = y.groupby(y).apply(lambda s: s.sample(frac=frac, random_state=42)).index.get_level_values(1)
        X, y = X.loc[sampled_idx], y.loc[sampled_idx]

        warnings.append(
            f"Dataset was subsampled from {len(combined):,} to {len(X):,} rows "
            "to keep training time manageable while maintaining the class distribution."
        )

    return X, y, warnings