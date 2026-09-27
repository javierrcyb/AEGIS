"""
Dos modos:
  1. Demo (MetroPT-3, preprocesado con los scripts de scripts/) — muestra el
     sistema completo, con modelo supervisado entrenado.
  2. Subir un CSV propio — corre el pipeline no-supervisado genérico
     (aegis_core/pipeline.py) sobre CUALQUIER dataset con timestamp +
     columnas numéricas. Sin modelo entrenado para ese dataset específico,
     así que lo dice explícitamente en vez de inventar una probabilidad.

Usage:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from aegis_core.data_profiling import profile_dataset
from aegis_core.ml_engine import build_feature_matrix
from aegis_core.explainability import compute_shap_values, global_feature_importance, top_features_for_row
from aegis_core.evidence_engine import build_evidence
from aegis_core.reasoning_agent import aegis_reasoning, DEFAULT_MODEL
from aegis_core.pipeline import analyze_dataset, auto_detect_timestamp_column, apply_supervised_layer
from aegis_core.ml_engine import (
    detect_label_column_candidates,
    check_trainability,
    estimate_training_seconds,
    prepare_training_data,
    train_and_evaluate_all,
    select_best_model,
    PIPELINE_OUTPUT_COLS,
)

from aegis_core.reasoning_agent import aegis_reasoning, list_ollama_models, DEFAULT_MODEL

DEMO_DATA_PATH = "data/processed/metropt3_final.csv"
DEMO_MODEL_PATH = "models/aegis_classifier.joblib"
DEMO_COLUMNS_PATH = "models/aegis_feature_columns.json"

DEMO_VALUE_COLUMNS = ["current", "temperature", "pressure_main", "pressure_panel",
                       "pressure_reservoir", "dv_pressure", "h1"]

st.set_page_config(page_title="AEGIS", layout="wide")


@st.cache_data
def load_demo_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col="timestamp", parse_dates=True).dropna()


@st.cache_resource
def load_demo_model(model_path: str, columns_path: str):
    model = joblib.load(model_path)
    with open(columns_path) as f:
        columns = json.load(f)
    return model, columns


# Main: selects the data source and renders the pages
# ---------------------------------------------------------------------------

def main():
    st.sidebar.title("AEGIS")
    st.sidebar.caption("Agentic Engineering Intelligence for Industrial Systems")

    mode = st.sidebar.radio("Data Source", ["Demo: MetroPT-3", "Upload My Own CSV"])

    if mode == "Demo: MetroPT-3":
        df, value_columns, has_model, warnings = _load_demo_mode()
    else:
        df, value_columns, has_model, warnings = _load_upload_mode()

    if df is None:
        return

    for w in warnings:
        st.sidebar.warning(w)

    page = st.sidebar.radio("Page", ["Overview", "Signals", "AI Analysis", "Models", "Data"])

    if page == "Overview":
        page_overview(df)
    elif page == "Signals":
        page_signals(df, value_columns)
    elif page == "AI Analysis":
        page_ai_analysis(df, has_model, is_demo=(mode == "Demo: MetroPT-3"))
    elif page == "Models":
        page_models(df, value_columns, has_model, is_demo=(mode == "Demo: MetroPT-3"))
    elif page == "Data":
        page_data(df, value_columns)


def _load_demo_mode():
    if not Path(DEMO_DATA_PATH).exists():
        st.error(f"Cannot find {DEMO_DATA_PATH}. Run the scripts in the ‘scripts/’ directory first (see README).")
        st.stop()
    df = load_demo_data(DEMO_DATA_PATH)
    return df, DEMO_VALUE_COLUMNS, True, []


def _load_upload_mode():
    uploaded = st.sidebar.file_uploader("Upload My Own CSV", type="csv")
    if uploaded is None:
        st.info("Upload a CSV file in the sidebar to get started. It needs at least one date/time column and numeric columns for sensor data.")
        return None, None, None, []

    raw_df = pd.read_csv(uploaded)
    st.sidebar.caption(f"{len(raw_df):,} rows, {len(raw_df.columns)} columns")

    ts_guess = auto_detect_timestamp_column(raw_df)
    ts_options = list(raw_df.columns)
    ts_index = ts_options.index(ts_guess) if ts_guess in ts_options else 0
    ts_col = st.sidebar.selectbox("Timestamp Column", ts_options, index=ts_index)

    numeric_candidates_all = raw_df.select_dtypes(include="number").columns.tolist()
    binary_like = [c for c in numeric_candidates_all if raw_df[c].dropna().nunique() <= 2]
    numeric_candidates = [c for c in numeric_candidates_all if c not in binary_like]
    default_cols = numeric_candidates[: min(6, len(numeric_candidates))]
    value_columns = st.sidebar.multiselect("Columns to Analyze", numeric_candidates, default=default_cols)
    if binary_like:
        st.sidebar.caption(f"Binary columns excluded from signal analysis (possible failure labels): {binary_like}")

    file_key = f"{uploaded.name}_{uploaded.size}_{ts_col}_{tuple(value_columns)}"
    analyze_clicked = st.sidebar.button("Analyze dataset", type="primary")

    if analyze_clicked:
        with st.spinner("Running profiling + features + regime + anomaly detection..."):
            try:
                result = analyze_dataset(raw_df, timestamp_col=ts_col, value_columns=value_columns or None)
                st.session_state["analysis_result"] = result
                st.session_state["analysis_key"] = file_key
            except Exception as e:
                st.error(f"Error analyzing the dataset: {e}")
                return None, None, None, []

    if "analysis_result" not in st.session_state or st.session_state.get("analysis_key") != file_key:
        st.info("Configure the columns in the sidebar and click **Analyze dataset**.")
        return None, None, None, []

    result = st.session_state["analysis_result"]
    _offer_supervised_training(raw_df, ts_col, value_columns, result, file_key)

    # If a model has been trained, the updated version of the result is used
    trained_key = file_key + "_trained"
    if trained_key in st.session_state:
        result = st.session_state[trained_key]

    return result.df, result.value_columns, result.has_supervised_model, result.warnings


def _offer_supervised_training(raw_df, ts_col, value_columns, result, file_key):
    """
    Detects whether the uploaded dataset contains a failure column (0/1),
    and if the user wants, trains on it, but NEVER automatically.
    """
    with st.sidebar.expander("Supervised training (optional)"):
        if f"train_summary_{file_key}" in st.session_state:
            st.success(st.session_state[f"train_summary_{file_key}"])
        candidates = detect_label_column_candidates(raw_df, exclude_cols=[ts_col] + value_columns)
        if not candidates:
            st.caption("No binary (0/1) column was detected that could be a failure label.")
            return

        label_col = st.selectbox("Failure column (0/1)", ["None"] + candidates, key=f"label_{file_key}")
        if label_col == "None":
            return

        ts_index = pd.to_datetime(raw_df[ts_col])
        label_lookup = pd.Series(raw_df[label_col].values, index=ts_index)
        y_full = label_lookup.reindex(result.df.index)

        issues = check_trainability(y_full.dropna())
        if issues:
            for issue in issues:
                st.error(issue)
            st.caption("Training is not offered as long as these problems persist -> a model trained under these conditions would not be reliable.")
            return

        st.success(f"{int(y_full.sum())} positive values out of {y_full.notna().sum():,} valid rows")

        estimate_key = f"estimate_{file_key}_{label_col}"
        if st.button("Estimate trainging time", key=f"btn_est_{file_key}"):
            df_with_label = result.df.copy()
            df_with_label["__label__"] = y_full
            X, y, prep_warnings = prepare_training_data(df_with_label, label_col="__label__")
            for w in prep_warnings:
                st.caption(f"{w}")
            est_seconds = estimate_training_seconds(X, y)
            st.session_state[estimate_key] = (est_seconds, X, y)

        if estimate_key in st.session_state:
            est_seconds, X, y = st.session_state[estimate_key]
            st.info(f"Estimated time: ~{est_seconds:.0f} seconds for {len(X):,} rows x {X.shape[1]} features.")

            if st.button(f"Train now (~{est_seconds:.0f}s)", type="primary", key=f"btn_train_{file_key}"):
                with st.spinner(f"Training (estimated ~{est_seconds:.0f}s; may vary)..."):
                    import time as _time
                    t0 = _time.time()
                    n_test = max(int(len(X) * 0.2), 50)
                    X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
                    y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]
                    try:
                        results = train_and_evaluate_all(X_train, y_train, X_test, y_test)
                        best = select_best_model(results, by="pr_auc")
                        elapsed = _time.time() - t0
                        summary_msg = f"Done in {elapsed:.1f} actual seconds. Best model: {best.name} (PR-AUC {best.pr_auc:.3f})"
                        st.session_state[f"train_summary_{file_key}"] = summary_msg
                        st.session_state["uploaded_trained_model"] = {"model": best.model,
                                                                      "feature_columns": list(X.columns)}
                        st.success(summary_msg)

                        new_df, has_model, layer_warnings = apply_supervised_layer(
                            result.df, model=best.model, model_feature_columns=list(X.columns)
                        )
                        from aegis_core.pipeline import AnalysisResult
                        carried_warnings = [
                            w for w in result.warnings
                            if "no trained supervised model" not in w.lower()
                        ]
                        new_result = AnalysisResult(
                            df=new_df, profile=result.profile, has_supervised_model=has_model,
                            value_columns=result.value_columns, warnings=carried_warnings + layer_warnings,
                        )
                        st.session_state[file_key + "_trained"] = new_result
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error while training: {e}")


# PAGE 1 — Overview
# ---------------------------------------------------------------------------

def page_overview(df: pd.DataFrame):
    st.header("Overview")

    latest = df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Health Score", f"{latest['health_score']:.0f}/100")
    col2.metric("Risk", _risk_from_health(latest["health_score"]))
    col3.metric("Confidence", latest["confidence_label"])
    col4.metric("Regime actual", latest["regime"])

    st.caption(f"Latest timestamp in the dataset: {df.index[-1]}")

    if "health_label" in df.columns:
        st.subheader("Distribution of actual states (entire dataset)")
        st.bar_chart(df["health_label"].value_counts())

    st.subheader("Health score over time (resampled daily)")
    daily = df["health_score"].resample("1D").mean()
    st.line_chart(daily)


def _risk_from_health(score: float) -> str:
    if score >= 80:
        return "LOW"
    elif score >= 60:
        return "MEDIUM"
    elif score >= 40:
        return "HIGH"
    return "CRITICAL"


# PAGE 2 — Signals
# ---------------------------------------------------------------------------

def page_signals(df: pd.DataFrame, value_columns: list):
    st.header("Signals")

    signal = st.selectbox("Signal", value_columns)

    date_range = st.date_input(
        "Date range",
        value=(df.index.min().date(), df.index.max().date()),
        min_value=df.index.min().date(),
        max_value=df.index.max().date(),
    )
    if len(date_range) == 2:
        start, end = date_range
        subset = df.loc[str(start):str(end)]
    else:
        subset = df

    st.caption(f"{len(subset):,} rows in the selected range")

    plot_data = _downsample_for_plot(subset[[signal, "anomaly_score"]])
    if len(plot_data) < len(subset):
        st.caption(f"Plotting a sample of {len(plot_data):,} points (from {len(subset):,})")

    st.line_chart(plot_data[[signal]])

    st.subheader("Anomaly score in the same range")
    st.line_chart(plot_data[["anomaly_score"]])

    if "health_label" in subset.columns:
        st.subheader("Rows marked as FAULT/DEGRADING in this range")
        abnormal = subset[subset["health_label"].isin(["FAULT", "DEGRADING"])]
        st.write(f"{len(abnormal):,} out of {len(subset):,} rows ({len(abnormal)/max(len(subset),1)*100:.2f}%)")
        if not abnormal.empty:
            st.dataframe(abnormal[[signal, "anomaly_score", "health_label"]].head(20))
    else:
        st.subheader("Rows with the highest anomaly_score in this range")
        st.dataframe(subset.nlargest(20, "anomaly_score")[[signal, "anomaly_score", "health_score"]])


def _downsample_for_plot(data, max_points: int = 3000):
    n = len(data)
    if n <= max_points:
        return data
    step = max(1, n // max_points)
    return data.iloc[::step]


# PAGE 3 — AI Analysis
# ---------------------------------------------------------------------------

def page_ai_analysis(df: pd.DataFrame, has_model: bool, is_demo: bool):
    st.header("AI Analysis")
    st.caption("The local LLM reasons ONLY based on the evidence that has already been computed")
    installed_models = list_ollama_models()
    if installed_models:
        llm_model = st.selectbox("Local Model (Ollama)", installed_models)
    else:
        st.warning(
            "I couldn't access Ollama at localhost:11434 -> Typed the model name manually.")

    if not has_model:
        st.warning(
            "This dataset does not have a trained supervised model (there is no labeled failure "
            "history). The diagnosis will be based SOLELY on unsupervised detection."
        )

    if "health_label" in df.columns:
        mode = st.radio("Choose a moment", ["Automatic (worst-case scenario)", "Manual"])
    else:
        mode = "Manual"

    if mode == "Automatic (worst-case scenario)":
        fault_rows = df[df["health_label"] == "FAULT"]
        if fault_rows.empty:
            st.warning("There are no FAULT rows in the dataset.")
            return
        target_ts = fault_rows["anomaly_score"].idxmax()
    else:
        options = df.index[::max(1, len(df) // 500)]  # limit the number of options listed
        target_ts = st.selectbox("Timestamp", options)

    st.write(f"**Timestamp analyzed :** {target_ts}")
    row = df.loc[target_ts]

    col1, col2, col3 = st.columns(3)
    col1.metric("Health Score", f"{row['health_score']:.1f}")
    col2.metric("Anomaly Score", f"{row['anomaly_score']:.3f}")
    col3.metric("Model Probability", f"{row['model_probability']:.3f}" if has_model and pd.notna(row.get("model_probability")) else "N/A")

    if st.button("Generate an AEGIS diagnostic report"):
        with st.spinner("Calculating evidence and calling the local LLM (may take 10–30 seconds)..."):
            try:
                top_feats = _get_top_features(df, target_ts, has_model, is_demo)
                evidence = build_evidence(row, top_feats, change_point_nearby=False)
                diagnosis = aegis_reasoning(evidence, model=llm_model)

                if "error" in diagnosis:
                    st.error("The local LLM did not return valid JSON. Raw response:")
                    st.code(diagnosis.get("raw_response", ""))
                else:
                    st.success("Generated Diagnosis")
                    st.markdown(f"**Machine Status:** {diagnosis['machine_state']}")
                    st.markdown(f"**What Changed:** {diagnosis['what_changed']}")
                    st.markdown(f"**Possible explanation:** {diagnosis['likely_explanation']}")
                    st.markdown("**Evidence used:**")
                    for e in diagnosis["evidence_used"]:
                        st.markdown(f"- {e}")
                    st.markdown(f"**Trust:** {diagnosis['confidence_statement']}")
                    st.markdown(f"**Recommended Action:** `{diagnosis['recommended_action']}`")
                    st.markdown(f"**Details:** {diagnosis['action_detail']}")

                    with st.expander("View structured evidence (what the LLM actually saw)"):
                        st.json(evidence.to_dict())

            except Exception as e:
                st.error(f"Error: {e}.")


def _get_top_features(df, target_ts, has_model, is_demo):
    """
    If there is a trained model (demo case), use the actual SHAP.
    If not, use a simple substitute: the columns whose z-score per regime is
    the most extreme for that row
    """
    if has_model and is_demo:
        model, feature_columns = load_demo_model(DEMO_MODEL_PATH, DEMO_COLUMNS_PATH)
        window = df.loc[target_ts - pd.Timedelta(hours=2): target_ts + pd.Timedelta(hours=2)]
        X_window = build_feature_matrix(window[DEMO_VALUE_COLUMNS + ["regime", "anomaly_score"]], exclude_cols=[])
        X_window = X_window.reindex(columns=feature_columns, fill_value=0)
        shap_df, _ = compute_shap_values(model, X_window, background_sample_size=200)
        return top_features_for_row(shap_df, target_ts, n=3)
    else:
        zscore_cols = [c for c in df.columns if c.endswith("_zscore_regime")]
        if not zscore_cols:
            return []
        row_z = df.loc[target_ts, zscore_cols].abs().sort_values(ascending=False)
        return [c.replace("_zscore_regime", "") for c in row_z.head(3).index]


# PAGE 4 — Models
# ---------------------------------------------------------------------------

def page_models(df: pd.DataFrame, value_columns: list, has_model: bool, is_demo: bool):
    st.header("Models")

    if not has_model:
        st.info(
            "This dataset does not have a trained supervised model -> there is no labeled "
            "failure history to train against. This page is for demo purposes only."
        )
        return

    st.subheader("Global feature importance (SHAP, based on a sample)")

    if is_demo:
        if st.button("Calculate global SHAP (takes ~1 min)"):
            with st.spinner("Calculating ..."):
                model, feature_columns = load_demo_model(DEMO_MODEL_PATH, DEMO_COLUMNS_PATH)
                sample = df.sample(min(5000, len(df)), random_state=42)
                X_sample = build_feature_matrix(sample[DEMO_VALUE_COLUMNS + ["regime", "anomaly_score"]], exclude_cols=[])
                X_sample = X_sample.reindex(columns=feature_columns, fill_value=0)
                shap_df, _ = compute_shap_values(model, X_sample, background_sample_size=200)
                importance = global_feature_importance(shap_df)
                st.bar_chart(importance)

        st.subheader("Model Comparison (leave-one-event-out)")
        st.caption("Averaged metrics across the 4 actual events, using the ‘core’ feature set.")
        comparison = pd.DataFrame({
            "F1": [0.375, 0.321, 0.177],
            "ROC-AUC": [0.732, 0.816, 0.860],
            "PR-AUC": [0.363, 0.281, 0.316],
        }, index=["LogisticRegression", "LightGBM", "RandomForest"])
        st.dataframe(comparison)
        st.caption(
            "Note: Performance varies significantly depending on the duration and severity of the excluded event."
        )

    elif "uploaded_trained_model" in st.session_state:
        if st.button("Calculate global SHAP (takes ~1 min)"):
            with st.spinner("Calculating ..."):
                trained = st.session_state["uploaded_trained_model"]
                model, feature_columns = trained["model"], trained["feature_columns"]
                sample = df.sample(min(5000, len(df)), random_state=42)
                X_sample = build_feature_matrix(sample, exclude_cols=PIPELINE_OUTPUT_COLS)
                X_sample = X_sample.reindex(columns=feature_columns, fill_value=0)
                shap_df, _ = compute_shap_values(model, X_sample, background_sample_size=200)
                importance = global_feature_importance(shap_df)
                st.bar_chart(importance)
        st.caption(
            "This model was trained on your uploaded dataset — there is no leave-one-event-out "
            "comparison available for it, since that requires multiple known historical events."
        )

    else:
        st.caption("No trained model available yet for this dataset.")


# PAGE 5 — Data
# ---------------------------------------------------------------------------

def page_data(df: pd.DataFrame, value_columns: list):
    st.header("Data")

    summary_cols = ["regime", "anomaly_score", "health_score", "model_probability", "confidence"]
    cols_to_profile = [c for c in value_columns + summary_cols if c in df.columns]

    st.caption("Highlighting only the “raw” columns + final results, not the rolling/lag features.")
    profile = profile_dataset(df[cols_to_profile])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{profile.n_rows:,}")
    col2.metric("Columns", profile.n_cols)
    col3.metric("Duplicate rows", profile.duplicate_rows)
    if profile.missing_timestamp_pct:
        col4.metric("Missing timestamps", f"{profile.missing_timestamp_pct:.1f}%")

    st.subheader("Data Quality Score")
    q = profile.quality_scores
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Overall", f"{q['overall']}/100")
    col2.metric("Completeness", q["completeness"])
    col3.metric("Consistency", q["consistency"])
    col4.metric("Temporal", q["temporal_quality"])
    col5.metric("Signal variance", q["signal_variance"])

    st.subheader("Warnings")
    for w in profile.warnings:
        st.write(w)

    st.subheader("Columns Analyzed")
    st.write(value_columns)


if __name__ == "__main__":
    main()
