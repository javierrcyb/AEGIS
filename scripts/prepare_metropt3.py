"""
AEGIS — Día 1: preparación de datos (MetroPT-3)
=================================================

Qué hace este script:
1. Carga el CSV crudo de MetroPT-3 (lo tenés que descargar vos, ver instrucciones abajo)
2. Normaliza nombres de columnas
3. Etiqueta cada timestamp en HEALTHY / DEGRADING / FAULT / RECOVERY
   usando las 4 fallas reales documentadas por UCI
4. Resamplea a una frecuencia manejable (default 1 minuto) para no reventar memoria/Streamlit
5. Guarda el dataset procesado en data/processed/

Cómo conseguir el CSV crudo (hacelo UNA vez, a mano):
------------------------------------------------------
Opción A (recomendada, más simple):
    1. Entrá a: https://archive.ics.uci.edu/dataset/791/metropt+3+dataset
    2. Bajá el zip, descomprimilo
    3. Poné el CSV en: aegis/data/raw/metropt3_raw.csv

Opción B (con la librería ucimlrepo, si tenés internet libre):
    pip install ucimlrepo
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=791)
    ds.data.features.to_csv("aegis/data/raw/metropt3_raw.csv", index=False)

Uso:
    python scripts/prepare_metropt3.py --input data/raw/metropt3_raw.csv --freq 1min
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# 1. DOCUMENTED ACTUAL FAILURES
# ---------------------------------------------------------------------------
FAILURES = [
    ("2020-04-18 00:00", "2020-04-18 23:59", "2020-04-30 12:00"),
    ("2020-05-29 23:30", "2020-05-30 06:00", "2020-06-08 16:00"),
    ("2020-06-05 10:00", "2020-06-07 14:30", "2020-06-08 16:00"),
    ("2020-07-15 14:30", "2020-07-15 19:00", "2020-07-16 00:00"),
]

# 2. SCHEMA MAPPING (this is what makes the rest of the pipeline
#    dataset-agnostic — the generic Layer 1 reads from here, not from fixed names)
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "timestamp": ["timestamp", "unnamed: 0", "time"],
    "current": ["motor_current", "motor current"],
    "temperature": ["oil_temperature", "oil temperature"],
    "pressure_main": ["tp2"],
    "pressure_panel": ["tp3"],
    "pressure_reservoir": ["reservoirs"],
    "dv_pressure": ["dv_pressure", "dv pressure"],
    "comp_signal": ["comp"],
    "dv_electric": ["dv_electric", "dv electric"],
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert it to simple snake_case so it doesn't depend on the uppercase letters/spaces in the original CSV."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def apply_schema_map(df: pd.DataFrame) -> pd.DataFrame:
    """Rename actual columns to the generic names used in the AEGIS schema, if they exist."""
    rename = {}
    for generic_name, candidates in COLUMN_MAP.items():
        for cand in candidates:
            if cand in df.columns:
                rename[cand] = generic_name
                break
    df = df.rename(columns=rename)
    return df


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_columns(df)
    df = apply_schema_map(df)

    if "timestamp" not in df.columns:
        raise ValueError(
            "Could not find the timestamp column. Check COLUMN_MAP and add the "
            "actual name of the date/time column from the CSV you downloaded."
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols] = df[numeric_cols].astype("float32")

    return df


def build_label_column(index: pd.DatetimeIndex, pre_window_hours: float = 4.0) -> pd.Series:
    """
    Vectorized label.
    States: HEALTHY, DEGRADING, FAULT, RECOVERY
    """
    labels = np.full(len(index), "HEALTHY", dtype=object)

    for start_str, end_str, maint_str in FAILURES:
        start = pd.Timestamp(start_str)
        end = pd.Timestamp(end_str)
        pre_start = start - pd.Timedelta(hours=pre_window_hours)

        degrading_mask = (index >= pre_start) & (index < start)
        fault_mask = (index >= start) & (index <= end)
        labels[degrading_mask] = "DEGRADING"
        labels[fault_mask] = "FAULT"

        if maint_str:
            maint = pd.Timestamp(maint_str)
            if maint > end:
                recovery_mask = (index > end) & (index <= maint)
                still_healthy = labels[recovery_mask] == "HEALTHY"
                idx_positions = np.where(recovery_mask)[0]
                labels[idx_positions[still_healthy]] = "RECOVERY"

    return pd.Series(labels, index=index, name="health_label")


def resample_dataset(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["float32", "float64"]).columns
    agg = {c: "mean" for c in numeric_cols}

    resampled = df[numeric_cols].resample(freq).agg(agg)

    # The label is not averaged: it keeps the “worst” label in the interval
    severity_order = {"HEALTHY": 0, "RECOVERY": 1, "DEGRADING": 2, "FAULT": 3}
    label_series = df["health_label"].map(severity_order)
    worst_label_code = label_series.resample(freq).max()
    inverse = {v: k for k, v in severity_order.items()}
    resampled["health_label"] = worst_label_code.map(inverse)

    resampled = resampled.dropna(subset=["health_label"])
    return resampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/metropt3_raw.csv")
    parser.add_argument("--output", default="data/processed/metropt3_labeled.csv")
    parser.add_argument("--freq", default="1min", help="Frecuencia de resampleo, ej 1min, 5min")
    parser.add_argument("--pre-window-hours", type=float, default=4.0)
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    df = load_raw(args.input)
    print(f"  -> {len(df):,} rows, columns: {list(df.columns)}")

    print("Labeling by actual failure windows ...")
    df["health_label"] = build_label_column(df.index, pre_window_hours=args.pre_window_hours)
    print(df["health_label"].value_counts())

    print(f"Resampling to {args.freq} ...")
    df_resampled = resample_dataset(df, freq=args.freq)
    print(f"  -> {len(df_resampled):,} rows after resampling")
    print(df_resampled["health_label"].value_counts())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_resampled.to_csv(out_path)
    print(f"Saved in {out_path.resolve()}")


if __name__ == "__main__":
    main()
