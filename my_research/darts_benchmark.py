#!/usr/bin/env python3
"""Darts-based benchmark for LTSF-Linear paper datasets and model families."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    file_name: str
    seq_len: int
    horizons: List[int]
    split: str


DATASETS: Dict[str, DatasetSpec] = {
    "ETTh1": DatasetSpec("ETTh1", "ETTh1.csv", 336, [96, 192, 336, 720], "ett_hour"),
    "ETTh2": DatasetSpec("ETTh2", "ETTh2.csv", 336, [96, 192, 336, 720], "ett_hour"),
    "ETTm1": DatasetSpec("ETTm1", "ETTm1.csv", 336, [96, 192, 336, 720], "ett_minute"),
    "ETTm2": DatasetSpec("ETTm2", "ETTm2.csv", 336, [96, 192, 336, 720], "ett_minute"),
    "Electricity": DatasetSpec("Electricity", "electricity.csv", 336, [96, 192, 336, 720], "custom"),
    "Traffic": DatasetSpec("Traffic", "traffic.csv", 336, [96, 192, 336, 720], "custom"),
    "Weather": DatasetSpec("Weather", "weather.csv", 336, [96, 192, 336, 720], "custom"),
    "Exchange-Rate": DatasetSpec("Exchange-Rate", "exchange_rate.csv", 336, [96, 192, 336, 720], "custom"),
    "ILI": DatasetSpec("ILI", "national_illness.csv", 104, [24, 36, 48, 60], "custom"),
}

MODELS = [
    "AutoARIMA",
    "Linear",
    "NLinear",
    "DLinear",
    "LSTM",
    "GRU",
    "N-BEATS",
    "Transformer",
]


def _split_points(length: int, split: str, seq_len: int) -> tuple[int, int]:
    if split == "ett_hour":
        train_end = 12 * 30 * 24
        val_end = train_end + 4 * 30 * 24
    elif split == "ett_minute":
        train_end = 12 * 30 * 24 * 4
        val_end = train_end + 4 * 30 * 24 * 4
    else:
        train_end = int(length * 0.7)
        val_end = int(length * 0.8)
    train_end = max(train_end, seq_len + 1)
    val_end = max(val_end, train_end + 1)
    return min(train_end, length - 2), min(val_end, length - 1)


def _load_csv_as_timeseries(csv_path: Path):
    from darts import TimeSeries

    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        time_col = "date"
    else:
        time_col = df.columns[0]

    df[time_col] = pd.to_datetime(df[time_col])
    value_cols = [c for c in df.columns if c != time_col]
    if not value_cols:
        raise ValueError(f"No value columns found in {csv_path}")

    return TimeSeries.from_dataframe(df, time_col=time_col, value_cols=value_cols, fill_missing_dates=False)


def _build_model_factory(model_name: str, seq_len: int, pred_len: int, epochs: int, seed: int) -> Callable[[], object]:
    from darts.models import (
        AutoARIMA,
        DLinearModel,
        LinearRegressionModel,
        NBEATSModel,
        NLinearModel,
        RNNModel,
        TransformerModel,
    )

    if model_name == "AutoARIMA":
        return lambda: AutoARIMA()
    if model_name == "Linear":
        return lambda: LinearRegressionModel(lags=seq_len, output_chunk_length=pred_len)
    if model_name == "NLinear":
        return lambda: NLinearModel(input_chunk_length=seq_len, output_chunk_length=pred_len, n_epochs=epochs, random_state=seed)
    if model_name == "DLinear":
        return lambda: DLinearModel(input_chunk_length=seq_len, output_chunk_length=pred_len, n_epochs=epochs, random_state=seed)
    if model_name == "LSTM":
        return lambda: RNNModel(model="LSTM", input_chunk_length=seq_len, training_length=seq_len + pred_len, n_epochs=epochs, random_state=seed)
    if model_name == "GRU":
        return lambda: RNNModel(model="GRU", input_chunk_length=seq_len, training_length=seq_len + pred_len, n_epochs=epochs, random_state=seed)
    if model_name == "N-BEATS":
        return lambda: NBEATSModel(input_chunk_length=seq_len, output_chunk_length=pred_len, n_epochs=epochs, random_state=seed)
    if model_name == "Transformer":
        return lambda: TransformerModel(
            input_chunk_length=seq_len,
            output_chunk_length=pred_len,
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
            dropout=0.1,
            batch_size=32,
            n_epochs=epochs,
            random_state=seed,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def _fit_predict(model_name: str, model_factory: Callable[[], object], train_series, pred_len: int):
    from darts.metrics import mae, mse

    model = model_factory()
    train_start = time.perf_counter()

    if model_name == "AutoARIMA" and train_series.n_components > 1:
        models = []
        for idx in range(train_series.n_components):
            component_model = model_factory()
            component_model.fit(train_series.univariate_component(idx))
            models.append(component_model)
        train_time = time.perf_counter() - train_start

        infer_start = time.perf_counter()
        preds = [m.predict(pred_len) for m in models]
        pred = preds[0]
        for p in preds[1:]:
            pred = pred.stack(p)
        inference_time = time.perf_counter() - infer_start
        return train_time, inference_time, pred

    model.fit(train_series)
    train_time = time.perf_counter() - train_start

    infer_start = time.perf_counter()
    pred = model.predict(pred_len)
    inference_time = time.perf_counter() - infer_start
    return train_time, inference_time, pred


def run_benchmark(args: argparse.Namespace) -> pd.DataFrame:
    rows = []

    selected_datasets = [DATASETS[d] for d in args.datasets]
    selected_models = args.models

    if args.dry_run:
        for ds in selected_datasets:
            for horizon in ds.horizons:
                for model_name in selected_models:
                    rows.append(
                        {
                            "dataset": ds.name,
                            "data_file": ds.file_name,
                            "model": model_name,
                            "seq_len": ds.seq_len,
                            "pred_len": horizon,
                            "mae": np.nan,
                            "mse": np.nan,
                            "train_time_sec": np.nan,
                            "inference_time_sec": np.nan,
                            "status": "DRY_RUN",
                            "error": "",
                        }
                    )
        return pd.DataFrame(rows)

    from darts.dataprocessing.transformers import Scaler
    from darts.metrics import mae, mse

    data_root = Path(args.data_root).resolve()

    for ds in selected_datasets:
        csv_path = data_root / ds.file_name
        if not csv_path.exists():
            for horizon in ds.horizons:
                for model_name in selected_models:
                    rows.append(
                        {
                            "dataset": ds.name,
                            "data_file": str(csv_path),
                            "model": model_name,
                            "seq_len": ds.seq_len,
                            "pred_len": horizon,
                            "mae": np.nan,
                            "mse": np.nan,
                            "train_time_sec": np.nan,
                            "inference_time_sec": np.nan,
                            "status": "MISSING_DATASET",
                            "error": f"Missing file: {csv_path}",
                        }
                    )
            continue

        full_series = _load_csv_as_timeseries(csv_path)
        train_end, val_end = _split_points(len(full_series), ds.split, ds.seq_len)
        train_series = full_series[:val_end]

        scaler = Scaler()
        scaled_train = scaler.fit_transform(train_series)
        scaled_full = scaler.transform(full_series)

        for horizon in ds.horizons:
            target = full_series[val_end : val_end + horizon]
            if len(target) < horizon:
                for model_name in selected_models:
                    rows.append(
                        {
                            "dataset": ds.name,
                            "data_file": str(csv_path),
                            "model": model_name,
                            "seq_len": ds.seq_len,
                            "pred_len": horizon,
                            "mae": np.nan,
                            "mse": np.nan,
                            "train_time_sec": np.nan,
                            "inference_time_sec": np.nan,
                            "status": "INSUFFICIENT_TEST_POINTS",
                            "error": f"Need {horizon} points after split, got {len(target)}",
                        }
                    )
                continue

            for model_name in selected_models:
                try:
                    factory = _build_model_factory(model_name, ds.seq_len, horizon, args.epochs, args.seed)
                    train_t, infer_t, pred_scaled = _fit_predict(model_name, factory, scaled_train, horizon)
                    pred = scaler.inverse_transform(pred_scaled)
                    actual = target

                    rows.append(
                        {
                            "dataset": ds.name,
                            "data_file": str(csv_path),
                            "model": model_name,
                            "seq_len": ds.seq_len,
                            "pred_len": horizon,
                            "mae": float(mae(actual, pred)),
                            "mse": float(mse(actual, pred)),
                            "train_time_sec": train_t,
                            "inference_time_sec": infer_t,
                            "status": "OK",
                            "error": "",
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    rows.append(
                        {
                            "dataset": ds.name,
                            "data_file": str(csv_path),
                            "model": model_name,
                            "seq_len": ds.seq_len,
                            "pred_len": horizon,
                            "mae": np.nan,
                            "mse": np.nan,
                            "train_time_sec": np.nan,
                            "inference_time_sec": np.nan,
                            "status": "FAILED",
                            "error": str(exc),
                        }
                    )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Darts benchmark for datasets used in arXiv:2205.13504v3")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/runner/work/LTSF-Linear/LTSF-Linear/dataset",
        help="Directory containing ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv, electricity.csv, traffic.csv, weather.csv, exchange_rate.csv, national_illness.csv",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
        help="Datasets to benchmark",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        choices=MODELS,
        help="Model families to benchmark",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Epochs for neural models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry_run", action="store_true", help="Only materialize benchmark matrix without fitting models")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/home/runner/work/LTSF-Linear/LTSF-Linear/my_research/results/benchmark_results.csv",
        help="CSV path for benchmark rows",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="/home/runner/work/LTSF-Linear/LTSF-Linear/my_research/results/benchmark_summary.csv",
        help="CSV path for aggregated summary",
    )
    parser.add_argument(
        "--run_metadata_json",
        type=str,
        default="/home/runner/work/LTSF-Linear/LTSF-Linear/my_research/results/run_metadata.json",
        help="JSON path for run metadata",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = pd.Timestamp.utcnow().isoformat()
    df = run_benchmark(args)

    output_csv = Path(args.output_csv)
    summary_csv = Path(args.summary_csv)
    metadata_json = Path(args.run_metadata_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_csv, index=False)

    summary = (
        df[df["status"] == "OK"]
        .groupby(["dataset", "model"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mse_mean=("mse", "mean"),
            train_time_mean_sec=("train_time_sec", "mean"),
            inference_time_mean_sec=("inference_time_sec", "mean"),
            runs=("status", "count"),
        )
    )
    summary.to_csv(summary_csv, index=False)

    metadata = {
        "started_at_utc": started_at,
        "finished_at_utc": pd.Timestamp.utcnow().isoformat(),
        "data_root": str(Path(args.data_root).resolve()),
        "datasets": args.datasets,
        "models": args.models,
        "epochs": args.epochs,
        "seed": args.seed,
        "dry_run": bool(args.dry_run),
        "rows": int(len(df)),
        "ok_rows": int((df["status"] == "OK").sum()),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
