# my_research

This directory contains a Darts-based benchmark implementation aligned with the datasets used in **arXiv-2205.13504v3**.

## Included artifacts

- `darts_benchmark.py`: benchmark runner for:
  - AutoARIMA
  - Linear
  - NLinear
  - DLinear
  - RNN-LSTM
  - RNN-GRU
  - N-BEATS
  - Transformer
- `model_explanations.tex`: detailed LaTeX explanation of every used model and the benchmark metric protocol.
- `output/`: generated outputs (`benchmark_results.csv`, `benchmark_summary.csv`, `run_metadata.json`).

## Datasets expected (same as paper)

Place these CSV files under `./dataset` in repository root:

- `ETTh1.csv`
- `ETTh2.csv`
- `ETTm1.csv`
- `ETTm2.csv`
- `electricity.csv`
- `traffic.csv`
- `weather.csv`
- `exchange_rate.csv`
- `national_illness.csv`

## Install dependencies

From repository root:

```bash
pip install -r requirements.txt
pip install "u8darts[torch]" pmdarima
```

## Run benchmark

Dry run (writes benchmark matrix without training):

```bash
python my_research/darts_benchmark.py --dry_run
```

Full run:

```bash
python my_research/darts_benchmark.py --epochs 10
```

## Output files

- `my_research/output/benchmark_results.csv`
- `my_research/output/benchmark_summary.csv`
- `my_research/output/run_metadata.json`

> Note: after `--dry_run`, `benchmark_summary.csv` can contain only headers because no model training/evaluation rows are marked as `OK`.
