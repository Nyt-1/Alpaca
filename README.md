# Alpaca Gradient Boosting

Pulls 1-minute TSLA bars from the Alpaca market data API, builds lookback-window features, and trains a `HistGradientBoostingClassifier` to predict short-horizon upward moves/breakouts. Includes a simple take-profit / time-stop backtest for picking a probability threshold.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your Alpaca keys
```

## Run

```bash
python Gradient_Boosting.py
```

`Work_in_progress.ipynb` is an exploratory notebook.
