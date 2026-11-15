import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment

from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["ALPACA_API_KEY"]
API_SECRET = os.environ["ALPACA_API_SECRET"]

client = StockHistoricalDataClient(API_KEY, API_SECRET)

# METRICS OF THE DATA PULL
start_time = datetime(2025, 10, 1, 10, 30, tzinfo=timezone.utc)
end_time = datetime(2025, 12, 1, 10, 30, tzinfo=timezone.utc)
symbol = "TSLA"
horizon = 10  # minutes
lookback = 20
eps = 0.0005 # tolerance for labeling
breakout_buffer = 0.0005


def fetch_1min_bars_single_symbol(
    symbol: str,
    start_utc: datetime,
    end_utc: datetime,
    feed: DataFeed = DataFeed.IEX,
) -> pd.DataFrame:
    assert start_utc.tzinfo is not None and end_utc.tzinfo is not None, "Use timezone-aware UTC datetimes"

    chunks = []
    day = start_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    while day < end_utc:
        day_end = min(day + timedelta(days=1), end_utc)

        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=max(day, start_utc),
            end=day_end,
            feed=feed,
            adjustment=Adjustment.RAW,
        )

        df = client.get_stock_bars(req).df
        if len(df) > 0:
            df = df.reset_index()
            if "symbol" in df.columns:
                df = df[df["symbol"] == symbol]
            df = df.sort_values("timestamp").set_index("timestamp")
            chunks.append(df[["open", "high", "low", "close", "volume"]])

        day = day + timedelta(days=1)

    if not chunks:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    out = pd.concat(chunks).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out

def build_dataset_from_bars(
    bars: pd.DataFrame,
    lookback: int = 20,
    horizon: int = 10,
    eps_close: float = 0.0,              # original close-to-close tolerance
    breakout_buffer: float = 0.0005,     # "a lil" buffer, e.g. 0.0005 = 5 bps
):
    bars = bars.sort_index().copy()

    # segmenting to avoid crossing gaps
    dt = bars.index.to_series().diff()
    new_seg = (dt.isna()) | (dt > pd.Timedelta(minutes=1))
    seg_id = new_seg.cumsum()

    # ---- features (NO future leakage) ----
    bars["oc_pct"]   = (bars["close"] - bars["open"]) / bars["open"]
    bars["hl_pct"]   = (bars["high"] - bars["low"]) / bars["close"]
    bars["logvol"]   = np.log1p(bars["volume"])
    bars["logret1"]  = bars.groupby(seg_id)["close"].transform(lambda s: np.log(s).diff())

    feature_cols = ["logvol", "oc_pct", "hl_pct", "logret1"]

    X_list, y_list, t_list, profit_list = [], [], [], []

    for _, seg in bars.groupby(seg_id):
        if len(seg) < (lookback + horizon + 1):
            continue

        seg_arr = seg[feature_cols].to_numpy(dtype=np.float32)

        # we need arrays for label/profit calculations
        seg_high  = seg["high"].to_numpy(dtype=np.float32)
        seg_close = seg["close"].to_numpy(dtype=np.float32)

        # i is the "current bar" index (end of lookback window)
        # forward window will be i+1 ... i+horizon
        last_i = len(seg) - horizon - 1

        for i in range(lookback - 1, last_i + 1):
            window = seg_arr[i - lookback + 1 : i + 1]
            if np.isnan(window).any():
                continue

            # past max high in lookback window (includes current bar)
            past_max = float(seg_high[i - lookback + 1 : i + 1].max())
            thr = past_max * (1.0 + breakout_buffer)

            # forward window: next horizon minutes
            fwd_slice = slice(i + 1, i + 1 + horizon)
            fwd_max = float(seg_high[fwd_slice].max())

            # original condition: close[t+horizon] > close[t] * (1+eps_close)
            close_now = float(seg_close[i])
            close_fwd = float(seg_close[i + horizon])
            close_up = close_fwd > close_now * (1.0 + eps_close)

            breakout = fwd_max >= thr

            y = 1 if (breakout or close_up) else 0

            # "profit" as best achievable return within the forward window
            profit_ret = (fwd_max / close_now) - 1.0

            X_list.append(window)
            y_list.append(y)
            t_list.append(seg.index[i])
            profit_list.append(profit_ret)

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, lookback, len(feature_cols)), dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    t = pd.DatetimeIndex(t_list)
    profit = np.array(profit_list, dtype=np.float32)

    return X, y, t, profit

def tp_or_time_stop_return(bars, t_index, horizon=10, tp=0.003, cost=0.0002):
    """
    tp: take-profit as fraction (0.003 = 0.3%)
    Returns array of realized returns for each timestamp in t_index.
    Entry: open at t+1
    Exit: tp hit (using highs) OR close at t+horizon
    """
    rets = []
    for ts in t_index:
        ts_entry = ts + pd.Timedelta(minutes=1)
        ts_exit  = ts + pd.Timedelta(minutes=horizon)

        # Need these timestamps available
        if ts_entry not in bars.index or ts_exit not in bars.index:
            rets.append(np.nan)
            continue

        entry = float(bars.loc[ts_entry, "open"])
        tp_price = entry * (1.0 + tp)

        # Check if TP was hit in minutes [t+1 ... t+horizon]
        fwd = bars.loc[ts_entry:ts_exit]
        hit = (fwd["high"] >= tp_price).any()

        if hit:
            exit_price = tp_price
        else:
            exit_price = float(bars.loc[ts_exit, "close"])

        r = (exit_price / entry) - 1.0 - cost
        rets.append(r)

    return np.array(rets, dtype=np.float64)








# MAIN SCRIPT

from sklearn.ensemble import HistGradientBoostingClassifier

# Fetch DATAFRAMES and FILTER TO REGULAR HOURS
bars_df = fetch_1min_bars_single_symbol(symbol, start_time, end_time)
bars_df = bars_df.tz_convert("America/New_York")
bars_df = bars_df.between_time("09:30", "16:00")
bars_df = bars_df.tz_convert("UTC")

#Build FEATURES, LABELS AND TIMESTAMPS
X, y, t, profit = build_dataset_from_bars(bars_df, lookback, horizon, eps, breakout_buffer)


#PRINT STUFF
print(bars_df)
print("bars:", bars_df.shape)
print("X_shape:", X.shape, "y_data:", y, "t_len:", len(t))

#FLATTEN FEATURES FOR CLASSICAL ML (3D TO 2D)
X_flat = X.reshape(X.shape[0], -1)

# SPLIT INTO TRAIN, VAL, TEST
N = len(y)
i_train = int(N * 0.70)
i_val   = int(N * 0.85)

X_train, y_train, t_train = X_flat[:i_train], y[:i_train], t[:i_train]
X_val,   y_val,   t_val   = X_flat[i_train:i_val], y[i_train:i_val], t[i_train:i_val]
X_test,  y_test,  t_test  = X_flat[i_val:], y[i_val:], t[i_val:]

# MODEL TRAINING
model = HistGradientBoostingClassifier(
    max_depth=3,
    learning_rate=0.05,
    max_iter=500,
    random_state=42
)

model.fit(X_train, y_train)

# EVALUATION
from sklearn.metrics import roc_auc_score, accuracy_score

p_val = model.predict_proba(X_val)[:, 1]
pred_val = (p_val >= 0.5).astype(int)

print("val AUC:", roc_auc_score(y_val, p_val))
print("val accuracy:", accuracy_score(y_val, pred_val))
print("majority baseline:", max(y_val.mean(), 1 - y_val.mean()))


# PICKING THRESHOLD
def forward_return(bars, t_index, horizon=10):
    # assumes bars index is timestamps and contains 'close'
    rets = []
    for ts in t_index:
        ts2 = ts + pd.Timedelta(minutes=horizon)
        if ts in bars.index and ts2 in bars.index:
            r = (bars.loc[ts2, "close"] / bars.loc[ts, "close"]) - 1.0
            rets.append(float(r))
        else:
            rets.append(np.nan)
    return np.array(rets, dtype=np.float64)

r_val = tp_or_time_stop_return(bars_df, t_val, horizon=horizon, tp=0.003, cost=0.0002)

mask = ~np.isnan(r_val)
p_val, y_val, r_val = p_val[mask], y_val[mask], r_val[mask]

def pick_threshold(p, r, cost_per_trade=0.0002):  # 2 bps example
    best = None
    for thr in np.arange(0.50, 0.71, 0.01):
        take = p >= thr
        if take.sum() < 20:   # avoid tiny sample thresholds
            continue
        avg_ret = r[take].mean() - cost_per_trade
        trades_per_day = take.sum()  # not really "per day" unless your slice is 1 day; just informative
        cand = (thr, avg_ret, take.sum())
        if best is None or avg_ret > best[1]:
            best = cand
    return best  # (threshold, avg_net_return, num_trades)

best = pick_threshold(p_val, r_val, cost_per_trade=0.0002)
print("best threshold:", best)