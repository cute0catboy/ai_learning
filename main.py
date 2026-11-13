import numpy as np
import pandas as pd
from pathlib import Path
import mpl_plots as mp
DATA = Path(__file__).parent.resolve() / "data"
PLOTS = Path(__file__).parent.resolve() / "plots"
DATA.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

np.random.seed(7)
n = 300
dates = pd.date_range("2023-01-02", periods=n, freq="B")

close = 100 * np.cumprod(1 + np.random.normal(0.0004, 0.012, n))
open_ = close * (1 + np.random.normal(0, 0.004, n))
high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.004, n)))
low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.004, n)))
volume = np.random.randint(800_000, 1_500_000, n)

df = pd.DataFrame({"open": open_, "high": high, "low": low,
                   "close": close, "volume": volume}, index=dates)
df.to_csv(DATA / "prices.csv", index_label="Date")

df2 = pd.read_csv(DATA / "prices.csv", index_col="Date", parse_dates=True)
assert df2.index.is_monotonic_increasing
print(df2.index.dtype)

print(df2.loc["2023-03"])
print(len(df2.loc["2023-03"]))

ret = df2["close"].pct_change()
hl_range = (df2["high"] - df2["low"]) / df2["close"]
body = (df2["close"] - df2["open"]) / df2["open"]
vol_change = df2["volume"].pct_change()

df2["dow"] = df2.index.day_of_week
wm_df = pd.DataFrame({"ret": ret, "dow": df2["dow"]})
week_mean = wm_df.groupby("dow")["ret"].mean()
print(week_mean)

next_close = df2["close"].shift(-1)
y_raw = (next_close > df2["close"]).astype(float)
y_raw[next_close.isna()] = np.nan

data = pd.DataFrame({"ret": ret, "hl_range": hl_range, "body": body, "vol_change": vol_change, "y": y_raw}).dropna()
print(data.shape)
print(data.iloc[0])
X = data[["ret", "hl_range", "body", "vol_change"]]
y = data["y"].astype(int)
assert not X.isna().any().any()

#weekly тут просто так
#weekly = df2.resample("W").agg({
#    "open": "first",
#    "high": "max",
#    "low": "min",
#    "close": "last",
#    "volume": "sum",
#})
#
#print(weekly.shape)
#print(weekly.head(3))
#first_week = df2.iloc[:5]
#assert weekly.iloc[0]["high"] == first_week["high"].max()
#assert weekly.iloc[0]["volume"] == first_week["volume"].sum()
# тут кончается

n_samples = X.shape[0]
split_70 = int(n_samples * 0.7)
split_85 = int(n_samples * 0.85)

X_train = X.iloc[:split_70]
y_train = y.iloc[:split_70]
X_validation = X.iloc[split_70:split_85]
y_validation = y.iloc[split_70:split_85]
X_test = X.iloc[split_85:]
y_test = y.iloc[split_85:]

print(y_train.mean(), y_validation.mean(), y_test.mean())

assert X_train.shape[0] == y_train.shape[0]
assert X_validation.shape[0] == y_validation.shape[0]
assert X_test.shape[0] == y_test.shape[0]

assert pd.concat([X_train, X_validation, X_test]).equals(X)
assert pd.concat([y_train, y_validation, y_test]).equals(y)

mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)
clip_low = mu - 3 * sd
clip_high = mu + 3 * sd

X_train_clipped = X_train.clip(lower=clip_low, upper=clip_high, axis=1)
X_validation_clipped = X_validation.clip(lower=clip_low, upper=clip_high, axis=1)
X_test_clipped = X_test.clip(lower=clip_low, upper=clip_high, axis=1)

print((X_train != X_train_clipped).sum())

X_min = X_train_clipped.min(axis=0)
X_max = X_train_clipped.max(axis=0)

mm = X_max - X_min
mm[mm == 0] = 1.0

X_train_normalized = (X_train_clipped - X_min) / mm
X_validation_normalized = (X_validation_clipped - X_min) / mm
X_test_normalized = (X_test_clipped - X_min) / mm

assert np.allclose(X_train_normalized.min(), 0)
assert np.allclose(X_train_normalized.max(), 1)

for name, arr in [("train", X_train_normalized),
                  ("val", X_validation_normalized),
                  ("test", X_test_normalized)]:
    print(name, arr.min(axis=0).to_list(), arr.max(axis=0).to_list())
    assert not arr.isna().any().any()

mp.plot_price_splits(df2["close"], X_train, X_validation, X_test)