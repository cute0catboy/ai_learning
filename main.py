import numpy as np
import pandas as pd
from pathlib import Path
import mpl_plots as mp

PLOTS = Path(__file__).parent.resolve() / "plots"
PLOTS.mkdir(exist_ok=True)
DATA = Path(__file__).parent.resolve() / "data"
DATA.mkdir(exist_ok=True)

#github.com/Pubmarks/datasets
df2 = pd.read_csv(DATA / "ohlcv.csv", index_col="date", parse_dates=True)
assert df2.index.is_monotonic_increasing
print(df2.index.dtype)

ret = df2["close"].pct_change()
hl_range = (df2["high"] - df2["low"]) / df2["close"]
body = (df2["close"] - df2["open"]) / df2["open"]
vol_rel = df2["volume"] / df2["volume"].rolling(20).mean() - 1      # объём относительно недавнего среднего

df2["dow"] = df2.index.day_of_week
wm_df = pd.DataFrame({"ret": ret, "dow": df2["dow"]})
week_mean = wm_df.groupby("dow")["ret"].mean()

next_close = df2["close"].shift(-1)
y_raw = (next_close > df2["close"]).astype(float)
y_raw[next_close.isna()] = np.nan

data = pd.DataFrame({"ret": ret, "hl_range": hl_range, "body": body, "vol_rel": vol_rel, "y": y_raw}).replace([np.inf, -np.inf], np.nan).dropna()
print(data.shape)
print(data.iloc[0])
X = data[["ret", "hl_range", "body", "vol_rel"]]
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

for name, a, b in [("train", X_train, X_train_clipped),
                   ("val", X_validation, X_validation_clipped),
                   ("test", X_test, X_test_clipped)]:
    print(name, ((a != b).mean() * 100).round(1).to_dict())

print(X_train_normalized.max())          # какой именно столбец не равен 1
print(X_train_clipped.max())
print(X_max, X_min, mm, sep="\n")
print(np.isinf(X_train).sum())           # есть ли inf
print(X_train.nunique())                 # 1 = константный столбец
print(clip_low, clip_high, sep="\n")     # не nan ли границы

assert np.allclose(X_train_normalized.min(), 0)
assert np.allclose(X_train_normalized.max(), 1)

for name, arr in [("train", X_train_normalized),
                  ("val", X_validation_normalized),
                  ("test", X_test_normalized)]:
    print(name, arr.min(axis=0).to_list(), arr.max(axis=0).to_list())
    assert not arr.isna().any().any()

mp.plot_price_splits(df2["close"], X_train, X_validation, X_test)