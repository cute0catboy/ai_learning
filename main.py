import numpy as np
import pandas as pd

np.random.seed(42)
n = 200
close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n))
close[[10, 50, 51]] = np.nan

dates = pd.date_range("2024-01-01", periods=n, freq="B")
close = pd.Series(close, index=dates, name="close")

close_ffilled = close.ffill()
daily_return = close_ffilled.pct_change()

window = 5
sma = close_ffilled.rolling(window).mean()
sma_ratio = close_ffilled / sma - 1

next_close = close_ffilled.shift(-1)
y = (next_close > close_ffilled).astype(float)
y[next_close.isna()] = np.nan   # у последнего дня нет настоящей цели

data = pd.DataFrame({"ret": daily_return, "sma_ratio": sma_ratio, "y": y}).dropna()
X = data[["ret", "sma_ratio"]]   # первая строка: день с индексом 4
y = data["y"].astype(int)        # y[i] = 1, если завтра цена выше сегодняшней
assert X.shape == (195, 2)
assert not X.isna().any().any()

n_samples = X.shape[0]
split_70 = int(n_samples * 0.7)
split_85 = int(n_samples * 0.85)

X_train = X.iloc[:split_70]
y_train = y.iloc[:split_70]
X_validation = X.iloc[split_70:split_85]
y_validation = y.iloc[split_70:split_85]
X_test = X.iloc[split_85:]
y_test = y.iloc[split_85:]

assert X_train.shape[0] == y_train.shape[0]
assert X_validation.shape[0] == y_validation.shape[0]
assert X_test.shape[0] == y_test.shape[0]
assert pd.concat([X_train, X_validation, X_test]).equals(X)
assert pd.concat([y_train, y_validation, y_test]).equals(y)

mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)
high = mu + 3 * sd

X_train_clipped = X_train.clip(lower=low, upper=high, axis=1)
X_validation_clipped = X_validation.clip(lower=low, upper=high, axis=1)
X_test_clipped = X_test.clip(lower=low, upper=high, axis=1)

X_min = X_train_clipped.min(axis=0)
X_max = X_train_clipped.max(axis=0)

mm = X_max - X_min
mm[mm == 0] = 1.0

X_train_normalized = (X_train_clipped - X_min) / mm
X_validation_normalized = (X_validation_clipped - X_min) / mm
X_test_normalized = (X_test_clipped - X_min) / mm

for name, arr in [("train", X_train_normalized),
                  ("val", X_validation_normalized),
                  ("test", X_test_normalized)]:
    print(name, arr.min(axis=0).to_list(), arr.max(axis=0).to_list())
    assert not arr.isna().any()