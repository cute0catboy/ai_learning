import numpy as np
import numpy_data_validator as nvd

np.random.seed(42)
n = 200
close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n))
close[[10, 50, 51]] = np.nan

close_ffilled = nvd.ffill_1d(close)
daily_return = nvd.return_counter(close_ffilled)
sma_ratio = nvd.sma_ratio(close_ffilled, 5)

X = np.column_stack(( #соответствует дню с idx == 4
    daily_return[3:-1],
    sma_ratio[:-1]
))
assert not np.isnan(X).any()
y = (close_ffilled[5:] > close_ffilled[4:-1]).astype(int) #boolean движение с дня с индексом 4 по 5

n_samples = X.shape[0]
split_70 = int(n_samples * 0.7)
split_85 = int(n_samples * 0.85)

X_train = X[:split_70]
y_train = y[:split_70]
X_validation = X[split_70:split_85]
y_validation = y[split_70:split_85]
X_test = X[split_85:]
y_test = y[split_85:]

assert X_train.shape[0] == y_train.shape[0]
assert X_validation.shape[0] == y_validation.shape[0]
assert X_test.shape[0] == y_test.shape[0]
X_restored = np.vstack((X_train, X_validation, X_test))
y_restored = np.concatenate((y_train, y_validation, y_test))
assert np.array_equal(X_restored, X)
assert np.array_equal(y_restored, y)

mu = X_train.mean(axis=0)     # форма (2,)
sd = X_train.std(axis=0)      # форма (2,)
low = mu - 3 * sd
high = mu + 3 * sd

X_train_clipped = np.clip(a=X_train, min=low, max=high)
X_validation_clipped = np.clip(a=X_validation, min=low, max=high)
X_test_clipped = np.clip(a=X_test, min=low, max=high)

X_min = np.min(X_train_clipped, axis=0)
X_max = np.max(X_train_clipped, axis=0)

mm = X_max - X_min
mm[mm == 0] = 1.0

X_train_normalized = (X_train_clipped-X_min) / mm
X_validation_normalized = (X_validation_clipped-X_min) / mm
X_test_normalized = (X_test_clipped-X_min) / mm

for name, arr in [("train", X_train_normalized),
                  ("val", X_validation_normalized),
                  ("test", X_test_normalized)]:
    print(name, arr.min(axis=0), arr.max(axis=0))
    assert not np.isnan(arr).any()
