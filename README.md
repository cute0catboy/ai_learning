# INTRODUCTION

this is the ai which predicts the stock exchange rate




# how to use nvd (numpy_data_validator.py) and how to create pipeline:

data validation pipeline:

1. ffill nan elements.

use ffill_1d() for 1d arrays

2. prepare X features.

use return_counter() and sma_ratio() and others features if you need

3. create X - it must be normalized by days.
4. create y - like X, but for 1 day later.
5. history split of X and y for train, validation, test:

n_samples = X.shape[0]
split_70 = int(n_samples * 0.7)
split_85 = int(n_samples * 0.85)

X_train = X[:split_70]
y_train = y[:split_70]
X_validation = X[split_70:split_85]
y_validation = y[split_70:split_85]
X_test = X[split_85:]
y_test = y[split_85:]

6. assert X_... with y_... - must be with similar shapes:

assert X_train.shape[0] == y_train.shape[0]
assert X_validation.shape[0] == y_validation.shape[0]
assert X_test.shape[0] == y_test.shape[0]

7. assert X and y restored with np.array_equal:

X_restored = np.vstack((X_train, X_validation, X_test))
y_restored = np.concatenate((y_train, y_validation, y_test))
assert np.array_equal(X_restored, X)
assert np.array_equal(y_restored, y)

8. clip X_train and X_test and X_validation:

mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)
low = mu - 3 * sd
high = mu + 3 * sd

X_train_clipped = np.clip(a=X_train, min=low, max=high)
X_validation_clipped = np.clip(a=X_validation, min=low, max=high)
X_test_clipped = np.clip(a=X_test, min=low, max=high)

9. minmax normalize X:

X_min = np.min(X_train_clipped, axis=0)
X_max = np.max(X_train_clipped, axis=0)

mm = X_max - X_min
mm[mm == 0] = 1.0

X_train_normalized = (X_train_clipped-X_min) / mm
X_validation_normalized = (X_validation_clipped-X_min) / mm
X_test_normalized = (X_test_clipped-X_min) / mm

10. final assert:

for name, arr in [("train", X_train_normalized),
                  ("val", X_validation_normalized),
                  ("test", X_test_normalized)]:
    print(name, arr.min(axis=0), arr.max(axis=0))
    assert not np.isnan(arr).any()