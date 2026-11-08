import numpy as np

X = np.random.rand(100, 4)
y = np.random.randint(0, 2, 100)

# 1. Граница разделения считаем от размера данных
n = X.shape[0]            # 100 наблюдений
split_70 = int(n * 0.7)
split_85 = int(n * 0.85)

# 2. Хронологическое разделение: один и тот же индекс для X и y
X_train = X[:split_70]
y_train = y[:split_70]
X_validation = X[split_70:split_85]
y_validation = y[split_70:split_85]
X_test = X[split_85:]
y_test = y[split_85:]

# 3. Размерности
print(X_train.shape, y_train.shape)
print(X_validation.shape, y_validation.shape)
print(X_test.shape, y_test.shape)

# 4. Соответствие X и y
assert X_train.shape[0] == y_train.shape[0]
assert X_validation.shape[0] == y_validation.shape[0]
assert X_test.shape[0] == y_test.shape[0]
X_restored = np.vstack((X_train, X_validation, X_test))
y_restored = np.concatenate((y_train, y_validation, y_test))
assert np.array_equal(X_restored, X)
assert np.array_equal(y_restored, y)

X_min = np.min(X_train, axis=0)
X_max = np.max(X_train, axis=0)

X_train_normalized = (X_train-X_min) / (X_max - X_min)
X_validation_normalized = (X_validation-X_min) / (X_max - X_min)
X_test_normalized = (X_test-X_min) / (X_max - X_min)

print(X_train_normalized)
print(X_validation_normalized)
print(X_test_normalized)