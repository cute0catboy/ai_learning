import numpy as np

data = np.array([
    [100, 105, 98, 103, 1500],
    [103, 108, 101, 106, 1800],
    [106, 110, 104, 105, 2100],
    [105, 109, 102, 108, 1700],
    [108, 115, 107, 113, 2500]
])

daily_return = np.diff(data[:, 3]) / data[:-1, 3] * 100
print(daily_return)

highlow_range = data[:, 1] - data[:, 2]
print(highlow_range)

body = data[:, 3] - data[:, 0]
print(body)

volume_change = np.diff(data[:, 4]) / data[:-1, 4] * 100
print(volume_change)

target = data[:-1, 3] < data[1:, 3]
print(target)

features = np.column_stack((
    daily_return,
    highlow_range[1:],
    body[1:],
    volume_change,
))
print(features)

x = features
y = target

max_x = np.max(x, axis=0)
min_x = np.min(x, axis=0)

scaled_x = (x - min_x) / (max_x - min_x)
print(scaled_x)