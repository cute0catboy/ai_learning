# INTRODUCTION

this is the ai which predicts the stock exchange rate

# how to create pipeline with pandas:

data validation pipeline:

1. ffill nan elements.

<blockquote>
<pre><code>s_ff = s.ffill()</code></pre>
</blockquote>

2. prepare features you need. ret and sma_ratio for example.

<blockquote>
<pre><code>ret = s_ff.pct_change()
sma = s_ff.rolling(window).mean()
sma_ratio = s_ff / sma - 1</code></pre>
</blockquote>

3. create y (1 if next day close is higher). use float, so the last row can be nan (it has no real target).

<blockquote>
<pre><code>next_close = s_ff.shift(-1)
y = (next_close > s_ff).astype(float)
y[next_close.isna()] = np.nan</code></pre>
</blockquote>

4. build X and y from one table with one dropna(), so rows always match by date.
the first row is the day with index window-1.

<blockquote>
<pre><code>data = pd.DataFrame({"ret": ret, "sma_ratio": sma_ratio, "y": y}).dropna()
X = data[["ret", "sma_ratio"]]
y = data["y"].astype(int)
assert not X.isna().any().any()</code></pre>
</blockquote>

5. history split of X and y for train, validation, test (only by position, never shuffle):

<blockquote>
<pre><code>n_samples = X.shape[0]
split_70 = int(n_samples * 0.7)
split_85 = int(n_samples * 0.85)

X_train = X.iloc[:split_70]
y_train = y.iloc[:split_70]
X_validation = X.iloc[split_70:split_85]
y_validation = y.iloc[split_70:split_85]
X_test = X.iloc[split_85:]
y_test = y.iloc[split_85:]</code></pre>
</blockquote>

6. assert X_... with y_... - must be with similar shapes:

<blockquote>
<pre><code>assert X_train.shape[0] == y_train.shape[0]
assert X_validation.shape[0] == y_validation.shape[0]
assert X_test.shape[0] == y_test.shape[0]</code></pre>
</blockquote>

7. assert X and y restored (pd.concat keeps dates, .equals compares values and index):

<blockquote>
<pre><code>assert pd.concat([X_train, X_validation, X_test]).equals(X)
assert pd.concat([y_train, y_validation, y_test]).equals(y)</code></pre>
</blockquote>

8. clip X_train, X_validation, X_test with bounds from train only, per column.
axis=1 here means "align low/high with columns".
Note: pandas std() uses ddof=1, numpy ddof=0.

<blockquote>
<pre><code>mu = X_train.mean()
sd = X_train.std()
low = mu - 3 * sd
high = mu + 3 * sd

X_train_clipped = X_train.clip(lower=low, upper=high, axis=1)
X_validation_clipped = X_validation.clip(lower=low, upper=high, axis=1)
X_test_clipped = X_test.clip(lower=low, upper=high, axis=1)</code></pre>
</blockquote>

9. minmax normalize X (min and max only from clipped train):

<blockquote>
<pre><code>X_min = X_train_clipped.min()
X_max = X_train_clipped.max()

mm = X_max - X_min
mm[mm == 0] = 1.0

X_train_normalized = (X_train_clipped - X_min) / mm
X_validation_normalized = (X_validation_clipped - X_min) / mm
X_test_normalized = (X_test_clipped - X_min) / mm</code></pre>
</blockquote>

10. final assert (train must be exactly [0, 1], val and test may go outside):

<blockquote>
<pre><code>for name, arr in [("train", X_train_normalized),
                  ("val", X_validation_normalized),
                  ("test", X_test_normalized)]:
    print(name, arr.min().to_list(), arr.max().to_list())
    assert not arr.isna().any().any()</code></pre>
</blockquote>