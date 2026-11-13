'''
mpl_plots - plots for the data pipeline.

rules:
- always fig, ax (never plt.plot)
- savefig BEFORE plt.show()
- one file name per plot
- same bins and same scale when two plots are compared

every function:
- draws, saves to PLOTS / name, shows (if show=True), closes the figure
- returns one number or None (count of values outside bounds, etc.)
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PLOTS = Path(__file__).parent.resolve() / "plots"

C_TRAIN = "#996A05"
C_VAL = "#1F3E80"
C_TEST = "#1F805F"
C_CLOSE = "#661A1A"
C_SMA = "#141A6E"


def _finish(fig, name: str, show: bool, plots_dir: Path) -> None:
    '''
    tight_layout -> savefig -> show -> close. order matters:
    savefig before show, otherwise the file can be empty.
    '''
    plots_dir.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(plots_dir / name, dpi=120)
    if show:
        plt.show()
    plt.close(fig)


def plot_price_splits(close: pd.Series,
                      X_train: pd.DataFrame,
                      X_validation: pd.DataFrame,
                      X_test: pd.DataFrame,
                      window: int = 20,
                      name: str = "price_splits.png",
                      show: bool = True,
                      plots_dir: Path = PLOTS) -> None:
    '''
    close + SMA(window) with colored train / validation / test ranges.

    params:
    close: pd.Series with DatetimeIndex
    X_train, X_validation, X_test: DataFrames, only their index (dates) is used

    return:
    None
    '''
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(close, color=C_CLOSE, lw=1.2, label="close")
    ax.plot(close.rolling(window).mean(), color=C_SMA, lw=1.5, label=f"SMA{window}")

    ax.axvspan(X_train.index[0], X_train.index[-1], color=C_TRAIN, alpha=0.2, label="train")
    ax.axvspan(X_validation.index[0], X_validation.index[-1], color=C_VAL, alpha=0.2, label="validation")
    ax.axvspan(X_test.index[0], X_test.index[-1], color=C_TEST, alpha=0.2, label="test")

    ax.set_title(f"Close and SMA{window}")
    ax.set_xlabel("date")
    ax.set_ylabel("price")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    _finish(fig, name, show, plots_dir)


def plot_hist_bounds(values: pd.Series,
                     low: float,
                     high: float,
                     bins: int = 30,
                     name: str = "hist_bounds.png",
                     show: bool = True,
                     plots_dir: Path = PLOTS) -> int:
    '''
    histogram of values with vertical lines at low and high (clip bounds).

    params:
    values: pd.Series (train only)
    low, high: numbers, for example clip_low["ret"], clip_high["ret"]

    return:
    int - how many values are strictly outside [low, high]
    '''
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values.dropna(), bins=bins, color="#4C72B0", edgecolor="white")
    ax.axvline(low, color="red", linestyle="--", label="clip low")
    ax.axvline(high, color="red", linestyle="--", label="clip high")

    ax.set_title(f"Train: {values.name}")
    ax.set_xlabel(str(values.name))
    ax.set_ylabel("count")
    ax.legend()
    _finish(fig, name, show, plots_dir)

    return int((~values.between(low, high)).sum())


def plot_before_after(before: pd.Series,
                      after: pd.Series,
                      bins: int = 30,
                      name: str = "before_after_clip.png",
                      show: bool = True,
                      plots_dir: Path = PLOTS) -> int:
    '''
    two histograms side by side: before and after clip.
    bins are made once from `before` and used for both, sharey=True,
    so the plots are comparable.

    params:
    before: pd.Series (for example X_train["vol_change"])
    after: pd.Series (for example X_train_clipped["vol_change"])

    return:
    int - how many values clip changed
    '''
    edges = np.linspace(before.min(), before.max(), bins + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    axes[0].hist(before, bins=edges, color="#4C72B0", edgecolor="white")
    axes[0].set_title("before clip")
    axes[1].hist(after, bins=edges, color="#55A868", edgecolor="white")
    axes[1].set_title("after clip")
    for ax in axes:
        ax.set_xlabel(str(before.name))
    axes[0].set_ylabel("count")
    _finish(fig, name, show, plots_dir)

    return int((before != after).sum())


# def plot_normalized(train: pd.Series,
#                     validation: pd.Series,
#                     test: pd.Series,
#                     name: str = "normalized.png",
#                     show: bool = True,
#                     plots_dir: Path = PLOTS) -> dict:
#     '''
#     one normalized column for train / validation / test on one axis.
#     dashed lines at 0 and 1 = train range (min and max come from train only).

#     params:
#     train, validation, test: pd.Series with dates in index

#     return:
#     dict - {"validation": n_outside, "test": n_outside}, outside [0, 1]
#     '''
#     fig, ax = plt.subplots(figsize=(11, 4))
#     ax.plot(train, color=C_TRAIN, label="train")
#     ax.plot(validation, color=C_VAL, label="validation")
#     ax.plot(test, color=C_TEST, label="test")
#     ax.axhline(0, color="black", linestyle="--")
#     ax.axhline(1, color="black", linestyle="--")

#     ax.set_title(f"Normalized {name}") #here was bug
#     ax.set_xlabel("date")
#     ax.set_ylabel("value")
#     ax.legend()
#     fig.autofmt_xdate()
#     _finish(fig, name, show, plots_dir)

#     return {
#         "validation": int((~validation.between(0, 1)).sum()), #here bug, "between"???
#         "test": int((~test.between(0, 1)).sum()),
#     }


def plot_price_volume(df: pd.DataFrame,
                      name: str = "price_volume.png",
                      show: bool = True,
                      plots_dir: Path = PLOTS) -> None:
    '''
    price on top, volume below, common x axis, heights 3 : 1.

    params:
    df: DataFrame with DatetimeIndex and columns "close", "volume"

    return:
    None
    '''
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax1.plot(df["close"], color=C_CLOSE, label="close")
    ax1.set_ylabel("price")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.bar(df.index, df["volume"], width=1.0, color="gray")
    ax2.set_ylabel("volume")

    fig.autofmt_xdate()
    _finish(fig, name, show, plots_dir)