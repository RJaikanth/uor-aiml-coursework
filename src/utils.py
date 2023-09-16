"""
car_prices.utils.py
Author: Raghhuveer Jaikanth
Date  : 08/03/2023

Contains additional utility functions.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    out_ = pd.DataFrame(
        df.isnull().sum().sort_values(ascending = False), columns = ["Num Missing Values"]
    )
    out_["Percentage"] = out_["Num Missing Values"] * 100 / len(df)

    return out_


def bin_and_count(df: pd.DataFrame, col: str, bins: list[int], min_count: int = 100) -> pd.Series:
    cut_ = pd.cut(df[col], bins, ordered = True, right = True)
    temp_ = df.groupby([cut_]).count().iloc[:, 0]
    return temp_[temp_ > min_count]


def get_iqr(df: pd.DataFrame, col: str) -> tuple[int, int]:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    return q1, q3


def get_iqr_mask(df: pd.DataFrame, col: str, multiplier: float = 1.5) -> pd.Series:
    # Get Quarters
    q1, q3 = get_iqr(df, col)
    iqr = (q3 - q1)

    print(q1, q3, iqr)

    mask = df[col].between(q1 - multiplier * iqr, q3 + multiplier * iqr, inclusive = "both")

    return mask


def filter_iqr(df: pd.DataFrame, col: str, multiplier: float = 1.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = get_iqr_mask(df, col, multiplier)

    outliers = df.loc[~mask, :]
    inliers = df.loc[mask, :]

    return inliers, outliers


class Logger(object):
    def __init__(self, filename: str, mode: str = 'w'):
        self.terminal = sys.stdout
        self.log = open(filename, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

    def close(self):
        self.log.close()


def set_mpl_settings():
    # Matplotlib Settings
    plt.style.use(["seaborn-ticks"])
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.grid.which"] = "both"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["figure.figsize"] = (10, 7.5)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["legend.fancybox"] = True
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.loc"] = "best"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    sns.set(rc = dict(plt.rcParams))

    # Output settings
    pd.set_option('display.float_format', lambda x: f"{x: 5.5f}")
    np.set_printoptions(precision = 5)
    np.set_printoptions(suppress = True)


def box_plot(df: pd.DataFrame, col: str, **kwargs):
    sns.boxplot(data = df, x = col, **kwargs)
    plt.title(col)
    plt.show()
