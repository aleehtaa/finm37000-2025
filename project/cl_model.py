"""Basic modeling harness for CL futures term structure and option skew."""

from __future__ import annotations

import sys
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg
from typing import Sequence
import statsmodels.api as sm

import warnings

warnings.filterwarnings("ignore")

# make repo imports work when run as script
repo_root = Path.cwd()
while not (repo_root / "pyproject.toml").exists() and repo_root != repo_root.parent:
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root))

from project.helpers import init_client
from project.cl_futures_data import (
    load_continuous_futures_data,
    build_term_structure_history,
)
from project.cl_options_data import (
    load_options_data,
    calculate_skew,
)


def prepare_data(
    start: str,
    end: str,
    client: db.Historical,
    constant_targets: Sequence[float] = (21.0, 42.0),
    parent_fut: str = "CL",
    parent_opt: str = "LO",
    reload: bool = False,
    target_delta: float = 0.25,
) -> pd.DataFrame:
    """pull futures + options, build slopes and skew, and merge for modeling"""
    # futures with slopes
    futures_df = load_continuous_futures_data(
        start=start,
        end=end,
        client=client,
        parent=parent_fut,
        reload=reload,
    )
    term_history = build_term_structure_history(futures_df, constant_targets)

    # options iv/delta/skew
    opt_df = load_options_data(
        client=client,
        futures_data=term_history,
        parent=parent_opt,
        reload=reload,
    )
    skew_df = calculate_skew(opt_df, target_delta=target_delta, vol_col="iv", interpolate=True)

    merged = term_history.merge(skew_df, on=["date"], how="inner", )
    
    # sign flag for curve shape (contango/backwardation)
    if "slope_m1_m2" in merged.columns:
        merged["slope_sign"] = np.sign(merged["slope_m1_m2"])
    return merged


def fit_ols_sm(df: pd.DataFrame, y_col: str, x_cols: list[str]):
    """statsmodels ols with intercept, returns fitted model"""
    df_clean = df.dropna(subset=[y_col] + x_cols)
    X = sm.add_constant(df_clean[x_cols])
    y = df_clean[y_col]
    return sm.OLS(y, X).fit()


def plot_heatmap(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    matrix: pd.DataFrame | None = None,
    title: str = "",
    cmap: str = "coolwarm",
    fmt: str = ".2f",
    vmin: float | None = -1,
    vmax: float | None = 1,
) -> None:
    """plot a heatmap for correlation/covariance data"""
    # build matrix from cols if not provided
    if matrix is None:
        if cols is None:
            raise ValueError("provide cols when matrix is not supplied")
        matrix = df[cols].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap=cmap, vmin=vmin, vmax=vmax, square=True)
    plt.title(title or "heatmap")
    plt.tight_layout()
    plt.show()


def main() -> None:
    start = "2015-01-01"
    end = "2025-11-01"
    client = init_client()

    data = prepare_data(start=start, end=end, client=client, constant_targets=(21.0, 42.0), target_delta=0.25)

    # simple diagnostics
    core_cols = ["slope_m1_m2", "slope_const_21_42", "skew_25d", "iv_25c", "iv_25p", "atm_iv"]
    available = [c for c in core_cols if c in data.columns]
    if available:
        print(corr_matrix(data, available).head())

    if {"skew_25d", "slope_m1_m2"} <= set(data.columns):
        res = fit_ols_sm(data, "skew_25d", ["slope_m1_m2"])
        print(res.summary())


if __name__ == "__main__":
    main()
