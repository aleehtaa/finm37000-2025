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
    parent_fut: str = "CL",
    parent_opt: str = "LO",
    reload: bool = False,
    target_delta: float = 0.25,
    constant_targets : Sequence[float] = (21.0, 42.0)
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
    term_history = build_term_structure_history(futures_df)

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


def fit_ols(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> dict[str, float]:
    """simple ols with intercept, returns params and r2"""
    X = df[x_cols].to_numpy()
    y = df[y_col].to_numpy()
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = linalg.lstsq(X, y)
    y_hat = X @ beta
    resid = y - y_hat
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = np.sum(resid ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    out = {"intercept": beta[0], "r2": r2}
    for i, col in enumerate(x_cols, start=1):
        out[col] = beta[i]
    return out


def corr_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """quick correlation matrix for selected columns"""
    return df[cols].corr()


def cov_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """quick covariance matrix for selected columns"""
    return df[cols].cov()


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
