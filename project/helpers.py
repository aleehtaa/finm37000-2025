import databento as db
import pandas as pd
from typing import Iterable
import numpy as np
from scipy.stats import norm
from pathlib import Path

from finm37000 import (
    get_databento_api_key,
    temp_env,
)

def init_client() -> db.Historical:
    """Create a Databento Historical client using the API key from env."""
    with temp_env(DATABENTO_API_KEY=get_databento_api_key()):
        client = db.Historical()
    return client

def get_save_dir():
    """Resolve a user-specific base path for saving data outputs."""
    if Path("/Users/rainc/OneDrive/Desktop/futures/finm37000").is_dir():
        save_dir = "/Users/rainc/OneDrive/Desktop/futures/finm37000"
    else:
        save_dir = "/Users/amylee/Desktop/finm37000"
    return save_dir

def linear_interp(
    x: Iterable[float],
    y: Iterable[float],
    target: float,
) -> float:
    """Sorted 1D interpolation wrapper around numpy.interp."""
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    idx = np.argsort(x_arr)
    x_sorted = x_arr[idx]
    y_sorted = y_arr[idx]
    return float(np.interp(target, x_sorted, y_sorted))
