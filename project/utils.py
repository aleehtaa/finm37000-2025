"""Utility helpers for CL data pipelines (Databento client init and save paths)."""

import databento as db
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


SAVE_DIR = get_save_dir()
