"""Aggregate detailed data into summaries."""

import pandas as pd


def make_ohlcv(
    trades_df: pd.DataFrame,
    rule: str,
    index_name: str = "ts_recv",
    out_index_name: str = "ts_event",
) -> pd.DataFrame:
    """Convert trade data to OHLCV data.

    Args:
        trades_df: A `pd.DataFrame` with columns `symbol`, `price`, and
                  `size` indexed on time.
        rule: `str` for the rule to pass to `resample`, e.g., "5s" for 5
               second data.
        index_name: `str` for the index name.
        out_index_name: `str` for the output index name.

    Returns:
        A `pd.DataFrame` with resampled according to the `rule` with
        `price` values aggregated into open, high, low, and close for
        each interval and `size` summed to produce volume for the interval.

    """
    resampled_df = (
        trades_df.groupby("symbol")
        .resample(rule, include_groups=False)
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("size", "sum"),
        )
        .dropna()
        .reset_index()
        .rename(columns={index_name: out_index_name})
        .sort_values(by=[out_index_name, "symbol"])
        .set_index(out_index_name)
    )
    return resampled_df


def aggregate_ohlcv(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate OHLCV by symbol, not resampled."""
    return trades.groupby("symbol").agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("size", "sum"),
    )
