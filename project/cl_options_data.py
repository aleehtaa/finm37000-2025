"""Compute CL options features (IV, delta, skew) and (optionally) join with futures term structure."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd
from scipy.stats import norm

from finm37000 import (
    us_business_day,
    tz_chicago,
    imply_american_vols,
)

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from project.helpers import (
    init_client,
    linear_interp,
    get_save_dir,
)

from project.cl_futures_data import (
    load_continuous_futures_data,
    build_term_structure_history,
)

SAVE_DIR = get_save_dir()

def _get_range_with_retry(client: db.Historical, retries: int = 3, backoff: float = 5.0, **kwargs):
    """Wrapper around client.timeseries.get_range with simple retry/backoff."""
    last_exc = None
    for attempt in range(retries):
        try:
            return client.timeseries.get_range(**kwargs)
        except Exception as exc:  # broad on purpose to catch timeouts
            last_exc = exc
            if attempt == retries - 1:
                raise
            time.sleep(backoff * (attempt + 1))
    if last_exc:
        raise last_exc
    raise RuntimeError("Unexpected retry loop state")

def _add_option_close_prices(option_chain, client):
    """Append daily close prices to an option chain from Databento."""
    # pull close prices for the option symbols
    # get close prices
    start = option_chain["date"].min().date()
    end = option_chain["date"].max().date() + us_business_day*2
    opt_symbols = option_chain["raw_symbol"].unique()
    
    close = _get_range_with_retry(
        client=client,
        dataset=db.Dataset.GLBX_MDP3,
        schema="ohlcv-1d",
        symbols=[*opt_symbols],
        stype_in="raw_symbol",
        start=start,
        end=end,
    )
    opt_df = (
        close.to_df()
        .reset_index()
        .rename(columns={"symbol": "raw_symbol"})
    )
    opt_df["date"] = opt_df["ts_event"].dt.tz_convert(tz_chicago).dt.normalize()
    option_chain = option_chain.merge(
        opt_df[["date", "instrument_id", "close", "raw_symbol"]],
        how="inner",
        on=["date", "raw_symbol"]
    )

    return option_chain

def _add_iv(
    option_data: pd.DataFrame,
    price_col: str = "close",
    forward_col: str = "uprc",
) -> pd.DataFrame:
    """Add American implied vols per date/underlying using the chosen price column."""
    out = []
    # loop over each date and underlying and get vols
    for _, subset in option_data.groupby(["date", "underlying"]):
        # check we have a forward for this slice
        if forward_col not in subset.columns:
            msg = f"Missing {forward_col}; cannot run imply_american_vols"
            raise ValueError(msg)
        futures_price = float(subset[forward_col].iloc[0])
        rate = float(subset["rate"].iloc[0])
        
        # get vols
        vols = imply_american_vols(
            option_df=subset,
            futures_price=futures_price,
            risk_free_rate=rate,
            price_cols = [price_col],
            use_actual_today=False,
        )
        subset = subset.assign(**vols)
        out.append(subset)

    option_data = pd.concat(out, ignore_index=True).rename(columns=({"iv_close": "iv"}))
    option_data["iv"] = option_data["iv"]
    return option_data


def _add_delta(
    option_data: pd.DataFrame,
    forward_col: str = "uprc",
) -> pd.DataFrame:
    """Compute Black forward deltas from implied vols and forward prices."""
    # get params
    F = option_data[forward_col]
    K = option_data["strike_price"]
    T = option_data["years_to_expiration"]
    vol = option_data["iv"]
    option_class = option_data["instrument_class"]
    # calc delta
    d1 = (np.log(F / K) + 0.5 * (vol**2) * T) / (vol * np.sqrt(T))
    option_data['delta'] = np.where(
        option_class == "C", 
        norm.cdf(d1),
        norm.cdf(d1) - 1.0
    )
    
    return option_data


def _add_rate(
    option_data: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the nearest risk-free rate observation on or before each date."""
    # merge latest available rate onto each date
    rf = (
        pd.read_csv(f"{SAVE_DIR}/data/risk-free-rate.csv")
        .rename(columns={"MCALDT": "date", "TMYTM": "rate"})
        .dropna()
    )
    rf["date"] = rf["date"].astype(str).str.replace(".0", "")
    rf["date"] = pd.to_datetime(rf["date"], format="%Y%m%d", utc=True).dt.tz_convert(tz_chicago).dt.normalize()
    rf = rf.sort_values("date").drop_duplicates("date", keep="last")
    option_data = pd.merge_asof(
        option_data.sort_values("date"),
        rf[["date", "rate"]],
        on="date",
        direction="backward",
        allow_exact_matches=True,
    )
    return option_data

    
def load_options_chain(
    parent: str,
    underlyings: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    client: db.Historical,
) -> pd.DataFrame:
    """Fetch option definitions for underlyings/date range and add time-to-expiry fields."""
    
    # pull definitions for the parent options and filter to underlyings
    options_def = _get_range_with_retry(
        client=client,
        dataset=db.Dataset.GLBX_MDP3,
        schema="definition",
        symbols=f"{parent}.OPT",
        stype_in="parent",
        start=start.date(),
        end=end.date() + us_business_day*2,
    )
    # filter
    opt_df = options_def.to_df()
    opt_df = opt_df[opt_df["underlying"].isin(underlyings)]
    opt_df = opt_df[opt_df["instrument_class"].isin(("C", "P"))]
    # time cols / tz conversions
    opt_df["date"] = opt_df["ts_event"].dt.tz_convert(tz_chicago).dt.normalize()
    opt_df["expiration"] = opt_df["expiration"].dt.tz_convert(tz_chicago).dt.normalize()
    opt_df["days_to_expiration"] = (opt_df["expiration"] - opt_df["date"]).dt.days.astype("Int64")
    opt_df["years_to_expiration"] = (
        (opt_df["expiration"] - opt_df["date"]).dt.total_seconds() / 365.0 / 24 / 60 / 60
    )
    opt_df = opt_df[opt_df['date'] >= start]
    opt_df = opt_df[opt_df['date'] < end]
    # get subset of data 
    cols = ["raw_symbol", "underlying", "instrument_class", "strike_price", "expiration"]
    opt_df = (
        opt_df.reset_index()
        [["date", "days_to_expiration", "years_to_expiration"] + cols]
        .sort_values("strike_price")
        .drop_duplicates(subset=cols)
    )
    return opt_df


def load_options_data(
    client: db.Historical | None = None,
    futures_data: pd.DataFrame | None = None,
    parent: str = "LO",
    reload: bool = False,
    ) -> pd.DataFrame:
    """Build options features per roll window: fetch chains, attach prices/IV/delta, merge with futures."""
    if reload:
        # ensure futures data supplied when reloading
        if futures_data is None:
            raise ValueError("futures_data must be provided when reload=True")
        futures_data = futures_data.rename(columns={"raw_symbol": "underlying", "close": "uprc"})

        segments = (
            futures_data[["symbol", "underlying", "d0", "d1", "expiration"]]
            .drop_duplicates()
        )

        opt_df = []
        for (window_start, window_end), grp in segments.groupby(["d0", "d1"]):
            underlying_symbols = list(grp["underlying"].unique())
            # check if client was provided
            if client is None:
                raise ValueError("client must be provided when reload=True")
            # fetch option chain for this window
            opt_chain = load_options_chain(
                parent=parent,
                underlyings=underlying_symbols,
                start=window_start,
                end=window_end,
                client=client,
            )
            opt_chain = _add_option_close_prices(opt_chain, client) 
            opt_df.append(opt_chain)

        opt_df = (
            pd.concat(opt_df)
            .sort_values("date")
            .reset_index(drop=True)
        )
        # add options metrics to dataframe
        uprc = (futures_data[["date", "symbol", "underlying", "uprc"]])
        opt_df = opt_df.merge(uprc, how="left", on=["date", "underlying"])
        opt_df = _add_rate(option_data=opt_df) 
        opt_df = _add_iv(option_data=opt_df)
        opt_df = _add_delta(option_data=opt_df)
        opt_df.to_parquet(f"{SAVE_DIR}/data/options_data.csv")
    else:
        opt_df = pd.read_parquet(f"{SAVE_DIR}/data/options_data.csv")

    return opt_df

def calculate_skew(
    options_df: pd.DataFrame,
    target_delta: float = 0.25,
    vol_col: str = "iv",
    interpolate: bool = True,
) -> pd.DataFrame:
    """
    For each (date, underlying, expiration), compute:

    - 25-delta call IV  (delta = +target_delta)
    - 25-delta put IV   (delta = -target_delta)
    - skew_25d  = iv_25c - iv_25p
    - slope_25d = skew_25d / (2 * target_delta)

    Uses linear interpolation in delta space when interpolate=True; otherwise picks
    the closest observed deltas to plus/minus target_delta.
    """

    group_cols = ["date", "underlying", "expiration"]
    rows = []

    df = options_df.dropna(subset=[vol_col, "delta"])

    for (dt, und, exp), grp in df.groupby(group_cols):
        # seperate calls and puts
        calls = grp[grp["instrument_class"] == "C"].sort_values("delta")
        puts  = grp[grp["instrument_class"] == "P"].sort_values("delta")
        if calls.empty or puts.empty:
            continue
        
        # linearly interpolate prices to get exact price of target_delta call/put
        if interpolate:
            if not (calls["delta"].min() <= target_delta <= calls["delta"].max()):
                continue
            if not (puts["delta"].min() <= -target_delta <= puts["delta"].max()):
                continue

            iv_25c = linear_interp(
                x=calls["delta"].to_numpy(),
                y=calls[vol_col].to_numpy(),
                target=target_delta,
            )

            iv_25p = linear_interp(
                x=puts["delta"].to_numpy(),
                y=puts[vol_col].to_numpy(),
                target=-target_delta,
            )

        else:
            call_idx = (calls["delta"] - target_delta).abs().idxmin()
            put_idx = (puts["delta"] + target_delta).abs().idxmin()

            iv_25c = calls.loc[call_idx, vol_col]
            iv_25p = puts.loc[put_idx, vol_col]

        skew = iv_25c - iv_25p
        slope = skew / (2.0 * target_delta)

        row_out = {
            "date": dt,
            "underlying": und,
            "expiration": exp,
            "iv_25c": iv_25c,
            "iv_25p": iv_25p,
            "skew_25d": skew,
            "slope_25d": slope,
        }
        rows.append(row_out)

    return pd.DataFrame(rows)

def main() -> None:
    """demonstration for workflow"""
    client = init_client()
    start = "2015-01-01"
    end = "2025-11-01"

    # futures
    futures_df = load_continuous_futures_data(start=start, end=end, client=client)
    term_history = build_term_structure_history(futures_df)

    # options data
    opt_df = load_options_data(
        client=client,
        futures_data=term_history,
        parent="LO",
        reload=True,
    )
    
    skew_df = calculate_skew(opt_df, target_delta=0.25, vol_col="iv", interpolate=True)


if __name__ == "__main__":
    main()
