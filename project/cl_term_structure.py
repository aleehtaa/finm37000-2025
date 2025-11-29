"""Build CL futures term-structure and slope time series using ohlcv-1d."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Iterable, Sequence

import databento as db
import numpy as np
import pandas as pd

from finm37000 import tz_chicago, us_business_day

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
from project.helpers import init_client


def load_definitions(required_ids, start, end, client, reload=False): 
    if reload:
        definitions = client.timeseries.get_range(
            dataset=db.Dataset.GLBX_MDP3,
            schema="definition",
            symbols=required_ids,
            stype_in="instrument_id",
            start=start,
            end=end,
        ).to_df()
        definitions.to_parquet("/Users/amylee/Desktop/finm37000/data/definitions.parquet")
    else:
        definitions = pd.read_parquet("/Users/amylee/Desktop/finm37000/data/definitions.parquet")

    if "instrument_class" in definitions.columns:
        definitions = definitions[
            definitions["instrument_class"]
            .isin([db.InstrumentClass.FUTURE, "FUTURE", "F"])
            ]
    definitions = definitions.drop_duplicates(subset=["expiration", "instrument_id", "symbol"], keep="last")
    #definitions["expiration"] = pd.to_datetime(definitions["expiration"]).dt.tz_convert(tz_chicago)
    exp = pd.to_datetime(definitions["expiration"], utc=True)  # treat as UTC
    definitions["expiration"] = exp.dt.tz_convert(tz_chicago).dt.normalize()
    definitions["instrument_id"] = definitions["instrument_id"].astype(int)

    return definitions[["expiration", "instrument_id", "symbol", "raw_symbol"]]

def _expand_roll_segments(
    roll_df: pd.DataFrame,
    trade_dates,
) -> pd.DataFrame:
    """
    Expand each roll segment to one row per date between d0 (inclusive) and d1 (exclusive); 
    Only keep dates that actually exist in OHLC
    """
    rows: list[pd.DataFrame] = []

    for _, r in roll_df.iterrows():
        d0 = r["d0"]
        d1 = r["d1"]
        # restrict to dates where we actually have OHLC bars
        dates = trade_dates[(trade_dates >= d0) & (trade_dates < d1)]
        if len(dates) == 0:
            continue

        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": r["symbol"],
                    "d0": d0,
                    "d1": d1,
                    "instrument_id": r.get("instrument_id", np.nan),
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def load_roll_specs(cont_symbols, start, end, client):
    roll_specs = client.symbology.resolve(
        dataset=db.Dataset.GLBX_MDP3,
        symbols=cont_symbols,
        stype_in="continuous",
        stype_out="instrument_id",
        start_date=start,
        end_date=end,
    )["result"]

    required_ids = sorted(
        {
            int(spec["s"])
            for spec_list in roll_specs.values()
            for spec in spec_list
        }
    )

    roll_df = (
        pd.concat(
            {symbol: pd.DataFrame(specs) for symbol, specs in roll_specs.items()},
            names=["symbol", "idx"],
        )
        .reset_index(level="symbol")
        .reset_index(drop=True)
    )
    #roll_df["date"] = pd.to_datetime(roll_df["date"]).dt.tz_localize(tz_chicago)
    roll_df["d0"] = pd.to_datetime(roll_df["d0"]).dt.tz_localize(tz_chicago)
    roll_df["d1"] = pd.to_datetime(roll_df["d1"]).dt.tz_localize(tz_chicago)
    roll_df = roll_df.rename(columns = {"s":"instrument_id"})
    roll_df["instrument_id"] = roll_df["instrument_id"].astype(int)

    return roll_df, required_ids

def load_ohlc(start, end, cont_symbols, client, reload=False):
    """Loads ohlc data from databento"""
    if reload:
        ohlcv = client.timeseries.get_range(
            dataset=db.Dataset.GLBX_MDP3,
            schema="ohlcv-1d",
            symbols=cont_symbols,
            stype_in="continuous",
            start=start,
            end=end,
        ).to_df()
        ohlcv.to_parquet("/Users/amylee/Desktop/finm37000/data/ohlcv.parquet") 
    else:
        ohlcv = pd.read_parquet("/Users/amylee/Desktop/finm37000/data/ohlcv.parquet")

    ohlcv = ohlcv.reset_index()
    ohlcv["date"] = ohlcv["ts_event"].dt.tz_convert(tz_chicago).dt.normalize()
    ohlcv = (
        ohlcv[["date", "ts_event", "instrument_id", "open", "high", "low", "close", "volume", "symbol"]]
        .sort_values(["date", "symbol"])
    )
    return ohlcv


def load_continuous_futures_data(
    client: db.Historical,
    start: datetime.date | str,
    end: datetime.date | str,
    parent: str = "CL",
) -> pd.DataFrame:
    """Fetch daily continuous futures data"""
    # define continous symbols for front month and next two months
    cont_symbols = [f"{parent}.c.{i}" for i in (0, 1, 2)]
    
    # get required data
    roll_df, required_ids = load_roll_specs(cont_symbols=cont_symbols, start=start, end=end, client=client)
    ohlcv = load_ohlc(cont_symbols=cont_symbols, start=start, end=end, client=client, reload=False)
    definitions = load_definitions(required_ids=required_ids, start=start, end=end, client=client, reload=False)
    
    # merge data together
    trade_dates = ohlcv["date"].unique()
    roll_df = _expand_roll_segments(roll_df=roll_df, trade_dates=trade_dates)
    df = (
        roll_df.merge(
            definitions[["instrument_id", "expiration", "raw_symbol"]],
            on="instrument_id",
            how="left",
        )
        .merge(ohlcv, on=["date", "symbol", "instrument_id"], how="left")
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
        .dropna(subset=["open", "close"])
    )
    # get days to expiration
    df["days_to_expiration"] = (
        (df["expiration"] - df["date"]).dt.days.astype("Int64")
    )

    return df


def interpolate_price(
    expiries_days: Iterable[int], prices: Iterable[float], target_days: float
) -> float:
    """Linearly interpolate a forward price to a target maturity in days."""
    pairs = sorted(zip(expiries_days, prices), key=lambda p: p[0])
    lower = [p for p in pairs if p[0] <= target_days]
    upper = [p for p in pairs if p[0] >= target_days]
    if not lower or not upper:
        msg = f"Cannot interpolate: target {target_days}d outside available maturities."
        raise ValueError(msg)
    lo_days, lo_px = lower[-1]
    hi_days, hi_px = upper[0]
    if hi_days == lo_days:
        return float(lo_px)
    weight = (target_days - lo_days) / (hi_days - lo_days)
    return float(lo_px + weight * (hi_px - lo_px))


def compute_slopes(
    front_df: pd.DataFrame,
    constant_targets: Sequence[float] = (35.0, 60.0), #FIXME: need to figure out what to do about when first month dte is > lower dte, etx
) -> dict[str, float]:
    """Compute term-structure slopes for a single trade date."""
    front_sorted = front_df.sort_values("expiration").reset_index(drop=True)
    if len(front_sorted) < 2:
        msg = "Need at least two maturities to compute slope."
        raise ValueError(msg)

    f1 = float(front_sorted.loc[0, "close"])
    f2 = float(front_sorted.loc[1, "close"])
    t1 = float(front_sorted.loc[0, "days_to_expiration"])
    t2 = float(front_sorted.loc[1, "days_to_expiration"])
    slope_m1_m2 = (np.log(f2) - np.log(f1)) / (t2 - t1)

    target_short, target_long = constant_targets
    days = front_sorted["days_to_expiration"]
    prices = front_sorted["close"]
    price_short = np.nan
    price_long = np.nan
    if days.min() <= target_short <= days.max():
        price_short = interpolate_price(days, prices, target_short)
    if days.min() <= target_long <= days.max():
        price_long = interpolate_price(days, prices, target_long)
    slope_const = np.nan
    if not np.isnan(price_short) and not np.isnan(price_long):
        slope_const = (np.log(price_long) - np.log(price_short)) / (target_long - target_short)

    return {
        "slope_m1_m2": slope_m1_m2,
        "slope_const_21_42": slope_const,
        "const_price_21d": price_short,
        "const_price_42d": price_long,
    }


def build_term_structure_history(futures_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily term-structure slopes from prepared futures data."""
    slope_rows = []
    front_rows = []
    for day in futures_df["date"].drop_duplicates():
        day_df = futures_df[futures_df["date"] == day]
        if day_df.empty:
            continue
        front = day_df.sort_values("expiration").head(3)
        if len(front) < 2:
            continue
        slopes = compute_slopes(front)
        slope_rows.append({"Trade date": day, **slopes})
        front_rows.append(front.assign(Trade_date=day))

    front_history = pd.concat(front_rows, ignore_index=True) if front_rows else pd.DataFrame()
    slope_history = pd.DataFrame(slope_rows)
    if front_history.empty:
        return slope_history
    return front_history.merge(slope_history, left_on="Trade_date", right_on="Trade date", how="right")


def main() -> None:
    start = "2025-01-01" # FIXME: have not tested on longer timeline yet
    end = "2025-11-01"
    client = init_client()
    futures_df = load_continuous_futures_data(
        client=client,
        start=start,
        end=end,
        parent="CL",
    )
    history = build_term_structure_history(futures_df)
    print(history.head())


if __name__ == "__main__":
    main()
