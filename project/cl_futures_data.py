"""Build CL futures term-structure and slope time series using ohlcv-1d."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Iterable, Sequence

import databento as db
import numpy as np
import pandas as pd

from finm37000 import tz_chicago

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from project.helpers import init_client, get_save_dir
SAVE_DIR = get_save_dir()

def load_definitions(required_ids, parent, start, end, client, reload=False): 
    """Load futures definitions for the given instrument IDs."""
    # fetch from api or cached parquet
    if reload:
        definitions = client.timeseries.get_range(
            dataset=db.Dataset.GLBX_MDP3,
            schema="definition",
            symbols=required_ids,
            stype_in="instrument_id",
            start=start,
            end=end,
        ).to_df()
        definitions.to_parquet(f"{SAVE_DIR}/data/definitions.parquet")
    else:
        definitions = pd.read_parquet(f"{SAVE_DIR}/data/definitions.parquet")

    if "instrument_class" in definitions.columns:
        definitions = definitions[
            definitions["instrument_class"]
            .isin([db.InstrumentClass.FUTURE, "FUTURE", "F"])
            ]
    # drop dups, normalize tz, filter by parent root
    definitions = definitions.drop_duplicates(subset=["expiration", "instrument_id", "symbol"], keep="last")
    exp = pd.to_datetime(definitions["expiration"], utc=True)  # treat as UTC
    definitions["expiration"] = exp.dt.tz_convert(tz_chicago).dt.normalize()
    definitions["instrument_id"] = definitions["instrument_id"].astype(int)
    definitions = definitions[definitions["raw_symbol"].str.startswith(parent)]

    return definitions[["expiration", "instrument_id", "symbol", "raw_symbol"]]


def _expand_roll_segments(
    roll_df: pd.DataFrame,
    trade_dates,
) -> pd.DataFrame:
    """Expand roll segments to one row per trade date between d0 (inclusive) and d1 (exclusive)."""
    # build one row per trade date within each roll window
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
    """Fetch and flatten roll specs for the continuous symbols."""
    # resolve continuous symbols to instrument ids and flatten to a dataframe
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
    roll_df["d0"] = pd.to_datetime(roll_df["d0"]).dt.tz_localize(tz_chicago)
    roll_df["d1"] = pd.to_datetime(roll_df["d1"]).dt.tz_localize(tz_chicago)
    roll_df = roll_df.rename(columns = {"s":"instrument_id"})
    roll_df["instrument_id"] = roll_df["instrument_id"].astype(int)

    return roll_df, required_ids


def load_ohlc(start, end, cont_symbols, client, reload=False):
    """Load daily ohlcv-1d data for the continuous symbols."""
    # load ohlcv per year (api or parquet), combine, add date column, sort
    start_year = pd.to_datetime(start).year
    end_year = pd.to_datetime(end).year

    dfs = []

    for year in range(start_year, end_year + 1):
        y_start = f"{year}-01-01"
        y_end = f"{year}-12-31"
        if year == end_year:
            y_end = end  

        print(f"Downloading {year}: {y_start} -> {y_end}")

        if reload:
            ohlcv_year = client.timeseries.get_range(
                dataset=db.Dataset.GLBX_MDP3,
                schema="ohlcv-1d",
                symbols=cont_symbols,
                stype_in="continuous",
                start=y_start,
                end=y_end,
            ).to_df()             

            ohlcv_year.to_parquet(f"{SAVE_DIR}/data/ohlcv_{year}.parquet")
        else:
            ohlcv_year = pd.read_parquet(f"{SAVE_DIR}/data/ohlcv_{year}.parquet")

        dfs.append(ohlcv_year)

    # Combine all years
    ohlcv = pd.concat(dfs)
    ohlcv = ohlcv.reset_index()  
    ohlcv["date"] = ohlcv["ts_event"].dt.tz_convert(tz_chicago).dt.normalize()
    ohlcv = (ohlcv[["date", "ts_event", "instrument_id", "open", "high", "low", "close", "volume", "symbol"]].sort_values(["date", "symbol"]))
    return ohlcv


def load_continuous_futures_data(
    client: db.Historical | None = None,
    start: datetime.date | str | None = None,
    end: datetime.date | str| None = None,
    parent: str = "CL",
    reload: bool = False,
) -> pd.DataFrame:
    """Fetch daily continuous futures with roll metadata attached."""
    # define continous symbols for front month and next two months
    cont_symbols = [f"{parent}.c.{i}" for i in (0, 1, 2)]
    
    # get roll specs, prices, definitions
    roll_df, required_ids = load_roll_specs(cont_symbols=cont_symbols, start=start, end=end, client=client)
    ohlcv = load_ohlc(cont_symbols=cont_symbols, start=start, end=end, client=client, reload=reload)
    definitions = load_definitions(required_ids=required_ids, parent=parent, start=start, end=end, client=client, reload=reload)
    # merge data together and add days to expiration
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
    expiries_days: Iterable[int],
    prices: Iterable[float],
    target_days: float,
) -> float:
    """Linearly interpolate (or extrapolate) a forward price to a target maturity in days.

    - If target_days is inside the available range, do standard linear interpolation
      between the surrounding maturities.
    - If target_days is outside the available range, linearly extrapolate using
      the nearest two points (front or back).
    """
    # sort maturities/prices, pick two points around target (or edge), do linear weight

    dtes = np.array(expiries_days)
    prcs = np.array(prices)

    # return price if we only have one point
    if len(prcs) == 1:
        return float(prcs[0])
    
    # get position of target days relative to what we ahve
    idx = np.argsort(dtes)
    dtes = dtes[idx]
    prcs = prcs[idx]
    pos = np.searchsorted(dtes, target_days)

    dte0, prc0, dte1, prc1 = 0.0, 0.0, 0.0, 0.0
    if pos == 0:  # before first, extrapolate with first two
        dte0, prc0 = dtes[0], prcs[0]
        dte1, prc1 = dtes[1], prcs[1]
    elif pos >= len(dtes):  # after last, extrapolate with last two
        dte0, prc0 = dtes[-2], prcs[-2]
        dte1, prc1 = dtes[-1], prcs[-1]
    else: # inside range, bracket with neighbors
        dte0, prc0 = dtes[pos - 1], prcs[pos - 1]
        dte1, prc1 = dtes[pos], prcs[pos]

    if dte1 == dte0:
        return float(prc0)

    weight = (target_days - dte0) / (dte1 - dte0)
    return float(weight*prc1 + (1-weight)*prc0)


def compute_slopes(
    front_df: pd.DataFrame,
    constant_targets: Sequence[float] = (30.0, 60.0),
) -> dict[str, float]:
    """Compute term-structure slopes for a single trade date."""

    # slope between first two maturities and optional constant-maturity slope
    front_sorted = front_df.sort_values("expiration").reset_index(drop=True)
    if len(front_sorted) < 2:
        msg = "Need at least two maturities to compute slope."
        raise ValueError(msg)

    f1 = float(front_sorted.iloc[0]["close"])
    f2 = float(front_sorted.iloc[1]["close"])
    t1 = float(front_sorted.iloc[0]["days_to_expiration"])
    t2 = float(front_sorted.iloc[1]["days_to_expiration"])
    slope_m1_m2 = (np.log(f2) - np.log(f1)) / (t2 - t1)

    target_short, target_long = constant_targets
    price_short, price_long = np.nan, np.nan
    
    # slop for constant expiration using interpolated prices
    days = front_sorted["days_to_expiration"]
    prices = front_sorted["close"]
    
    price_short = interpolate_price(days, prices, target_short)
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
    # loop over dates, compute slopes, keep front contracts, merge results
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
        slope_rows.append({"date": day, **slopes})
        front_rows.append(front.assign(date=day))

    front_history = pd.concat(front_rows, ignore_index=True) if front_rows else pd.DataFrame()
    slope_history = pd.DataFrame(slope_rows)
    if front_history.empty:
        return slope_history
    return front_history.merge(slope_history, on="date", how="right")


def main() -> None:
    start = "2015-01-01" 
    end = "2025-11-01"
    client = init_client()
    futures_df = load_continuous_futures_data(
        client=client,
        start=start,
        end=end,
        parent="CL",
        reload=False,
    )
    history = build_term_structure_history(futures_df)
    print(history.head())
    print(history.tail())


if __name__ == "__main__":
    main()
