"""Calculate constant maturity roll specifications and weighted values."""

import datetime
import sys
from dataclasses import dataclass
from typing import Union
from zoneinfo import ZoneInfo

import pandas as pd


@dataclass
class ConstantMaturitySpec:
    """The constant maturity roll specification."""

    d0: datetime.date
    d1: datetime.date
    pre_id: int
    next_id: int

    def to_json(self) -> dict[str, str]:
        """Convert to JSON."""
        return {
            "d0": self.d0.strftime("%Y-%m-%d"),
            "d1": self.d1.strftime("%Y-%m-%d"),
            "p": str(self.pre_id),
            "n": str(self.next_id),
        }


def _extract_maturity_days(symbol: str) -> datetime.timedelta:
    product, roll_type, maturity_str = symbol.split(".")
    maturity_days = datetime.timedelta(days=int(maturity_str))
    if roll_type != "cm":
        msg = "get_roll_spec is only designed for constant maturity 'cm' type"
        raise ValueError(msg)
    return maturity_days


def get_roll_spec(
    symbol: str,
    instrument_defs: pd.DataFrame,
    start: datetime.date,
    end: datetime.date,
) -> list[dict[str, str]]:
    """Compute the constant maturity instruments and roll dates.

    Args:
        symbol: The name of the continuous contract using the form
                         `f"{product}.cm.{dtm}"` where product is the common
                         symbol like `CL` and `dtm` is the days to maturity.
        instrument_defs: DataFrame with the instrument specifications
                         `ts_recv`, `instrument_class`, `instrument_id`, `raw_symbol,
                          and `expiration`. Per Databento standards, `ts_recv` and
                          `expiration` are in UTC.
        start: The start date of the roll spec.
        end: The end date of the roll spec.

    Returns:
        List of dicts, each with members `"d0"`, `"d1"`, and `"s"`
        containing the first date, one past the last date, and
        the instrument id of the instrument in the spliced contract. See
        `databento.Historical.symbology.resolve()["results"]` for
        a dictionary with values of this type.A pandas DataFrame containing
        the adjusted data.

    """
    utc = ZoneInfo("UTC")
    dates = pd.date_range(start=start, end=end, tz=utc)
    maturity_days = _extract_maturity_days(symbol)
    required_cols = ["expiration", "ts_recv", "instrument_id", "raw_symbol"]
    is_leg = instrument_defs["instrument_class"] == "F"
    by_exp = instrument_defs[is_leg][required_cols]
    by_exp = by_exp.set_index("expiration").drop_duplicates()
    by_exp = by_exp.sort_index().reset_index()
    cm: Union[ConstantMaturitySpec, None] = None
    one_day = datetime.timedelta(days=1)
    specs = []
    for date in dates:
        mask = (by_exp["ts_recv"].dt.date <= date.date()) & (
            by_exp["expiration"].dt.date >= date.date()
        )
        available = by_exp[mask]
        if sys.version_info <= (3, 9):
            maturity = available["expiration"].dt.date - date.date()  # type: ignore
        else:
            maturity = available["expiration"].dt.date - date.date()
        pre_df = available[maturity < maturity_days]["instrument_id"]
        if len(pre_df) == 0:
            msg = f"No futures with maturity < {maturity_days} on {date}."
            raise ValueError(msg)
        pre_id = pre_df.iloc[-1]
        next_df = available[maturity >= maturity_days]["instrument_id"]
        if len(next_df) == 0:
            msg = f"No futures with maturity >= {maturity_days} on {date}."
            raise ValueError(msg)
        next_id = next_df.iloc[0]
        available = available.copy()
        available["maturity"] = maturity
        if cm is None:
            cm = ConstantMaturitySpec(
                d0=date, d1=date + one_day, pre_id=pre_id, next_id=next_id
            )
        elif pre_id != cm.pre_id or next_id != cm.next_id:
            specs.append(cm.to_json())
            cm = ConstantMaturitySpec(
                d0=date, d1=date + one_day, pre_id=pre_id, next_id=next_id
            )
        else:
            cm.d1 += one_day
    if cm is not None:
        cm.d1 = dates[-1]
        if cm.d0 != cm.d1:
            specs.append(cm.to_json())
    return specs


def _splice_pair(
    roll_spec: list[dict[str, str]],
    df: pd.DataFrame,
    date_col: str,
    price_col: str,
) -> pd.DataFrame:
    grouped = df.groupby("instrument_id")
    pieces = []
    tz = df[date_col].dt.tz
    for spec in roll_spec:
        d0 = pd.Timestamp(spec["d0"], tz=tz)
        d1 = pd.Timestamp(spec["d1"], tz=tz)
        p = int(spec["p"])
        n = int(spec["n"])
        group_n = grouped.get_group(n)
        group_p = grouped.get_group(p)
        piece_n = group_n[(group_n[date_col] >= d0) & (group_n[date_col] < d1)].copy()
        piece_p = group_p[(group_p[date_col] >= d0) & (group_p[date_col] < d1)].copy()
        piece_p = piece_p.rename(
            columns={
                price_col: f"pre_{price_col}",
                "instrument_id": "pre_id",
                "expiration": "pre_expiration",
            }
        )
        piece_n = piece_n.rename(
            columns={
                price_col: f"next_{price_col}",
                "instrument_id": "next_id",
                "expiration": "next_expiration",
            }
        )
        pre_cols = [date_col, f"pre_{price_col}", "pre_id", "pre_expiration"]
        next_cols = [date_col, f"next_{price_col}", "next_id", "next_expiration"]
        piece = piece_p[pre_cols].merge(
            piece_n[next_cols],
            on=date_col,
            how="outer",
            validate="one_to_one",
        )
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def calc_maturity_weight(
    maturity_days: datetime.timedelta,
    df: pd.DataFrame,
    date_col: str,
) -> pd.Series:
    """Compute weights for constant maturity contracts.

    Args:
        maturity_days: Number of days maturity contracts.
        df: A pandas DataFrame containing the pre and next expiration columns
            and the current date column.
        date_col: The date column name.

    Returns:
        A pandas Series containing the maturity weights.

    """
    pre_exp = df["pre_expiration"]
    next_exp = df["next_expiration"]
    t = df[date_col]
    weight = (next_exp - (t + maturity_days)) / (next_exp - pre_exp)
    return weight


def constant_maturity_splice(  # noqa: PLR0913
    symbol: str,
    roll_spec: list[dict[str, str]],
    all_data: pd.DataFrame,
    date_col: str,
    price_col: str,
) -> pd.DataFrame:
    """Adjust and splice pairs of futures data adjusted for constant maturity.

    Args:
        symbol: The name of the continuous contract using the form
                `f"{product}.cm.{dtm}"` where product is the common
                symbol like `CL` and `dtm` is the days to maturity.
        roll_spec: List of dicts, each with members `"d0"`, `"d1"`, and `"s"`
                   containing the first date, one past the last date, and
                   the instrument id of the instrument in the spliced contract. See
                   `databento.Historical.symbology.resolve()["results"]` for
                   a dictionary with values of this type.
        all_data: A pandas DataFrame containing the raw data to splice including
                  the `expiration` column.
        date_col: The name of the column in `df` that contains the date.
        price_col: The name of the column in `df` that contains the column to adjust.

    Returns:
        A pandas DataFrame containing the adjusted data.

    """
    unadjusted_splice = _splice_pair(
        roll_spec,
        all_data,
        date_col=date_col,
        price_col=price_col,
    )
    maturity_days = _extract_maturity_days(symbol)
    maturity_weight = calc_maturity_weight(
        maturity_days,
        unadjusted_splice,
        date_col=date_col,
    )
    pre_price = unadjusted_splice[f"pre_{price_col}"]
    next_price = unadjusted_splice[f"next_{price_col}"]
    unadjusted_splice["pre_weight"] = maturity_weight
    unadjusted_splice[symbol] = (
        maturity_weight * pre_price + (1 - maturity_weight) * next_price
    )
    return unadjusted_splice
