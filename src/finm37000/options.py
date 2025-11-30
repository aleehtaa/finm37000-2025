"""Simple option pricing implementations."""

from enum import Enum
from typing import Iterable, cast

import databento as db
import numpy as np
import pandas as pd
import QuantLib as ql  # noqa: N813
from scipy.optimize import root_scalar
from scipy.stats import norm


class OptionType(Enum):
    """Enumeration of option types."""

    CALL = "Call"
    PUT = "Put"


def calc_black(  # noqa: PLR0913
    F: float | np.ndarray,  # noqa: N803
    K: float | np.ndarray,  # noqa: N803
    T: float | np.ndarray,  # noqa: N803
    vol: float | np.ndarray,
    r: float | np.ndarray,
    option_type: OptionType,
) -> float | np.ndarray:
    """Compute Black 76 model for European options on forwards."""
    d1 = (np.log(F / K) + (vol**2 / 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    discount_factor = np.exp(-r * T)

    cp = 1 if option_type == OptionType.CALL else -1
    return cast(
        "float", discount_factor * cp * (F * norm.cdf(cp * d1) - K * norm.cdf(cp * d2))
    )


def get_options_chain(
    parent: str,
    underlying: str,
    start: pd.Timestamp,
    client: db.Historical,
    days_per_year: float = 365.0,
) -> pd.DataFrame:
    """Retrieve the definitions of shared-parent options contracts.

    Parameters
    ----------
    parent : str
        The parent symbol, such as ES.
    underlying : str
        The underlying contract for the option.
    start : pd.Timestamp
        The date to obtain the definitions for.
    client: db.Historical
        The Historical client to retrieve databento data.
    days_per_year: float
        The number of days to use to normalize day counts.

    Returns:
    -------
    pd.DataFrame

    """
    options_def = client.timeseries.get_range(
        dataset=db.Dataset.GLBX_MDP3,
        schema="definition",
        symbols=f"{parent}.OPT",
        stype_in="parent",
        start=start.date(),
    )

    df = options_def.to_df()
    df = df[df["underlying"] == underlying]
    df = df[df["instrument_class"].isin(("C", "P"))]
    df["years_to_expiration"] = (
        (df["expiration"] - start).dt.total_seconds() / days_per_year / 24 / 60 / 60
    )
    return df.sort_values("strike_price")


def get_top_of_book(
    symbols: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    client: db.Historical,
) -> pd.DataFrame:
    """Get the last top-of-book and calculate midprice.

    Parameters
    ----------
    symbols : Iterable[str]
        A collection of symbols to retrieve the midprices for.
    start : pd.Timestamp
        The start time.
    end : pd.Timestamp
        The end time (exclusive).
    client: db.Historical
        The Historical client to retrieve databento data.

    Returns:
    -------
    pd.DataFrame

    """
    price_df = client.timeseries.get_range(
        dataset=db.Dataset.GLBX_MDP3,
        schema="mbp-1",
        symbols=symbols,
        start=start,
        end=end,
    ).to_df()

    price_df = price_df.groupby("symbol").last()
    price_df["bid"] = price_df["bid_px_00"]
    price_df["ask"] = price_df["ask_px_00"]
    price_df["bidq"] = price_df["bid_sz_00"]
    price_df["askq"] = price_df["ask_sz_00"]
    price_df["midprice"] = np.mean(price_df[["bid", "ask"]], axis=1)
    wt = price_df["bidq"] / (price_df["bidq"] + price_df["askq"])
    price_df["weighted_midprice"] = price_df["bid"] * (1.0 - wt) + price_df["ask"] * wt

    cols = ["bid", "ask", "midprice", "bidq", "askq", "weighted_midprice"]
    return price_df[cols]


def imply_european_vol(
    row: pd.Series,
    price_col: str = "midprice",
) -> float:
    """Find the roots of the Black-76 model by varying sigma, implied volatility.

    This function is for use with `pandas.Dataframe.apply`. Each row should contain
    a column for "strike_price", "years_to_expiration", "instrument_class", "midprice",
    and "underlying_price",

    If the optimization fails, `numpy.nan` is returned.

    Parameters
    ----------
    row : pd.Series
        A series of data to process.
    price_col : str, optional
        The name of the column in `row` that contains the price.

    Returns:
    -------
    float | numpy.nan

    """
    target = float(row[price_col])
    option_type = OptionType.CALL if row["instrument_class"] == "C" else OptionType.PUT

    def model_price(vol: float) -> float:
        return cast(
            "float",
            calc_black(
                F=row["underlying_price"],
                K=row["strike_price"],
                T=row["years_to_expiration"],
                vol=vol,
                r=row["interest_rate"],
                option_type=option_type,
            ),
        )

    def f(vol: float) -> float:
        return target - model_price(vol)

    lb = 0.00001
    ub = 4
    f_lb = f(lb)
    f_ub = f(ub)
    if f_ub * f_lb >= 0:
        lower_vol = float(model_price(lb))
        upper_vol = float(model_price(ub))
        print(
            (
                f"Cannot find {option_type} vol between "
                f"{lb=} and {ub=} "
                f"at strike {row['strike_price']}: "
                f"{lower_vol=} {target=} {upper_vol=}"
            )
        )
        print(
            (
                f"  F={row['underlying_price']} "
                f"T={row['years_to_expiration']} "
                f"r={row['interest_rate']} "
                f"mid={row['midprice']}"
            )
        )
        return np.nan
    result = root_scalar(f, bracket=[lb, ub], method="brentq")  # type: ignore[call-overload]
    if result.converged:
        return cast("float", result.root)
    print(
        f"Could not find sigma for {row['raw_symbol']} with midprice {row['midprice']}",
    )
    return np.nan


def imply_american_vols(  # noqa: PLR0913
    option_df: pd.DataFrame,
    futures_price: float,
    risk_free_rate: float,
    min_vol: float = 1e-4,
    max_vol: float = 4.0,
    max_evaluations: int = 200,
    accuracy: float = 1e-6,
    days_per_year: float = 365.0,
    price_cols: list[str] = ["bid", "midprice", "ask", "weighted_midprice"],
    use_actual_today: bool = True,
) -> dict[str, pd.Series]:
    """Use QuantLib's BAW model to get implied vols."""
    # --- Market setup ---
    today = ql.Date.todaysDate()
    if not use_actual_today:
        eval_dt = pd.to_datetime(option_df['date'].iloc[0]).date()
        today = ql.Date(eval_dt.day, eval_dt.month, eval_dt.year)

    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(futures_price))
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free_rate, day_count))
    q_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free_rate, day_count))

    def imply_vol(row: pd.Series, price_col: str = "mid") -> float:
        option_price = row[price_col]
        maturity_date = today + int(days_per_year * row.years_to_expiration)
        payoff = ql.PlainVanillaPayoff(
            ql.Option.Call if row.instrument_class == "C" else ql.Option.Put,
            row.strike_price,
        )
        exercise = ql.AmericanExercise(today, maturity_date)
        process = ql.GeneralizedBlackScholesProcess(
            spot_handle,
            q_ts,
            r_ts,
            ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(today, calendar, 0.2, day_count)
            ),
        )
        engine = ql.BaroneAdesiWhaleyApproximationEngine(process)
        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(engine)

        try:
            return cast(
                "float",
                option.impliedVolatility(
                    option_price,
                    process,
                    accuracy,
                    max_evaluations,
                    min_vol,
                    max_vol,
                ),
            )
        except RuntimeError:
            return float("nan")

    ivs = {}
    for col in price_cols:
        ivs[f"iv_{col}"] = option_df.apply(imply_vol, axis=1, price_col=col)
    return ivs
