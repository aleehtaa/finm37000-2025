"""Functions to fit basic skews."""

import logging
from dataclasses import dataclass
from typing import Callable, Self, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

from .options import imply_american_vols, imply_european_vol

logger = logging.getLogger(__name__)


def filter_valid(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Keep pairs that are neither NaN."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    return x[mask], y[mask]


def calculate_option_vols(
    top_df: pd.DataFrame,
    underlying_symbol: str,
    option_chain: pd.DataFrame,
    interest_rate: float,
) -> tuple[pd.DataFrame, float]:
    """Convenience function doing multiple implied vol calculations."""
    underlying_price = top_df["midprice"].loc[underlying_symbol]
    with_top_prices = option_chain.join(top_df, on="symbol").dropna(subset="midprice")
    with_top_prices["underlying_price"] = underlying_price
    with_top_prices["interest_rate"] = interest_rate
    with_vols = with_top_prices.assign(
        **imply_american_vols(
            with_top_prices,
            futures_price=underlying_price,
            risk_free_rate=interest_rate,
        )
    )
    with_vols["european_vol"] = with_vols.apply(
        imply_european_vol,
        axis=1,
    )
    return with_vols, underlying_price


def split_call_put(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into the call and put rows."""
    call = df[df["instrument_class"] == "C"]
    put = df[df["instrument_class"] == "P"]
    return call, put


def filter_otm(df: pd.DataFrame, underlying_price: float) -> pd.DataFrame:
    """Extract OTM rows from the options dataframe."""
    call, put = split_call_put(df)
    otm = pd.concat(
        [
            call[call["strike_price"] >= underlying_price],
            put[put["strike_price"] <= underlying_price],
        ]
    ).sort_values(by="strike_price")
    return otm


def fit_polynomial_skew(k: np.ndarray, sigma: np.ndarray, degree: int) -> np.poly1d:
    """Fit a polynomial skew curve to volatility."""
    k, sigma = filter_valid(k, sigma)
    coefficients = np.polyfit(k, sigma, degree)
    return np.poly1d(coefficients)


def fit_weighted_piecewise_polynomial_skew(
    k: np.ndarray, sigma: np.ndarray, atm: float, degree: int, atm_weight: float = 1e6
) -> Callable[[pd.Series | np.ndarray], np.ndarray]:
    """Fit a piecewise polynomial skew-like function.

    Join pieces at-the-money by using extra weight in the regression.
    """
    k, sigma = filter_valid(k, sigma)
    put_k = k[k < atm]
    put_wt = np.ones_like(put_k, dtype=float)
    put_wt[-1] = atm_weight
    put_skew_coef = np.polyfit(put_k, sigma[k < atm], degree, w=put_wt)
    call_k = k[k >= atm]
    call_wt = np.ones_like(call_k, dtype=float)
    call_wt[0] = atm_weight
    call_skew_coef = np.polyfit(call_k, sigma[k >= atm], degree, w=call_wt)
    put_skew = np.poly1d(put_skew_coef)
    call_skew = np.poly1d(call_skew_coef)

    def piecewise_polynomial_skew(x: pd.Series | np.ndarray) -> np.ndarray:
        if isinstance(x, pd.Series):
            x = x.to_numpy()
        return np.piecewise(x, [x < atm, x >= atm], [put_skew, call_skew])  # type: ignore[operator]

    return piecewise_polynomial_skew


def fit_spline_skew(
    k: npt.NDArray[np.float64],
    sigma: npt.NDArray[np.float64],
    pct: float = 0.1,
    bc_type: str = "clamped",
    extrapolate: bool = False,
) -> CubicSpline:
    """Fit a basic cubic spline."""
    k, sigma = filter_valid(k, sigma)
    n = int(len(k) * pct)
    spline = CubicSpline(k[::n], sigma[::n], bc_type=bc_type, extrapolate=extrapolate)  # type: ignore[call-overload]
    return cast("CubicSpline", spline)


def calc_raw_svi(  # noqa: PLR0913
    k: npt.NDArray[np.float64],
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> npt.NDArray[np.float64]:
    """The raw SVI curve.

    Positive variance guaranteed when a + b * siqma * sqrt(1-rho^2) >= 0.
    """
    out = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))
    return cast("npt.NDArray[np.float64]", out)


@dataclass
class RawSVIParams:
    """Parameters for raw SVI curve."""

    a: float = 0.1  # real:  vertical level of the skew
    b: float = 0.1  # nonnegative: higher b raises both wings
    rho: float = 0  # |rho| < 1: Lower rho raises the puts relative to the calls.
    m: float = 0  # real: higher m moves to the right
    sigma: float = 0.1  # positive: Higher sigma reduces ATM curvature.

    def __array__(self) -> np.ndarray:
        """Cast RawSVIParams to a numpy array."""
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    @classmethod
    def from_array(cls, x: np.ndarray) -> Self:
        """Convert numpy array to RawSVIParams."""
        return cls(
            a=float(x[0]),
            b=float(x[1]),
            rho=float(x[2]),
            m=float(x[3]),
            sigma=float(x[4]),
        )

    def calc(self, k: np.ndarray) -> float | np.ndarray:
        """Evaluate the raw svi curve at the given log-strikes."""
        return calc_raw_svi(k, self.a, self.b, self.rho, self.m, self.sigma)


def fit_raw_svi(k: np.ndarray, w: np.ndarray, tol: float = 1e-6) -> RawSVIParams:
    """Fit raw SVI curve to given log-strikes."""
    k, w = filter_valid(k, w)
    lower: npt.ArrayLike = np.array([-np.inf, 0, -1 + tol, -np.inf, 0], dtype=float)
    upper: npt.ArrayLike = np.array([np.inf, 1, 1 - tol, np.inf, np.inf], dtype=float)
    bounds: tuple[npt.ArrayLike, npt.ArrayLike] = (lower, upper)
    p0 = np.asarray(RawSVIParams())
    fit, _ = curve_fit(
        f=calc_raw_svi,
        xdata=k,
        ydata=w,
        p0=p0,
        bounds=bounds,
        maxfev=1 / tol,  # type: ignore[call-overload]
    )
    return RawSVIParams.from_array(fit)


def calc_call_price_implied_density(K: np.ndarray, C: np.ndarray) -> np.ndarray:  # noqa: N803
    """Calculate implied density.

    If assumptions are violated, it won't be a density.
    Need no butterfly arb, no calendar arb.
    """
    q_bar = -np.diff(C, prepend=-1 + C[0]) / np.diff(K, prepend=1 + K[0])
    if np.any(q_bar < 0) or np.any(q_bar > 1):
        n = ((q_bar < 0) | (q_bar > 1)).sum()
        logger.warning("%d call spreads are outside of [0, 1]", n)
    q = -np.diff(q_bar, append=np.nan)
    return q
