"""Functions to support plotting."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def add_vol_plot(  # noqa: PLR0913
    fig: go.Figure,
    vol_df: pd.DataFrame,
    name: str,
    mode: str = "lines+markers",
    x_col: str = "strike_price",
    y_col: str = "iv_midprice",
    strike_range: tuple[float, float] | None = None,
) -> None:
    """Add volatility plot to figure."""
    plot_df = vol_df
    if strike_range is not None:
        plot_df = vol_df[
            (vol_df[x_col] >= strike_range[0]) & (vol_df[x_col] <= strike_range[1])
        ]
    fig.add_trace(
        go.Scatter(
            x=plot_df[x_col],
            y=plot_df[y_col],
            mode=mode,
            name=name,
        )
    )


def add_vol_range(
    fig: go.Figure,
    vol_df: pd.DataFrame,
    range_cols: tuple[str, str] | None = None,
    strike_range: tuple[float, float] | None = None,
) -> None:
    """Add bid-ask vol range to figure."""
    if range_cols is None:
        range_cols = ("iv_bid", "iv_ask")
    df = vol_df
    if strike_range is not None:
        df = vol_df[
            (vol_df["strike_price"] >= strike_range[0])
            & (vol_df["strike_price"] <= strike_range[1])
        ]
    x = np.repeat(df["strike_price"], 3)
    y: np.ndarray = np.column_stack(
        [df[range_cols[0]], df[range_cols[1]], [None] * len(df)]  # type: ignore[arg-type]
    ).ravel()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color="gray", width=2),  # noqa: C408
            name=f"{range_cols[0]} - {range_cols[1]}",
        )
    )


def add_underlying(
    fig: go.Figure,
    underlying_price: float,
    text: str = "Underlying price",
    position: str = "top right",
) -> None:
    """Add vertical line at underlying price to figure."""
    fig.add_vline(
        x=underlying_price,
        line_dash="dot",
        line_color="gray",
        annotation_text=text,
        annotation_position=position,
    )


def layout_total_variance(fig: go.Figure, label: str, detail: str) -> None:
    """Layout implied volatility plot."""
    fig.update_layout(
        title=f"{label} Total Variance by Strike - {detail}",
        xaxis_title="Strike Price",
        yaxis_title="Total Variance (w)",
        template="plotly_white",
    )


def layout_vol(fig: go.Figure, label: str, detail: str) -> None:
    """Layout implied volatility plot."""
    fig.update_layout(
        title=f"{label} Implied Volatility by Strike - {detail}",
        xaxis_title="Strike Price",
        yaxis_title="Implied Volatility (σ)",
        template="plotly_white",
    )
    fig.update_yaxes(tickformat=".0%")


def layout_volume(fig: go.Figure, label: str, detail: str) -> None:
    """Layout volume plot."""
    fig.update_layout(
        title=f"{label} Volume by Strike - {detail}",
        xaxis_title="Strike Price",
        yaxis_title="Volume",
        template="plotly_white",
    )


def add_top_quantity(
    fig: go.Figure, df: pd.DataFrame, label: str, row: int = 1, col: int = 1
) -> None:
    """Add top-of-book quanityt to plotly figure."""
    fig.add_trace(
        go.Bar(
            x=df["strike_price"],
            y=df["bidq"],
            name=f"{label} Bid quantity",
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Bar(
            x=df["strike_price"],
            y=-df["askq"],
            name=f"{label} Ask quantity",
        ),
        row=row,
        col=col,
    )
    fig.update_yaxes(
        title_text="Quantity",
        row=row,
        col=col,
    )


def add_width(
    fig: go.Figure, df: pd.DataFrame, label: str, row: int = 2, col: int = 1
) -> None:
    """Add bid-ask spread to plotly figure."""
    fig.add_trace(
        go.Bar(
            x=df["strike_price"],
            y=df["ask"] - df["bid"],
            name=f"{label} Bid-ask width",
        ),
        row=row,
        col=col,
    )
    fig.update_yaxes(
        title_text="Price spread",
        row=row,
        col=col,
    )


def make_top_subplots(label: str) -> go.Figure:
    """Make subplots for top-of-book quantity and spread."""
    titles = [
        f"{label} Top-of-book Quantity",
        f"{label} Top-of-book Bid-ask spread",
    ]
    return make_subplots(rows=2, cols=1, subplot_titles=titles)


def add_volume_plot(fig: go.Figure, volume_df: pd.DataFrame, name: str) -> None:
    """Add volume to plotly figure."""
    fig.add_trace(
        go.Bar(
            x=volume_df["strike_price"],
            y=volume_df["volume"],
            name=name,
        )
    )
