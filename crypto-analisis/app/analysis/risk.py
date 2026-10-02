import numpy as np
import pandas as pd


def calculate_returns(
    df: pd.DataFrame,
) -> pd.Series:
    return (
        df["close"]
        .astype(float)
        .pct_change()
        .dropna()
    )


def calculate_volatility(
    df: pd.DataFrame,
    periods_per_day: int = 6,
) -> dict:
    """
    Data 4H:
    6 candle / hari.
    """

    returns = calculate_returns(df)

    if len(returns) < 30:
        raise ValueError(
            "Not enough data for volatility"
        )

    daily_vol = (
        returns.std()
        * np.sqrt(periods_per_day)
    )

    weekly_vol = (
        daily_vol
        * np.sqrt(7)
    )

    return {
        "4h_volatility":
            float(returns.std()),

        "daily_volatility":
            float(daily_vol),

        "weekly_volatility":
            float(weekly_vol),
    }


def calculate_drawdown(
    df: pd.DataFrame,
) -> float:
    prices = df["close"].astype(float)

    peak = prices.cummax()

    drawdown = (
        prices / peak - 1.0
    )

    return float(
        drawdown.min()
    )


def calculate_atr_percent(
    df: pd.DataFrame,
    period: int = 14,
) -> float:

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(
        period
    ).mean()

    latest_atr = atr.iloc[-1]
    latest_close = close.iloc[-1]

    if latest_close == 0:
        return 0.0

    return float(
        latest_atr / latest_close
    )


def build_risk_range(
    expected_return: float,
    daily_volatility: float,
    horizon_days: int,
) -> dict:

    horizon_volatility = (
        daily_volatility
        * np.sqrt(horizon_days)
    )

    # Range ± 1.5 standard deviation.
    range_width = (
        1.5
        * horizon_volatility
    )

    upside = (
        expected_return
        + range_width
    )

    downside = (
        expected_return
        - range_width
    )

    return {
        "expected_return":
            float(expected_return),

        "upside":
            float(upside),

        "downside":
            float(downside),

        "volatility":
            float(horizon_volatility),
    }


def classify_risk(
    volatility: float,
    drawdown: float,
) -> str:

    abs_drawdown = abs(drawdown)

    if (
        volatility >= 0.12
        or abs_drawdown >= 0.30
    ):
        return "high"

    if (
        volatility >= 0.07
        or abs_drawdown >= 0.18
    ):
        return "medium"

    return "low"


def calculate_risk_profile(
    df: pd.DataFrame,
) -> dict:

    volatility = calculate_volatility(df)

    drawdown = calculate_drawdown(df)

    atr_pct = calculate_atr_percent(df)

    risk = classify_risk(
        volatility["weekly_volatility"],
        drawdown,
    )

    return {
        "daily_volatility":
            volatility["daily_volatility"],

        "weekly_volatility":
            volatility["weekly_volatility"],

        "atr_percent":
            atr_pct,

        "maximum_drawdown":
            drawdown,

        "risk_level":
            risk,
    }
