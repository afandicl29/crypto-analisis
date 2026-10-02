import numpy as np
import pandas as pd


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    result = df.copy()

    for column in required:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    return result.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for period in (1, 3, 6, 12, 18, 30, 42):
        df[f"return_{period}"] = df["close"].pct_change(period)

    return df


def add_ema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for period in (9, 20, 50, 100, 200):
        df[f"ema_{period}"] = (
            df["close"]
            .ewm(span=period, adjust=False)
            .mean()
        )

    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi_14"] = 100 - (100 / (1 + rs))

    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    ema12 = df["close"].ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = df["close"].ewm(
        span=26,
        adjust=False,
    ).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = (
        df["macd"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    df["macd_hist"] = (
        df["macd"] - df["macd_signal"]
    )

    return df


def add_bollinger(
    df: pd.DataFrame,
    period: int = 20,
) -> pd.DataFrame:
    df = df.copy()

    middle = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()

    df["bb_middle"] = middle
    df["bb_upper"] = middle + (2 * std)
    df["bb_lower"] = middle - (2 * std)

    width = df["bb_upper"] - df["bb_lower"]

    df["bb_width"] = (
        width / middle.replace(0, np.nan)
    )

    df["bb_position"] = (
        (df["close"] - df["bb_lower"]) /
        width.replace(0, np.nan)
    )

    return df


def add_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    df = df.copy()

    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - previous_close).abs()
    tr3 = (df["low"] - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    df["atr_14"] = (
        true_range
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    df["atr_pct"] = (
        df["atr_14"] /
        df["close"].replace(0, np.nan)
    )

    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["volume_ma_20"] = (
        df["volume"].rolling(20).mean()
    )

    df["volume_ratio"] = (
        df["volume"] /
        df["volume_ma_20"].replace(0, np.nan)
    )

    df["volume_change"] = df["volume"].pct_change()

    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema_20_distance"] = (
        df["close"] / df["ema_20"] - 1
    )

    df["ema_50_distance"] = (
        df["close"] / df["ema_50"] - 1
    )

    df["ema_200_distance"] = (
        df["close"] / df["ema_200"] - 1
    )

    df["ema_20_50_spread"] = (
        df["ema_20"] / df["ema_50"] - 1
    )

    df["ema_50_200_spread"] = (
        df["ema_50"] / df["ema_200"] - 1
    )

    df["trend_up"] = (
        (df["ema_20"] > df["ema_50"]) &
        (df["ema_50"] > df["ema_200"])
    ).astype(int)

    df["trend_down"] = (
        (df["ema_20"] < df["ema_50"]) &
        (df["ema_50"] < df["ema_200"])
    ).astype(int)

    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    returns = df["close"].pct_change()

    for period in (12, 18, 30, 42):
        df[f"volatility_{period}"] = (
            returns.rolling(period).std()
        ) * np.sqrt(period)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _validate(df)

    df = add_returns(df)
    df = add_ema(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_atr(df)
    df = add_volume_features(df)
    df = add_trend_features(df)
    df = add_volatility_features(df)

    return df
