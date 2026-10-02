import pandas as pd


HORIZONS = {
    "3d": 18,
    "5d": 30,
    "7d": 42,
}


def add_forecast_targets(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    for name, periods in HORIZONS.items():
        future_close = result["close"].shift(-periods)

        result[f"target_return_{name}"] = (
            future_close / result["close"] - 1
        )

        result[f"target_direction_{name}"] = (
            result[f"target_return_{name}"] > 0
        ).astype(int)

    return result
