import numpy as np
import pandas as pd

from app.analysis.forecast import HORIZONS


TARGET_PREFIX = "target_return_"

EXCLUDED_COLUMNS = {
    "open_time",
    "close_time",

    "target_return_3d",
    "target_return_5d",
    "target_return_7d",

    "target_direction_3d",
    "target_direction_5d",
    "target_direction_7d",
}


def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:

    columns = []

    for column in df.columns:

        if column in EXCLUDED_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):
            columns.append(column)

    return columns


class RidgePredictor:

    def __init__(
        self,
        alpha: float = 10.0,
    ):
        self.alpha = alpha
        self.feature_columns = []
        self.mean_ = None
        self.std_ = None
        self.coef_ = None
        self.intercept_ = 0.0
        self.fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        target_column: str,
    ):

        self.feature_columns = get_feature_columns(
            df
        )

        data = (
            df[
                self.feature_columns
                + [target_column]
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        if len(data) < 100:
            raise ValueError(
                f"Not enough training rows: {len(data)}"
            )

        X = data[
            self.feature_columns
        ].to_numpy(dtype=float)

        y = data[
            target_column
        ].to_numpy(dtype=float)

        self.mean_ = X.mean(axis=0)

        self.std_ = X.std(axis=0)

        self.std_[
            self.std_ == 0
        ] = 1.0

        X_scaled = (
            X - self.mean_
        ) / self.std_

        X_design = np.column_stack(
            [
                np.ones(len(X_scaled)),
                X_scaled,
            ]
        )

        identity = np.eye(
            X_design.shape[1]
        )

        identity[0, 0] = 0.0

        matrix = (
            X_design.T @ X_design
            + self.alpha * identity
        )

        vector = (
            X_design.T @ y
        )

        weights = np.linalg.solve(
            matrix,
            vector,
        )

        self.intercept_ = weights[0]
        self.coef_ = weights[1:]
        self.fitted = True

        return self

    def predict(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:

        if not self.fitted:
            raise RuntimeError(
                "Predictor has not been fitted"
            )

        X = (
            df[
                self.feature_columns
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

        X = X.fillna(
            pd.Series(
                self.mean_,
                index=self.feature_columns,
            )
        )

        X = X.to_numpy(
            dtype=float
        )

        X_scaled = (
            X - self.mean_
        ) / self.std_

        return (
            self.intercept_
            + X_scaled @ self.coef_
        )

    def predict_one(
        self,
        df: pd.DataFrame,
    ) -> float:

        return float(
            self.predict(
                df.tail(1)
            )[0]
        )


def calculate_confidence(
    predictions: np.ndarray,
    actual: np.ndarray,
) -> float:

    if len(predictions) == 0:
        return 0.0

    prediction_direction = np.sign(
        predictions
    )

    actual_direction = np.sign(
        actual
    )

    directional_accuracy = float(
        np.mean(
            prediction_direction
            == actual_direction
        )
    )

    errors = (
        actual - predictions
    )

    scale = float(
        np.std(actual)
    )

    if scale == 0:
        stability = 0.0
    else:
        error_score = (
            np.mean(
                np.abs(errors)
            ) / scale
        )

        stability = (
            1.0
            / (1.0 + error_score)
        )

    confidence = (
        0.65
        * directional_accuracy
        + 0.35
        * stability
    )

    return float(
        np.clip(
            confidence,
            0.0,
            1.0,
        )
    )


def evaluate_horizon_model(
    df: pd.DataFrame,
    horizon: str,
    n_splits: int = 5,
    min_train_size: int = 300,
    test_size: int = 80,
    alpha: float = 10.0,
) -> dict:
    """
    Evaluate one horizon using purged walk-forward validation.
    """

    from app.backtest.walk_forward import (
        walk_forward_validate,
    )

    return walk_forward_validate(
        df=df,
        horizon=horizon,
        n_splits=n_splits,
        min_train_size=min_train_size,
        test_size=test_size,
        alpha=alpha,
    )


def fit_final_horizon_model(
    df: pd.DataFrame,
    horizon: str,
    alpha: float = 10.0,
) -> dict:
    """
    Train the production model using all currently
    available rows whose future target is already known.

    The latest incomplete target row is excluded automatically
    because target_return_* is NaN for it.
    """

    if horizon not in HORIZONS:
        raise ValueError(
            f"Unsupported horizon: {horizon}"
        )

    target = (
        f"{TARGET_PREFIX}{horizon}"
    )

    clean = (
        df
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=[target]
        )
        .reset_index(drop=True)
    )

    if len(clean) < 100:
        raise ValueError(
            f"Not enough data for final model: {len(clean)}"
        )

    model = RidgePredictor(
        alpha=alpha
    )

    model.fit(
        clean,
        target,
    )

    return {
        "model": model,
        "horizon": horizon,
        "target": target,
        "train_rows": len(clean),
    }


def train_horizon_model(
    df: pd.DataFrame,
    horizon: str,
    train_ratio: float = 0.8,
    alpha: float = 10.0,
):
    """
    Backward-compatible evaluation function.

    New code should use:
        evaluate_horizon_model()
        fit_final_horizon_model()

    This function is retained temporarily so existing callers
    do not immediately break.
    """

    if horizon not in HORIZONS:
        raise ValueError(
            f"Unsupported horizon: {horizon}"
        )

    target = (
        f"{TARGET_PREFIX}{horizon}"
    )

    clean = (
        df
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=[target]
        )
        .reset_index(drop=True)
    )

    split = int(
        len(clean)
        * train_ratio
    )

    if split < 100:
        raise ValueError(
            "Not enough historical data"
        )

    train = clean.iloc[:split]
    test = clean.iloc[split:]

    model = RidgePredictor(
        alpha=alpha
    )

    model.fit(
        train,
        target,
    )

    predictions = model.predict(
        test
    )

    actual = (
        test[target]
        .to_numpy(dtype=float)
    )

    errors = actual - predictions

    mae = float(
        np.mean(
            np.abs(errors)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )

    directional_accuracy = float(
        np.mean(
            np.sign(predictions)
            == np.sign(actual)
        )
    )

    confidence = calculate_confidence(
        predictions,
        actual,
    )

    if (
        len(predictions) > 1
        and np.std(predictions) > 0
        and np.std(actual) > 0
    ):
        correlation = float(
            np.corrcoef(
                predictions,
                actual,
            )[0, 1]
        )
    else:
        correlation = 0.0

    return {
        "model": model,
        "horizon": horizon,
        "target": target,
        "train_rows": len(train),
        "test_rows": len(test),
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy":
            directional_accuracy,
        "correlation":
            correlation,
        "confidence":
            confidence,
        "predictions":
            predictions,
        "actual":
            actual,
    }
