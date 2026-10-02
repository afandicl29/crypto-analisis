import numpy as np
import pandas as pd

from app.analysis.forecast import HORIZONS
from app.models.predictor import RidgePredictor


def walk_forward_validate(
    df: pd.DataFrame,
    horizon: str,
    n_splits: int = 5,
    min_train_size: int = 300,
    test_size: int = 80,
    alpha: float = 10.0,
) -> dict:
    """
    Purged expanding-window walk-forward validation.

    Training samples whose future target overlaps the test period
    are removed from the end of the training set.

    Example:
        7d horizon = 42 candles
        -> purge 42 candles before every test window.
    """

    if horizon not in HORIZONS:
        raise ValueError(
            f"Unsupported horizon: {horizon}"
        )

    target = f"target_return_{horizon}"
    horizon_periods = HORIZONS[horizon]

    if target not in df.columns:
        raise ValueError(
            f"Missing target column: {target}"
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

    required = (
        min_train_size
        + horizon_periods
        + n_splits * test_size
    )

    if len(clean) < required:
        raise ValueError(
            f"Not enough data for purged walk-forward: "
            f"{len(clean)} < {required}"
        )

    predictions = []
    actual = []
    windows = []

    for split in range(n_splits):

        test_start = (
            min_train_size
            + split * test_size
        )

        test_end = min(
            test_start + test_size,
            len(clean),
        )

        if test_start >= len(clean):
            break

        test = clean.iloc[
            test_start:test_end
        ]

        if len(test) == 0:
            continue

        # Purge training observations whose future target
        # could overlap the test period.
        train_end = (
            test_start - horizon_periods
        )

        if train_end <= 0:
            continue

        train = clean.iloc[
            :train_end
        ]

        if len(train) < min_train_size:
            continue

        model = RidgePredictor(
            alpha=alpha
        )

        model.fit(
            train,
            target,
        )

        pred = model.predict(test)

        act = test[
            target
        ].to_numpy(dtype=float)

        predictions.extend(
            pred.tolist()
        )

        actual.extend(
            act.tolist()
        )

        windows.append(
            {
                "window": split + 1,
                "train_start": 0,
                "train_end": train_end,
                "train_rows": len(train),
                "purge_rows": horizon_periods,
                "test_start": test_start,
                "test_end": test_end,
                "test_rows": len(test),
            }
        )

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    actual = np.asarray(
        actual,
        dtype=float,
    )

    if len(predictions) == 0:
        raise ValueError(
            "No valid walk-forward windows were produced."
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

    direction = float(
        np.mean(
            np.sign(predictions)
            == np.sign(actual)
        )
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

    error_scale = float(
        np.std(actual)
    )

    if error_scale > 0:
        error_ratio = (
            mae / error_scale
        )

        stability = (
            1.0
            / (1.0 + error_ratio)
        )
    else:
        stability = 0.0

    confidence = float(
        np.clip(
            0.65 * direction
            + 0.35 * stability,
            0.0,
            1.0,
        )
    )

    return {
        "horizon": horizon,
        "samples": len(actual),
        "windows": windows,
        "predictions": predictions,
        "actual": actual,
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": direction,
        "correlation": correlation,
        "confidence": confidence,
        "purge_rows": horizon_periods,
    }
