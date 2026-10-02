import numpy as np


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return float(
        np.clip(
            value,
            minimum,
            maximum,
        )
    )


def score_forecast(
    expected_return: float,
    confidence: float,
    correlation: float = 0.0,
) -> float:
    """
    Forecast score.

    Besarnya prediction tidak cukup.
    Confidence dan correlation ikut menentukan
    apakah forecast layak dipercaya.
    """

    normalized_return = np.tanh(
        expected_return / 0.10
    )

    confidence_component = (
        confidence * 2.0 - 1.0
    )

    # Correlation negatif adalah penalti keras.
    correlation_component = np.clip(
        (correlation + 0.20) / 0.60,
        0.0,
        1.0,
    )

    quality = (
        0.50 * confidence
        + 0.50 * correlation_component
    )

    direction_component = (
        normalized_return
        * (0.40 + 0.60 * quality)
    )

    return (
        0.60 * direction_component
        + 0.40 * confidence_component
    )


def score_backtest(
    directional_accuracy: float,
    correlation: float,
) -> float:

    direction_score = (
        directional_accuracy - 0.50
    ) / 0.20

    correlation_score = (
        correlation + 0.20
    ) / 0.60

    return (
        0.60 * clamp(
            direction_score
        )
        + 0.40 * clamp(
            correlation_score
        )
    )


def score_trend(
    trend_up: float,
    trend_down: float,
) -> float:

    if trend_up >= 1:
        return 1.0

    if trend_down >= 1:
        return 0.0

    return 0.5


def score_momentum(
    rsi: float,
) -> float:
    """
    RSI tidak dianggap sederhana:
    RSI tinggi bukan otomatis bullish.
    Extreme momentum mendapat sedikit penalti.
    """

    if rsi < 30:
        return 0.70

    if rsi < 45:
        return 0.60

    if rsi <= 65:
        return 0.70

    if rsi <= 75:
        return 0.55

    if rsi <= 85:
        return 0.40

    return 0.25


def score_risk(
    risk_level: str,
) -> float:

    mapping = {
        "low": 1.0,
        "medium": 0.70,
        "high": 0.40,
    }

    return mapping.get(
        risk_level,
        0.50,
    )


def calculate_downside_penalty(
    downside: float,
) -> float:

    downside_abs = abs(
        min(downside, 0.0)
    )

    return clamp(
        downside_abs / 0.15
    )


def calculate_signal_score(
    expected_return: float,
    confidence: float,
    directional_accuracy: float,
    correlation: float,
    trend_up: float,
    trend_down: float,
    rsi: float,
    risk_level: str,
    downside: float,
) -> dict:

    forecast_score = score_forecast(
    expected_return,
    confidence,
    correlation,
    )

    backtest_score = score_backtest(
        directional_accuracy,
        correlation,
    )

    trend_score = score_trend(
        trend_up,
        trend_down,
    )

    momentum_score = score_momentum(
        rsi
    )

    risk_score = score_risk(
        risk_level
    )

    downside_penalty = (
        calculate_downside_penalty(
            downside
        )
    )

    raw_score = (
        0.30 * forecast_score
        + 0.20 * backtest_score
        + 0.15 * trend_score
        + 0.10 * momentum_score
        + 0.15 * risk_score
        - 0.10 * downside_penalty
    )

    normalized_score = (
        (raw_score + 1.0)
        / 2.0
    )

    normalized_score = clamp(
        normalized_score
    )

    if normalized_score >= 0.70:
        signal = "strong"
    elif normalized_score >= 0.55:
        signal = "positive"
    elif normalized_score >= 0.45:
        signal = "neutral"
    elif normalized_score >= 0.30:
        signal = "negative"
    else:
        signal = "weak"

    return {
        "score": normalized_score,
        "signal": signal,
        "forecast_score":
            forecast_score,
        "backtest_score":
            backtest_score,
        "trend_score":
            trend_score,
        "momentum_score":
            momentum_score,
        "risk_score":
            risk_score,
        "downside_penalty":
            downside_penalty,
    }


def rank_candidates(
    candidates: list[dict],
) -> list[dict]:

    return sorted(
        candidates,
        key=lambda item: item.get(
            "score",
            0.0,
        ),
        reverse=True,
    )


def evaluate_model_quality(
    directional_accuracy: float,
    correlation: float,
    confidence: float,
) -> dict:
    """
    Menentukan apakah forecast layak digunakan
    sebagai sinyal utama.
    """

    reasons = []

    if directional_accuracy < 0.52:
        reasons.append(
            "direction_accuracy_low"
        )

    if correlation < 0:
        reasons.append(
            "negative_correlation"
        )

    if confidence < 0.50:
        reasons.append(
            "low_confidence"
        )

    if (
        directional_accuracy >= 0.60
        and correlation >= 0.20
        and confidence >= 0.60
    ):
        quality = "strong"

    elif (
        directional_accuracy >= 0.55
        and correlation >= 0.05
        and confidence >= 0.52
    ):
        quality = "usable"

    elif (
        directional_accuracy >= 0.52
        and correlation >= 0
    ):
        quality = "weak"

    else:
        quality = "poor"

    return {
        "quality": quality,
        "usable": quality != "poor",
        "reasons": reasons,
    }

def apply_model_quality_penalty(
    score: float,
    model_quality: dict,
) -> float:
    """
    Penalti score berdasarkan kualitas model
    pada horizon 3D / 5D / 7D.
    """

    weights = {
        "3d": 0.30,
        "5d": 0.30,
        "7d": 0.40,
    }

    quality_values = {
        "strong": 1.00,
        "usable": 0.75,
        "weak": 0.45,
        "poor": 0.10,
    }

    weighted_quality = 0.0

    for horizon, weight in weights.items():
        quality = model_quality.get(
            horizon,
            {},
        )

        label = quality.get(
            "quality",
            "poor",
        )

        weighted_quality += (
            weight
            * quality_values.get(
                label,
                0.10,
            )
        )

    adjusted_score = (
        score * weighted_quality
    )

    return clamp(adjusted_score)

    return float(
        max(
            0.0,
            min(
                adjusted_score,
                1.0,
            ),
        )
    )
