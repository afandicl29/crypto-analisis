import pandas as pd

from app.data.repository import get_candles
from app.data.binance import BinanceClient
from app.features.indicators import build_features
from app.analysis.forecast import add_forecast_targets
from app.analysis.risk import (
    calculate_risk_profile,
    build_risk_range,
)
from app.analysis.signal import (
    calculate_signal_score,
    evaluate_model_quality,
    apply_model_quality_penalty,
)
from app.models.predictor import (
    train_horizon_model,
)
from app.ai.analyst import analyze


HORIZONS = ["3d", "5d", "7d"]


def analyze_symbol(
    symbol: str,
    interval: str = "4h",
    limit: int = 1000,
) -> dict:

    rows = get_candles(
        symbol=symbol,
        interval=interval,
        limit=limit,
    )

    if len(rows) < 300:
        raise ValueError(
            f"Not enough candles for {symbol}: "
            f"{len(rows)}"
        )

    df = pd.DataFrame(rows)

    df = build_features(df)
    df = add_forecast_targets(df)

    latest = df.iloc[-1]

    client = BinanceClient()

    try:
        current_price = client.get_ticker_price(symbol)
    finally:
        client.close()

    risk = calculate_risk_profile(df)

    horizons = {}

    for horizon in HORIZONS:

        result = train_horizon_model(
            df,
            horizon=horizon,
            train_ratio=0.8,
            alpha=10.0,
        )

        model = result["model"]

        prediction = model.predict_one(df)

        target = (
            f"target_return_{horizon}"
        )

        historical = (
            df[target]
            .replace(
                [float("inf"), float("-inf")],
                pd.NA,
            )
            .dropna()
        )

        # Residual error dari out-of-sample test.
        residual_std = (
            (
                result["actual"]
                - result["predictions"]
            ).std()
        )

        days = int(
            horizon.replace("d", "")
        )

        range_result = build_risk_range(
            expected_return=prediction,
            daily_volatility=(
                risk["daily_volatility"]
            ),
            horizon_days=days,
        )

        # Gabungkan volatilitas pasar dan
        # error model untuk range yang lebih realistis.
        model_error_width = (
            1.5 * residual_std
        )

        downside = min(
            range_result["downside"],
            prediction - model_error_width,
        )

        upside = max(
            range_result["upside"],
            prediction + model_error_width,
        )

        horizons[horizon] = {
            "prediction": float(
                prediction
            ),
            "upside": float(upside),
            "downside": float(downside),
            "volatility": float(
                range_result["volatility"]
            ),
            "directional_accuracy": float(
                result[
                    "directional_accuracy"
                ]
            ),
            "mae": float(
                result["mae"]
            ),
            "rmse": float(
                result["rmse"]
            ),
            "correlation": float(
                result["correlation"]
            ),
            "confidence": float(
                result["confidence"]
            ),
            "samples": int(
                result["test_rows"]
            ),
        }

    # Horizon utama untuk ranking:
    # 7 hari karena target aplikasi adalah 3–7 hari
    primary = horizons["7d"]
    quality_3d = evaluate_model_quality(
    horizons["3d"]["directional_accuracy"],
    horizons["3d"]["correlation"],
    horizons["3d"]["confidence"],
    )

    quality_5d = evaluate_model_quality(
        horizons["5d"]["directional_accuracy"],
        horizons["5d"]["correlation"],
        horizons["5d"]["confidence"],
    )

    quality_7d = evaluate_model_quality(
        horizons["7d"]["directional_accuracy"],
        horizons["7d"]["correlation"],
        horizons["7d"]["confidence"],
    )

    quality_map = {
        "3d": quality_3d,
        "5d": quality_5d,
        "7d": quality_7d,
    }

    signal = calculate_signal_score(
        expected_return=primary[
            "prediction"
        ],
        confidence=primary[
            "confidence"
        ],
        directional_accuracy=primary[
            "directional_accuracy"
        ],
        correlation=primary[
            "correlation"
        ],
        trend_up=float(
            latest.get(
                "trend_up",
                0,
            )
        ),
        trend_down=float(
            latest.get(
                "trend_down",
                0,
            )
        ),
        rsi=float(
            latest["rsi_14"]
        ),
        risk_level=risk[
            "risk_level"
        ],
        downside=primary[
            "downside"
        ],
    )

    raw_score = signal["score"]

    final_score = apply_model_quality_penalty(
        raw_score,
        quality_map,
    )

    signal["raw_score"] = raw_score
    signal["score"] = final_score

    if final_score < 0.45:
        signal["signal"] = "no_edge"

    elif final_score < 0.55:
        signal["signal"] = "weak"

    else:
        signal["signal"] = "positive"

    ai_analysis = analyze(
        analysis={
            "symbol": symbol,
            "interval": interval,
           "current_price": float(
                current_price
            ),
            "indicators": {
                "rsi": float(
                    latest["rsi_14"]
                ),
                "ema20": float(
                    latest["ema_20"]
                ),
                "ema50": float(
                    latest["ema_50"]
                ),
                "ema200": float(
                    latest["ema_200"]
                ),
                "atr_percent": float(
                    latest["atr_pct"]
                ),
                "volume_ratio": float(
                    latest["volume_ratio"]
                ),
                "trend_up": bool(
                    latest["trend_up"]
                ),
                "trend_down": bool(
                    latest["trend_down"]
                ),
            },
            "risk": risk,
            "model_quality": quality_map,
            "horizons": horizons,
            "signal": signal,
        },
    )
        
    return {
        "symbol": symbol,
        "interval": interval,
        "candles": len(df),
       "current_price": float(
            current_price
        ),
        "indicators": {
            "rsi": float(
                latest["rsi_14"]
            ),
            "ema20": float(
                latest["ema_20"]
            ),
            "ema50": float(
                latest["ema_50"]
            ),
            "ema200": float(
                latest["ema_200"]
            ),
            "atr_percent": float(
                latest["atr_pct"]
            ),
            "volume_ratio": float(
                latest["volume_ratio"]
            ),
            "trend_up": bool(
                latest["trend_up"]
            ),
            "trend_down": bool(
                latest["trend_down"]
            ),
        },
        "risk": risk,
        "model_quality": quality_map,
        "horizons": horizons,
        "signal": signal,
        "ai_analysis": ai_analysis,
    }


def analyze_market(
    symbols: list[str],
    interval: str = "4h",
    limit: int = 1000,
) -> list[dict]:

    results = []

    for symbol in symbols:
        try:
            result = analyze_symbol(
                symbol,
                interval,
                limit,
            )

            results.append(result)

        except Exception as exc:
            results.append(
                {
                    "symbol": symbol,
                    "error": str(exc),
                }
            )

    valid = [
        item
        for item in results
        if "score" in item.get(
            "signal",
            {}
        )
    ]

    valid.sort(
        key=lambda item: item[
            "signal"
        ]["score"],
        reverse=True,
    )

    for rank, item in enumerate(
        valid,
        start=1,
    ):
        item["rank"] = rank
    
    return results
