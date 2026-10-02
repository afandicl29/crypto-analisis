const state = {
    symbol: "BTCUSDT",
    interval: "4h",
    loading: false
};

const USDT_IDR = 16600;

function formatPrice(value) {
    if (
        value === null ||
        value === undefined ||
        !Number.isFinite(value)
    ) {
        return "—";
    }

    const idrValue = value * USDT_IDR;

    return new Intl.NumberFormat("id-ID", {
        style: "currency",
        currency: "IDR",
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(idrValue);
}


function formatPercent(value, digits = 2) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) {
        return "—";
    }

    const number = Number(value);

    return `${number >= 0 ? "+" : ""}${number.toFixed(digits)}%`;
}


function formatConfidence(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) {
        return "—";
    }

    return `${(Number(value) * 100).toFixed(1)}%`;
}


function formatNumber(value, digits = 3) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) {
        return "—";
    }

    return Number(value).toFixed(digits);
}


function directionFromPrediction(prediction) {
    if (prediction > 0.002) {
        return {
            label: "↑ NAIK",
            className: "up"
        };
    }

    if (prediction < -0.002) {
        return {
            label: "↓ TURUN",
            className: "down"
        };
    }

    return {
        label: "— NETRAL",
        className: "neutral"
    };
}


function renderForecast(id, forecast, currentPrice) {
    const element = document.getElementById(id);

    if (!element || !forecast) {
        return;
    }

    const direction = element.querySelector(".direction");
    const target = element.querySelector(".target");
    const change = element.querySelector(".change");
    const confidence = element.querySelector(".confidence");

    const prediction = Number(forecast.prediction);
    const targetPrice = currentPrice * (1 + prediction);

    const directionData = directionFromPrediction(prediction);

    element.classList.remove(
        "up",
        "down",
        "neutral"
    );

    element.classList.add(
        directionData.className
    );

    direction.textContent = directionData.label;

    target.textContent = formatPrice(
        targetPrice
    );

    change.textContent = formatPercent(
        prediction * 100
    );

    confidence.textContent =
        `Confidence: ${formatConfidence(forecast.confidence)}`;
}


function setText(id, value) {
    const element = document.getElementById(id);

    if (element) {
        element.textContent = value;
    }
}


function renderAnalysisStatus(text) {
    const element = document.querySelector(
        ".analysis-status"
    );

    if (element) {
        element.textContent = text;
    }
}


function getBias(signal) {
    if (!signal) {
        return "—";
    }

    switch (signal.signal) {
        case "strong":
            return "STRONG";
        case "positive":
            return "POSITIVE";
        case "weak":
            return "WEAK";
        case "negative":
            return "NEGATIVE";
        case "no_edge":
            return "NO EDGE";
        default:
            return String(signal.signal).toUpperCase();
    }
}


function getBestHorizon(horizons) {
    let best = null;

    for (const horizon of ["3d", "5d", "7d"]) {
        const item = horizons?.[horizon];

        if (!item) {
            continue;
        }

        if (
            best === null ||
            Number(item.confidence) >
            Number(best.confidence)
        ) {
            best = {
                horizon,
                confidence: item.confidence
            };
        }
    }

    if (!best) {
        return "—";
    }

    return best.horizon.toUpperCase();
}


function getSupportResistance(result) {
    const current = Number(
        result.current_price
    );

    const risk = result.risk || {};

    /*
     * Jika risk engine sudah menyediakan level,
     * gunakan level tersebut.
     */
    const supportCandidates = [
        risk.support,
        risk.support_price,
        risk.downside_price
    ];

    const resistanceCandidates = [
        risk.resistance,
        risk.resistance_price,
        risk.upside_price
    ];

    const support = supportCandidates.find(
        value =>
            Number.isFinite(Number(value))
    );

    const resistance = resistanceCandidates.find(
        value =>
            Number.isFinite(Number(value))
    );

    /*
     * Fallback sementara apabila risk engine belum
     * menyediakan support/resistance eksplisit.
     */
    return {
        support: support !== undefined
            ? Number(support)
            : null,

        resistance: resistance !== undefined
            ? Number(resistance)
            : null,

        current
    };
}


function buildConclusion(result) {
    if (
        result.ai_analysis &&
        result.ai_analysis.conclusion
    ) {
        return result.ai_analysis.conclusion;
    }

    const signal = result.signal || {};
    const primary = result.horizons?.["7d"];

    if (!primary) {
        return "Data analisis belum tersedia.";
    }

    const prediction =
        Number(primary.prediction) * 100;

    const confidence =
        Number(primary.confidence) * 100;

    const direction =
        prediction >= 0
            ? "kenaikan"
            : "penurunan";

    return (
        `Model 7D menunjukkan estimasi ${direction} ` +
        `${Math.abs(prediction).toFixed(2)}% ` +
        `dengan confidence ${confidence.toFixed(1)}%. ` +
        `Signal saat ini ${getBias(signal)}.`
    );
}


function renderModelMetrics(result) {
    /*
     * Tambahkan informasi model ke subtitle
     * tanpa mengubah struktur dashboard utama.
     */
    const primary = result.horizons?.["7d"];

    if (!primary) {
        return;
    }

    const panel = document.querySelector(
        ".analysis-panel .panel-title p"
    );

    if (panel) {
        panel.textContent =
            `7D Accuracy ${formatConfidence(primary.directional_accuracy)} · ` +
            `Correlation ${formatNumber(primary.correlation, 3)} · ` +
            `MAE ${formatPercent(primary.mae * 100)}`;
    }
}


function render(result) {
    if (!result) {
        return;
    }

    const currentPrice =
        Number(result.current_price);

    const symbol =
        result.symbol || state.symbol;

    const indicators =
        result.indicators || {};

    const risk =
        result.risk || {};

    const signal =
        result.signal || {};

    const horizons =
        result.horizons || {};

    const assetName =
        symbol.replace(
            "USDT",
            "/USDT"
        );

    setText(
        "asset-name",
        assetName
    );

    setText(
        "current-price",
        formatPrice(currentPrice)
    );

    setText(
        "level-current",
        formatPrice(currentPrice)
    );

    /*
     * Forecast
     */
    renderForecast(
        "forecast-3d",
        horizons["3d"],
        currentPrice
    );

    renderForecast(
        "forecast-5d",
        horizons["5d"],
        currentPrice
    );

    renderForecast(
        "forecast-7d",
        horizons["7d"],
        currentPrice
    );

    /*
     * Signal / AI summary
     */
    setText(
        "market-bias",
        getBias(signal)
    );

    setText(
        "market-risk",
        risk.risk_level
            ? String(risk.risk_level).toUpperCase()
            : "—"
    );

    setText(
        "best-horizon",
        getBestHorizon(horizons)
    );

    setText(
        "ai-conclusion",
        buildConclusion(result)
    );

    /*
     * Support / resistance.
     */
    const levels =
        getSupportResistance(result);

    setText(
        "support",
        levels.support !== null
            ? formatPrice(levels.support)
            : "—"
    );

    setText(
        "resistance",
        levels.resistance !== null
            ? formatPrice(levels.resistance)
            : "—"
    );

    /*
     * Status.
     */
    const quality =
        result.model_quality?.["7d"]?.quality;

    renderAnalysisStatus(
        quality
            ? `7D MODEL: ${quality.toUpperCase()}`
            : "ANALYSIS READY"
    );

    /*
     * Current price change placeholder.
     *
     * Kita tidak mengarang perubahan harga intraday.
     * Karena API saat ini belum mengirim perubahan
     * terhadap candle sebelumnya.
     */
    setText(
        "price-change",
        `RSI ${formatNumber(indicators.rsi, 1)} · ` +
        `ATR ${formatPercent(indicators.atr_percent * 100)}`
    );

    renderModelMetrics(result);

    setText(
        "last-update",
        new Date().toLocaleTimeString(
            "id-ID"
        )
    );
}


async function loadAnalysis() {
    if (state.loading) {
        return;
    }

    state.loading = true;

    renderAnalysisStatus(
        "LOADING..."
    );

    try {
        const response = await fetch(
            `/api/analysis/${state.symbol}?interval=${state.interval}&limit=1000`,
            {
                cache: "no-store"
            }
        );

        if (!response.ok) {
            const text = await response.text();

            throw new Error(
                `${response.status}: ${text}`
            );
        }

        const result =
            await response.json();

        render(result);

    } catch (error) {
        console.error(
            "Analysis error:",
            error
        );

        renderAnalysisStatus(
            "ANALYSIS ERROR"
        );

        setText(
            "ai-conclusion",
            `Gagal mengambil analisis: ${error.message}`
        );

    } finally {
        state.loading = false;
    }
}


function setup() {
    const selector =
        document.getElementById("symbol");

    if (selector) {
        state.symbol =
            selector.value;

        selector.addEventListener(
            "change",
            () => {
                state.symbol =
                    selector.value;

                loadAnalysis();
            }
        );
    }

    loadAnalysis();

    /*
     * Refresh setiap 5 menit.
     *
     * Tidak terlalu agresif karena proses
     * analisis model cukup berat.
     */
   setInterval(
    loadAnalysis,
    60 * 1000
    );
}


document.addEventListener(
    "DOMContentLoaded",
    setup
);
