import json
from typing import Any

import requests

from app.config.settings import (
    AI_API_KEY,
    AI_BASE_URL,
    AI_MODEL,
    AI_PROVIDER,
    AI_TIMEOUT,
)


SYSTEM_PROMPT = """
You are the AI Analyst for a cryptocurrency quantitative
decision-support application.

Analyze ONLY the quantitative data provided.

Rules:
- Do not invent market data.
- Do not change supplied numbers.
- Do not guarantee profit.
- Do not execute trades.
- Do not provide entry or order execution instructions.
- The application evaluates 3-day, 5-day and 7-day horizons.

Return ONLY valid JSON.
Do not use markdown.
Do not use ``` fences.

Required schema:

{
  "summary": "short objective summary",
  "bullish_factors": [],
  "risk_factors": [],
  "outlook": {
    "3d": "",
    "5d": "",
    "7d": ""
  },
  "model_assessment": "",
  "conclusion": ""
}

Keep the analysis concise and objective.
"""


def _fallback_analysis(
    reason: str,
) -> dict[str, Any]:
    """
    Fallback agar kegagalan AI tidak membuat
    seluruh endpoint quantitative analysis gagal.
    """

    return {
        "summary": (
            "AI Analyst tidak tersedia. "
            "Hasil quantitative model tetap dapat digunakan."
        ),

        "bullish_factors": [],

        "risk_factors": [
            "AI analysis unavailable"
        ],

        "outlook": {
            "3d": "AI unavailable",
            "5d": "AI unavailable",
            "7d": "AI unavailable",
        },

        "model_assessment": (
            "Quantitative analysis tersedia, "
            "tetapi interpretasi AI gagal."
        ),

        "conclusion": (
            "Analisis kuantitatif berhasil. "
            "AI Analyst tidak dapat memberikan "
            "interpretasi pada saat ini."
        ),

        "error": reason,
    }


def _clean_json_content(
    content: str,
) -> str:

    content = (
        content
        .strip()
        .replace("\ufeff", "")
    )

    if not content:
        raise ValueError(
            "AI returned empty content."
        )

    # Hapus markdown fence jika model tetap
    # mengembalikannya meskipun sudah dilarang.
    if content.startswith("```"):

        lines = content.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        content = "\n".join(
            lines
        ).strip()

    # Cari object JSON jika model menambahkan
    # teks sebelum/sesudah JSON.
    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "AI response does not contain a JSON object."
        )

    return content[start:end + 1]


def analyze(
    analysis: dict[str, Any],
) -> dict[str, Any]:

    if AI_PROVIDER != "openrouter":
        return _fallback_analysis(
            f"Unsupported AI provider: {AI_PROVIDER}"
        )

    if not AI_API_KEY:
        return _fallback_analysis(
            "AI_API_KEY is not configured."
        )

    payload = {
        "symbol": analysis.get("symbol"),
        "interval": analysis.get("interval"),
        "current_price": analysis.get(
            "current_price"
        ),
        "indicators": analysis.get(
            "indicators",
            {},
        ),
        "risk": analysis.get(
            "risk",
            {},
        ),
        "horizons": analysis.get(
            "horizons",
            {},
        ),
        "model_quality": analysis.get(
            "model_quality",
            {},
        ),
        "signal": analysis.get(
            "signal",
            {},
        ),
    }

    try:

        response = requests.post(
            f"{AI_BASE_URL}/chat/completions",

            headers={
                "Authorization":
                    f"Bearer {AI_API_KEY}",

                "Content-Type":
                    "application/json",

                "HTTP-Referer":
                    "http://localhost:8000",

                "X-Title":
                    "Crypto Analyzer",
            },

            json={
                "model": AI_MODEL,

                "temperature": 0.2,

                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },

                    {
                        "role": "user",
                        "content": (
                            "Analyze this quantitative result:\n\n"
                            + json.dumps(
                                payload,
                                indent=2,
                            )
                        ),
                    },
                ],
            },

            timeout=AI_TIMEOUT,
        )

        if not response.ok:

            return _fallback_analysis(
                f"OpenRouter API error "
                f"{response.status_code}"
            )

        data = response.json()

        choices = data.get(
            "choices",
            [],
        )

        if not choices:

            return _fallback_analysis(
                "AI response contains no choices."
            )

        message = choices[0].get(
            "message",
            {},
        )

        content = message.get(
            "content",
            "",
        )

        content = _clean_json_content(
            content
        )

        result = json.loads(
            content
        )

        if not isinstance(
            result,
            dict,
        ):
            raise ValueError(
                "AI JSON response is not an object."
            )

        return result

    except json.JSONDecodeError as exc:

        return _fallback_analysis(
            f"Invalid AI JSON: {exc}"
        )

    except requests.RequestException as exc:

        return _fallback_analysis(
            f"AI network error: {exc}"
        )

    except Exception as exc:

        return _fallback_analysis(
            f"AI analysis error: {exc}"
        )
