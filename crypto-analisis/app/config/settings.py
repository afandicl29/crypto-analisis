import os
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = DATA_DIR / "crypto.db"

BINANCE_BASE_URL = "https://data-api.binance.vision"

SUPPORTED_INTERVALS = (
    "1h",
    "4h",
    "1d",
)

DEFAULT_INTERVAL = "4h"

DEFAULT_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
)

# ============================================================
# AI / OpenRouter
# ============================================================

import os

AI_PROVIDER = os.getenv(
    "AI_PROVIDER",
    "openrouter",
).lower()

AI_API_KEY = os.getenv(
    "AI_API_KEY",
    "",
)

AI_BASE_URL = os.getenv(
    "AI_BASE_URL",
    "https://openrouter.ai/api/v1",
).rstrip("/")

AI_MODEL = os.getenv(
    "AI_MODEL",
    "openrouter/free",
)

AI_TIMEOUT = int(
    os.getenv(
        "AI_TIMEOUT",
        "60",
    )
)
