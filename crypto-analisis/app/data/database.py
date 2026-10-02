'PY'
import sqlite3
from contextlib import contextmanager

from app.config.settings import DATABASE_PATH


def initialize_database():
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                open_time INTEGER NOT NULL,
                close_time INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                quote_volume REAL NOT NULL,
                trades INTEGER NOT NULL,
                taker_buy_base_volume REAL NOT NULL,
                taker_buy_quote_volume REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,

                UNIQUE(symbol, interval, open_time)
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_candles_symbol_interval_time
            ON candles(symbol, interval, open_time)
            """
        )

        conn.commit()


@contextmanager
def get_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row

    try:
        yield conn
    finally:
        conn.close()
