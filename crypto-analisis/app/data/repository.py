from app.data.database import get_connection


def save_candles(
    symbol: str,
    interval: str,
    rows: list,
) -> dict:
    symbol = symbol.upper()

    inserted = 0
    updated = 0

    with get_connection() as conn:
        for row in rows:
            if len(row) < 11:
                continue

            cursor = conn.execute(
                """
                INSERT INTO candles (
                    symbol,
                    interval,
                    open_time,
                    close_time,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    quote_volume,
                    trades,
                    taker_buy_base_volume,
                    taker_buy_quote_volume
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, interval, open_time)
                DO UPDATE SET
                    close_time = excluded.close_time,
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    volume = excluded.volume,
                    quote_volume = excluded.quote_volume,
                    trades = excluded.trades,
                    taker_buy_base_volume =
                        excluded.taker_buy_base_volume,
                    taker_buy_quote_volume =
                        excluded.taker_buy_quote_volume
                """,
                (
                    symbol,
                    interval,
                    int(row[0]),
                    int(row[6]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                    float(row[7]),
                    int(row[8]),
                    float(row[9]),
                    float(row[10]),
                ),
            )

            if cursor.rowcount:
                inserted += 1

        conn.commit()

    return {
        "symbol": symbol,
        "interval": interval,
        "rows": len(rows),
        "processed": inserted,
    }


def get_candles(
    symbol: str,
    interval: str = "4h",
    limit: int = 500,
) -> list[dict]:
    symbol = symbol.upper()

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                open_time,
                close_time,
                open,
                high,
                low,
                close,
                volume,
                quote_volume,
                trades,
                taker_buy_base_volume,
                taker_buy_quote_volume
            FROM candles
            WHERE symbol = ?
              AND interval = ?
            ORDER BY open_time DESC
            LIMIT ?
            """,
            (
                symbol,
                interval,
                limit,
            ),
        ).fetchall()

    result = []

    for row in reversed(rows):
        result.append(
            {
                "open_time": row["open_time"],
                "close_time": row["close_time"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "quote_volume": row["quote_volume"],
                "trades": row["trades"],
                "taker_buy_base_volume": row[
                    "taker_buy_base_volume"
                ],
                "taker_buy_quote_volume": row[
                    "taker_buy_quote_volume"
                ],
            }
        )

    return result


def count_candles(
    symbol: str,
    interval: str = "4h",
) -> int:
    symbol = symbol.upper()

    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS total
            FROM candles
            WHERE symbol = ?
              AND interval = ?
            """,
            (
                symbol,
                interval,
            ),
        ).fetchone()

    return int(row["total"])
