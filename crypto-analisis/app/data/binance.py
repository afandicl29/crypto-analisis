import time
import requests

from app.config.settings import BINANCE_BASE_URL


class BinanceClient:

    def __init__(self):
        self.base_url = BINANCE_BASE_URL
        self.session = requests.Session()

    def _get(self, endpoint, params=None):
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        return response.json()

    def ping(self):
        return self._get("/api/v3/ping")

    def get_klines(
        self,
        symbol,
        interval="4h",
        limit=1000,
        start_time=None,
        end_time=None,
    ):
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1000),
        }

        if start_time is not None:
            params["startTime"] = start_time

        if end_time is not None:
            params["endTime"] = end_time

        return self._get(
            "/api/v3/klines",
            params=params,
        )

    def get_historical_klines(
        self,
        symbol,
        interval="4h",
        limit=3000,
    ):
        all_rows = []
        end_time = None
        remaining = limit

        while remaining > 0:
            batch_size = min(1000, remaining)

            rows = self.get_klines(
                symbol=symbol,
                interval=interval,
                limit=batch_size,
                end_time=end_time,
            )

            if not rows:
                break

            all_rows = rows + all_rows
            remaining -= len(rows)

            if len(rows) < batch_size:
                break

            end_time = rows[0][0] - 1

            time.sleep(0.15)

        unique = {
            row[0]: row
            for row in all_rows
        }

        return sorted(
            unique.values(),
            key=lambda row: row[0],
        )

    def get_ticker_price(self, symbol):
        data = self._get(
            "/api/v3/ticker/price",
            params={
                "symbol": symbol.upper(),
            },
        )

        return float(data["price"])

    def close(self):
        self.session.close()