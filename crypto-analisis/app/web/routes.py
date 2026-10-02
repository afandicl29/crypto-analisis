from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config.settings import DEFAULT_INTERVAL, DEFAULT_SYMBOLS
from app.data.binance import BinanceClient
from app.data.database import initialize_database
from app.data.repository import (
    get_candles,
    save_candles,
    count_candles,
)
from app.services.analysis_service import (
    analyze_symbol,
    analyze_market,
)

router = APIRouter()

templates = Jinja2Templates(
    directory="app/web/templates"
)

initialize_database()


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "title": "Crypto Analyzer",
        },
    )


@router.get("/api/market/status")
def market_status():
    return {
        "database": "connected",
        "symbols": list(DEFAULT_SYMBOLS),
        "default_interval": DEFAULT_INTERVAL,
        "candles": {
            symbol: count_candles(
                symbol,
                DEFAULT_INTERVAL,
            )
            for symbol in DEFAULT_SYMBOLS
        },
    }


@router.post("/api/market/sync/{symbol}")
def sync_market(
    symbol: str,
    interval: str = DEFAULT_INTERVAL,
    limit: int = 1000,
):
    symbol = symbol.upper()

    if interval not in ("1h", "4h", "1d"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported interval",
        )

    if limit < 1 or limit > 5000:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 5000",
        )

    client = BinanceClient()

    try:
        rows = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
        )

        if not rows:
            raise HTTPException(
                status_code=404,
                detail="No market data returned",
            )

        result = save_candles(
            symbol=symbol,
            interval=interval,
            rows=rows,
        )

        return {
            "status": "ok",
            **result,
            "database_total": count_candles(
                symbol,
                interval,
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        )

    finally:
        client.close()


@router.get("/api/analysis/market")
def analysis_market(
    interval: str = "4h",
    limit: int = 1000,
):
    symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
    ]

    return {
        "status": "ok",
        "interval": interval,
        "results": analyze_market(
            symbols=symbols,
            interval=interval,
            limit=limit,
        ),
    }
@router.get("/api/analysis/{symbol}")
def analysis_symbol(
    symbol: str,
    interval: str = "4h",
    limit: int = 1000,
):
    return analyze_symbol(
        symbol=symbol.upper(),
        interval=interval,
        limit=limit,
    )

@router.get("/api/market/price/{symbol}")
def market_price(symbol: str):
    symbol = symbol.upper()

    if symbol not in DEFAULT_SYMBOLS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported symbol",
        )

    client = BinanceClient()

    try:
        ticker = client.get_ticker_price(symbol)

        return {
            "status": "ok",
            "symbol": symbol,
            "price": float(ticker["price"]),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        )

    finally:
        client.close()
