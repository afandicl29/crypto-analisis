from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.web.routes import router

app = FastAPI(
    title="Crypto Analyzer",
    version="0.1.0",
)

app.mount(
    "/static",
    StaticFiles(directory="app/web/static"),
    name="static",
)

app.include_router(router)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "crypto-analyzer",
    }
