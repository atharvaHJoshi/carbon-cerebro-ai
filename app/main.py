from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.router_main import api_router
from .config import settings
from .database import create_tables

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Sustainable AI infrastructure middleware: dynamic token pruning, intelligent context "
        "compression, adaptive LoRA selection, layer-wise computation optimization, carbon-aware "
        "multi-objective routing and sustainability analytics for LLM inference."
    ),
    version=settings.APP_VERSION,
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    create_tables()


app.include_router(api_router)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DATA_DIR.mkdir(exist_ok=True)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(STATIC_DIR), html=True), name="dashboard")


@app.get("/")
def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "tagline": "Make LLMs do the same useful work with less computation.",
        "endpoints": {
            "generate": "POST /api/generate",
            "generate_baseline": "POST /api/generate/baseline",
            "estimate_carbon": "POST /api/estimate",
            "models": "GET /api/models",
            "adapters": "GET /api/adapters",
            "regions": "GET /api/regions",
            "benchmark": "POST /api/benchmark/run",
            "analytics_summary": "GET /api/analytics/summary",
            "analytics_timeseries": "GET /api/analytics/timeseries",
            "analytics_requests": "GET /api/analytics/requests",
            "health": "GET /health",
            "dashboard": "/dashboard",
            "docs": "/docs",
        },
        "objective_weights_default": settings.DEFAULT_WEIGHTS,
    }