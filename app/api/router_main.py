from fastapi import APIRouter

from .routes_generate import router as generate_router
from .routes_models import router as models_router
from .routes_benchmark import router as benchmark_router
from .routes_analytics import router as analytics_router
from .routes_health import router as health_router

api_router = APIRouter()
api_router.include_router(generate_router)
api_router.include_router(models_router)
api_router.include_router(benchmark_router)
api_router.include_router(analytics_router)
api_router.include_router(health_router)
