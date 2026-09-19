from fastapi import APIRouter

from ..config import settings
from ..core.carbon import energy_for_tokens_constants
from ..core.lora import ADAPTER_CATALOG
from ..core.router import model_catalog_overview

router = APIRouter(prefix="/api", tags=["catalog"])


def _profile(id_):
    p = energy_for_tokens_constants(id_).to_dict()
    p["cost_per_token_usd"] = {
        "tinygpt": 0.0, "flan-t5-base": 1e-7, "gemma-2b": 3e-7, "llama-3-8b": 8e-7,
        "mistral-7b": 7e-7, "qwen-7b": 8e-7, "gpt-large-70b": 5e-6, "gemini-pro": 3e-6,
        "claude-3.5": 3e-6,
    }.get(id_, 8e-7)
    p["quality_capability"] = {
        "tinygpt": 0.35, "flan-t5-base": 0.55, "gemma-2b": 0.78, "llama-3-8b": 0.9,
        "mistral-7b": 0.88, "qwen-7b": 0.9, "gpt-large-70b": 0.98, "gemini-pro": 0.97,
        "claude-3.5": 0.98,
    }.get(id_, 0.9)
    return p


@router.get("/models")
def list_models():
    ids = ["tinygpt", "flan-t5-base", "gemma-2b", "llama-3-8b", "mistral-7b", "qwen-7b",
           "gpt-large-70b", "gemini-pro", "claude-3.5"]
    models = []
    for mid in ids:
        prof = _profile(mid)
        models.append({
            "id": mid,
            "params": prof["params"],
            "backend": "real-from-scratch" if mid == "tinygpt" else ("remote" if mid in ("gemini-pro", "claude-3.5") else "simulated"),
            "energy_per_token_in_kwh": prof["ept_in"],
            "energy_per_token_out_kwh": prof["ept_out"],
            "device_power_watts": prof["device_power_watts"],
            "cost_per_token_usd": prof["cost_per_token_usd"],
            "quality_capability": prof["quality_capability"],
            "prefill_tps": prof["prefill_tps"],
            "decode_tps": prof["decode_tps"],
        })
    return {"models": models}


@router.get("/adapters")
def list_adapters():
    return {"adapters": [{"id": a["id"], "name": a["name"], "domain": a["domain"]} for a in ADAPTER_CATALOG]}


@router.get("/regions")
def list_regions():
    return {"regions": [{"code": k, "grid_intensity_kgco2_per_kwh": round(v, 4)} for k, v in settings.GRID_INTENSITY_TABLE.items()],
            "default_region": settings.DEFAULT_GRID_REGION}


@router.get("/catalog")
def catalog():
    return {"overview": model_catalog_overview()}