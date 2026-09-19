from fastapi import APIRouter, Depends, HTTPException

from ..database import Session, get_db
from ..schemas import BaselineRequest, EstimateRequestLegacy, GenerateRequest
from ..services.baseline import run_baseline
from ..services.orchestrator import compute_delta, run_generate

router = APIRouter(prefix="/api", tags=["generate"])


@router.post("/generate")
def generate(req: GenerateRequest, db: Session = Depends(get_db)):
    try:
        out = run_generate(req, db=db)
        out["comparison"]["delta"] = compute_delta(out["result"]["structured_metrics"], out["comparison"]["baseline"])
        return out
    except Exception as e:  # pragma: no cover
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"generate failed: {e}") from e


@router.post("/generate/baseline")
def generate_baseline(req: BaselineRequest, db: Session = Depends(get_db)):
    try:
        return run_baseline(req, db=db)
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"baseline failed: {e}") from e


@router.post("/estimate")
def legacy_estimate(req: EstimateRequestLegacy, db: Session = Depends(get_db)):
    """Backwards-compatible single carbon estimate (same shape as v0.1)."""
    from ..core.carbon import energy_for_tokens_constants, estimate_carbon
    from ..tokenizer import count_tokens

    model = req.model_name
    tokens_in = count_tokens(req.prompt)
    tokens_out = req.max_tokens
    profile = energy_for_tokens_constants(model)
    latency = tokens_in / profile.prefill_tps * 1000 + tokens_out / profile.decode_tps * 1000
    est = estimate_carbon(profile, tokens_in, tokens_out, latency_ms=latency, region=req.region, layers_ratio=1.0)
    rec = {
        "prompt": req.prompt,
        "model_name": model,
        "provider": "modeled",
        "tokens_input": tokens_in,
        "tokens_output": tokens_out,
        "total_tokens": tokens_in + tokens_out,
        "inference_time_ms": int(latency),
        "energy_consumed_kwh": est.energy_kwh,
        "carbon_emitted_kgco2": est.carbon_kg,
        "estimation_method": est.method,
        "grid_region": est.grid_region,
        "grid_intensity": est.grid_intensity,
    }
    from ..database import RequestRecord
    db.add(RequestRecord.from_dict({
        "kind": "estimate", "prompt": req.prompt, "model_name": model,
        "tokens_in": tokens_in, "tokens_out": tokens_out, "latency_ms": latency,
        "energy_kwh": est.energy_kwh, "carbon_kg": est.carbon_kg,
        "grid_region": est.grid_region, "grid_intensity": est.grid_intensity,
    }))
    db.commit()
    return rec