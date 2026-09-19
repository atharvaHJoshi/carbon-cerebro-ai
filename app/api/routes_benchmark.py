from fastapi import APIRouter, Body, Depends

from ..database import Session, get_db
from ..schemas import BenchmarkRequest
from ..services.benchmark import aggregate, benchmark_scenarios

router = APIRouter(prefix="/api", tags=["benchmark"])


@router.post("/benchmark/run")
def run_benchmark(req: BenchmarkRequest, db: Session = Depends(get_db)):
    out = benchmark_scenarios(
        req.scenarios,
        baseline_model=req.baseline_model,
        default_model=req.default_model,
        name=req.name,
        region=req.region,
        save_records=req.save_records,
        db=db,
    )
    return out


@router.post("/benchmark/aggregate")
def aggregate_only(results: list[dict] = Body(...)):
    return aggregate(results)