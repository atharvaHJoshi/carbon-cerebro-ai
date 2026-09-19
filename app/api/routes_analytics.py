from fastapi import APIRouter, Depends, HTTPException, Query

from ..database import Session, get_db
from ..services import analytics

router = APIRouter(prefix="/api", tags=["analytics"])


@router.get("/analytics/summary")
def get_summary(db: Session = Depends(get_db)):
    return analytics.summary(db)


@router.get("/analytics/timeseries")
def get_timeseries(points: int = Query(20, ge=2, le=200), db: Session = Depends(get_db)):
    return {"timeseries": analytics.timeseries(db, points=points)}


@router.get("/analytics/benchmarks")
def get_benchmarks(limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)):
    return {"benchmarks": analytics.benchmark_runs(db, limit=limit)}


@router.get("/analytics/requests")
def get_requests(limit: int = Query(50, ge=1, le=500), kind: str | None = None, db: Session = Depends(get_db)):
    rows = analytics.recent_requests(db, limit=limit)
    if kind:
        rows = [r for r in rows if r["kind"] == kind]
    return {"requests": rows}


@router.get("/analytics/requests/{request_id}")
def get_request(request_id: int, db: Session = Depends(get_db)):
    from ..database import RequestRecord
    row = db.query(RequestRecord).filter(RequestRecord.id == request_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="request not found")
    return row.to_dict()