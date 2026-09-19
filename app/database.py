from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .config import settings

URL = settings.DATABASE_URL
IS_SQLITE = URL.startswith("sqlite")

engine = create_engine(URL, connect_args={"check_same_thread": False} if IS_SQLITE else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class JSONText(Text):
    def __init__(self):
        super().__init__()


def _dump_json(obj) -> str:
    return json.dumps(obj, default=str)


class RequestRecord(Base):
    """One executed request through either the GMA pipeline or a baseline run."""

    __tablename__ = "requests"

    id = Column(Integer, primary_key=True, index=True)
    kind = Column(String(24), default="gma")          # gma | baseline | estimate
    session_id = Column(String(64), nullable=True)
    prompt = Column(Text, default="")
    model_name = Column(String(64), default="")
    adapter_name = Column(String(64), nullable=True)
    layers_run = Column(Integer, nullable=True)
    layers_total = Column(Integer, nullable=True)
    prune_ratio = Column(Float, default=0.0)
    context_ratio = Column(Float, default=1.0)
    tokens_in = Column(Integer, default=0)
    tokens_out = Column(Integer, default=0)
    tokens_original = Column(Integer, default=0)
    latency_ms = Column(Float, default=0.0)
    energy_kwh = Column(Float, default=0.0)
    carbon_kg = Column(Float, default=0.0)
    cost_usd = Column(Float, default=0.0)
    quality_score = Column(Float, default=0.0)
    grid_region = Column(String(8), default="IND")
    grid_intensity = Column(Float, default=0.708)
    response = Column(Text, default="")
    explanation = Column(JSONText, default="{}")
    metrics = Column(JSONText, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)

    @classmethod
    def from_dict(cls, d: dict) -> "RequestRecord":
        return cls(
            kind=d.get("kind", "gma"),
            session_id=d.get("session_id"),
            prompt=d.get("prompt", ""),
            model_name=d.get("model_name", ""),
            adapter_name=d.get("adapter_name"),
            layers_run=d.get("layers_run"),
            layers_total=d.get("layers_total"),
            prune_ratio=d.get("prune_ratio", 0.0),
            context_ratio=d.get("context_ratio", 1.0),
            tokens_in=d.get("tokens_in", 0),
            tokens_out=d.get("tokens_out", 0),
            tokens_original=d.get("tokens_original", 0),
            latency_ms=d.get("latency_ms", 0.0),
            energy_kwh=d.get("energy_kwh", 0.0),
            carbon_kg=d.get("carbon_kg", 0.0),
            cost_usd=d.get("cost_usd", 0.0),
            quality_score=d.get("quality_score", 0.0),
            grid_region=d.get("grid_region", "IND"),
            grid_intensity=d.get("grid_intensity", 0.708),
            response=d.get("response", ""),
            explanation=_dump_json(d.get("explanation", {})),
            metrics=_dump_json(d.get("metrics", {})),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "session_id": self.session_id,
            "prompt": self.prompt,
            "model_name": self.model_name,
            "adapter_name": self.adapter_name,
            "layers_run": self.layers_run,
            "layers_total": self.layers_total,
            "prune_ratio": self.prune_ratio,
            "context_ratio": self.context_ratio,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_original": self.tokens_original,
            "latency_ms": self.latency_ms,
            "energy_kwh": self.energy_kwh,
            "carbon_kg": self.carbon_kg,
            "cost_usd": self.cost_usd,
            "quality_score": self.quality_score,
            "grid_region": self.grid_region,
            "grid_intensity": self.grid_intensity,
            "response": self.response,
            "explanation": json.loads(self.explanation or "{}"),
            "metrics": json.loads(self.metrics or "{}"),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class BenchmarkRun(Base):
    __tablename__ = "benchmarks"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(128), default="benchmark")
    scenarios = Column(Integer, default=0)
    results = Column(JSONText, default="{}")       # full per-request comparison
    summary = Column(JSONText, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "scenarios": self.scenarios,
            "results": json.loads(self.results or "{}"),
            "summary": json.loads(self.summary or "{}"),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()