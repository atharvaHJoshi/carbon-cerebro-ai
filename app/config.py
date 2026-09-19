import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), env_file_encoding="utf-8", extra="ignore")

    # ------------------------------------------------------------------ app
    APP_NAME: str = "Green Model Advisor"
    APP_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ------------------------------------------------------------------ db
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'data' / 'green_model_advisor.db'}"

    # ------------------------------------------------------------------ grid
    # Regional grid carbon intensity (kg CO2 / kWh). Used by the Carbon Engine.
    DEFAULT_GRID_REGION: str = "IND"
    DEFAULT_GRID_INTENSITY: float = 0.708
    GRID_INTENSITY_TABLE: dict[str, float] = {
        "IND": 0.708,
        "USA": 0.382,
        "CHN": 0.582,
        "GBR": 0.212,
        "DEU": 0.336,
        "FRA": 0.052,
        "BRA": 0.098,
        "CAN": 0.120,
        "JPN": 0.461,
        "AUS": 0.581,
        "GLOBAL": 0.475,
    }

    # ------------------------------------------------------------------ pipeline tuning
    MAX_PRUNE_RATIO: float = 0.45          # max fraction of input tokens the pruner may drop
    MIN_PRUNE_KEEP: float = 0.55           # never drop below this keep fraction for important content
    DEFAULT_CONTEXT_BUDGET: float = 0.5    # default context retention budget (fraction of tokens)
    TOP_K_CONTEXT: int = 12                 # default max messages kept in compressed context
    CONTEXT_REDUNDANCY_LAMBDA: float = 0.6  # MMR redundancy penalty
    EARLY_EXIT_CONFIDENCE: float = 0.72     # target token confidence for early exit
    MIN_LAYERS_RATIO: float = 0.25          # floor for how little of a network may be used

    # ------------------------------------------------------------------ routing
    DEFAULT_WEIGHTS: dict[str, float] = {
        "quality": 0.35,
        "carbon": 0.25,
        "energy": 0.10,
        "cost": 0.10,
        "latency": 0.10,
        "resource": 0.10,
    }
    MIN_REQUIRED_QUALITY: float = 0.65
    LORA_MIN_CONFIDENCE: float = 0.45       # below this, fall back to base model (no adapter)
    LORA_BASE_REWARD: float = 0.05          # reward "base" in scoring so fallback is a fair candidate

    # ------------------------------------------------------------------ energy model
    # kWh per prefill/input token and per decode/output token, per model class.
    # These are calibrated literature-informed estimates (see docs/ARCHITECTURE.md).
    JOULE_TO_KWH: float = 1.0 / 3.6e6
    KJ_PER_1K_TOKENS_SMALL: float = 1.5
    KJ_PER_1K_TOKENS_7B: float = 7.0
    KJ_PER_1K_TOKENS_70B: float = 60.0

    # ------------------------------------------------------------------ tinygpt
    TINYGPT_WEIGHTS: Path = BASE_DIR / "data" / "models" / "tinygpt.npz"
    TINYGPT_TRAIN_STEPS: int = 600
    TINYGPT_SEED: int = 1337

    # ------------------------------------------------------------------ misc
    SIMULATOR_LATENCY_SECONDS_PER_TOKEN: float = 0.004   # per output token (decode)
    SIMULATOR_PREFILL_TPS: float = 280.0                 # prefill tokens per second
    REQUEST_TIMEOUT: int = 60


settings = Settings()

# Realms of the object store that maps HumanHash names -> model names.
# (legacy convenience for the /estimate endpoint)
ALIASES = {
    "aliases": ["estimate", "models"],
}