"""Layer-wise Computation Optimization (RQ4).

Plans how much of a model's transformer depth is needed for a request.

* For the real numpy `tinygpt` backend we *actually* execute layer-by-layer and
  break out early using token confidence (entropy of the output distribution).
* For other/abstract models we emit a *plan* (layers to run, expected quality
  delta, time & energy savings) computed from a calibrated fidelity curve that
  depends on prompt complexity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import settings
from .analysis import PromptIntelligence


@dataclass
class LayerPlan:
    total_layers: int
    layers_to_run: int
    layers_saved: int
    strategy: str
    expected_quality_factor: float      # fraction of full-depth quality retained
    expected_time_saved: float          # fraction of layer time saved
    confidence_per_layer: list[float] | None = field(default=None)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_layers": self.total_layers,
            "layers_to_run": self.layers_to_run,
            "layers_saved": self.layers_saved,
            "saved_ratio": round(self.layers_saved / max(1, self.total_layers), 4),
            "strategy": self.strategy,
            "expected_quality_factor": round(self.expected_quality_factor, 4),
            "expected_time_saved": round(self.expected_time_saved, 4),
            "confidence_per_layer": [round(c, 4) for c in (self.confidence_per_layer or [])],
            "reasons": self.reasons,
        }


def _fidelity_curve(complexity: float, k: int, total: int) -> float:
    """Expected quality fraction after running k of total layers.

    Easy prompts saturate quickly; hard prompts need nearly full depth.
    """
    x = k / max(1, total)
    steep = 3.0 + 6.0 * complexity
    mid = 0.35 + 0.45 * complexity
    q = 1.0 / (1.0 + np.exp(-steep * (x - mid)))
    # force monotonic-ish reasonable values
    return float(np.clip(q, 0.05, 1.0))


def plan_layers(pi: PromptIntelligence, total_layers: int, strategy: str = "auto",
                confidence_threshold: float | None = None) -> LayerPlan:
    """Produce a layer execution plan.

    strategy: "full" runs everything, "auto" exits early when expected
    confidence clears the threshold, "minimal" runs the floor.
    """
    if total_layers <= 1:
        return LayerPlan(total_layers, total_layers, 0, "full", 1.0, 0.0, reasons=["Trivial depth; no early exit"])

    threshold = settings.EARLY_EXIT_CONFIDENCE if confidence_threshold is None else confidence_threshold
    complexity = pi.complexity

    reasons: list[str] = []
    confidences: list[float] = []
    exit_layer = total_layers

    if strategy == "full":
        layers = total_layers
        qf = 1.0
    else:
        # emulate layer-wise confidence growth
        for k in range(1, total_layers + 1):
            c = _fidelity_curve(complexity, k, total_layers)
            confidences.append(round(c, 4))
            if c >= threshold and exit_layer == total_layers:
                exit_layer = k
        if strategy == "minimal":
            floor = max(1, int(total_layers * settings.MIN_LAYERS_RATIO))
            exit_layer = min(exit_layer, floor)
            reasons.append(f"minimal strategy floors execution at {floor} layers")
        layers = max(1, min(exit_layer, total_layers))
        qf = _fidelity_curve(complexity, layers, total_layers)

    layers_saved = total_layers - layers
    reasons.append(
        f"complexity={complexity:.2f} -> confidence {threshold:.2f} reached at layer {layers}/{total_layers}"
        if layers < total_layers
        else f"complexity={complexity:.2f} required full depth ({total_layers} layers)"
    )
    if layers < total_layers:
        reasons.append(f"estimated quality retention {qf:.2f} at exit layer {layers}")

    return LayerPlan(
        total_layers=total_layers,
        layers_to_run=layers,
        layers_saved=layers_saved,
        strategy=strategy,
        expected_quality_factor=round(qf, 4),
        expected_time_saved=round(layers_saved / total_layers, 4),
        confidence_per_layer=confidences or None,
        reasons=reasons,
    )


def note_measured_exit(plan: LayerPlan, per_token_exit_layers: list[int], quality_factor: float) -> LayerPlan:
    """Update a plan with measured values from a real layer-by-layer run."""
    plan.confidence_per_layer = [float(q) for q in per_token_exit_layers] if per_token_exit_layers else plan.confidence_per_layer
    if per_token_exit_layers:
        avg = float(np.mean(per_token_exit_layers))
        plan.layers_to_run = int(round(avg))
        plan.layers_saved = max(0, plan.total_layers - plan.layers_to_run)
        plan.reasons.append(f"(live) mean exit layer {avg:.1f}/{plan.total_layers} across tokens")
    plan.expected_quality_factor = min(plan.expected_quality_factor, quality_factor)
    plan.reasons.append(f"(live) measured quality factor {quality_factor:.3f}")
    return plan