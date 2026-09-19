"""Deterministic simulator backend.

Produces a stable, keyword-aware response for every (prompt, model, config)
tuple and reports realistic measured metadata: input/output token counts, wall
latency derived from a per-model throughput model, depth used and energy.

Determinism matters: the benchmark measures *the same* prompt under baseline and
optimized pipelines, so differences come from the pipeline itself, not sampling.
"""

from __future__ import annotations

import re
from hashlib import sha256

import numpy as np

from ..core.analysis import analyze_prompt
from ..core.carbon import energy_for_tokens_constants
from ..core.embeddings import cached_embed, cosine
from ..tokenizer import count_tokens
from .base import InferenceBackend, InferenceResult

_RESPONSE_FRAGMENTS = {
    "software": [
        "A robust approach is to decompose the task into clear functions, validate inputs, and test edge cases.",
        "You can implement this with a well-tested module; keep functions pure and handle errors explicitly.",
        "Use a data structure with predictable complexity for this operation.",
    ],
    "finance": [
        "The financial position depends on liquidity, cash flow and risk exposure; monitor all three.",
        "A prudent strategy balances returns with risk tolerance and a clear budget.",
        "Key metrics include revenue, margin, and capital allocation over the period.",
    ],
    "medical": [
        "Clinical guidance: consider symptoms, history and contraindications before choosing a course.",
        "A careful review of presenting signs and relevant guidelines supports the recommendation.",
        "Safety first: verify dosage, interactions and monitoring before prescribing.",
    ],
    "legal": [
        "Review the governing clause and applicable jurisdiction before reaching a conclusion.",
        "The obligation is conditioned on the terms and any compliance requirements.",
        "Documented terms and prior practice should guide the interpretation.",
    ],
    "math": [
        "Set up the problem, apply the appropriate identity and verify by substitution.",
        "The result follows by substituting values and simplifying step by step.",
        "Derive the expression, then check consistency at boundary cases.",
    ],
    "science": [
        "Evidence suggests the effect is real; control variables and replicate for confidence.",
        "The mechanism depends on energy and matter flows in the system under study.",
        "Measured quantities support the hypothesis within stated uncertainty.",
    ],
    "general": [
        "The answer depends on a few key factors; consider the context and constraints carefully.",
        "Here is a clear way to think about it, with the main trade-offs highlighted.",
        "Reasonable conclusions follow once the essential facts are separated from noise.",
    ],
}

_OPENERS = {
    "summarization": "In brief, ",
    "classification": "Based on the evidence, the most likely category is ",
    "question_answering": "Here is the answer: ",
    "code_generation": "Here is a clean implementation: ",
    "generation": "Here is the draft: ",
    "translation": "The translated text reads: ",
    "extraction": "The extracted items are: ",
    "reasoning": "Reasoning through it: ",
    "editing": "Revised version: ",
    "planning": "A workable plan: ",
    "recommendation": "I recommend: ",
    "prediction": "Projection: ",
    "math_problem": "The computed result: ",
}


class SimulatorBackend(InferenceBackend):
    def run(self, prompt: str, model_id: str, *, max_tokens: int = 128,
            early_exit: bool = True, temperature: float = 0.7,
            quality_factor: float = 1.0, depth_used: float = 1.0) -> InferenceResult:
        pi = analyze_prompt(prompt)
        profile = energy_for_tokens_constants(model_id)
        tokens_in = count_tokens(prompt)

        seed = int(sha256(f"{prompt}|{model_id}|{depth_used:.2f}".encode()).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)

        domain = pi.top_domain
        frag = _RESPONSE_FRAGMENTS.get(domain, _RESPONSE_FRAGMENTS["general"])[seed % len(_RESPONSE_FRAGMENTS.get(domain, _RESPONSE_FRAGMENTS["general"]))]
        opener = _OPENERS.get(pi.intent, "")
        ext = f" ({model_id} output)" if model_id != "tinygpt" else ""
        body = opener + frag
        # length scaled by complexity and task
        n_tokens = min(max_tokens, max(12, int(30 + 40 * pi.complexity)))
        tokens_out = n_tokens

        # measure wall latency from throughput model
        prefill_ms = tokens_in / profile.prefill_tps * 1000.0
        decode_ms = tokens_out / profile.decode_tps * 1000.0 * (1.0 if early_exit else 1.0)
        # early exit shortens decode path proportionally to depth saved
        decode_ms *= (0.5 + 0.5 * depth_used)
        latency_ms = prefill_ms + decode_ms

        # anchored token-per-degree energy is computed by the carbon engine in
        # the orchestrator; here we just carry deterministic metadata.
        residual = body
        # enforce output token budget roughly: pad/truncate on whitespace
        while count_tokens(residual) < n_tokens and len(residual) < 2000:
            residual += f" {body[:120]}"[: 120]
        residual = residual[: max(200, int(len(residual) * (0.6 + 0.4 * depth_used)))]
        tokens_out = count_tokens(residual)
        latency_ms = prefill_ms + count_tokens(residual) / profile.decode_tps * 1000.0 * (0.5 + 0.5 * depth_used)

        return InferenceResult(
            text=residual.strip() + ext,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            depth_used=depth_used,
            measured={"simulated": True, "domain": domain, "intent": pi.intent},
            metrics={"quality_factor": quality_factor},
        )

    def _probe(self) -> None:
        self.run("probe", "flan-t5-base", max_tokens=4)


simulator_backend = SimulatorBackend()