# Green Model Advisor — Complete Codebase Walkthrough (Interview Guide)

> This document explains the **entire** codebase of the Green Model Advisor project in a
> structured, interview-ready way. After studying this you should be able to:
> - Explain the project in one sentence and in five minutes
> - Trace one HTTP request through every layer of the system
> - Explain why each module exists and how it maps to a research objective
> - Answer deep technical questions about the design decisions and trade-offs
> - Explain what "accuracy" means here and how it is measured

---

## Table of Contents

1. [Elevator Pitch](#1-elevator-pitch)
2. [The Problem & The Research Questions](#2-the-problem--the-research-questions)
3. [System Architecture (Big Picture)](#3-system-architecture-big-picture)
4. [Project Structure](#4-project-structure)
5. [Tech Stack & Package Layout](#5-tech-stack--package-layout)
6. [Data Flow: Tracing One Request End-to-End](#6-data-flow-tracing-one-request-end-to-end)
7. [Module-by-Module Deep Dive](#7-module-by-module-deep-dive)
    - [7.1 config.py — Single Source of Truth](#71-configpy--single-source-of-truth)
    - [7.2 main.py — Application Bootstrap](#72-mainpy--application-bootstrap)
    - [7.3 schemas.py — API Contracts](#73-schemaspy--api-contracts)
    - [7.4 database.py — Persistence Layer](#74-databasepy--persistence-layer)
    - [7.5 tokenizer.py — Token Counting](#75-tokenizerpy--token-counting)
    - [7.6 core/embeddings.py — Offline Semantic Embeddings](#76-coreembeddingspy--offline-semantic-embeddings)
    - [7.7 core/analysis.py — Prompt Intelligence (Objective: Analyze)](#77-coreanalysispy--prompt-intelligence)
    - [7.8 core/pruning.py — Dynamic Token Pruning (RQ1)](#78-corepruningpy--dynamic-token-pruning-rq1)
    - [7.9 core/context.py — Intelligent Context Compression (RQ2)](#79-corecontextpy--intelligent-context-compression-rq2)
    - [7.10 core/lora.py — Adaptive LoRA Selection (RQ3)](#710-corelorapy--adaptive-lora-selection-rq3)
    - [7.11 core/early_exit.py — Layer-wise Optimization (RQ4)](#711-coreearly_exitpy--layer-wise-optimization-rq4)
    - [7.12 core/carbon.py — Carbon Prediction Engine](#712-corecarbonpy--carbon-prediction-engine)
    - [7.13 core/router.py — Multi-Objective Router + Explainability](#713-corerouterpy--multi-objective-router--explainability)
    - [7.14 core/evaluation.py — Response Quality Metrics](#714-coreevaluationpy--response-quality-metrics)
    - [7.15 backends/base.py — Backend Abstraction](#715-backendsbasepy--backend-abstraction)
    - [7.16 backends/simulator.py — Deterministic Simulator](#716-backendssimulatorpy--deterministic-simulator)
    - [7.17 backends/tinygpt.py — Real From-Scratch Transformer](#717-backendstinygptpy--real-from-scratch-transformer)
    - [7.18 backends/tinygpt_backend.py — tinygpt Adapter](#718-backendstinygpt_backendpy--tinygpt-adapter)
    - [7.19 backends/remote.py — Live Remote Backend](#719-backendsremotepy--live-remote-backend)
    - [7.20 services/orchestrator.py — The Heart of the Pipeline](#720-servicesorchestratorpy--the-heart-of-the-pipeline)
    - [7.21 services/baseline.py — The "Do Nothing" Counterfactual](#721-servicesbaselinepy--the-do-nothing-counterfactual)
    - [7.22 services/benchmark.py — The Benchmark Harness](#722-servicesbenchmarkpy--the-benchmark-harness)
    - [7.23 services/analytics.py — Sustainability Analytics](#723-servicesanalyticspy--sustainability-analytics)
    - [7.24 api/ — REST Layer](#724-apiroutes--rest-layer)
8. [The Four Research Questions — Implementation Summary](#8-the-four-research-questions--implementation-summary)
9. [How "Greener" Is Measured: Evaluation & Benchmarking](#9-how-greener-is-measured-evaluation--benchmarking)
10. [Design Decisions & Trade-offs](#10-design-decisions--trade-offs)
11. [Bugs Found & Fixed (what to say if asked "what did you debug")](#11-bugs-found--fixed)
12. [Testing Strategy](#12-testing-strategy)
13. [How To Run](#13-how-to-run)
14. [Common Interview Questions & Talking Points](#14-common-interview-questions--talking-points)

---

## 1. Elevator Pitch

**One sentence:**

> Green Model Advisor is an intelligent middleware platform that sits between an AI
> application and its LLM backends and decides, before every inference, **what is the
> minimum computation required to produce an acceptable high-quality response** — using
> token pruning, context compression, adaptive LoRA selection, early-exit layer planning,
> and carbon-aware multi-objective routing.

**One paragraph (good for "tell me about your project"):**

> Most LLM applications send every request, with its full history, to the same large
> model — so simple questions burn the same compute as hard reasoning problems. Green
> Model Advisor treats *efficiency* as a routing problem in front of inference. It
> analyzes the prompt to extract intent, domain and complexity; prunes low-value tokens;
> compresses long conversation history; picks a domain-specific LoRA adapter; decides how
> many transformer layers are actually needed; and then scores every candidate model
> configuration across quality, carbon, energy, cost, latency and resource use. The final
> decision is fully explainable, every run is measured against a "do-nothing" baseline,
> and the results are stored for sustainability analytics. Nothing is claimed to be green
> — everything is reported as a measured reduction versus the baseline.

**Why it's a good BE/engineering project to talk about:**
- It is **system-level engineering**, not "I called an API".
- It has a **real measurable outcome** (token / carbon / energy reduction %).
- It shows **systems thinking**: abstraction layers, dependency injection, deterministic
  testing, pluggable backends, persistence, analytics.
- It includes a **from-scratch transformer** (tinygpt, numpy-only) — great for talking
  about how you actually understand model internals, not just APIs.

---

## 2. The Problem & The Research Questions

**Core research question:**

> Can an intelligent inference optimization layer reduce the computational and
> environmental cost of LLM applications while preserving an acceptable level of response
> quality?

**Four technical research questions** (each maps to one reusable module):

| # | Research Question | Module |
|---|-------------------|--------|
| RQ1 | Can semantically low-value input tokens be removed while maintaining task performance? | `core/pruning.py` |
| RQ2 | Can long conversational contexts be compressed while preserving needed information? | `core/context.py` |
| RQ3 | Can routing automatically select the right domain-specific LoRA adapter? | `core/lora.py` |
| RQ4 | Can adaptive computation reduce unnecessary transformer-layer execution? | `core/early_exit.py` + `backends/tinygpt.py` |

**Supporting objectives (the glue):**
- Prompt intelligence / request analysis (`core/analysis.py`)
- Carbon-aware estimation (`core/carbon.py`)
- Multi-objective model selection (`core/router.py`)
- Explainability (`core/router.py::build_explanation`)
- Impact measurement (`services/benchmark.py`, `services/analytics.py`)
- Feedback/learning loop (persisted records → `services/analytics.py`)

---

## 3. System Architecture (Big Picture)

```
        User / AI Application
                 │
                 ▼
        ┌────────────────────────────────────┐
        │            FastAPI  (api/)          │
        │                                    │
        │  POST /api/generate                 │
        │  POST /api/generate/baseline        │
        │  GET  /api/models · /adapters       │
        │  GET  /api/analytics/...            │
        │  POST /api/benchmark/run            │
        └───────────────┬────────────────────┘
                        ▼
        ┌────────────────────────────────────┐
        │  services/orchestrator.run_generate │  ← the pipeline
        │  1. Prompt Intelligence (analysis)  │
        │  2. Token Pruning          (RQ1)    │
        │  3. Context Compression    (RQ2)    │
        │  4. Adaptive LoRA          (RQ3)    │
        │  5. Layer Plan             (RQ4)    │
        │  6. Multi-objective Router          │
        │  7. Execute via a Backend           │
        │  8. Energy/Carbon/Cost estimation   │
        │  9. Quality evaluation              │
        │ 10. Persist record + delta          │
        └───────────────┬────────────────────┘
                        │
        ┌───────────────┼────────────────────┐
        ▼               ▼                    ▼
  backends/simulator  backends/tinygpt   backends/remote
  (abstract models)  (real numpy GPT)   (vLLM/OpenAI, optional)
        │               │                    │
        └───────────────┴────────────────────┘
                        ▼
              Persisted to SQLite (database.py)
                        ▼
            services/analytics  →  /api/analytics/*
```

**Key architectural principles you should be able to articulate:**
1. **Backend abstraction** — the pipeline never cares whether it talks to a simulator,
   a real from-scratch transformer, or a hosted vLLM. Swappable behind a single interface.
2. **Determinism for science** — the simulator and the embedding model are deterministic,
   so a benchmark measures *pipeline* differences, not sampling noise.
3. **Explainability as a first-class output** — every decision returns structured "why".
4. **Measure, don't assert** — the "greener" claim is always a measured delta vs baseline.
5. **Offline-first** — no network/model download needed for the core path;
   remote backend is optional and gracefully degrades.

---

## 4. Project Structure

```
green-model-advisor/
├── README.md
├── requirements.txt          # pinned dependencies
├── install.sh                # Arch/pacman + venv setup script
├── run.py                    # uvicorn entry point
├── carbon_estimates.db        # legacy sqlite file
├── data/
│   ├── green_model_advisor.db  # main sqlite database (created at runtime)
│   └── models/tinygpt.npz      # trained tinygpt weights (numpy archive)
├── venv/                     # python virtual environment
├── tests/                    # 100+ pytest tests
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI app bootstrap + route mount
│   ├── config.py             # all tunable constants (Settings)
│   ├── database.py           # SQLAlchemy ORM models + session
│   ├── schemas.py            # Pydantic request models
│   ├── tokenizer.py          # token counting (tiktoken w/ fallback)
│   ├── api/                  # REST endpoints
│   │   ├── router_main.py      # aggregates sub-routers
│   │   ├── routes_generate.py
│   │   ├── routes_models.py
│   │   ├── routes_benchmark.py
│   │   ├── routes_analytics.py
│   │   └── routes_health.py
│   ├── backends/             # inference backends
│   │   ├── base.py             # InferenceBackend ABC + result dataclass
│   │   ├── simulator.py
│   │   ├── tinygpt.py          # real from-scratch transformer (numpy)
│   │   ├── tinygpt_backend.py
│   │   └── remote.py
│   ├── core/                 # the optimization engines (research contribution)
│   │   ├── analysis.py         # prompt intelligence
│   │   ├── pruning.py          # RQ1
│   │   ├── context.py          # RQ2
│   │   ├── lora.py             # RQ3
│   │   ├── early_exit.py       # RQ4 planning
│   │   ├── embeddings.py       # offline hashed char n-gram vectors
│   │   ├── carbon.py           # carbon/energy estimation
│   │   ├── router.py           # multi-objective routing
│   │   └── evaluation.py       # quality scoring
│   ├── services/             # orchestration + analytics layers
│   │   ├── orchestrator.py
│   │   ├── baseline.py
│   │   ├── benchmark.py
│   │   └── analytics.py
│   └── data/corpus/          # training text for tinygpt
│       ├── samples.txt
│       └── green_ai.txt
```

---

## 5. Tech Stack & Package Layout

| Component | Technology | Why |
|-----------|-----------|-----|
| Web framework | FastAPI + Uvicorn | async-ready, typed request validation via Pydantic, auto OpenAPI docs |
| Data validation | Pydantic v2 (BaseModel + pydantic-settings) | request contracts + env-configurable settings |
| ORM / DB | SQLAlchemy 2.0 + SQLite | relational persistence, zero-config, file-based |
| Tokens | tiktoken (cl100k_base) with fallback | real tokenizer when available; calibrated fallback offline |
| Number crunching | numpy | model math, embeddings, vector ops |
| Carbon tracking | codecarbon | installed; the app has its own estimation engine on top |
| Testing | pytest + httpx TestClient | fast, covers API + core + orchestration + backends |
| HTTP | requests | optional remote backend |

> **Interview tip:** you can say — *"The core value is framework-agnostic; FastAPI and
> SQLAlchemy are just plumbing. The actual research lives in `app/core/` with pure Python
> + numpy, which makes every optimization unit-testable in isolation."*

---

## 6. Data Flow: Tracing One Request End-to-End

Walk through `POST /api/generate` with a prompt like *"Write a Python function using
binary search"*.

```
1. Client → FastAPI
   routes_generate.generate(req)                  # validates GenerateRequest
     │
2. services/orchestrator.run_generate(req, db)
     │
   ├─ (1) Prompt Intelligence  core/analysis.analyze_prompt(prompt)
   │        → intent="code_generation", top_domain="software",
   │          complexity=0.62, required_context=0.4, tokens=…, reasons=[…]
   │
   ├─ (2) Dynamic Token Pruning  core/pruning.prune_prompt(prompt)
   │        → drops low-importance chunks, keeps STOPWORDS penalized,
   │          protects snake_case identifiers ("binary_search"), ≤45% budget
   │
   ├─ (3) Context Compression  core/context.compress_context(history, query)
   │        (only if history exists)
   │        → embeds every message, MMR relevance ranking, budget trim
   │
   ├─ (4) Adaptive LoRA  core/lora.select_adapter(optimized_prompt)
   │        → domain lexicon + keyword + semantic match
   │        → if confidence ≥ 0.45: selected="lora-code" else fallback to base
   │
   ├─ (5) Layer Plan  core/early_exit.plan_layers(pi, 32, strategy)
   │        → fidelity curve on complexity: e.g. 22/32 layers needed
   │
   ├─ (6) Router  core/router.route(pi, adapter, layers_ratio, …)
   │        → builds N candidates (tinygpt … claude-3.5)
   │        → normalizes each dimension, weights, scores, sorts
   │        → selected model + full explanation dict
   │
   ├─ (7) Execute
   │        backend = _backend_for(selected_model)   # tinygpt/simulator/remote
   │        result  = backend.run(optimized_input, model, max_tokens, …)
   │
   ├─ (8) Measure
   │        estimate_carbon(profile, tokens_in, tokens_out, latency, layers_ratio)
   │        cost = tokens × per-token price
   │
   ├─ (9) Evaluate
   │        evaluate_response(response, reference|None)
   │        → quality score (semantic similarity / keyword coverage / length)
   │
   ├─(10) Persist (if save_record)
   │        RequestRecord → SQLite
   │
   └─ return bundle:
        { request_id, record_id,
          prompt_intelligence, optimizations, execution_plan, decision,
          result{response, structured_metrics, evaluation, backend_details},
          comparison{baseline, delta}, explanation }
```

The last step also computes a **counterfactual** — `baseline_counterfactual()` simulates
"same request, no optimization, fixed model, full depth". The **comparison.delta** block
is exactly what makes this project "measured green", not "claimed green".

---

## 7. Module-by-Module Deep Dive

### 7.1 `config.py` — Single Source of Truth

A `Settings(BaseSettings)` class loaded from `.env` (via `pydantic-settings`), instantiated
once as `settings`.

**What every group of constants controls:**

| Group | What it tunes | Example values |
|-------|---------------|----------------|
| Grid carbon intensity | kg CO₂ per kWh per country | IND 0.708, FRA 0.052, USA 0.382 |
| Pipeline tuning | pruning budget, context budget, early-exit confidence | max prune 45%, context budget 50%, exit confidence 0.72 |
| Routing | objective weights + hard constraints | quality 0.35, carbon 0.25, …; min quality 0.65 |
| Energy model | energy per token constants | small 1.5 kJ/1k, 7B 7 kJ/1k, 70B 60 kJ/1k |
| tinygpt | training hyper-parameters | 6 layers, 48 embd, 4 heads, 600 steps |
| Simulator | latency model | 0.004 s/decoded token, 280 prefill tps |

> **Interview tip:** always reference this file when tuning — *"All experimental
> parameters are centralised in config so the whole platform is reproducible from one
> settings object."*

---

### 7.2 `main.py` — Application Bootstrap

- Creates the `FastAPI` app with title/description/version.
- Global CORS middleware (all origins — dev-facing).
- `@app.on_event("startup")` → `create_tables()` (idempotent `create_all`).
- Mounts `/dashboard` if `app/static` exists (empty now — a hook for a dashboard UI).
- `GET /` returns a self-describing endpoint map + default objective weights.

`router_main.py` aggregates all sub-routers so `main.py` imports just one `api_router`.

---

### 7.3 `schemas.py` — API Contracts

Pydantic models that define the request shapes and enforce validation:

- **GenerateRequest** — the main one. Note the optimization toggles:
  `use_pruning`, `use_context_compression`, `use_early_exit`, `use_lora`, plus
  `strategy` (`auto | full | minimal`), optional `region`, `weights`, `constraints`,
  and `reference` (ground truth for quality evaluation).
- **BaselineRequest** — same prompt but explicitly un-optimized.
- **BenchmarkRequest** — a list of scenarios to run through both pipelines.
- **EstimateRequestLegacy** — backwards-compatible single carbon estimate.
- **RouteRequest** — for a dedicated routing endpoint.

> **Why toggle flags?** The flags let experiments answer *"which optimization actually
> contributes the savings?"* by turning one module off at a time.

---

### 7.4 `database.py` — Persistence Layer

SQLAlchemy ORM, two tables:

**`RequestRecord`** — one row per executed request:
`kind` (`gma | baseline | estimate`), `session_id`, prompt, model, adapter, `layers_run/total`
(layer-wise), `prune_ratio`, `context_ratio`, token counts (in/out/original), latency,
energy, carbon, cost, quality, grid region/intensity, response, plus JSON `explanation`
and `metrics`.

- `from_dict()` serializes nested dicts with `json.dumps(default=str)`.
- `to_dict()` deserializes them back — this round-trip is what analytics reads.

**`BenchmarkRun`** — stores a whole benchmark batch: name, scenario count, JSON `results`
and `summary`.

`create_tables()` → `Base.metadata.create_all` (idempotent). `get_db()` is the FastAPI
dependency that yields a session.

> **Interview tip:** the JSON columns store the *explainability* and *metrics* payloads
> because they have a varying schema; the fixed columns hold the numbers we aggregate on.

---

### 7.5 `tokenizer.py` — Token Counting

Two strategies, with graceful degradation:
- If `tiktoken` is installed → real cl100k-base encoding (what OpenAI-style tokenz count).
- Else → calibrated heuristic: `words × 1.25 + punct × 0.45 + 2`.

Also provides:
- `tokenize()` — actual token list for importance analysis.
- `split_word_chunks()` — whitespace-aware `(start, end, chunk)` spans used by the pruner.
- `join_chunks()` — re-joins kept chunks losslessly.

> **Why spans?** The pruner needs to know exactly *which byte ranges* each chunk covered
> so it can protect user-specified substrings and later report what it removed.

---

### 7.6 `core/embeddings.py` — Offline Semantic Embeddings

Hash-based **bag of character n-grams** (n = 2,3,4) into a 384-d vector:

1. Normalise (lowercase, strip non-alphanumerics).
2. Slide overlapping n-grams; sub-sample (~25%) to de-bias frequent grams (word2vec trick).
3. Hash each gram → index + sign into a 384-d vector.
4. L2-normalise.

`cosine(a, b)` gives semantic similarity; `embed_many` batches; `cached_embed` is
`lru_cache`d for speed and determinism.

> **Why not a real sentence transformer / BERT?** Because benchmarks must be
> *offline and reproducible* — a neural embedding model would add download weight and
> nondeterminism. Char n-grams capture morphology and word overlap well enough for
> domain/routing decisions. This is a deliberate, defensible engineering trade-off.

---

### 7.7 `core/analysis.py` — Prompt Intelligence

Turns raw text into a structured `PromptIntelligence` dataclass:

- **Intent** — keyword-pattern matching over ~13 intent classes (summarization, code
  generation, reasoning, math, translation, …), with confidence normalization + a small
  question-mark boost.
- **Domain** — lexicon keyword scoring over 9 domains, boosted by:
  - a **semantic push** against per-domain prototype embeddings (`_DOMAIN_PROTOTYPES`),
  - a **code-context hint** (`_CODE_INDICATOR_RE`) that boosts "software" and removes
    ambiguous commerce words ("return", "support") from customer_support — this is what
    stops "return mid" in code from being classified as e-commerce.
- **Complexity** — linear combination of text features: length, rare-word ratio,
  math symbols, code constructs, questions, lexical diversity, avg word length,
  uppercase ratio → score 0..1 + level `low|medium|high|critical` + reasoning score.
- **Required context** — how much history matters (explicit references, anaphora,
  short query, task type).
- **Reasons** — every step appends a human-readable reason. This feeds explainability.

`analyze_conversation()` blends the latest message with the prior 8 messages so a short
follow-up like "what about tail recursion?" still gets domain context.

> **Interview tip:** this module is the "understanding" stage — everything downstream
> consumes these derived signals, never the raw string.

---

### 7.8 `core/pruning.py` — Dynamic Token Pruning (RQ1)

Goal: **drop semantically redundant tokens while protecting task-critical content.**

Process in `prune_prompt()`:
1. Split into word chunks (with spans).
2. Score each chunk's importance with `_importance()` — a weighted heuristic:
   - **Penalized**: stopwords, common words, pure punctuation/whitespace.
   - **Boosted**: cue/action verbs, rare vocab, entities, numbers, code tokens, and
     **snake_case/camelCase identifiers** (added so `binary_search` survives).
   - Position bias: later tokens (instruction tail) matter more.
3. **Protect** user-specified substrings and instruction-like lines
   (`you are | system:| role | # | ````, …).
4. Greedily remove the lowest-importance *non-protected* chunks up to a budget
   (`MAX_PRUNE_RATIO` = 45%), then clamp to a keep floor (`MIN_PRUNE_KEEP` = 55%).
5. Rebuild text; if nothing was removed, return the original untouched (no artifact
   token drift from whitespace strip).

Returns a `PruneResult` with original/pruned text, token counts, retention ratio,
compression ratio, importance map, protected spans and removed spans — **evidence for
explainability**.

> **Interview tip:** the research criterion is *not* "how many tokens did we remove" but
> *"how many tokens can we remove while preserving downstream task performance"* — which
> is why evaluation (7.14) is measured, not assumed.

---

### 7.9 `core/context.py` — Intelligent Context Compression (RQ2)

Goal: keep only the conversation history the current query actually needs.

Pipeline:
1. Embed query + every message.
2. Compute cosine relevance to the query.
3. **MMR (Maximal Marginal Relevance)** selection — greedily picks messages that are
   relevant *and* not redundant with already-picked ones (`relevance − λ·redundancy`).
   This removes duplicate/repetitive turns.
4. Enforce a token budget (default 50% retention) and `top_k` (12 messages),
   preserving conversation order for coherence.
5. If even one message can't fit the budget → emit an **extractive summary** (concatenated
   heads of top messages) as fallback.

Returns `ContextResult` with kept/dropped messages, relevance scores, retention ratio,
redundancy count and reasons.

---

### 7.10 `core/lora.py` — Adaptive LoRA Selection (RQ3)

Holds a static `ADAPTER_CATALOG` of domain adapters:
`lora-code, lora-finance, lora-medical, lora-legal, lora-math, lora-science, lora-general`.

For a query, `select_adapter()` scores every adapter:
```
score = 0.55 × domain_belief + 0.25 × semantic + 0.20 × keyword_overlap
```
where:
- `domain_belief` comes from prompt-intelligence domain scores (Section 7.7),
- `semantic` = cosine(query, adapter prototype embedding),
- `keyword_overlap` = fraction of adapter keywords present.

If the best score is below `LORA_MIN_CONFIDENCE` (0.45) → **fallback to base model**
(no adapter), which protects against forcing a wrong adapter. Helpers quantify the
quality uplift (+8%) and energy overhead (+3%) of applying an adapter — those feed the
router.

> **Interview tip:** this is the "instead of manually testing 4 LoRA adapters" story —
> the system *predicts* the right one and even knows when to admit "no adapter".

---

### 7.11 `core/early_exit.py` — Layer-wise Optimization (RQ4)

Two layers of implementation:

**Planning** (`plan_layers`): given complexity, builds a **fidelity curve** —
`quality_ratio(k) = sigmoid(steepness · (k/total − midpoint))`. Easy prompts saturate
early (exit at ~low k); hard prompts need nearly full depth. Strategies:
- `full` → run everything,
- `auto` → exit where expected confidence clears `EARLY_EXIT_CONFIDENCE`,
- `minimal` → additionally floor to `MIN_LAYERS_RATIO`.

Returns a `LayerPlan` (layers to run, saved, expected quality factor, time saved,
confidence per layer, reasons).

**Live execution** (`note_measured_exit`): real per-token exit layers from the actual
tinygpt run update the plan — so *planned* savings become *measured* savings.

> **Interview answer:** "For proprietary models you can only *plan* early exit; for our
> from-scratch tinygpt we *actually break out of the forward pass* using token
> confidence — see Section 7.17."

---

### 7.12 `core/carbon.py` — Carbon Prediction Engine

`ModelEnergyProfile` per model: params, `ept_in`/`ept_out` (kWh per token), device power,
prefill/decode throughput.

For a given token count + measured/estimated latency:
```
E_token = in·ept_in + out·ept_out              (token-based energy)
E_time  = device_power × latency × utilization (time-based energy)
E       = E_token + E_time + fixed_overhead    (hybrid)
CO₂     = E × grid_intensity(region)
```
`layers_ratio` scales both terms for early-exit plans. Grid intensities come from
`config.GRID_INTENSITY_TABLE` (e.g., France 0.052 vs India 0.708 kg CO₂/kWh).

Returns a `CarbonEstimate` with full component breakdown + the formula used — again,
explainability and *calibrated* estimation over a naive single constant.

---

### 7.13 `core/router.py` — Multi-Objective Router + Explainability

For every candidate model, the router computes six dimensions:
- **quality** = model capability × task-fit multiplier − complexity penalty + adapter boost,
  scaled by layers_ratio.
- **energy / carbon** via `estimate_carbon()` for the candidate config.
- **latency** = prefill + decode from throughput model + fixed overhead × depth.
- **cost** = tokens × price/1k.
- **resource** = normalized model footprint.

Then:
1. Apply hard constraints (`min_quality`, `max_carbon_g`, `max_latency_ms`,
   `max_cost_usd`, `exclude_models`). If nothing survives → relax (documented in the
   explanation).
2. Min-max **normalize** every dimension across candidates.
3. Weighted sum with user-provided or default weights
   (quality 0.35, carbon 0.25, energy 0.10, cost 0.10, latency 0.10, resource 0.10).
4. Sort; pick the winner and runner-up.

`build_explanation()` produces the "Selected model: X" narrative: task complexity, domain,
intent, required context, reasoning, estimated quality/latency/carbon, LoRA, layer plan,
optimization ratios, how much it beat the runner-up by, and dimension scores — i.e.
**Objective 8 (explainability) end-to-end**.

---

### 7.14 `core/evaluation.py` — Response Quality Metrics

`evaluate_response(prediction, reference, …)` produces the **quality_score**:
```
quality = 0.55·semantic_similarity + 0.25·keyword_coverage + 0.20·length_adequacy
```
- **semantic_similarity** — cosine of char n-gram embeddings (reference required).
- **keyword_coverage** — fraction of reference keywords appearing in the prediction.
- **length_ratio** — bounded alignment with the reference length.
- No reference → conservative self-calibrated baseline (so quality stays comparable).

> **Why keyword coverage + length?** Embedding similarity alone rewards paraphrase but
> misses "did they actually answer it"; keyword coverage captures topical faithfulness and
> length catches hollow outputs.

---

### 7.15 `backends/base.py` — Backend Abstraction

`InferenceResult` dataclass: text, tokens_in/out, latency_ms, `depth_used` (0..1) and
extra `measured`/`metrics` dicts.

`InferenceBackend` ABC: `run(prompt, model_id, …)` + `available()` probe. Three concrete
backs satisfy it — simulator, tinygpt adapter, and remote. **The orchestrator only talks
to this interface**, so swapping a backend never touches the pipeline.

---

### 7.16 `backends/simulator.py` — Deterministic Simulator

For abstract models (llama-3-8b, gemma-2b, flan-t5, …):

1. Analyze the prompt → domain + intent.
2. Pick a domain-specific response fragment and an intent-specific opener.
3. Seed a RNG with `sha256(prompt|model|depth)` → **fully deterministic** output.
4. Latency from the model's throughput profile; early exit shortens decode time
   proportionally to `depth_used`.

Because it is deterministic, benchmark deltas reflect the pipeline's choices, not
sampling luck.

---

### 7.17 `backends/tinygpt.py` — Real From-Scratch Transformer

A complete GPT-style transformer in **pure numpy**:

- **Architecture**: LayerNorm → causal multi-head self-attention → MLP, ×6 layers,
  48-d embeddings, 4 heads, block length 96, char-level vocab (~built from corpus).
- **Training**: manual backprop (attention, layernorm, GELU gradients all hand-derived),
  Adam optimizer with bias correction, batches on the bundled corpus, weights cached to
  `data/models/tinygpt.npz`. Lazy: trains only if the cache is missing.
- **Inference**: 
  - `_forward_last_position()` runs each layer and returns the logits **after every
    layer** for the final token — this is what enables early exit.
  - `generate()` with early-exit **actually stops the forward pass** when either
    (a) token confidence ≥ threshold AND min layers met, or (b) confidence plateaued
    (`Δ < plateau_eps`) — i.e. adaptive computation in a real transformer.
  - Returns `exit_layers` per token, confidences, mean exit layer, so the platform
    reports *measured* layer savings (RQ4, live).

> **Interview gold:** this is your strongest evidence that you understand transformer
> internals — attention score computation, causal masking, residual streams, LayerNorm
> gradients, GELU derivatives, and how next-token entropy can be used as an exit signal.

---

### 7.18 `backends/tinygpt_backend.py` — tinygpt Adapter

Implements `InferenceBackend` for tinygpt:
- probes with a 2-token generation,
- runs `m.generate(...)`, converts `mean_exit_layer` → `depth_used` (0..1),
- carries real measured metadata: per-token exit layers, confidences,
  `real_transformer: True`.

Also noticeably cheaper latency math than the simulator, so routed carbon/energy for
tinygpt is realistic.

---

### 7.19 `backends/remote.py` — Live Remote Backend

Optional; enabled only if `GMA_OPENAI_BASE_URL` + `GMA_OPENAI_API_KEY` are set (e.g., a
local vLLM server). Speaks `/v1/chat/completions`, reads `usage.prompt_tokens` /
`completion_tokens`, and reports wall latency. If unconfigured, `available()` returns
False → orchestrator silently falls back to the simulator. This is the production
integration seam for real deployments.

---

### 7.20 `services/orchestrator.py` — The Heart of the Pipeline

`run_generate()` implements the full 10-step pipeline (Section 6). Key points to know:

- `_backend_for(model, early_exit)` picks tinygpt / remote / simulator.
- `_output_tokens_estimate()` predicts output length from complexity + model class — used
  by routing before execution.
- It computes a **baseline counterfactual** (`baseline_counterfactual()`) so every GMA
  run returns a `comparison.{baseline, delta}` — tokens/latency/energy/carbon/cost/quality
  % change vs doing nothing.
- `compute_delta()` — pure function turning two metric dicts into percentage deltas,
  tolerant of both nested (`{"structured_metrics": …}`) and flat inputs.
- Handles `req.model` override (force a model, e.g. for tests), `req.adapter` override,
  and `save_record` persistence.
- For tinygpt, quality is computed from the *decision* when no reference is given
  (`quality = selected.quality × layers expected factor`), plus an `evaluate_response`
  pass with no reference for self-consistency.

`build_execution_prompt()` re-composes `"Role: content … User: latest"` so the model sees
compressed history + optimized prompt.

---

### 7.21 `services/baseline.py` — The "Do Nothing" Counterfactual

`run_baseline()` runs the same request **un-optimized** on a fixed model
(`llama-3-8b` default): full raw prompt + full history, full depth (`depth_used=1.0`, no
early exit, no adapter, `prune_ratio=0`, `context_ratio=1.0`). Persists a
`kind="baseline"` record. This is the reference every delta is computed against.

> **Interview tip:** A project is scientific only if it has a control group. The baseline
> is the control; the GMA pipeline is the treatment; the benchmark reports the effect size.

---

### 7.22 `services/benchmark.py` — The Benchmark Harness

`benchmark_scenarios()` runs a list of prompt/reference scenarios through **both**
pipelines and records per-request results + a `BenchmarkRun` row with the full JSON.

`aggregate()` collapses results into:
- average baseline vs GMA metrics,
- `reductions_pct`: tokens / latency / energy / carbon / cost reduction %,
- quality delta,
- `carbon_saved_g` and `energy_saved_kwh` absolute savings,
- model & adapter usage distribution (did routing actually diversify?).

Built-in default scenarios cover summarization, code, explanation, classification,
carbon math, drafting, translation, and best-practices — a decent cross-domain eval set.

---

### 7.23 `services/analytics.py` — Sustainability Analytics

Reads persisted `RequestRecord`s and exposes:
- **summary** — totals, per-model carbon/energy/quality, and cost savings aggregates
  (token/energy/carbon reduction %, absolute gCO₂ saved), plus pairwise baseline-vs-GMA
  deltas.
- **timeseries** — bucketed carbon/latency over time.
- **benchmark_runs** / **recent_requests** — raw history for dashboards.

This is the "measure → learn → optimize better" feedback loop (Objective 12): the DB is
both a record of decisions and the dataset a future prediction model could train on.

---

### 7.24 `api/routes_*` — REST Layer

| File | Endpoints | Purpose |
|------|-----------|---------|
| `routes_generate.py` | `POST /api/generate`, `/api/generate/baseline`, `/api/estimate` | main execution + legacy estimate |
| `routes_models.py` | `GET /api/models`, `/api/adapters`, `/api/regions`, `/api/catalog` | catalog/introspection |
| `routes_benchmark.py` | `POST /api/benchmark/run`, `/api/benchmark/aggregate` | run & aggregate benchmarks |
| `routes_analytics.py` | `GET /api/analytics/{summary,timeseries,benchmarks,requests[/{id}]}` | sustainability analytics |
| `routes_health.py` | `GET /health` | liveness |

`router_main.py` bundles them all into `api_router`; `main.py` includes it. Route
handlers are thin — business logic lives in `services/`, so the API layer is trivially
testable.

---

## 8. The Four Research Questions — Implementation Summary

| RQ | Technique | Where | Deliverable |
|----|-----------|-------|-------------|
| RQ1 | Importance-scored token pruning with protection & budget | `core/pruning.py` | `PruneResult` (ratio saved, protected spans, removed spans) |
| RQ2 | MMR relevance selection + budget trimming + extractive fallback | `core/context.py` | `ContextResult` (retention, redundancy, reasons) |
| RQ3 | Domain/semantic/keyword adapter scoring with base fallback | `core/lora.py` | `LoRADecision` (selected id, confidence) |
| RQ4 | Fidelity-curve layer planning + real early-exit in numpy transformer | `core/early_exit.py` + `backends/tinygpt.py` | `LayerPlan` + measured exit layers |

The **system-level contribution** is that these four act in one pipeline, each feeding
evidence into a multi-objective router, and every run is compared to a no-optimization
counterfactual (see Sections 9 & 11).

---

## 9. How "Greener" Is Measured: Evaluation & Benchmarking

**Per request** (returned in `comparison.delta`):
- `token_reduction_pct` (flip of input token ratio vs baseline)
- `latency_change_pct`, `energy_change_pct`, `carbon_change_pct`, `cost_change_pct`
- `depth_savings_pct`, absolute `carbon_saved_g`, `energy_saved_kwh`

**Per benchmark** (`reductions_pct` in summary): averaged tokens/latency/energy/carbon/
cost reduction %, plus absolute `carbon_saved_g` and `energy_saved_kwh`, plus
`avg_quality_delta`.

**Quality is not assumed:** every response is scored by `evaluate_response` — semantic
similarity, keyword coverage, length adequacy — so you can *simultaneously* say "we cut
carbon by X% **and** quality changed by only Y" — the exact trade-off that answers the
core research question.

---

## 10. Design Decisions & Trade-offs

Be ready to defend these:

1. **Deterministic simulator + deterministic embeddings** — needed for reproducible
   A/B benchmarking; the cost is that "generations" are templated, not creative.
   → *"We optimise the inference *pipeline*, so we removed downstream randomness to make
   the comparison fair."*
2. **Embeddings = hashed char n-grams, not BERT** — 0 download, 0 nondeterminism,
   fast, good enough for domain/routing. Downside: weaker on long-range semantics.
3. **tinygpt char-level & tiny (6×48)** — it fits on a laptop in pure numpy, trains in
   seconds, and still demonstrates *real layer-wise early exit*. It is not a production
   model; it's an experimental apparatus to answer RQ4 honestly.
4. **Estimates over measurements** for energy/carbon on abstract models — we use
   calibrated literature-informed constants + measured latency; real hardware telemetry
   (codecarbon / NVIDIA) is available but optional.
5. **SQLite** — zero-config, file-based, perfectly adequate for a research prototype;
   switching to Postgres is a config change (`DATABASE_URL`).
6. **Everything optimised is recorded** (`save_record`) so the analytics module doubles
   as a training set for future learned routing — closing the feedback loop.

---

## 11. Bugs Found & Fixed

If asked "what did you actually fix / debug", you can speak to these real defects:
- `app/database.py` — `self.responseesponse` typo broke `RequestRecord.to_dict()`.
- `app/api/router_main.py` & `routes_health.py` — corrupted (null-byte) files; recreated
  from scratch with correct router aggregation + `/health` endpoint.
- `app/core/early_exit.py:96` — f-string formatted `confidence_threshold` (= `None`)
  instead of `threshold` → TypeError on every `auto`-strategy `/api/generate` call.
- `app/services/orchestrator.py` — `compute_delta()` expected nested
  `optimized["structured_metrics"]` but callers handed it the flattened metrics dict →
  KeyError on every response; made it tolerant of both shapes.
- `app/core/pruning.py` — short-prompt early return passed 10 positional args for an
  11-field dataclass, so `compression_ratio` received a dict; switched to keyword args.
- `app/services/benchmark.py` — passed raw lists/dicts into `JSONText` columns →
  `sqlite3.ProgrammingError`; fixed with `json.dumps`.
- Domain detection bug — literal "return mid" in code was classified as e-commerce
  (`customer_support`) because "return" is a commerce keyword; added code-context
  awareness + ambiguity dampening.
- Pruning dropped identifiers like `binary_search`; added snake_case/camelCase boosts.

---

## 12. Testing Strategy

`tests/` has 108 tests organized by layer:

| File | What it verifies |
|------|------------------|
| `test_api.py` | every endpoint: status codes, response shape, validation (422 on empty prompt), tinygpt end-to-end, region override |
| `test_analysis.py` | intent/domain/complexity/context detection incl. code & medical/finance cases |
| `test_pruning.py` | budgets, short-prompt identity, identifier protection, retention floor, zero-ratio no-op |
| `test_context.py` | no history, single turn, relevance, budget extremes, relevance score bounds |
| `test_lora.py` | correct adapter per domain, fallback, pipeline steps |
| `test_early_exit.py` | full/auto/minimal strategies, threshold monotonicity, measured-exit updates |
| `test_carbon.py` | grid intensity, monotonicity w/ tokens, France-vs-India, layers scaling, dict shape |
| `test_router.py` | candidate generation, constraints, exclusions, weights, runner-up |
| `test_evaluation.py` | quality scoring, similarity bounds, empty prediction |
| `test_benchmark.py` | aggregation math, compute_delta, multi-scenario runs |
| `test_orchestrator.py` | end-to-end generate w/ and w/o history, persistence, disabled optimizations, reference quality |
| `test_backends.py` | simulator determinism, early-exit speedup, domain mapping |
| `test_embeddings.py` | cosine properties, batch shape, caching, edge cases |
| `test_database.py` | RequestRecord round-trip, missing fields, session smoke |

Run: `python -m pytest tests/ -v`

---

## 13. How To Run

```bash
source venv/bin/activate
python -m pytest tests/ -q     # 108 passed
python run.py                  # uvicorn on 0.0.0.0:8000
```

Try it:
```bash
curl -X POST localhost:8000/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Write a Python function using binary search","max_tokens":64}'
```

---

## 14. Common Interview Questions & Talking Points

**Q: What is the actual research contribution?**
> Not the individual techniques — token pruning, context compression, LoRA routing and
> early exit all exist in literature. The contribution is the *integration*: one
> middleware that runs them together, routes on six objectives, explains every decision,
> and measures everything against a no-optimization baseline.

**Q: How do you know your savings are real and not just because you used the simulator?**
> Three answers: (1) the simulator is deterministic, so the *same prompt* under baseline
> vs GMA differs only by pipeline choices; (2) quality is scored, so we report the
> carbon-vs-quality trade-off, not carbon alone; (3) we validate the layer-wise early
> exit on a *real* from-scratch numpy transformer (tinygpt), not just a formula.

**Q: Why a from-scratch transformer instead of using Hugging Face?**
> RQ4 requires reaching inside the model between layers to measure confidence and exit —
> impossible with a black-box hosted API and hard with opaque HF wrappers. A small
> char-level numpy transformer gives us full control to *demonstrate* adaptive
> computation, while the simulator/remote backends cover production-scale models.

**Q: How would you scale / productionise this?**
> - Swap SQLite → Postgres via `DATABASE_URL`.
> - Write a remote vLLM backend per model family (already designed: `backends/remote.py`).
> - Feed `analytics` records into a learned predictor to replace heuristic constants.
> - Add a dashboard (the `/dashboard` static mount is already wired).
> - Use codecarbon/RAPL/NVIDIA telemetry for real power draw instead of calibrated
>   constants.

**Q: Explain the routing math.**
> For each candidate model: compute quality/energy/carbon/latency/cost/resource; min-max
> normalize each dimension across candidates; weighted sum with user weights (default
> quality 0.35, carbon 0.25); hard constraints filter first; highest score wins; all of
> it explained in `explanation.reasons`.

**Q: What was the hardest bug?**
> Good examples from Section 11 — the `pruning.py` dataclass arg misalignment and the
> `compute_delta` KeyError both silently broke the *main* endpoint while unit tests on
> sub-modules still passed, which is a classic "system vs component" testing lesson —
> hence the end-to-end tests I added.

---

## Appendix — Endpoint Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Self-describing app info |
| GET | `/health` | Liveness probe |
| POST | `/api/generate` | Full optimized pipeline |
| POST | `/api/generate/baseline` | Un-optimized counterfactual |
| POST | `/api/estimate` | Legacy single carbon estimate |
| GET | `/api/models` | Model catalog with cost/energy profiles |
| GET | `/api/adapters` | LoRA adapter catalog |
| GET | `/api/regions` | Grid carbon intensities |
| GET | `/api/catalog` | Energy profiles overview |
| POST | `/api/benchmark/run` | Run a scenario suite through both pipelines |
| POST | `/api/benchmark/aggregate` | Aggregate supplied results |
| GET | `/api/analytics/summary` | Totals + savings aggregates |
| GET | `/api/analytics/timeseries` | Carbon/latency over time |
| GET | `/api/analytics/benchmarks` | Historical benchmark runs |
| GET | `/api/analytics/requests` | Recent request records |
| GET | `/api/analytics/requests/{id}` | Single record detail |
| GET | `/docs` | OpenAPI interactive docs |