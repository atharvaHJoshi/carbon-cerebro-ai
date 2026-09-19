"""Prompt Intelligence - turns a raw prompt into structured signals.

Produces:
  * intent        - what the task is (summarise, classify, code, translate, ...)
  * domain        - knowledge domain (software, finance, medical, legal, math, ...)
  * complexity    - 0..1 difficulty estimate + level
  * required_context - how much conversational history matters
  * style / sensitivity / reasoning hints used downstream by pruning & routing
"""

from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass, field

from ..tokenizer import count_tokens
from .embeddings import cached_embed
from .embeddings import cosine

# --------------------------------------------------------------------------- lexicons


STOPWORDS = set(
    """
    a an the and or but if then else for while do does did i you he she it we they
    with without of from to in on at by as is are was were be been being am not no
    this that these those there here what which who whom whose why how when where
    can could will would shall should may might must have has had having do did does
    get got getting make makes making like just really very about into over under
    through again further once also all any both each few more most other some such
    only own same so too very s t don now your their its our
    """.split()
)

_COMMON_WORDS = set(
    "the of and to in a that is was he for it with as his on be at by i this had not are but from or have an they which one you were her all there when we can him said what about its has if who will would then them may some she their so no these after other than there our first could how area do each did much here time".split()
)

_CUE_TERMS = {
    "explain": 0.30, "describe": 0.25, "summarize": 0.25, "list": 0.20,
    "compare": 0.30, "contrast": 0.30, "define": 0.25, "compute": 0.30,
    "calculate": 0.30, "justify": 0.30, "interpret": 0.25, "outline": 0.20,
    "what": 0.15, "why": 0.25, "how": 0.20, "solve": 0.30, "write": 0.15,
    "translate": 0.25, "classify": 0.25, "recommend": 0.25, "predict": 0.25,
    "draft": 0.20, "review": 0.20, "find": 0.15, "identify": 0.20, "argue": 0.30,
    "evaluate": 0.30, "analyze": 0.30, "convert": 0.25, "format": 0.15,
}

DOMAIN_KEYWORDS: dict[str, set[str]] = {
    "software": {
        "code", "program", "function", "api", "bug", "debug", "compile", "python",
        "javascript", "java", "golang", "rust", "algorithm", "docker", "kubernetes",
        "syntax", "error", "exception", "sql", "database", "variable", "loop",
        "recursion", "framework", "library", "deploy", "endpoint", "json", "http",
        "refactor", "array", "object", "class", "thread", "memory", "compiler",
        "repository", "git", "branch", "commit", "test", "unit test", "pytest",
        "lint", "type", "import", "module", "package", "server", "client", "latency",
        "regex", "shell", "script", "stack", "queue", "tree", "graph", "dp",
    },
    "finance": {
        "stock", "finance", "financial", "market", "investment", "revenue", "profit",
        "tax", "bank", "loan", "interest", "asset", "liability", "budget", "credit",
        "debit", "dividend", "equity", "bond", "portfolio", "pricing", "forecast",
        "earnings", "balance sheet", "cash flow", "capital", "risk", "liquidity",
        "inflation", "currency", "equity", "fund", "hedge", "gdp", "audit", "ledger",
    },
    "medical": {
        "patient", "doctor", "diagnosis", "symptom", "treatment", "disease", "drug",
        "medicine", "clinical", "dosage", "pandemic", "virus", "vaccine", "surgery",
        "cardiac", "neurology", "renal", "diabetes", "infection", "therapy",
        "prescription", "screening", "tumor", "staging", "prognosis", "antibiotic",
        "pharmaceutical", "anatomy", "physiology", "MRI", "CT scan", "healthy",
    },
    "legal": {
        "contract", "lawyer", "court", "legal", "lawsuit", "plaintiff", "defendant",
        "liability", "clause", "jurisdiction", "statute", "amendment", "tort",
        "patent", "trademark", "copyright", "compliance", "regulation", "arbitration",
        "settlement", "verdict", "appeal", "criminal", "civil", "testimony", "deposition",
    },
    "math": {
        "equation", "integral", "derivative", "matrix", "calculus", "algebra",
        "geometry", "probability", "variance", "statistics", "mean", "median",
        "hypothesis", "proof", "theorem", "sum", "summation", "differential",
        "linear", "vector", "eigen", "log", "logarithm", "exponential", "graph",
        "polynomial", "prime", "fraction", "decimal", "percent", "ratio",
    },
    "science": {
        "physics", "chemistry", "biology", "energy", "carbon", "climate", "temperature",
        "molecule", "atom", "cell", "experiment", "laboratory", "hypothesis", "quantum",
        "gravity", "velocity", "force", "acceleration", "photons", "electricity",
        "ecosystem", "biodiversity", "genome", "protein", "enzyme", "reaction",
    },
    "education_writing": {
        "essay", "paragraph", "thesis", "argument", "summarize", "outline", "rewrite",
        "grammar", "vocabulary", "spelling", "plagiarism", "citation", "works cited",
        "tone", "audience", "hook", "conclusion", "introduction", "draft", "edit",
    },
    "customer_support": {
        "refund", "return", "order", "shipping", "tracking", "account", "password",
        "reset", "billing", "invoice", "warranty", "complaint", "cancel", "support",
        "help", "error message", "subscription", "charge", "delivery",
    },
    "general": set(),
}

INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("summarization", ["summarize", "summary", "tl;dr", "concise version", "in short", "key points of", "condense"]),
    ("classification", ["classify", "categorize", "label", "which category", "detect", "identify the type", "sentiment"]),
    ("question_answering", ["what is", "what are", "who is", "when did", "where is", "why does", "how does", "how to", "explain", "meaning of", "define", "what does"]),
    ("code_generation", ["write code", "generate code", "implement", "function that", "program to", "script to", "code that", "bug fix", "refactor", "sql query", "regex for"]),
    ("generation", ["write a", "create a", "compose", "draft", "generate a", "story", "email", "poem", "blog post"]),
    ("translation", ["translate", "in french", "in spanish", "in german", "in hindi", "convert to", "language"]),
    ("extraction", ["extract", "pull out", "find all", "list the", "identify entities", "parse", "scrape"]),
    ("reasoning", ["if .* then", "logical", "deduce", "infer", "compare and contrast", "evaluate the argument", "prove", "solve", "puzzle"]),
    ("editing", ["rewrite", "proofread", "correct grammar", "improve", "simplify", "rephrase", "that translates to"]),
    ("planning", ["plan", "roadmap", "steps to", "strategy", "schedule", "itinerary", "checklist"]),
    ("recommendation", ["recommend", "suggest", "best options", "which model", "compare", "top 3"]),
    ("prediction", ["predict", "forecast", "will happen", "projected", "estimate the"]),
    ("math_problem", ["solve", "calculate", "compute", "equation", "integral", "derivative", "probability of", "what is 2", "="]),
]

_INTENT_WEIGHT = {"summarization": 0.9, "classification": 0.9, "question_answering": 0.8,
                  "code_generation": 1.0, "generation": 0.7, "translation": 0.8,
                  "extraction": 0.8, "reasoning": 1.0, "editing": 0.7, "planning": 0.8,
                  "recommendation": 0.8, "prediction": 0.9, "math_problem": 1.0}


@dataclass
class PromptIntelligence:
    intent: str
    intent_confidence: float
    domains: dict[str, float]
    top_domain: str
    domain_confidence: float
    complexity: float
    complexity_level: str
    required_context: float            # 0..1 how much history matters
    reasoning_required: float          # 0..1
    tokens: int
    words: int
    chars: int
    features: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "intent_confidence": round(self.intent_confidence, 4),
            "domains": {k: round(v, 4) for k, v in sorted(self.domains.items(), key=lambda x: -x[1])},
            "top_domain": self.top_domain,
            "domain_confidence": round(self.domain_confidence, 4),
            "complexity": round(self.complexity, 4),
            "complexity_level": self.complexity_level,
            "required_context": round(self.required_context, 4),
            "reasoning_required": round(self.reasoning_required, 4),
            "tokens": self.tokens,
            "words": self.words,
            "chars": self.chars,
            "reasons": self.reasons,
        }


_STOP = set(STOPWORDS)


def _features(text: str) -> dict:
    words = re.findall(r"\b[\w’']+\b", text.lower())
    n_words = len(words)
    unique = set(words) if words else set()
    n_unique = len(unique)
    rare_ratio = 0.0
    if n_words:
        rare_ratio = sum(1 for w in words if w not in _COMMON_WORDS) / n_words
    code_indicators = re.search(r"\b(def |class |function|import |return |=>|\{\s|\}\s|->|```|SELECT |INSERT |plt\.)", text)
    math_symbols = re.findall(r"[+\-*/^=√∫∑≤≥±×÷]", text)
    questions = len(re.findall(r"[?]", text))
    sentences = max(1, len(re.findall(r"[.!?]", text)))
    avg_word_len = (len(re.findall(r"[a-z]", text.lower())) / n_words) if n_words else 0.0
    upper_ratio = (len(re.findall(r"[A-Z]", text)) / max(1, len(text)))
    numbers = len(re.findall(r"\b\d+(?:[.,]\d+)*\b", text))
    return {
        "n_words": n_words,
        "n_unique": n_unique,
        "unique_ratio": (n_unique / n_words) if n_words else 0.0,
        "rare_ratio": rare_ratio,
        "code_indicators": bool(code_indicators),
        "math_count": len(math_symbols),
        "questions": questions,
        "sentences": sentences,
        "avg_word_len": avg_word_len,
        "upper_ratio": upper_ratio,
        "numbers": numbers,
    }


def detect_intent(text: str) -> tuple[str, float, list[str]]:
    low = text.lower()
    scores: dict[str, float] = {}
    for intent, pats in INTENT_PATTERNS:
        s = 0.0
        for p in pats:
            if p in low:
                s += _INTENT_WEIGHT.get(intent, 0.7) * (1.0 if len(p) > 3 else 0.6)
        scores[intent] = s
    best = max(scores, key=scores.get)
    total = sum(scores.values())
    conf = scores[best] / total if total else 0.0
    # bias for clear question fragments
    reasons = []
    if "?" in text and best == "question_answering":
        reasons.append("Explicit interrogative phrasing detected")
    if any(k in ("code_generation", "math_problem") and scores.get(k, 0) > 0 for k in ("code_generation", "math_problem")):
        reasons.append("Action verbs map to structured generation task")
    return best, min(conf + 0.15, 1.0) if scores[best] > 0 else 0.0, reasons


_CODE_INDICATOR_RE = re.compile(
    r"\b(def |class |import |from |return |function |lambda |void |int |float |"
    r"def\b|class\b|import\b|from\b|=>|->|```|\w+\(.*\)|SELECT |INSERT |UPDATE |WHERE |FROM )",
    re.IGNORECASE,
)
_COMMERCE_AMBIGUOUS = frozenset({
    "return", "support", "account", "charge", "tracking", "order",
    "claim", "cancel", "renew", "subscription",
})


def detect_domain(text: str, query_embed: object = None) -> tuple[dict[str, float], str, float, list[str]]:
    low = text.lower()
    words = set(re.findall(r"\w+", low))
    code_hint = bool(_CODE_INDICATOR_RE.search(low))
    scores: dict[str, float] = {}
    for domain, kws in DOMAIN_KEYWORDS.items():
        s = 0.0
        for kw in kws:
            if kw in low:
                s += 1.0
        overlap = words & kws
        if code_hint and domain == "customer_support":
            overlap = overlap - _COMMERCE_AMBIGUOUS
        s += len(overlap) * 0.7
        scores[domain] = s

    if code_hint:
        scores["software"] = scores.get("software", 0.0) + 2.5

    top = max(scores, key=scores.get)
    total = sum(scores.values()) or 1.0
    conf = scores[top] / total if total else 0.0
    reasons = []
    if scores[top] > 0:
        reasons.append(f"{top} lexicon matched ({int(scores[top])} signals)")
    else:
        reasons.append("No domain vocabulary matched -> general")
    # semantic push using emitter vectors against domain prototypes
    proto = _DOMAIN_PROTOTYPES.get(top)
    if proto is not None and query_embed is not None:
        sim = cosine(query_embed, proto)
        scores[top] = scores.get(top, 0) * 0.85 + sim * 2.0
        top = max(scores, key=scores.get)
        conf = scores[top] / (sum(scores.values()) or 1.0)
    return scores, top, min(max(conf, 0.25), 1.0), reasons


_DOMAIN_PROTOTYPES = {
    "software": cached_embed("writing and debugging code, functions, apis, databases, algorithms"),
    "finance": cached_embed("stock market, investment, revenue, tax, banking, financial statements"),
    "medical": cached_embed("patient symptoms, diagnosis, treatment, drugs, clinical care"),
    "legal": cached_embed("contracts, courts, laws, lawsuits, legal clauses and compliance"),
    "math": cached_embed("equations, calculus, algebra, probability, proofs and statistics"),
    "science": cached_embed("physics chemistry biology experiments, energy, climate, molecules"),
    "education_writing": cached_embed("essay writing, grammar, summarizing, editing and composition"),
    "customer_support": cached_embed("refunds, returns, orders, accounts, billing and support helpdesk"),
    "general": cached_embed("everyday questions and general knowledge"),
}


def complexity_features(f: dict, text: str) -> dict:
    length = f["n_words"]
    score = 0.0
    reasons = []

    if length > 250:
        score += 0.30; reasons.append("long prompt (+0.30)")
    elif length > 100:
        score += 0.20; reasons.append("lengthy prompt (+0.20)")
    elif length > 40:
        score += 0.10; reasons.append("moderate length (+0.10)")

    score += min(f["rare_ratio"] * 0.35, 0.35)
    if f["rare_ratio"] > 0.6:
        reasons.append("technical vocabulary raises difficulty")

    score += min(f["math_count"] * 0.06, 0.30)
    if f["math_count"] > 3:
        reasons.append("quantitative / symbolic demands (+%.2f)" % min(f["math_count"] * 0.06, 0.30))

    if f["code_indicators"]:
        score += 0.20; reasons.append("code constructs detected (+0.20)")

    if f["questions"] >= 2:
        score += 0.10; reasons.append("multiple sub-questions (+0.10)")

    if f["unique_ratio"] > 0.7:
        score += 0.10; reasons.append("high lexical diversity (+0.10)")

    if f["avg_word_len"] > 6.5:
        score += 0.10; reasons.append("long multi-syllabic terms (+0.10)")

    if f["upper_ratio"] > 0.05:
        score += 0.05; reasons.append("acronym heavy (+0.05)")

    score = min(max(score, 0.0), 1.0)

    if score < 0.25:
        level = "low"
    elif score < 0.5:
        level = "medium"
    elif score < 0.75:
        level = "high"
    else:
        level = "critical"

    # reasoning requirement: proportional to complexity but with contextual nuance
    reasoning = min(0.2 + score * 0.8, 1.0)
    return {"score": score, "level": level, "reasoning": reasoning, "reasons": reasons}


def required_context(text: str, intent: str, f: dict) -> tuple[float, list[str]]:
    low = text.lower()
    s = 0.0
    reasons = []
    if any(w in low for w in ["earlier", "previously", "as we discussed", "above", "in the conversation", "context", "from before", "last message"]):
        s += 0.45; reasons.append("explicit reference to prior discussion")
    if any(w in low for w in ["this", "that", "it", "them", "their", "the following", "given the above"]):
        s += 0.15; reasons.append("anaphoric references suggest history use")
    if intent in ("summarization", "extraction", "reasoning"):
        s += 0.15; reasons.append("task typically benefits from history")
    if f["n_words"] < 20:
        s += 0.25; reasons.append("very short query likely follows prior turns")
    return min(s, 1.0), reasons


def analyze_prompt(text: str | None) -> PromptIntelligence:
    text = text or ""
    stats = _features(text)
    emb = cached_embed(text)
    intent, intent_conf, ireasons = detect_intent(text)
    domains, top_domain, domain_conf, dreasons = detect_domain(text, emb)
    comp = complexity_features(stats, text)
    ctx, creasons = required_context(text, intent, stats)
    tokens = count_tokens(text)
    reasons = ireasons + dreasons + comp["reasons"] + creasons
    if not reasons:
        reasons = ["Prompt forwarded as-is; no strong signals found"]
    return PromptIntelligence(
        intent=intent,
        intent_confidence=round(intent_conf, 4),
        domains=domains,
        top_domain=top_domain,
        domain_confidence=round(domain_conf, 4),
        complexity=round(comp["score"], 4),
        complexity_level=comp["level"],
        required_context=round(ctx, 4),
        reasoning_required=round(comp["reasoning"], 4),
        tokens=tokens,
        words=stats["n_words"],
        chars=len(text),
        features=stats,
        reasons=reasons,
    )


def analyze_conversation(messages: list[dict]) -> PromptIntelligence:
    """Analyze a full conversation: combine latest user message + history summary."""
    if not messages:
        return analyze_prompt("")
    latest = messages[-1].get("content", "")
    combined = " ".join(m.get("content", "") for m in messages[-8:])
    pi = analyze_prompt(latest)
    pi2 = analyze_prompt(combined)
    pi.complexity = round(max(pi.complexity, pi2.complexity * 0.85), 4)
    pi.reasons.append("Conversation-level analysis blended with last message")
    return pi