"""A real, from-scratch tiny autoregressive transformer (numpy only).

Implements GPT-style blocks (LayerNorm -> causal self-attention -> MLP) with a
manual backprop + Adam trainer, so the layer-wise early-exit machinery (RQ4) is
demonstrated with *actual* transformer layers rather than only simulated ones.

Weights are cached to `data/models/tinygpt.npz`. If the cache is missing the
module lazily trains on the bundled corpus in `app/data/corpus/`.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..config import settings

_CONFIG = {
    "n_layer": 6,
    "n_embd": 48,
    "n_head": 4,
    "block": 96,
    "seed": settings.TINYGPT_SEED,
}

EPS = 1e-5


class TinyGPT:
    def __init__(self, model_id: str = "tinygpt", *, force_train: bool = False, train_steps: int | None = None):
        self.model_id = model_id
        self.cfg = dict(_CONFIG)
        self.train_steps = train_steps or settings.TINYGPT_TRAIN_STEPS
        self.params: dict[str, np.ndarray] = {}
        self.vocab: list[str] = [" ", "\n"]
        self._vocab_index: dict[str, int] = {}
        self.weights_path = settings.TINYGPT_WEIGHTS

        if force_train or not self.weights_path.exists():
            self._train()
        else:
            self.load()

    # ------------------------------------------------------------------ data & vocab
    def _load_corpus(self) -> str:
        corpus_dir = Path(__file__).resolve().parent.parent / "data" / "corpus"
        texts: list[str] = []
        for f in sorted(corpus_dir.glob("*.txt")):
            texts.append(f.read_text(encoding="utf-8", errors="ignore"))
        if not texts:
            texts = ["hello world. green model advisor reduces carbon emissions.\n"]
        return "\n".join(texts)

    def _build_vocab(self, data: str) -> list[str]:
        chars = sorted(set(data))
        vocab = [" ", "\n"] + [c for c in chars if c not in (" ", "\n")]
        return vocab

    # --------------------------------------------------------------- core nn ops
    @staticmethod
    def _init(*n: int) -> np.ndarray:
        return np.random.randn(*n) * 0.02

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=axis, keepdims=True)

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    @staticmethod
    def _gelu_bwd(dy: np.ndarray, pre: np.ndarray) -> np.ndarray:
        x = pre
        c = np.sqrt(2.0 / np.pi)
        inner = x + 0.044715 * x**3
        d = 0.5 * (1.0 + np.tanh(c * inner)) + 0.5 * x * (1 - np.tanh(c * inner) ** 2) * c * (1 + 0.134145 * x**2)
        return dy * d

    @staticmethod
    def _layernorm_fwd(x: np.ndarray, g: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        xn = (x - mean) / np.sqrt(var + EPS)
        return xn * g + b, xn, np.sqrt(var + EPS)

    @staticmethod
    def _layernorm_bwd(dy: np.ndarray, xn: np.ndarray, g: np.ndarray, std: np.ndarray) -> np.ndarray:
        dg = np.sum(dy * xn, axis=-1, keepdims=True)
        db = np.sum(dy, axis=-1, keepdims=True)
        dx = (g * dy - np.mean(g * dy, axis=-1, keepdims=True) - xn * np.mean(g * dy * xn, axis=-1, keepdims=True)) / std
        return dx

    @classmethod
    def _attn_fwd(cls, x: np.ndarray, p: dict, layer: int, n_head: int, head_dim: int):
        B, T, C = x.shape
        q = x @ p[f"q{layer}w"] + p[f"q{layer}b"]
        k = x @ p[f"k{layer}w"] + p[f"k{layer}b"]
        v = x @ p[f"v{layer}w"] + p[f"v{layer}b"]
        qr = q.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
        kr = k.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
        vr = v.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
        s = np.einsum("bhtd,bhTd->bhtT", qr, kr) / np.sqrt(head_dim)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        s = np.where(mask, -1e9, s)
        a = cls._softmax(s, axis=-1)
        o = np.einsum("bhtT,bhTd->bhtd", a, vr)
        o = o.transpose(0, 2, 1, 3).reshape(B, T, C)
        out = o @ p[f"o{layer}w"] + p[f"o{layer}b"]
        return out, {"q": qr, "k": kr, "v": vr, "a": a, "o_res": o}

    @classmethod
    def _attn_bwd(cls, dx2: np.ndarray, x: np.ndarray, p: dict, layer: int, cache: dict, n_head: int, head_dim: int):
        B, T, C = x.shape
        q, k, v, a, o_res = cache["q"], cache["k"], cache["v"], cache["a"], cache["o_res"]
        dout = dx2
        # out = o_res @ wo + bo
        dwo = np.einsum("btc,btd->cd", o_res, dout)
        dbo = dout.sum((0, 1))
        d_o_res = dout @ p[f"o{layer}w"].T
        d_attn = d_o_res.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
        # attn = a @ v
        d_v = np.einsum("bhtT,bhtd->bhTd", a, d_attn)
        d_a = np.einsum("bhtd,bhTd->bhtT", d_attn, v)
        d_s = a * (d_a - np.sum(d_a * a, axis=-1, keepdims=True))
        d_q = np.einsum("bhtT,bhTd->bhtd", d_s, k) / np.sqrt(head_dim)
        d_k = np.einsum("bhtT,bhtd->bhTd", d_s, q) / np.sqrt(head_dim)
        d_v = d_v
        dq = d_q.transpose(0, 2, 1, 3).reshape(B, T, C)
        dk = d_k.transpose(0, 2, 1, 3).reshape(B, T, C)
        dv = d_v.transpose(0, 2, 1, 3).reshape(B, T, C)
        dqw = np.einsum("btc,btd->cd", x, dq)
        dqb = dq.sum((0, 1))
        dkw = np.einsum("btc,btd->cd", x, dk)
        dkb = dk.sum((0, 1))
        dvw = np.einsum("btc,btd->cd", x, dv)
        dvb = dv.sum((0, 1))
        dh1 = dq @ p[f"q{layer}w"].T + dk @ p[f"k{layer}w"].T + dv @ p[f"v{layer}w"].T
        return ({
            f"q{layer}w": dqw, f"q{layer}b": dqb, f"k{layer}w": dkw, f"k{layer}b": dkb,
            f"v{layer}w": dvw, f"v{layer}b": dvb, f"o{layer}w": dwo, f"o{layer}b": dbo,
        }, dh1)

    def _forward(self, idx: np.ndarray, p: dict, *, save_cache: bool = True):
        B, T = idx.shape
        C = self.cfg["n_embd"]
        H = self.cfg["n_head"]
        HD = C // H
        V = self._vp
        x = p["wte"][idx] + p["wpe"][:T]
        caches: list[dict] = []
        xl = x
        for layer in range(self.cfg["n_layer"]):
            h1, xn1, std1 = self._layernorm_fwd(xl, p[f"ln1{layer}w"], p[f"ln1{layer}b"])
            out, acache = self._attn_fwd(h1, p, layer, H, HD)
            x2 = xl + out
            h2, xn2, std2 = self._layernorm_fwd(x2, p[f"ln2{layer}w"], p[f"ln2{layer}b"])
            pre = h2 @ p[f"w1{layer}"] + p[f"b1{layer}"]
            g = self._gelu(pre)
            proj = g @ p[f"w2{layer}"] + p[f"b2{layer}"]
            x3 = x2 + proj
            if save_cache:
                caches.append({"xn1": xn1, "std1": std1, "x1": xl, "x2": x2, "xn2": xn2, "std2": std2,
                               "g": g, "pre": pre, "attn": acache})
            xl = x3
        hf, xnf, stdf = self._layernorm_fwd(xl, p["lnfw"], p["lnfb"])
        logits = hf @ p["head"] + p["headb"]
        if save_cache:
            return logits, caches, (hf, xnf, stdf)
        return logits

    # ------------------------------------------------------------------ training
    def _train(self):
        start = time.time()
        np.random.seed(self.cfg["seed"])
        data = self._load_corpus()
        self.vocab = self._build_vocab(data)
        self._vocab_index = {c: i for i, c in enumerate(self.vocab)}
        self._vp = len(self.vocab)
        V, C, Tmax = self._vp, self.cfg["n_embd"], self.cfg["block"]
        L = self.cfg["n_layer"]

        p: dict[str, np.ndarray] = {
            "wte": self._init(V, C), "wpe": self._init(Tmax, C),
            "lnfw": np.ones(C), "lnfb": np.zeros(C),
            "head": self._init(C, V), "headb": np.zeros(V),
        }
        for l in range(L):
            p[f"ln1{l}w"] = np.ones(C); p[f"ln1{l}b"] = np.zeros(C)
            p[f"ln2{l}w"] = np.ones(C); p[f"ln2{l}b"] = np.zeros(C)
            p[f"q{l}w"] = self._init(C, C); p[f"q{l}b"] = np.zeros(C)
            p[f"k{l}w"] = self._init(C, C); p[f"k{l}b"] = np.zeros(C)
            p[f"v{l}w"] = self._init(C, C); p[f"v{l}b"] = np.zeros(C)
            p[f"o{l}w"] = self._init(C, C); p[f"o{l}b"] = np.zeros(C)
            p[f"w1{l}"] = self._init(C, 4 * C); p[f"b1{l}"] = np.zeros(4 * C)
            p[f"w2{l}"] = self._init(4 * C, C); p[f"b2{l}"] = np.zeros(C)

        ids = np.array([self._vocab_index.get(c, 0) for c in data], dtype=np.int64)
        B = 4
        m: dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in p.items()}
        vv: dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in p.items()}
        self._vp = V
        self.params = p

        for step in range(1, self.train_steps + 1):
            ix = np.random.randint(0, max(1, len(ids) - Tmax - 1), size=B)
            xb = np.stack([ids[i : i + Tmax] for i in ix])
            yb = np.stack([ids[i + 1 : i + Tmax + 1] for i in ix])

            logits, caches, (hf, xnf, stdf) = self._forward(xb, p)
            sm = self._softmax(logits, axis=-1)
            loss = -np.mean(np.log(sm[np.arange(B)[:, None], np.arange(Tmax)[None, :], yb] + 1e-9))

            dlogits = sm.copy()
            dlogits[np.arange(B)[:, None], np.arange(Tmax)[None, :], yb] -= 1.0
            dlogits /= (B * Tmax)

            dhf = dlogits @ p["head"].T
            dhead = hf.reshape(-1, C).T @ dlogits.reshape(-1, V)
            dheadb = dlogits.reshape(-1, V).sum(0)
            dxl = self._layernorm_bwd(dhf, xnf, p["lnfw"], stdf)

            grads: dict[str, np.ndarray] = {
                "head": dhead, "headb": dheadb,
                "lnfw": np.sum(dhf * xnf, axis=(0, 1)), "lnfb": np.sum(dhf, axis=(0, 1)),
            }

            for layer in reversed(range(L)):
                ca = caches[layer]
                # ---- MLP / ln2 path: x3 = x2 + proj
                dproj = dxl
                dg = dproj @ p[f"w2{layer}"].T
                dw2 = np.einsum("btc,btd->cd", ca["g"], dproj)
                db2 = dproj.sum((0, 1))
                dpre = self._gelu_bwd(dg, ca["pre"])
                dw1 = np.einsum("btc,btd->cd", ca["x2"], dpre)
                db1 = dpre.sum((0, 1))
                grads[f"w1{layer}"] = dw1
                grads[f"b1{layer}"] = db1
                grads[f"w2{layer}"] = dw2
                grads[f"b2{layer}"] = db2

                # x2 += (via ln2/h2 path)
                dh2 = dpre @ p[f"w1{layer}"].T
                dx2 = dxl + self._layernorm_bwd(dh2, ca["xn2"], p[f"ln2{layer}w"], ca["std2"])
                grads[f"ln2{layer}w"] = np.sum(dh2 * ca["xn2"], axis=(0, 1))
                grads[f"ln2{layer}b"] = np.sum(dh2, axis=(0, 1))

                # ---- attention path: x2 = x1 + out
                attn_grads, dh1 = self._attn_bwd(dx2, ca["x1"], p, layer, ca["attn"],
                                                 self.cfg["n_head"], C // self.cfg["n_head"])
                grads.update(attn_grads)
                dx1 = self._layernorm_bwd(dh1, ca["xn1"], p[f"ln1{layer}w"], ca["std1"]) + dx2
                grads[f"ln1{layer}w"] = np.sum(dh1 * ca["xn1"], axis=(0, 1))
                grads[f"ln1{layer}b"] = np.sum(dh1, axis=(0, 1))
                dxl = dx1

            grad_wte = np.zeros_like(p["wte"])
            for b in range(B):
                np.add.at(grad_wte, xb[b], dxl[b])
            grads["wte"] = grad_wte
            grad_wpe = np.sum(dxl, axis=0)[: Tmax]
            grads["wpe"] = grad_wpe

            for k in p:
                g = np.clip(grads[k], -5, 5)
                m[k] = 0.9 * m[k] + 0.1 * g
                vv[k] = 0.999 * vv[k] + 0.001 * g * g
                mhat = m[k] / (1 - 0.9**step)
                vhat = vv[k] / (1 - 0.999**step)
                p[k] -= 3e-4 * mhat / (np.sqrt(vhat) + 1e-8)

            if step % 50 == 0 or step == 1:
                print(f"[tinygpt] step {step}/{self.train_steps} loss {loss:.4f}")

        self.params = p
        self.save()
        self._train_time = time.time() - start
        print(f"[tinygpt] trained in {self._train_time:.1f}s | vocab={V} | final loss {loss:.4f}")

    def save(self):
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"vocab": self.vocab, "cfg": self.cfg}
        np.savez_compressed(self.weights_path, _meta=np.array(meta, dtype=object), **self.params)

    def load(self):
        if not self.params:
            d = np.load(self.weights_path, allow_pickle=True)
            meta = d["_meta"].item()
            self.vocab = meta["vocab"]
            self.cfg.update(meta["cfg"])
            self._vocab_index = {c: i for i, c in enumerate(self.vocab)}
            self._vp = len(self.vocab)
            self.params = {k: d[k] for k in d.files if k != "_meta"}
        return self

    # ------------------------------------------------------------------ inference
    def _encode(self, text: str) -> list[int]:
        return [self._vocab_index.get(c, 0) for c in text]

    def _decode(self, ids: list[int]) -> str:
        pad = self._vocab_index.get(" ", 0)
        out = [self.vocab[i] for i in ids if 0 <= i < len(self.vocab)]
        return "".join(out)

    def _forward_last_position(self, idx: np.ndarray) -> list[np.ndarray]:
        """Run all layers, returning logits for the final token after each layer."""
        p = self.params
        B, T = idx.shape
        C = self.cfg["n_embd"]
        H = self.cfg["n_head"]
        HD = C // H
        x = p["wte"][idx] + p["wpe"][:T]
        per_layer: list[np.ndarray] = []
        xl = x
        for layer in range(self.cfg["n_layer"]):
            h1, _, _ = self._layernorm_fwd(xl, p[f"ln1{layer}w"], p[f"ln1{layer}b"])
            out, _ = self._attn_fwd(h1, p, layer, H, HD)
            x2 = xl + out
            h2, _, _ = self._layernorm_fwd(x2, p[f"ln2{layer}w"], p[f"ln2{layer}b"])
            pre = h2 @ p[f"w1{layer}"] + p[f"b1{layer}"]
            x3 = x2 + self._gelu(pre) @ p[f"w2{layer}"] + p[f"b2{layer}"]
            xl = x3
            hf, _, _ = self._layernorm_fwd(xl[:, -1][None], p["lnfw"], p["lnfb"])
            per_layer.append((hf[0, -1] @ p["head"] + p["headb"]))
        return per_layer

    def generate(self, prompt: str, max_new: int = 64, temperature: float = 0.8,
                 early_exit: bool = True, confidence_threshold: float | None = None,
                 min_layers: int | None = None, plateau_eps: float = 0.02) -> dict:
        self.load()
        threshold = settings.EARLY_EXIT_CONFIDENCE if confidence_threshold is None else confidence_threshold
        min_layers = min_layers or max(1, self.cfg["n_layer"] // 3)
        ids = self._encode(prompt) or [self._vocab_index.get(" ", 0)]
        idx = np.array(ids[: self.cfg["block"] - max_new], dtype=np.int64).reshape(1, -1)
        rng = np.random.RandomState(settings.TINYGPT_SEED)
        gen_ids: list[int] = []
        exit_layers: list[int] = []
        confidences: list[float] = []

        for _ in range(max_new):
            ctx = idx[:, -self.cfg["block"] :]
            used = self.cfg["n_layer"]
            logits = None
            if early_exit:
                layers = self._forward_last_position(ctx)
                prev_conf = 0.0
                for li, lg in enumerate(layers):
                    conf = float(np.max(self._softmax(lg[np.newaxis, :])[0]))
                    used_candidate = li + 1
                    # primary exit: confidence sawtooth cleared
                    if conf >= threshold and used_candidate >= min_layers:
                        used, logits = used_candidate, lg
                        break
                    # secondary exit: confidence plateaued (adaptive computation)
                    if (used_candidate >= min_layers and li > 0
                            and (conf - prev_conf) < plateau_eps
                            and conf >= max(0.15, threshold - 0.3)):
                        used, logits = used_candidate, lg
                        break
                    prev_conf = conf
                if logits is None:
                    logits = layers[-1]
                    used = self.cfg["n_layer"]
            else:
                logits = self._forward_last_position(ctx)[-1]

            probs = self._softmax(logits[np.newaxis, :])[0]
            conf = float(np.max(probs))
            if temperature <= 0:
                tok = int(np.argmax(probs))
            else:
                pp = self._softmax((np.log(np.clip(probs, 1e-12, 1.0)) / temperature)[np.newaxis, :])[0]
                tok = int(rng.choice(len(pp), p=pp))
            gen_ids.append(tok)
            exit_layers.append(used)
            confidences.append(conf)
            idx = np.concatenate([idx, np.array([[tok]], dtype=np.int64)], axis=1)

        return {
            "text": self._decode(gen_ids),
            "exit_layers": exit_layers,
            "confidences": confidences,
            "mean_exit_layer": float(np.mean(exit_layers)) if exit_layers else 0.0,
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "layers_total": self.cfg["n_layer"],
            "tokens_out": len(gen_ids),
            "prompt_tokens": len(ids),
        }


_tinygpt_singleton: TinyGPT | None = None


def get_tinygpt(*, force_train: bool = False) -> TinyGPT:
    global _tinygpt_singleton
    if _tinygpt_singleton is None:
        _tinygpt_singleton = TinyGPT(force_train=force_train)
    return _tinygpt_singleton


def train_cli(steps: int | None = None) -> None:
    model = TinyGPT(force_train=True, train_steps=steps)
    _ = model  # trained and saved in __init__