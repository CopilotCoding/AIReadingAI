"""Corpus -> tensors for the interpreter network.

Weights-as-input encoding: each hidden unit becomes one TOKEN. A set encoder
over unit tokens (no positional encoding) is permutation-invariant BY
CONSTRUCTION, dissolving the "same function, many orderings" problem
architecturally.

Two lessons from the lab's own programmatic readers are baked in:

  ENGINEERED INVARIANTS -- exp1/exp2 readers work off derived per-unit
  quantities: the contribution vector w_out * w_in, the weight norm, the
  knot position -b/||w||. Tokens carry these directly rather than hoping
  the interpreter reinvents them from raw weights.

  SYMMETRY AUGMENTATION -- ReLU units have an exact function-preserving
  scaling symmetry (w1 -> a*w1, b -> a*b, w_out -> w_out/a). Training tokens
  are rebuilt each epoch under random per-unit scalings; the engineered
  features above are precisely the invariants of that symmetry, so
  augmentation teaches the reader to rely on what is real.

v0 limitation, stated: for depth-2 specimens the inter-layer matrix W2 is
summarized per-unit (row/column stats), so exact cross-layer connectivity
is not visible. If depth-2 families underperform, that is the first suspect.
"""

import json
from pathlib import Path

import numpy as np
import torch

from interpretability_lab.corpus.generate import BASIS, LOGIC_TABLES, GENERATED_ROOT

BASIS_NAMES = list(BASIS)                     # 13 canonical term names
TASK_CLASSES = ["regression"] + list(LOGIC_TABLES) + ["none"]  # 8 classes
TOKEN_DIM = 17
GLOBAL_DIM = 6
EPS = 1e-8


def _stats(v: np.ndarray) -> list[float]:
    return [float(v.mean()), float(v.std()), float(v.min()), float(v.max())]


def extract_raw(sd: dict, arch: dict) -> dict:
    t = arch["type"]
    if t == "TinyLinear":
        return {"w": sd["out.weight"].numpy().ravel().copy(),
                "b": float(sd["out.bias"])}
    if t == "TinyMLP":
        return {"W1": sd["hidden.weight"].numpy().copy(),
                "b1": sd["hidden.bias"].numpy().copy(),
                "w2": sd["out.weight"].numpy().ravel().copy(),
                "b2": float(sd["out.bias"])}
    if t == "TinyMLP2":
        return {"W1": sd["hidden1.weight"].numpy().copy(),
                "b1": sd["hidden1.bias"].numpy().copy(),
                "W2": sd["hidden2.weight"].numpy().copy(),
                "b2": sd["hidden2.bias"].numpy().copy(),
                "w3": sd["out.weight"].numpy().ravel().copy(),
                "b3": float(sd["out.bias"])}
    raise ValueError(t)


def _unit_token(role, w_in3, b, w_out, extra4):
    """role one-hot(3) | w_in(3) | b | w_out | stats(4) | contribution(3) |
    ||w_in|| | knot  -> 17 dims. contribution & knot are the ReLU-scaling
    invariants the mechanistic readers use."""
    norm = float(np.linalg.norm(w_in3))
    contrib = (w_out * np.asarray(w_in3)).tolist()
    knot = float(np.clip(-b / (norm + EPS), -10, 10))
    return list(role) + list(w_in3) + [b, w_out] + list(extra4) \
        + contrib + [norm, knot]


def tokens_from_raw(raw: dict, arch: dict, rng=None):
    """Build tokens; if rng given, apply random per-unit ReLU scaling
    augmentation (function-preserving)."""
    t = arch["type"]
    toks = []

    def alpha(n):
        if rng is None:
            return np.ones(n)
        return np.exp(rng.uniform(-0.7, 0.7, size=n))  # ~[0.5, 2]

    if t == "TinyLinear":
        w = np.pad(raw["w"], (0, 3 - len(raw["w"])))
        toks.append(_unit_token((0, 0, 1), w, raw["b"], 1.0, (0, 0, 0, 0)))
        out_bias, h_total = 0.0, 0
    elif t == "TinyMLP":
        h = len(raw["b1"])
        a = alpha(h)
        for j in range(h):
            w = np.pad(raw["W1"][j] * a[j], (0, 3 - raw["W1"].shape[1]))
            toks.append(_unit_token((1, 0, 0), w, float(raw["b1"][j] * a[j]),
                                    float(raw["w2"][j] / a[j]), (0, 0, 0, 0)))
        out_bias, h_total = raw["b2"], h
    elif t == "TinyMLP2":
        h1, h2 = len(raw["b1"]), len(raw["b2"])
        a = alpha(h1)
        bta = alpha(h2)
        W2 = raw["W2"] / a[None, :]          # undo L1 scaling on its columns
        W2 = W2 * bta[:, None]               # apply L2 scaling on its rows
        for j in range(h1):
            w = np.pad(raw["W1"][j] * a[j], (0, 3 - raw["W1"].shape[1]))
            col = W2[:, j]
            w_out_eff = float(col.mean())
            toks.append(_unit_token((1, 0, 0), w, float(raw["b1"][j] * a[j]),
                                    w_out_eff, _stats(col)))
        for k in range(h2):
            row = W2[k, :]
            toks.append(_unit_token((0, 1, 0), (0.0, 0.0, 0.0),
                                    float(raw["b2"][k] * bta[k]),
                                    float(raw["w3"][k] / bta[k]), _stats(row)))
        out_bias, h_total = raw["b3"], h1 + h2
    else:
        raise ValueError(t)

    g = [arch["in_dim"] / 3.0, h_total / 128.0,
         float(t == "TinyLinear"), float(t == "TinyMLP"),
         float(t == "TinyMLP2"), out_bias]
    return (torch.tensor(np.array(toks, dtype=np.float32)),
            torch.tensor(g, dtype=torch.float32))


def specimen_labels(meta: dict):
    support = torch.zeros(len(BASIS_NAMES))
    coefs = torch.zeros(len(BASIS_NAMES))
    if meta["task_type"] == "none":
        cls = TASK_CLASSES.index("none")
    elif meta["task_type"] == "binary_classification":
        cls = TASK_CLASSES.index(meta["rule"]["logic"])
    else:
        cls = 0
        for coef, name in meta["rule"]["terms"]:
            i = BASIS_NAMES.index(name)
            support[i] = 1.0
            coefs[i] = float(coef)
    return cls, support, coefs


def load_corpus(root: Path = GENERATED_ROOT):
    items = []
    for line in (root / "manifest.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        d = root / row["path"]
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        sd = torch.load(d / "weights.pt", weights_only=True)
        raw = extract_raw(sd, meta["arch"])
        tokens, gfeat = tokens_from_raw(raw, meta["arch"])
        cls, support, coefs = specimen_labels(meta)
        items.append({"uid": meta["uid"], "family": meta["family"],
                      "meta": meta, "raw": raw, "tokens": tokens,
                      "gfeat": gfeat, "cls": cls, "support": support,
                      "coefs": coefs})
    return items


def split_by_family(items, seed=0, frac=(0.8, 0.1, 0.1)):
    rng = np.random.default_rng(seed)
    by_fam = {}
    for it in items:
        by_fam.setdefault(it["family"], []).append(it)
    splits = {"train": [], "val": [], "test": []}
    for fam, lst in by_fam.items():
        idx = rng.permutation(len(lst))
        n_tr = int(frac[0] * len(lst))
        n_va = int(frac[1] * len(lst))
        for i in idx[:n_tr]:
            splits["train"].append(lst[i])
        for i in idx[n_tr:n_tr + n_va]:
            splits["val"].append(lst[i])
        for i in idx[n_tr + n_va:]:
            splits["test"].append(lst[i])
    return splits


def collate(batch, device="cpu", rng=None):
    """rng != None applies symmetry augmentation (training only)."""
    toks = []
    for b in batch:
        if rng is not None:
            t, _ = tokens_from_raw(b["raw"], b["meta"]["arch"], rng)
        else:
            t = b["tokens"]
        toks.append(t)
    T = max(len(t) for t in toks)
    x = torch.zeros(len(batch), T, TOKEN_DIM)
    mask = torch.ones(len(batch), T, dtype=torch.bool)   # True = padding
    for i, t in enumerate(toks):
        x[i, :len(t)] = t
        mask[i, :len(t)] = False
    g = torch.stack([b["gfeat"] for b in batch])
    cls = torch.tensor([b["cls"] for b in batch])
    support = torch.stack([b["support"] for b in batch])
    coefs = torch.stack([b["coefs"] for b in batch])
    return (x.to(device), mask.to(device), g.to(device),
            cls.to(device), support.to(device), coefs.to(device))
