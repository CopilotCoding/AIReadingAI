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


def _assemble(role_idx, W_in, b, w_out, extra4):
    """Vectorized token block: role one-hot(3) | w_in(3) | b | w_out |
    stats(4) | contribution(3) | ||w_in|| | knot  -> (n, 17).
    contribution & knot are the ReLU-scaling invariants the mechanistic
    readers use."""
    n = len(b)
    role = np.zeros((n, 3))
    role[:, role_idx] = 1.0
    W3 = np.zeros((n, 3))
    if W_in.shape[1]:
        W3[:, :W_in.shape[1]] = W_in
    norm = np.linalg.norm(W3, axis=1)
    contrib = w_out[:, None] * W3
    knot = np.clip(-b / (norm + EPS), -10, 10)
    return np.concatenate([role, W3, b[:, None], w_out[:, None], extra4,
                           contrib, norm[:, None], knot[:, None]], axis=1)


def _rowstats(M):
    return np.stack([M.mean(1), M.std(1), M.min(1), M.max(1)], axis=1)


def tokens_from_raw(raw: dict, arch: dict, rng=None):
    """Build tokens (vectorized); if rng given, apply random per-unit ReLU
    scaling augmentation (function-preserving symmetry)."""
    t = arch["type"]

    def alpha(n):
        if rng is None:
            return np.ones(n)
        return np.exp(rng.uniform(-0.7, 0.7, size=n))  # ~[0.5, 2]

    if t == "TinyLinear":
        toks = _assemble(2, raw["w"][None, :], np.array([raw["b"]]),
                         np.array([1.0]), np.zeros((1, 4)))
        out_bias, h_total = 0.0, 0
    elif t == "TinyMLP":
        a = alpha(len(raw["b1"]))
        toks = _assemble(0, raw["W1"] * a[:, None], raw["b1"] * a,
                         raw["w2"] / a, np.zeros((len(a), 4)))
        out_bias, h_total = raw["b2"], len(a)
    elif t == "TinyMLP2":
        a = alpha(len(raw["b1"]))
        bta = alpha(len(raw["b2"]))
        W2 = (raw["W2"] / a[None, :]) * bta[:, None]
        t1 = _assemble(0, raw["W1"] * a[:, None], raw["b1"] * a,
                       W2.mean(0), _rowstats(W2.T))
        t2 = _assemble(1, np.zeros((len(bta), 0)), raw["b2"] * bta,
                       raw["w3"] / bta, _rowstats(W2))
        toks = np.concatenate([t1, t2], axis=0)
        out_bias, h_total = raw["b3"], len(a) + len(bta)
    else:
        raise ValueError(t)

    g = [arch["in_dim"] / 3.0, h_total / 128.0,
         float(t == "TinyLinear"), float(t == "TinyMLP"),
         float(t == "TinyMLP2"), out_bias]
    return (torch.tensor(toks.astype(np.float32)),
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


def split_holdout_families(items, holdout, seed=0, val_frac=0.1):
    """Family-generalization split: `holdout` families go ENTIRELY to test;
    the interpreter never trains on them. Remaining families split train/val.
    Tests whether coefficient reading generalizes to unseen function classes.
    The refusal ('none') family always stays in train (it is not a rule class).
    """
    rng = np.random.default_rng(seed)
    holdout = set(holdout)
    train, val, test = [], [], []
    by_fam = {}
    for it in items:
        by_fam.setdefault(it["family"], []).append(it)
    for fam, lst in by_fam.items():
        if fam in holdout:
            test.extend(lst)
            continue
        idx = rng.permutation(len(lst))
        n_va = int(val_frac * len(lst))
        for i in idx[:n_va]:
            val.append(lst[i])
        for i in idx[n_va:]:
            train.append(lst[i])
    return {"train": train, "val": val, "test": test}


def precompute_tensors(items, n_aug=8, seed=0, device="cpu"):
    """Tokenize every specimen ONCE into `n_aug` augmented variants (plus the
    clean one), pad to a global max length, and stack into GPU tensors. Training
    then indexes into this pool instead of re-tokenizing on CPU every epoch --
    the fix for the augmentation being the bottleneck at corpus scale.

    Returns a dict of stacked tensors; variant 0 is un-augmented (for eval).
    """
    rng = np.random.default_rng(seed)
    per_item = []           # list of (V, Ti, TOKEN_DIM) tensors
    maxT = 0
    for it in items:
        variants = [it["tokens"]]
        for _ in range(n_aug):
            t, _ = tokens_from_raw(it["raw"], it["meta"]["arch"], rng)
            variants.append(t)
        stk = torch.stack(variants)          # (V, Ti, D)  (Ti same within item)
        per_item.append(stk)
        maxT = max(maxT, stk.shape[1])
    N, V = len(items), n_aug + 1
    X = torch.zeros(N, V, maxT, TOKEN_DIM)
    mask = torch.ones(N, maxT, dtype=torch.bool)
    for i, stk in enumerate(per_item):
        Ti = stk.shape[1]
        X[i, :, :Ti] = stk
        mask[i, :Ti] = False
    pool = {
        "X": X.to(device), "mask": mask.to(device),
        "gfeat": torch.stack([it["gfeat"] for it in items]).to(device),
        "cls": torch.tensor([it["cls"] for it in items]).to(device),
        "support": torch.stack([it["support"] for it in items]).to(device),
        "coefs": torch.stack([it["coefs"] for it in items]).to(device),
        "n_variants": V,
    }
    return pool


def pool_batch(pool, idx, augment=True, gen=None):
    """Gather one batch from a precomputed pool. Picks a random augmentation
    variant per sample when augment=True (all on-device, no CPU work)."""
    idx = torch.as_tensor(idx, device=pool["X"].device)
    if augment and pool["n_variants"] > 1:
        v = torch.randint(1, pool["n_variants"], (len(idx),), device=idx.device)
    else:
        v = torch.zeros(len(idx), dtype=torch.long, device=idx.device)
    x = pool["X"][idx, v]                     # (B, maxT, D)
    mask = pool["mask"][idx]
    return (x, mask, pool["gfeat"][idx], pool["cls"][idx],
            pool["support"][idx], pool["coefs"][idx])


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
