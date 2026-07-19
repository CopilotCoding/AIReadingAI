"""Corpus -> tensors for the interpreter network.

Weights-as-input encoding: each hidden unit becomes one TOKEN --
[layer-role one-hot | incoming weights (padded to 3) | bias | outgoing
weight | connectivity stats]. A set encoder over unit tokens (no positional
encoding) is permutation-invariant BY CONSTRUCTION, which dissolves the
"same function, many weight orderings" problem architecturally instead of
via canonicalization or augmentation.

v0 limitation, stated: for depth-2 specimens the full inter-layer matrix
W2 is summarized per-unit (outgoing/incoming row stats), so exact
cross-layer connectivity is not visible to the interpreter. If depth-2
families underperform, that is the first suspect.
"""

import json
from pathlib import Path

import numpy as np
import torch

from interpretability_lab.corpus.generate import BASIS, LOGIC_TABLES, GENERATED_ROOT

BASIS_NAMES = list(BASIS)                     # 13 canonical term names
TASK_CLASSES = ["regression"] + list(LOGIC_TABLES) + ["none"]  # 8 classes
TOKEN_DIM = 12
GLOBAL_DIM = 6


def _stats(v: np.ndarray) -> list[float]:
    return [float(v.mean()), float(v.std()), float(v.min()), float(v.max())]


def specimen_tokens(sd: dict, arch: dict):
    """(tokens [T, TOKEN_DIM], global_feats [GLOBAL_DIM])"""
    t = arch["type"]
    toks = []
    if t == "TinyLinear":
        w = sd["out.weight"].numpy().ravel()
        b = float(sd["out.bias"])
        row = [0, 0, 1] + list(np.pad(w, (0, 3 - len(w)))) + [b, 1.0, 0, 0, 0, 0]
        toks.append(row)
        out_bias, h_total = 0.0, 0
    elif t == "TinyMLP":
        W1 = sd["hidden.weight"].numpy()
        b1 = sd["hidden.bias"].numpy()
        w2 = sd["out.weight"].numpy().ravel()
        out_bias = float(sd["out.bias"])
        for j in range(len(b1)):
            w = np.pad(W1[j], (0, 3 - W1.shape[1]))
            toks.append([1, 0, 0] + list(w) + [float(b1[j]), float(w2[j]),
                                               0, 0, 0, 0])
        h_total = len(b1)
    elif t == "TinyMLP2":
        W1 = sd["hidden1.weight"].numpy(); b1 = sd["hidden1.bias"].numpy()
        W2 = sd["hidden2.weight"].numpy(); b2 = sd["hidden2.bias"].numpy()
        w3 = sd["out.weight"].numpy().ravel()
        out_bias = float(sd["out.bias"])
        for j in range(len(b1)):
            w = np.pad(W1[j], (0, 3 - W1.shape[1]))
            toks.append([1, 0, 0] + list(w) + [float(b1[j]), 0.0]
                        + _stats(W2[:, j]))
        for k in range(len(b2)):
            toks.append([0, 1, 0] + [0.0, 0.0, 0.0] + [float(b2[k]),
                                                       float(w3[k])]
                        + _stats(W2[k, :]))
        h_total = len(b1) + len(b2)
    else:
        raise ValueError(t)
    g = [arch["in_dim"] / 3.0, h_total / 128.0,
         float(t == "TinyLinear"), float(t == "TinyMLP"),
         float(t == "TinyMLP2"), out_bias]
    return (torch.tensor(toks, dtype=torch.float32),
            torch.tensor(g, dtype=torch.float32))


def specimen_labels(meta: dict):
    """(task_class, support [13], coefs [13])"""
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
    """Returns a list of dicts, one per specimen, tensors ready."""
    items = []
    for line in (root / "manifest.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        d = root / row["path"]
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        sd = torch.load(d / "weights.pt", weights_only=True)
        tokens, gfeat = specimen_tokens(sd, meta["arch"])
        cls, support, coefs = specimen_labels(meta)
        items.append({"uid": meta["uid"], "family": meta["family"],
                      "meta": meta, "tokens": tokens, "gfeat": gfeat,
                      "cls": cls, "support": support, "coefs": coefs})
    return items


def split_by_family(items, seed=0, frac=(0.8, 0.1, 0.1)):
    """Stratified train/val/test split."""
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


def collate(batch, device="cpu"):
    T = max(len(b["tokens"]) for b in batch)
    x = torch.zeros(len(batch), T, TOKEN_DIM)
    mask = torch.ones(len(batch), T, dtype=torch.bool)   # True = padding
    for i, b in enumerate(batch):
        n = len(b["tokens"])
        x[i, :n] = b["tokens"]
        mask[i, :n] = False
    g = torch.stack([b["gfeat"] for b in batch])
    cls = torch.tensor([b["cls"] for b in batch])
    support = torch.stack([b["support"] for b in batch])
    coefs = torch.stack([b["coefs"] for b in batch])
    return (x.to(device), mask.to(device), g.to(device),
            cls.to(device), support.to(device), coefs.to(device))
