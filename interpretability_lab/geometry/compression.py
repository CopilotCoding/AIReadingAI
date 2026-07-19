"""Algorithmic compression ledger.

Tracks, for every specimen in the corpus, three tiers of description size:

  parameters  ->  effective mechanistic objects  ->  symbolic terms

'Effective mechanistic objects' is NOT nominal region count -- exp2 showed a
network with 9569 nominal regions whose gradient is constant over 98.5% of
the domain (regions measure architecture, not computation). Instead we
measure it: the minimal number of hidden units that preserves the network's
own function to R^2 >= 0.999, found by ranking units by single-ablation
impact and jointly pruning the least-impactful (rank-once joint-prune; an
upper bound on the true minimum, stated as such).

If effective objects >> symbolic terms while extraction keeps validating,
the readers are collapsing many learned geometric objects into few
conceptual operations -- the compression the project is looking for.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from interpretability_lab.models.tiny import TinyLinear, TinyMLP, TinyMLP2

CORPUS_ROOT = Path(__file__).parent.parent / "corpus" / "data"

# evaluation inputs per (experiment, task-prefix): (dim, low, high) or "binary"
DOMAINS = {
    ("exp0", "xor"): "binary",
    ("exp1", "1a_quadratic"): (1, -3, 3), ("exp1", "1b_sine"): (1, -np.pi, np.pi),
    ("exp1", "1c_abs"): (1, -3, 3),
    ("exp2", "2a_add"): (2, -2, 2), ("exp2", "2b_subtract"): (2, -2, 2),
    ("exp2", "2c_multiply"): (2, -2, 2),
    ("exp3", "3a_compose"): (3, -2, 2),
}

# symbolic degrees of freedom of the final human-readable description
SYMBOLIC_TERMS = {
    "0a_linear_1d": 2, "0b_linear_2d": 3, "xor": 3,          # 2 gates + combine
    "1a_quadratic": 3, "1b_sine": 1, "1c_abs": 1,
    "2a_add": 2, "2b_subtract": 2, "2c_multiply": 1,
    "3a_compose": 2,                                          # ac + bc
}


def _build(arch: dict):
    t = arch["type"]
    if t == "TinyLinear":
        return TinyLinear(arch["in_dim"])
    if t == "TinyMLP":
        return TinyMLP(arch["in_dim"], arch["hidden"])
    if t == "TinyMLP2":
        return TinyMLP2(arch["in_dim"], arch["h1"], arch["h2"])
    raise ValueError(t)


def _eval_inputs(spec, n=4096, seed=0):
    if spec == "binary":
        import itertools
        return torch.tensor(list(itertools.product([0.0, 1.0], repeat=2)))
    d, lo, hi = spec
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, d, generator=g) * (hi - lo) + lo


def effective_units(model, X, r2_floor=0.999):
    """Minimal hidden units preserving the model's own function to r2_floor.
    Rank units by single-ablation MSE, then binary-search the largest set of
    low-impact units that can be zeroed jointly. Returns (kept, total)."""
    relus = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.ReLU)]
    if not relus:
        return 0, 0
    with torch.no_grad():
        full = model(X).ravel()
    var = float(((full - full.mean()) ** 2).mean()) or 1e-12

    # discover widths
    widths = {}
    handles = []
    def size_hook(name):
        def fn(_m, _i, out):
            widths[name] = out.shape[1]
        return fn
    for n, m in relus:
        handles.append(m.register_forward_hook(size_hook(n)))
    with torch.no_grad():
        model(X[:1])
    for h in handles:
        h.remove()

    units = [(n, j) for n, _ in relus for j in range(widths[n])]

    def run_zeroed(zeroset):
        byname = {}
        for n, j in zeroset:
            byname.setdefault(n, []).append(j)
        hs = []
        for n, m in relus:
            if n not in byname:
                continue
            idx = byname[n]
            def fn(_m, _i, out, idx=idx):
                out = out.clone()
                out[:, idx] = 0.0
                return out
            hs.append(m.register_forward_hook(fn))
        try:
            with torch.no_grad():
                y = model(X).ravel()
        finally:
            for h in hs:
                h.remove()
        return y

    impact = []
    for u in units:
        y = run_zeroed([u])
        impact.append(float(((y - full) ** 2).mean()))
    order = np.argsort(impact)  # least impactful first

    def ok(k):  # can the k least-impactful units be removed jointly?
        if k == 0:
            return True
        y = run_zeroed([units[i] for i in order[:k]])
        return 1 - float(((y - full) ** 2).mean()) / var >= r2_floor

    lo_k, hi_k = 0, len(units)
    while lo_k < hi_k:  # find max k with ok(k)
        mid = (lo_k + hi_k + 1) // 2
        if ok(mid):
            lo_k = mid
        else:
            hi_k = mid - 1
    return len(units) - lo_k, len(units)


def build_ledger(out_dir: Path) -> list[dict]:
    rows = []
    for meta_path in sorted(CORPUS_ROOT.glob("*/*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        key = next((k for k in DOMAINS if k[0] == meta["experiment"]
                    and meta["task"].startswith(k[1])), None)
        if key is None:
            continue
        model = _build(meta["arch"])
        model.load_state_dict(torch.load(meta_path.parent / "weights.pt",
                                         weights_only=True))
        model.eval()
        X = _eval_inputs(DOMAINS[key])
        kept, total = effective_units(model, X)
        nominal = meta.get("n_regions") or meta.get("n_pieces")
        sym_terms = next((v for k2, v in SYMBOLIC_TERMS.items()
                          if meta["task"].startswith(k2) or k2 in meta["task"]), None)
        rows.append({
            "specimen": f"{meta['experiment']}/{meta['task']}",
            "params": meta["param_count"],
            "nominal_objects": nominal,
            "effective_units": kept if total else None,
            "total_units": total,
            "symbolic_terms": sym_terms,
            "params_per_term": round(meta["param_count"] / sym_terms, 1) if sym_terms else None,
            "units_per_term": round(kept / sym_terms, 1) if (sym_terms and total) else None,
            "ground_truth": meta["ground_truth"],
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ledger.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        "# Algorithmic compression ledger",
        "",
        "effective units = minimal hidden units preserving the model's own "
        "function to R^2 >= 0.999 (rank-once joint-prune; upper bound).",
        "",
        "| specimen | params | nominal objs | effective units | symbolic terms "
        "| params/term | units/term |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['specimen']} | {r['params']} | {r['nominal_objects'] or '-'} "
            f"| {r['effective_units'] if r['total_units'] else '-'}"
            f"/{r['total_units'] or '-'} | {r['symbolic_terms']} "
            f"| {r['params_per_term']} | {r['units_per_term'] or '-'} |")
    (out_dir / "ledger.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows
