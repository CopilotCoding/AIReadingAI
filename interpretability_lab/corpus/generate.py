"""Corpus v2 generator: mass-produce (trained network, ground-truth rule)
pairs as training data for the Phase 3 interpreter.

Design decisions (from the corpus audit):
  - CANONICAL rule language: rules are structured term lists over a fixed
    basis registry with canonical variables x0, x1, x2. The human-readable
    string is DERIVED from the structure, never hand-written, so the target
    language is uniform across the whole corpus.
  - SELF-CONTAINED metas: domain, input dim, task type, arch constructor
    kwargs all stored per specimen; no experiment code needed to use one.
  - MANY instances per rule family, random seeds/widths/coefficients --
    an interpreter must learn "many weight configurations -> same rule",
    not memorize one specimen per rule.
  - REFUSAL class: untrained networks saved with rule = null. An interpreter
    that cannot say "no rule here" is worthless.
  - Quality gate: regression specimens must fit their rule to R^2 >= 0.99
    (recorded), logic specimens must be exactly correct; failures retry a
    fresh seed once, then are skipped and counted.

The old corpus/data/ specimens remain as pinned experiment subjects but are
not part of this training set.

Usage:  python -m interpretability_lab.corpus.generate --n 300
"""

import argparse
import hashlib
import itertools
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch

from interpretability_lab.models.tiny import (TinyLinear, TinyMLP, TinyMLP2,
                                              param_count, train_full_batch)

GENERATED_ROOT = Path(__file__).parent / "generated"

# Fixed basis registry -- the corpus's entire rule language. Term names are
# canonical; the interpreter's output vocabulary is exactly these names.
BASIS = {
    "1":       lambda X: torch.ones(len(X)),
    "x0":      lambda X: X[:, 0],
    "x1":      lambda X: X[:, 1],
    "x2":      lambda X: X[:, 2],
    "x0^2":    lambda X: X[:, 0] ** 2,
    "x0^3":    lambda X: X[:, 0] ** 3,
    "x1^2":    lambda X: X[:, 1] ** 2,
    "sin(x0)": lambda X: torch.sin(X[:, 0]),
    "cos(x0)": lambda X: torch.cos(X[:, 0]),
    "|x0|":    lambda X: torch.abs(X[:, 0]),
    "x0*x1":   lambda X: X[:, 0] * X[:, 1],
    "x0*x2":   lambda X: X[:, 0] * X[:, 2],
    "x1*x2":   lambda X: X[:, 1] * X[:, 2],
}

LOGIC_TABLES = {  # over inputs (0,0),(0,1),(1,0),(1,1)
    "XOR": [0, 1, 1, 0], "XNOR": [1, 0, 0, 1], "AND": [0, 0, 0, 1],
    "OR": [0, 1, 1, 1], "NAND": [1, 1, 1, 0], "NOR": [1, 0, 0, 0],
}


def rule_to_string(terms):
    parts = []
    for coef, name in terms:
        mag = f"{abs(coef):g}"
        body = mag if name == "1" else (f"{mag}*{name}" if mag != "1" else name)
        parts.append(("- " if coef < 0 else "+ ") + body)
    s = " ".join(parts)
    return "y = " + (s[2:] if s.startswith("+ ") else "-" + s[2:])


def eval_terms(terms, X):
    y = torch.zeros(len(X))
    for coef, name in terms:
        y += coef * BASIS[name](X)
    return y


def _coef(rng, lo=0.5, hi=6.0, nd=2):
    c = rng.uniform(lo, hi) * rng.choice([-1, 1])
    return round(float(c), nd)


# ---- rule samplers: each returns (terms, input_dim, domain) -----------------

def sample_linear(rng):
    d = int(rng.integers(1, 4))
    terms = [(_coef(rng), f"x{i}") for i in range(d)]
    if rng.random() < 0.8:
        terms.append((_coef(rng), "1"))
    return terms, d, (-3.0, 3.0)


def sample_poly(rng):
    deg = int(rng.integers(2, 4))
    names = ["1", "x0", "x0^2", "x0^3"][:deg + 1]
    keep = [n for n in names if n in ("x0^2", "x0^3")[deg - 2:] or rng.random() < 0.7]
    if f"x0^{deg}" not in keep:
        keep.append(f"x0^{deg}")
    terms = [(_coef(rng, 0.5, 4.0), n) for n in keep]
    return terms, 1, (-3.0, 3.0)


def sample_trig(rng):
    name = "sin(x0)" if rng.random() < 0.5 else "cos(x0)"
    terms = [(_coef(rng, 0.5, 3.0), name)]
    if rng.random() < 0.4:
        terms.append((_coef(rng, 0.5, 2.0), "1"))
    return terms, 1, (-float(np.pi), float(np.pi))


def sample_abs(rng):
    terms = [(_coef(rng, 0.5, 3.0), "|x0|")]
    if rng.random() < 0.4:
        terms.append((_coef(rng, 0.5, 2.0), "1"))
    return terms, 1, (-3.0, 3.0)


def sample_product(rng):
    terms = [(_coef(rng, 0.5, 3.0), "x0*x1")]
    if rng.random() < 0.3:
        terms.append((_coef(rng), "x0"))
    return terms, 2, (-2.0, 2.0)


def sample_compose(rng):
    a = _coef(rng, 0.5, 2.0)          # a*(x0+x1)*x2 = a*x0*x2 + a*x1*x2
    return [(a, "x0*x2"), (a, "x1*x2")], 3, (-2.0, 2.0)


REGRESSION_FAMILIES = {
    "linear": sample_linear, "poly": sample_poly, "trig": sample_trig,
    "abs": sample_abs, "product": sample_product, "compose": sample_compose,
}


# ---- arch samplers ----------------------------------------------------------

def sample_arch(rng, family, input_dim):
    if family == "compose":
        h = int(rng.choice([32, 48, 64]))
        return TinyMLP2(input_dim, h, h), {"type": "TinyMLP2", "in_dim": input_dim,
                                           "h1": h, "h2": h}
    if family == "linear" and rng.random() < 0.3:
        return TinyLinear(input_dim), {"type": "TinyLinear", "in_dim": input_dim}
    h = int(rng.choice([16, 32, 64]))
    return TinyMLP(input_dim, h), {"type": "TinyMLP", "in_dim": input_dim,
                                   "hidden": h}


# ---- specimen builders ------------------------------------------------------

def build_regression(rng, family, dev):
    terms, d, dom = REGRESSION_FAMILIES[family](rng)
    for attempt in range(2):
        seed = int(rng.integers(0, 2 ** 31))
        torch.manual_seed(seed)
        model, arch = sample_arch(rng, family, d)
        g = torch.Generator().manual_seed(seed)
        X = torch.rand(2048, d, generator=g) * (dom[1] - dom[0]) + dom[0]
        Y = eval_terms(terms, X).unsqueeze(1)
        epochs = 3000 if arch["type"] == "TinyMLP2" else \
            (800 if arch["type"] == "TinyLinear" else 2000)
        train_full_batch(model, X, Y, epochs=epochs, lr=0.01, device=dev)
        Xt = torch.rand(1024, d, generator=g) * (dom[1] - dom[0]) + dom[0]
        Yt = eval_terms(terms, Xt)
        with torch.no_grad():
            pred = model(Xt).ravel()
        r2 = 1 - float(((Yt - pred) ** 2).sum()) / float(((Yt - Yt.mean()) ** 2).sum())
        if r2 >= 0.99:
            meta = {"schema": 2, "family": family, "task_type": "regression",
                    "rule": {"terms": [[c, n] for c, n in terms]},
                    "rule_str": rule_to_string(terms), "input_dim": d,
                    "domain": list(dom), "arch": arch, "seed": seed,
                    "fit_r2": r2}
            return model, meta
    return None, None


def build_logic(rng, dev):
    gate_name = str(rng.choice(list(LOGIC_TABLES)))
    table = LOGIC_TABLES[gate_name]
    X = torch.tensor(list(itertools.product([0.0, 1.0], repeat=2)))
    Y = torch.tensor([[float(v)] for v in table])
    for attempt in range(5):
        seed = int(rng.integers(0, 2 ** 31))
        torch.manual_seed(seed)
        h = int(rng.choice([2, 3, 4]))
        model = TinyMLP(2, h)
        train_full_batch(model, X, Y, epochs=4000, lr=0.05)
        with torch.no_grad():
            bits = (model(X).ravel() > 0.5).to(torch.int64).tolist()
        if bits == table:
            meta = {"schema": 2, "family": "logic",
                    "task_type": "binary_classification",
                    "rule": {"logic": gate_name, "truth_table": table},
                    "rule_str": f"y = {gate_name}(x0,x1)", "input_dim": 2,
                    "domain": "binary",
                    "arch": {"type": "TinyMLP", "in_dim": 2, "hidden": h},
                    "seed": seed, "fit_r2": 1.0}
            return model, meta
    return None, None


def build_untrained(rng):
    seed = int(rng.integers(0, 2 ** 31))
    torch.manual_seed(seed)
    d = int(rng.integers(1, 4))
    if rng.random() < 0.3:
        h = int(rng.choice([32, 48]))
        model, arch = TinyMLP2(d, h, h), {"type": "TinyMLP2", "in_dim": d,
                                          "h1": h, "h2": h}
    else:
        h = int(rng.choice([16, 32, 64]))
        model, arch = TinyMLP(d, h), {"type": "TinyMLP", "in_dim": d, "hidden": h}
    meta = {"schema": 2, "family": "none", "task_type": "none", "rule": None,
            "rule_str": None, "input_dim": d, "domain": [-3.0, 3.0],
            "arch": arch, "seed": seed, "fit_r2": None}
    return model, meta


def build_specimen(family, task_seed):
    """Worker-process entry: build one specimen on CPU (at this model size a
    CPU core beats GPU kernel-launch overhead; parallelism comes from running
    many workers). Returns (state_dict, meta) or None on quality-gate failure."""
    torch.set_num_threads(1)
    rng = np.random.default_rng(task_seed)
    if family == "none":
        model, meta = build_untrained(rng)
    elif family == "logic":
        model, meta = build_logic(rng, None)
    else:
        model, meta = build_regression(rng, family, None)
    if model is None:
        return None
    meta["task_seed"] = task_seed
    return model.state_dict(), meta


# ---- progress UI ------------------------------------------------------------

class _PlainUI:
    def __init__(self, total, t0):
        self.total, self.t0 = total, t0

    def update(self, made, counts, skips, last_meta):
        if made % 20 == 0:
            rate = made / max(time.time() - self.t0, 1e-9)
            print(f"  {made}/{self.total} specimens ({rate:.1f}/s, "
                  f"{skips} skipped) {counts}", flush=True)

    def close(self):
        pass


class _RichUI:
    def __init__(self, total, t0):
        from rich.console import Group
        from rich.live import Live
        from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                                   SpinnerColumn, TextColumn,
                                   TimeElapsedColumn, TimeRemainingColumn)
        from rich.table import Table
        self._Group, self._Table = Group, Table
        self.t0 = t0
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]corpus v2[/]"),
            BarColumn(bar_width=32),
            MofNCompleteColumn(),
            TextColumn("[green]{task.fields[rate]:.1f}/s[/]"),
            TimeElapsedColumn(),
            TextColumn("eta"),
            TimeRemainingColumn(),
        )
        self.task = self.progress.add_task("gen", total=total, rate=0.0)
        self.live = Live(self.progress, refresh_per_second=4)
        self.live.start()

    def _table(self, counts, skips, last_meta):
        t = self._Table(show_edge=False, pad_edge=False, box=None)
        t.add_column("family", style="bold")
        t.add_column("count", justify="right")
        for fam in sorted(counts):
            t.add_row(fam, str(counts[fam]))
        t.add_row("[dim]skipped[/]", f"[dim]{skips}[/]")
        if last_meta is not None:
            rule = last_meta["rule_str"] or "[dim](untrained -- refusal class)[/]"
            t.add_row("[dim]latest[/]",
                      f"[dim]{last_meta['arch']['type']} <- {rule}[/]")
        return t

    def update(self, made, counts, skips, last_meta):
        rate = made / max(time.time() - self.t0, 1e-9)
        self.progress.update(self.task, completed=made, rate=rate)
        self.live.update(self._Group(self.progress,
                                     self._table(counts, skips, last_meta)))

    def close(self):
        self.live.stop()


def make_ui(total, t0):
    try:
        return _RichUI(total, t0)
    except Exception:
        return _PlainUI(total, t0)


# ---- main -------------------------------------------------------------------

def weights_hash(state_dict) -> str:
    """Content hash of a state dict: two specimens are true duplicates iff
    this collides. Checked against the manifest so duplicates cannot enter
    the corpus even if the generator is rerun with a reused seed."""
    h = hashlib.sha256()
    for k, v in sorted(state_dict.items()):
        h.update(k.encode())
        h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main(n_total: int, seed=None, workers=None):
    if seed is None:  # entropy by default so reruns never replay the stream;
        seed = int.from_bytes(os.urandom(4), "little")  # recorded per specimen
    if workers is None:
        workers = max(4, (os.cpu_count() or 8) - 4)
    print(f"generator stream seed: {seed}  ({workers} worker processes)")
    rng = np.random.default_rng(seed)
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = GENERATED_ROOT / "manifest.jsonl"
    existing, seen_hashes = 0, set()
    if manifest_path.exists():
        for line in manifest_path.open(encoding="utf-8"):
            existing += 1
            row = json.loads(line)
            if "whash" in row:
                seen_hashes.add(row["whash"])

    # ~72% regression across 6 families, ~14% logic, ~14% refusal class
    schedule = (list(REGRESSION_FAMILIES) * 6 + ["logic"] * 5 + ["none"] * 5)
    counts, skips = {}, 0
    t0 = time.time()
    made = 0
    ui = make_ui(n_total, t0)
    ex = ProcessPoolExecutor(max_workers=workers)
    pending = set()
    i = 0

    def submit_one():
        nonlocal i
        fam = schedule[i % len(schedule)]
        i += 1
        pending.add(ex.submit(build_specimen, fam, int(rng.integers(0, 2 ** 63))))

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for _ in range(workers * 2):   # keep the pool saturated
            submit_one()
        while made < n_total:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                if made >= n_total:
                    break
                res = fut.result()
                submit_one()
                if res is None:
                    skips += 1
                    ui.update(made, counts, skips, None)
                    continue
                sd, meta = res
                wh = weights_hash(sd)
                if wh in seen_hashes:  # true duplicate: refuse
                    skips += 1
                    ui.update(made, counts, skips, None)
                    continue
                seen_hashes.add(wh)
                uid = f"{meta['family']}_{existing + made:06d}"
                d = GENERATED_ROOT / meta["family"] / uid
                d.mkdir(parents=True, exist_ok=True)
                torch.save(sd, d / "weights.pt")
                meta.update({"uid": uid,
                             "param_count": sum(int(v.numel()) for v in sd.values()),
                             "generator_seed": seed, "whash": wh,
                             "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")})
                (d / "meta.json").write_text(json.dumps(meta, indent=2),
                                             encoding="utf-8")
                manifest.write(json.dumps(
                    {"uid": uid, "path": f"{meta['family']}/{uid}",
                     "family": meta["family"], "rule_str": meta["rule_str"],
                     "input_dim": meta["input_dim"],
                     "param_count": meta["param_count"], "whash": wh}) + "\n")
                manifest.flush()  # durability: a killed run must not orphan dirs
                counts[meta["family"]] = counts.get(meta["family"], 0) + 1
                made += 1
                ui.update(made, counts, skips, meta)

    ex.shutdown(wait=False, cancel_futures=True)
    ui.close()
    dt = time.time() - t0
    print(f"\nDONE: {made} specimens in {dt / 60:.1f} min "
          f"({skips} skips: quality gate or duplicate hash)")
    for fam, c in sorted(counts.items()):
        print(f"  {fam:<10} {c}")

    # uniqueness report over the WHOLE corpus, not just this run
    rules, hashes, total = {}, set(), 0
    for line in manifest_path.open(encoding="utf-8"):
        row = json.loads(line)
        total += 1
        hashes.add(row.get("whash", row["uid"]))
        if row["rule_str"]:
            rules[row["rule_str"]] = rules.get(row["rule_str"], 0) + 1
    repeated = {r: c for r, c in rules.items() if c > 1}
    print(f"\nuniqueness report ({total} specimens in corpus):")
    print(f"  unique weight hashes: {len(hashes)}/{total} "
          f"({'OK -- no duplicate networks' if len(hashes) == total else 'DUPLICATES PRESENT'})")
    print(f"  distinct rules: {len(rules)}; rules with >1 network: "
          f"{len(repeated)} (desirable: same rule, different weights)")
    print(f"corpus root: {GENERATED_ROOT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG stream seed; default = fresh entropy (recorded)")
    ap.add_argument("--workers", type=int, default=None,
                    help="worker processes; default = CPU count - 4")
    args = ap.parse_args()
    main(args.n, args.seed, args.workers)
