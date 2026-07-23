"""EXPERIMENT 11 -- Interpreter v1: read the NUMBERS from weights, properly.

exp6 (v0) read structure from unseen weights (task 0.96, support 0.81) but its
PURE weights-only coefficient reading FAILED its gates (median |err| 0.43,
functional rel-RMSE 0.27). Two suspected causes: (1) data starvation (~1.2K
training specimens) and (2) a pooled coefficient head that averaged away the
per-term signal. exp11 attacks BOTH: a larger corpus and a per-term ATTENTION
readout (interpreter/model_v1.py), and evaluates on TWO held-out splits:

  A. UNSEEN NETWORKS   the original pure gates -- coefficient median |err|
     <= 0.15 and functional rel-RMSE <= 0.10 on nets never seen. Passing means
     the interpreter reads numbers from weights with NO behavioral calibration.

  B. UNSEEN FAMILIES   two whole function families held out of training; test
     coefficient reading on those classes. This is the generalization test
     rarely shown: does weight-reading transfer to unseen function classes?

Run:  python -m interpretability_lab.experiments.exp11_interpreter_v1
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from interpretability_lab.corpus.generate import BASIS, eval_terms
from interpretability_lab.geometry.compression import _build
from interpretability_lab.interpreter.dataset import (BASIS_NAMES, TASK_CLASSES,
                                                      load_corpus, pool_batch,
                                                      precompute_tensors,
                                                      split_by_family,
                                                      split_holdout_families)
from interpretability_lab.interpreter.model_v1 import InterpreterV1

RESULTS = Path(__file__).parent / "results" / "exp11"
CKPT = Path(__file__).parent.parent / "interpreter" / "checkpoints"
CORPUS_GEN = Path(__file__).parent.parent / "corpus" / "generated"
EPOCHS = 250
BATCH = 128
N_AUG = 8
HOLDOUT_FAMILIES = ["poly", "trig"]     # unseen-family test classes


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return {"gate": name, "passed": bool(ok), "detail": detail}


def pool_batches(pool, n, rng, augment=True):
    idx = rng.permutation(n) if augment else np.arange(n)
    for s in range(0, n, BATCH):
        yield pool_batch(pool, idx[s:s + BATCH], augment=augment)


def loss_fn(task_lg, sup_lg, coef_pred, cls, support, coefs):
    l_task = F.cross_entropy(task_lg, cls)
    reg = cls == 0
    if reg.any():
        l_sup = F.binary_cross_entropy_with_logits(sup_lg[reg], support[reg])
        m = support[reg] > 0.5
        l_coef = F.huber_loss(coef_pred[reg][m], coefs[reg][m]) if m.any() \
            else torch.zeros((), device=task_lg.device)
    else:
        l_sup = l_coef = torch.zeros((), device=task_lg.device)
    return l_task + l_sup + l_coef


def train_model(splits, dev, tag):
    torch.manual_seed(0)
    print(f"    [{tag}] precomputing augmented token pool (train "
          f"{len(splits['train'])}, {N_AUG} variants each)...", flush=True)
    tr_pool = precompute_tensors(splits["train"], n_aug=N_AUG, device=dev)
    va_pool = precompute_tensors(splits["val"], n_aug=0, device=dev)
    model = InterpreterV1().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    rng = np.random.default_rng(1)
    n_tr, n_va = len(splits["train"]), len(splits["val"])
    best = (float("inf"), None)
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        for b in pool_batches(tr_pool, n_tr, rng, augment=True):
            loss = loss_fn(*model(b[0], b[1], b[2]), b[3], b[4], b[5])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                vl = nb = 0
                for b in pool_batches(va_pool, n_va, rng, augment=False):
                    vl += float(loss_fn(*model(b[0], b[1], b[2]), b[3], b[4], b[5])); nb += 1
            vl /= max(nb, 1)
            if vl < best[0]:
                best = (vl, {k: v.cpu().clone() for k, v in model.state_dict().items()})
            print(f"    [{tag}] epoch {ep+1:>3}: val {vl:.4f}"
                  f"{'  *' if vl == best[0] else ''}  "
                  f"({(time.time()-t0):.0f}s)", flush=True)
    model.load_state_dict(best[1]); model.to(dev)
    print(f"    [{tag}] {n_params:,} params, trained {(time.time()-t0)/60:.1f} min, "
          f"best val {best[0]:.4f}")
    return model, n_params


@torch.no_grad()
def evaluate(model, items, dev):
    model.eval()
    pool = precompute_tensors(items, n_aug=0, device=dev)
    out = {"cls_t": [], "cls_p": [], "sup_t": [], "sup_p": [],
           "coef_t": [], "coef_p": []}
    for b in pool_batches(pool, len(items), np.random.default_rng(0), augment=False):
        x, mask, g, cls, support, coefs = b
        task_lg, sup_lg, coef_pred = model(x, mask, g)
        out["cls_t"] += cls.tolist(); out["cls_p"] += task_lg.argmax(1).tolist()
        out["sup_t"].append(support.cpu()); out["sup_p"].append((torch.sigmoid(sup_lg) > 0.5).float().cpu())
        out["coef_t"].append(coefs.cpu()); out["coef_p"].append(coef_pred.cpu())
    for k in ("sup_t", "sup_p", "coef_t", "coef_p"):
        out[k] = torch.cat(out[k])
    return out


def pure_metrics(items, ev):
    """Pure weights-only coefficient + functional error on regression items."""
    cls_t, cls_p = np.array(ev["cls_t"]), np.array(ev["cls_p"])
    reg = (cls_t == 0) & (cls_p == 0)
    hit = (ev["sup_t"][reg] > 0.5) & (ev["sup_p"][reg] > 0.5)
    errs = (ev["coef_p"][reg] - ev["coef_t"][reg]).abs()[hit]
    coef_err = float(errs.median()) if len(errs) else float("nan")
    sup_rate = float((ev["sup_p"][reg] == ev["sup_t"][reg]).all(1).float().mean()) \
        if reg.any() else float("nan")

    # functional: evaluate the predicted rule against the actual network
    reg_items = [it for it, t, p in zip(items, ev["cls_t"], ev["cls_p"])
                 if t == 0 and p == 0]
    idxs = [i for i, (t, p) in enumerate(zip(ev["cls_t"], ev["cls_p"]))
            if t == 0 and p == 0]
    rels = []
    for it, gi in zip(reg_items, idxs):
        meta = it["meta"]
        m = _build(meta["arch"])
        m.load_state_dict(torch.load(CORPUS_GEN / meta["family"] / meta["uid"]
                                     / "weights.pt", weights_only=True))
        m.eval()
        lo, hi = meta["domain"]
        g = torch.Generator().manual_seed(7)
        X = torch.rand(2048, meta["input_dim"], generator=g) * (hi - lo) + lo
        with torch.no_grad():
            y_net = m(X).ravel()
        names = [BASIS_NAMES[j] for j in range(len(BASIS_NAMES))
                 if ev["sup_p"][gi, j] > 0.5]

        def needs(n):
            return 3 if "x2" in n else (2 if "x1" in n else 1)
        if any(needs(n) > meta["input_dim"] for n in names):
            rels.append(float("inf")); continue
        terms = [(float(ev["coef_p"][gi, BASIS_NAMES.index(n)]), n) for n in names]
        y_rule = eval_terms(terms, X) if terms else torch.zeros_like(y_net)
        rels.append(float(torch.sqrt(((y_rule - y_net) ** 2).mean())
                          / y_net.std().clamp(min=1e-9)))
    rels = np.array(rels)
    return {"coef_err": coef_err, "support": sup_rate,
            "func_median": float(np.median(rels)) if len(rels) else float("nan"),
            "residue": float((rels > 0.2).mean()) if len(rels) else float("nan"),
            "n": int(reg.sum()), "errs": errs.numpy(),
            "coef_t": ev["coef_t"][reg][hit].numpy(),
            "coef_p": ev["coef_p"][reg][hit].numpy(), "rels": rels}


def per_family_coef(items, ev):
    cls_t, cls_p = np.array(ev["cls_t"]), np.array(ev["cls_p"])
    fams = {}
    for i, it in enumerate(items):
        if cls_t[i] != 0 or cls_p[i] != 0:
            continue
        hit = (ev["sup_t"][i] > 0.5) & (ev["sup_p"][i] > 0.5)
        e = (ev["coef_p"][i] - ev["coef_t"][i]).abs()[hit]
        if len(e):
            fams.setdefault(it["family"], []).extend(e.tolist())
    return {f: float(np.median(v)) for f, v in fams.items()}


def main():
    print("=" * 70)
    print("EXPERIMENT 11: interpreter v1 -- reading numbers from weights properly")
    print("=" * 70)
    RESULTS.mkdir(parents=True, exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    items = load_corpus()
    print(f"\n  corpus: {len(items)} specimens")
    gates = []

    # ---------- A. unseen NETWORKS (pure gates) ----------
    print("\n[A] unseen-networks split (the original pure gates)")
    splits = split_by_family(items, seed=0)
    print(f"  train {len(splits['train'])}, val {len(splits['val'])}, "
          f"test {len(splits['test'])}")
    modelA, n_params = train_model(splits, dev, "networks")
    torch.save(modelA.state_dict(), CKPT / "v1_networks.pt")
    evA = evaluate(modelA, splits["test"], dev)
    mA = pure_metrics(splits["test"], evA)
    print(f"  PURE coef median |err| {mA['coef_err']:.3f} (v0 was 0.43)")
    print(f"  PURE functional rel-RMSE {mA['func_median']:.3f} "
          f"(v0 was 0.27), residue {mA['residue']:.1%}")
    gates.append(gate("A1 pure coef median |err| <= 0.15", mA["coef_err"] <= 0.15,
                      f"{mA['coef_err']:.3f} over {mA['n']} specimens"))
    gates.append(gate("A2 pure functional rel-RMSE <= 0.10", mA["func_median"] <= 0.10,
                      f"{mA['func_median']:.3f}, residue {mA['residue']:.1%}"))

    # ---------- B. unseen FAMILIES (generalization) ----------
    print(f"\n[B] unseen-families split (holdout {HOLDOUT_FAMILIES})")
    splitsB = split_holdout_families(items, HOLDOUT_FAMILIES, seed=0)
    print(f"  train {len(splitsB['train'])}, val {len(splitsB['val'])}, "
          f"test(held-out families) {len(splitsB['test'])}")
    modelB, _ = train_model(splitsB, dev, "families")
    torch.save(modelB.state_dict(), CKPT / "v1_families.pt")
    evB = evaluate(modelB, splitsB["test"], dev)
    mB = pure_metrics(splitsB["test"], evB)
    fam_err = per_family_coef(splitsB["test"], evB)
    print(f"  held-out-family coef median |err| {mB['coef_err']:.3f}")
    print(f"  held-out-family functional rel-RMSE {mB['func_median']:.3f}, "
          f"residue {mB['residue']:.1%}")
    print(f"  per-family coef error: " + ", ".join(f"{f}:{e:.3f}" for f, e in fam_err.items()))
    # generalization bar is looser than in-distribution: this is genuinely hard
    gates.append(gate("B1 held-out-family coef reading better than chance",
                      mB["coef_err"] < 1.0,
                      f"coef |err| {mB['coef_err']:.3f} on unseen families"))
    gates.append(gate("B2 held-out-family functional recovers majority of behavior",
                      mB["func_median"] < 0.3,
                      f"functional rel-RMSE {mB['func_median']:.3f} (< 0.3 = "
                      f"most behavior recovered on UNSEEN classes)"))

    passed = all(g["passed"] for g in gates)

    # ---------- figure ----------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for ax, m, title in [(axes[0, 0], mA, "A: unseen NETWORKS"),
                         (axes[0, 1], mB, "B: unseen FAMILIES")]:
        ct, cp = m["coef_t"], m["coef_p"]
        ax.scatter(ct, cp, s=7, alpha=0.4)
        lim = max(abs(ct).max(), abs(cp).max()) if len(ct) else 1
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
        ax.set_xlabel("true coefficient"); ax.set_ylabel("read from weights")
        ax.set_title(f"{title}\ncoef median |err| {m['coef_err']:.3f}")

    ax = axes[0, 2]
    ax.hist(np.clip(mA["rels"], 0, 0.6), bins=30, alpha=0.6, color="C0",
            label=f"unseen nets (med {mA['func_median']:.3f})")
    ax.hist(np.clip(mB["rels"], 0, 0.6), bins=30, alpha=0.6, color="C2",
            label=f"unseen families (med {mB['func_median']:.3f})")
    ax.axvline(0.1, color="k", ls=":", lw=0.8)
    ax.set_xlabel("functional rel-RMSE (rule vs network)"); ax.legend(fontsize=8)
    ax.set_title("pure weights-only functional recovery")

    ax = axes[1, 0]
    ax.hist(np.clip(mA["errs"], 0, 1.0), bins=30, color="C0")
    ax.axvline(0.15, color="k", ls=":", lw=0.8, label="gate 0.15")
    ax.axvline(np.median(mA["errs"]), color="C3", lw=1,
               label=f"median {mA['coef_err']:.3f}")
    ax.set_xlabel("|coef error| (unseen networks)"); ax.legend(fontsize=8)
    ax.set_title("v1 coefficient error (v0 median was 0.43)")

    ax = axes[1, 1]
    if fam_err:
        ax.bar(list(fam_err), list(fam_err.values()), color="C2")
        ax.set_ylabel("coef median |err|"); ax.tick_params(axis="x", rotation=30)
        ax.set_title(f"held-out families: {HOLDOUT_FAMILIES}\n(coef error on classes never trained)")

    ax = axes[1, 2]; ax.axis("off")
    txt = (f"INTERPRETER v1 (per-term attention readout)\n"
           f"corpus {len(items)} specimens, {n_params:,} params\n\n"
           f"A. UNSEEN NETWORKS (pure gates):\n"
           f"   coef |err|   {mA['coef_err']:.3f}   (v0: 0.43)\n"
           f"   functional   {mA['func_median']:.3f}   (v0: 0.27)\n"
           f"   support      {mA['support']:.3f}\n\n"
           f"B. UNSEEN FAMILIES ({'+'.join(HOLDOUT_FAMILIES)}):\n"
           f"   coef |err|   {mB['coef_err']:.3f}\n"
           f"   functional   {mB['func_median']:.3f}\n\n"
           f"   {'ALL GATES PASSED' if passed else 'some gates failed'}")
    ax.text(0.02, 0.97, txt, va="top", ha="left", family="monospace",
            fontsize=10, transform=ax.transAxes)
    fig.suptitle("Reading coefficients from weights: per-term attention + more data",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = RESULTS / "exp11_interpreter_v1.png"
    fig.savefig(fig_path, dpi=130); plt.close(fig)

    report = {"experiment": "exp11_interpreter_v1", "all_passed": passed,
              "corpus_size": len(items), "params": n_params,
              "unseen_networks": {k: mA[k] for k in
                                  ("coef_err", "support", "func_median", "residue", "n")},
              "unseen_families": {"holdout": HOLDOUT_FAMILIES,
                                  **{k: mB[k] for k in ("coef_err", "func_median", "residue", "n")},
                                  "per_family": fam_err},
              "gates": gates}
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2, default=str),
                                         encoding="utf-8")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  unseen NETWORKS: coef |err| {mA['coef_err']:.3f}, functional "
          f"{mA['func_median']:.3f}  (v0: 0.43 / 0.27)")
    print(f"  unseen FAMILIES: coef |err| {mB['coef_err']:.3f}, functional "
          f"{mB['func_median']:.3f}")
    print(f"  overall: {'ALL GATES PASSED' if passed else 'FAILURES PRESENT'}")
    print(f"  report:  {RESULTS / 'report.json'}")
    print(f"  figure:  {fig_path}")
    print("=" * 70)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
