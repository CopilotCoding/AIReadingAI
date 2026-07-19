"""EXPERIMENT 6 -- Interpreter v0: an AI trained from scratch to read AIs.

Phase 3 begins. A permutation-invariant set-transformer is trained on the
generated corpus (1472 specimens): INPUT = another network's raw weights,
OUTPUT = the rule that network was trained on (canonical basis terms +
coefficients), a logic-gate class, or REFUSE for untrained networks.
No activations, no input/output probing -- weights only.

Pre-registered gates on the held-out test split:
  G1 task-type accuracy >= 0.95           (regression / 6 gates / refusal)
  G2 refusal precision & recall >= 0.95   (the axiom: must be able to refuse)
  G3 support exact-match >= 0.70          (right terms, no spurious terms)
  G4 coefficient median |err| <= 0.15     (on correctly-supported terms)
  G5 FUNCTIONAL verification: the predicted rule, evaluated as a function,
     reproduces the specimen NETWORK's actual behavior -- median rel RMSE
     <= 0.10 over test regression specimens. This is the behavioral gate:
     the readout must describe the network, not just match labels.
Per the core axiom, the residue (specimens no story captures) is reported,
not hidden.

Run:  python -m interpretability_lab.experiments.exp6_interpreter_v0
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
                                                      collate, load_corpus,
                                                      split_by_family)
from interpretability_lab.interpreter.model import Interpreter

RESULTS = Path(__file__).parent / "results" / "exp6"
CKPT = Path(__file__).parent.parent / "interpreter" / "checkpoints"
EPOCHS = 400
BATCH = 64


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return {"gate": name, "passed": bool(ok), "detail": detail}


def batches(items, rng, device, train=True):
    idx = rng.permutation(len(items)) if train else np.arange(len(items))
    for s in range(0, len(idx), BATCH):
        yield collate([items[i] for i in idx[s:s + BATCH]], device)


def loss_fn(task_lg, sup_lg, coef_pred, cls, support, coefs):
    l_task = F.cross_entropy(task_lg, cls)
    reg = cls == 0
    if reg.any():
        l_sup = F.binary_cross_entropy_with_logits(sup_lg[reg], support[reg])
        # coefficient loss masked to PRESENT terms: the 11-of-13 zero
        # background otherwise drowns the signal and the head learns "say 0"
        m = support[reg] > 0.5
        l_coef = F.huber_loss(coef_pred[reg][m], coefs[reg][m]) if m.any() \
            else torch.zeros((), device=task_lg.device)
    else:
        l_sup = l_coef = torch.zeros((), device=task_lg.device)
    return l_task + l_sup + l_coef, l_task, l_sup, l_coef


@torch.no_grad()
def evaluate(model, items, device):
    model.eval()
    out = {"cls_true": [], "cls_pred": [], "support_true": [], "support_pred": [],
           "coefs_true": [], "coefs_pred": [], "uid": [], "family": []}
    for b in batches(items, np.random.default_rng(0), device, train=False):
        x, mask, g, cls, support, coefs = b
        task_lg, sup_lg, coef_pred = model(x, mask, g)
        out["cls_true"] += cls.tolist()
        out["cls_pred"] += task_lg.argmax(1).tolist()
        out["support_true"].append(support.cpu())
        out["support_pred"].append((torch.sigmoid(sup_lg) > 0.5).float().cpu())
        out["coefs_true"].append(coefs.cpu())
        out["coefs_pred"].append(coef_pred.cpu())
    for k in ("support_true", "support_pred", "coefs_true", "coefs_pred"):
        out[k] = torch.cat(out[k])
    out["uid"] = [it["uid"] for it in items]
    out["family"] = [it["family"] for it in items]
    model.train()
    return out


def functional_check(items, ev):
    """Evaluate each predicted rule AS A FUNCTION against the specimen
    network's actual outputs on its own domain."""
    rel_rmses, matched_uids = [], []
    for i, it in enumerate(items):
        if ev["cls_true"][i] != 0 or ev["cls_pred"][i] != 0:
            continue
        meta = it["meta"]
        model = _build(meta["arch"])
        sd = torch.load(Path(__file__).parent.parent / "corpus" / "generated"
                        / meta["family"] / meta["uid"] / "weights.pt",
                        weights_only=True)
        model.load_state_dict(sd)
        model.eval()
        lo, hi = meta["domain"]
        g = torch.Generator().manual_seed(7)
        X = torch.rand(2048, meta["input_dim"], generator=g) * (hi - lo) + lo
        with torch.no_grad():
            y_net = model(X).ravel()
        terms = [(float(ev["coefs_pred"][i, j]), BASIS_NAMES[j])
                 for j in range(len(BASIS_NAMES))
                 if ev["support_pred"][i, j] > 0.5]
        # a predicted term referencing a variable the network does not have
        # is an ILL-FORMED rule: counted as infinite error (residue), per the
        # axiom -- not silently forgiven, not a crash
        def needs_dim(name):
            return 3 if "x2" in name else (2 if "x1" in name else 1)
        if any(needs_dim(n) > meta["input_dim"] for _, n in terms):
            rel_rmses.append(float("inf"))
            matched_uids.append(it["uid"])
            continue
        y_rule = eval_terms(terms, X) if terms else torch.zeros_like(y_net)
        rel = float(torch.sqrt(((y_rule - y_net) ** 2).mean())
                    / y_net.std().clamp(min=1e-9))
        rel_rmses.append(rel)
        matched_uids.append(it["uid"])
    return np.array(rel_rmses), matched_uids


def main():
    print("=" * 70)
    print("EXPERIMENT 6: interpreter v0 -- an AI reading other AIs' weights")
    print("=" * 70)
    RESULTS.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n  loading corpus...")
    items = load_corpus()
    splits = split_by_family(items, seed=0)
    n = {k: len(v) for k, v in splits.items()}
    print(f"  corpus: {len(items)} specimens -> train {n['train']}, "
          f"val {n['val']}, test {n['test']} (stratified by family)")

    torch.manual_seed(0)
    model = Interpreter().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  interpreter: set-transformer, {n_params:,} params "
          f"(permutation-invariant by construction), device {dev}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    rng = np.random.default_rng(1)
    hist, best = [], (float("inf"), None)
    t0 = time.time()
    for ep in range(EPOCHS):
        tl = 0.0
        nb = 0
        for b in batches(splits["train"], rng, dev):
            x, mask, g, cls, support, coefs = b
            loss, *_ = loss_fn(*model(x, mask, g), cls, support, coefs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += float(loss)
            nb += 1
        sched.step()
        if (ep + 1) % 10 == 0:
            with torch.no_grad():
                vl = nvb = 0.0
                for b in batches(splits["val"], rng, dev, train=False):
                    x, mask, g, cls, support, coefs = b
                    l, *_ = loss_fn(*model(x, mask, g), cls, support, coefs)
                    vl += float(l)
                    nvb += 1
            vl /= nvb
            hist.append({"epoch": ep + 1, "train": tl / nb, "val": vl})
            if vl < best[0]:
                best = (vl, {k: v.cpu().clone() for k, v in model.state_dict().items()})
            print(f"  epoch {ep + 1:>3}: train {tl / nb:.4f}  val {vl:.4f}"
                  f"{'  *' if vl == best[0] else ''}")
    model.load_state_dict(best[1])
    model.to(dev)
    print(f"  trained in {(time.time() - t0) / 60:.1f} min; "
          f"best val loss {best[0]:.4f}")
    torch.save(model.state_dict(), CKPT / "v0.pt")

    # ---- evaluation on held-out test specimens
    ev = evaluate(model, splits["test"], dev)
    cls_t = np.array(ev["cls_true"])
    cls_p = np.array(ev["cls_pred"])
    gates = []

    acc = float((cls_t == cls_p).mean())
    gates.append(gate("G1 task-type accuracy >= 0.95", acc >= 0.95,
                      f"accuracy {acc:.3f} on {len(cls_t)} test specimens"))
    print("    per-class recall: " + "  ".join(
        f"{TASK_CLASSES[k]}:{float((cls_p[cls_t == k] == k).mean()):.2f}"
        for k in sorted(set(cls_t.tolist()))))

    none_i = TASK_CLASSES.index("none")
    tp = int(((cls_p == none_i) & (cls_t == none_i)).sum())
    fp = int(((cls_p == none_i) & (cls_t != none_i)).sum())
    fn = int(((cls_p != none_i) & (cls_t == none_i)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    gates.append(gate("G2 refusal precision & recall >= 0.95",
                      prec >= 0.95 and rec >= 0.95,
                      f"precision {prec:.3f}, recall {rec:.3f}"))

    reg = (cls_t == 0) & (cls_p == 0)
    sup_match = (ev["support_pred"][reg] == ev["support_true"][reg]).all(1)
    sup_rate = float(sup_match.float().mean())
    gates.append(gate("G3 support exact-match >= 0.70", sup_rate >= 0.70,
                      f"{sup_rate:.3f} over {int(reg.sum())} regression specimens"))

    hit = (ev["support_true"][reg] > 0.5) & (ev["support_pred"][reg] > 0.5)
    errs = (ev["coefs_pred"][reg] - ev["coefs_true"][reg]).abs()[hit]
    med_err = float(errs.median()) if len(errs) else float("nan")
    gates.append(gate("G4 coefficient median |err| <= 0.15", med_err <= 0.15,
                      f"median {med_err:.3f} over {len(errs)} recovered terms"))

    print("  functional verification (predicted rule vs actual network)...")
    rel_rmses, _ = functional_check(splits["test"], ev)
    med_rel = float(np.median(rel_rmses))
    residue = float((rel_rmses > 0.2).mean())
    gates.append(gate("G5 functional: median rel RMSE <= 0.10",
                      med_rel <= 0.10,
                      f"median {med_rel:.3f}; residue (rel RMSE > 0.2): "
                      f"{residue:.1%} of specimens"))

    # per-family breakdown (reported)
    fams = sorted(set(ev["family"]))
    fam_rows = []
    print("  per-family test performance:")
    for fam in fams:
        m = np.array([f == fam for f in ev["family"]])
        facc = float((cls_t[m] == cls_p[m]).mean())
        if fam in ("logic", "none"):
            fam_rows.append((fam, facc, None))
            print(f"    {fam:<8} task acc {facc:.3f}")
        else:
            mm = m & (cls_t == 0) & (cls_p == 0)
            fs = (ev["support_pred"][mm] == ev["support_true"][mm]).all(1)
            fsr = float(fs.float().mean()) if mm.any() else float("nan")
            fam_rows.append((fam, facc, fsr))
            print(f"    {fam:<8} task acc {facc:.3f}, support match {fsr:.3f}")

    passed = all(g["passed"] for g in gates)

    # ---- figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax = axes[0, 0]
    ax.plot([h["epoch"] for h in hist], [h["train"] for h in hist], label="train")
    ax.plot([h["epoch"] for h in hist], [h["val"] for h in hist], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(fontsize=8)
    ax.set_title(f"interpreter training ({n_params:,} params)")

    ax = axes[0, 1]
    K = len(TASK_CLASSES)
    cm = np.zeros((K, K))
    for t, p in zip(cls_t, cls_p):
        cm[t, p] += 1
    cm = cm / cm.sum(1, keepdims=True).clip(min=1)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels(TASK_CLASSES, rotation=45,
                                                fontsize=7, ha="right")
    ax.set_yticks(range(K)); ax.set_yticklabels(TASK_CLASSES, fontsize=7)
    ax.set_title(f"task confusion (acc {acc:.3f})")
    ax.set_ylabel("true"); ax.set_xlabel("predicted")

    ax = axes[0, 2]
    names = [f for f, _, s in fam_rows if s is not None]
    vals = [s for _, _, s in fam_rows if s is not None]
    ax.bar(names, vals, color="C2")
    ax.axhline(0.7, color="k", ls=":", lw=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_title("support exact-match by family")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1, 0]
    ct = ev["coefs_true"][reg][hit].numpy()
    cp = ev["coefs_pred"][reg][hit].numpy()
    ax.scatter(ct, cp, s=6, alpha=0.4)
    lim = max(abs(ct).max(), abs(cp).max())
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax.set_xlabel("true coefficient"); ax.set_ylabel("read from weights")
    ax.set_title(f"coefficients read from raw weights (median |err| {med_err:.3f})")

    ax = axes[1, 1]
    ax.hist(np.clip(rel_rmses, 0, 1), bins=40, color="C0")
    ax.axvline(med_rel, color="C3", lw=1, label=f"median {med_rel:.3f}")
    ax.axvline(0.2, color="k", ls=":", lw=0.8, label=f"residue >0.2: {residue:.1%}")
    ax.set_xlabel("rel RMSE (predicted rule vs actual network)")
    ax.legend(fontsize=8)
    ax.set_title("functional verification + residue (the axiom's number)")

    ax = axes[1, 2]
    txt = (f"INTERPRETER v0 (weights -> rule, no probing):\n\n"
           f"  test task accuracy      {acc:.3f}\n"
           f"  refusal prec/recall     {prec:.3f} / {rec:.3f}\n"
           f"  support exact-match     {sup_rate:.3f}\n"
           f"  coef median |err|       {med_err:.3f}\n"
           f"  functional median rel   {med_rel:.3f}\n"
           f"  residue (>0.2)          {residue:.1%}\n\n"
           f"  {'ALL GATES PASSED' if passed else 'failures -- see report'}")
    ax.text(0.02, 0.97, txt, va="top", ha="left", fontsize=10,
            family="monospace", transform=ax.transAxes)
    ax.axis("off")
    fig.tight_layout()
    fig_path = RESULTS / "exp6_interpreter_v0.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    report = {"experiment": "exp6_interpreter_v0", "all_passed": passed,
              "interpreter_params": n_params, "corpus_size": len(items),
              "splits": n, "task_accuracy": acc,
              "refusal": {"precision": prec, "recall": rec},
              "support_exact_match": sup_rate,
              "coef_median_abs_err": med_err,
              "functional": {"median_rel_rmse": med_rel,
                             "residue_frac_gt_0.2": residue,
                             "n_checked": int(len(rel_rmses))},
              "per_family": [{"family": f, "task_acc": a, "support_match": s}
                             for f, a, s in fam_rows],
              "history": hist, "gates": gates}
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2,
                                                    default=str), encoding="utf-8")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  an AI read {len(cls_t)} unseen networks' weights: task acc "
          f"{acc:.3f}, support {sup_rate:.3f}, functional median {med_rel:.3f}")
    print(f"  overall: {'ALL GATES PASSED' if passed else 'FAILURES PRESENT'}")
    print(f"  report:  {RESULTS / 'report.json'}")
    print(f"  figure:  {fig_path}")
    print("=" * 70)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
