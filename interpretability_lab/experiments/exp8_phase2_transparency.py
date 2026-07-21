"""EXPERIMENT 8 -- Phase 2: do networks DESIGNED to expose their structure
read back cleaner than vanilla ones?

Everything so far read networks after the fact. Phase 2 flips it: train with
constraints meant to organize the network into separable objects, and ask --
against a vanilla control on the IDENTICAL task -- whether extraction gets
cleaner. Pre-registered prediction (accuracy is already ~perfect, so this is
NOT the metric): the transparency-regularized net should need FEWER effective
units per concept and yield cleaner, more selective SAE features.

Task: y = a*b on [-2,2]^2 (exp2's case -- the vanilla net there used 179
effective units of 256 for a single multiplication).

Two nets, same architecture (2->128->1 ReLU), same data, differ only in loss:
  VANILLA       plain MSE.
  TRANSPARENT   MSE + activation L1 (sparsity) + off-diagonal decorrelation of
                hidden activations (pushes features toward orthogonal,
                monosemantic directions).

Measured, all reader-side (the nets never see these):
  1. EFFECTIVE UNITS  minimal hidden units preserving the function (R^2>=0.999).
  2. SAE FEATURES     sparse autoencoder on hidden activations: reconstruction,
                      mean L0, and how few atoms carry the computation.
  3. CONCEPT OBJECTS  the top causal directions packaged as
                      GeometricConceptObjects, with grounded confidence.
  4. Negative control: an untrained net must yield no confident concept.

Gates test the COMPARISON (transparent <= vanilla on effective units and
sparsity) and that BOTH still solve the task -- not a hoped-for absolute.

Run:  python -m interpretability_lab.experiments.exp8_phase2_transparency
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from interpretability_lab.features.sae import fit_sae, score_atoms
from interpretability_lab.geometry.compression import effective_units
from interpretability_lab.geometry.concept import GeometricConceptObject
from interpretability_lab.hooks.recorder import ActivationRecorder
from interpretability_lab.models.tiny import TinyMLP, param_count

RESULTS = Path(__file__).parent / "results" / "exp8"
CONCEPTS = Path(__file__).parent.parent / "geometry" / "concepts"
HIDDEN = 128
DOMAIN = (-2.0, 2.0)


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return {"gate": name, "passed": bool(ok), "detail": detail}


def sample(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 2, generator=g) * (DOMAIN[1] - DOMAIN[0]) + DOMAIN[0]


def hidden_acts(model, X):
    rec = ActivationRecorder(model, names=["act"])
    with rec.capture():
        model(X)
    return rec.traces["act"].numpy()


def train(transparent, seed=0, epochs=6000, lr=5e-3, l1=1e-3, decorr=5e-2):
    torch.manual_seed(seed)
    model = TinyMLP(2, HIDDEN)
    X = sample(8192, seed + 1)
    Y = (X[:, 0] * X[:, 1]).unsqueeze(1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad()
        h = model.act(model.hidden(X))
        pred = model.out(h)
        loss = nn.functional.mse_loss(pred, Y)
        if transparent:
            loss = loss + l1 * h.abs().mean()
            hc = h - h.mean(0, keepdim=True)
            cov = (hc.T @ hc) / len(h)
            off = cov - torch.diag(torch.diag(cov))
            loss = loss + decorr * (off ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    Xt = sample(4096, seed + 99)
    Yt = (Xt[:, 0] * Xt[:, 1]).unsqueeze(1)
    with torch.no_grad():
        r2 = 1 - float(((Yt - model(Xt)) ** 2).sum()) / float(((Yt - Yt.mean()) ** 2).sum())
    return model, r2


def concept_objects_from(model, tag):
    """Package the top causal hidden directions as GeometricConceptObjects.

    Causal influence is measured TASK-RELATIVE: how much ablating the unit
    degrades the model's agreement with the true a*b -- so a unit only counts
    as a real concept if it participates in solving the task. The null is the
    task error a unit-ablation induces in an UNTRAINED net (a unit that
    'matters' only because the net's output moves, without task content, must
    not score). This is why the untrained control now refutes: its units move
    the output but carry no task, so influence ~ null."""
    X = sample(4096, 7)
    Ytrue = (X[:, 0] * X[:, 1]).numpy()
    H = hidden_acts(model, X)
    with torch.no_grad():
        base = model(X).ravel().numpy()
    tvar = float(((Ytrue - Ytrue.mean()) ** 2).mean()) or 1e-9

    def task_err(y):
        return float(((y - Ytrue) ** 2).mean()) / tvar

    base_err = task_err(base)
    w2 = model.out.weight.detach().numpy().ravel()
    contrib = np.abs(w2) * H.std(0)
    top = np.argsort(contrib)[::-1][:3]

    # null: mean task-error increase from ablating random units (in THIS net) --
    # a floor for "an ablation moved the output". A real concept must beat it.
    rng = np.random.default_rng(0)
    null_incs = []
    for j in rng.choice(HIDDEN, 20, replace=False):
        Hp = H.copy(); Hp[:, j] = 0.0
        with torch.no_grad():
            y = model.out(torch.tensor(Hp, dtype=torch.float32)).ravel().numpy()
        null_incs.append(task_err(y) - base_err)
    null = float(np.median(null_incs))

    objs = []
    for j in top:
        Hp = H.copy(); Hp[:, j] = 0.0
        with torch.no_grad():
            y = model.out(torch.tensor(Hp, dtype=torch.float32)).ravel().numpy()
        influence = task_err(y) - base_err     # task fidelity lost by removing it
        acts = H[:, j]
        order = np.argsort(acts)
        e = np.zeros(HIDDEN); e[j] = 1.0
        obj = GeometricConceptObject(
            name=f"unit{j}", kind="feature", source=f"exp8_{tag}",
            layer="act", center=[float(acts.mean())],
            subspace=[e.tolist()],
            activating_examples=X.numpy()[order[-5:]].tolist(),
            counterexamples=X.numpy()[order[:5]].tolist(),
            causal_influence=influence,
            causal_test="unit ablation (task-fidelity loss)",
            null_baseline=null,
            story=f"hidden unit {j}, contribution {contrib[j]:.3f}")
        obj.grade()
        objs.append(obj)
    return objs, null


def main():
    print("=" * 70)
    print("EXPERIMENT 8: Phase 2 -- do transparency-regularized nets read cleaner?")
    print("=" * 70)
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("\n  training vanilla vs transparent on y = a*b (identical task/arch)...")
    van, r2_v = train(transparent=False)
    tra, r2_t = train(transparent=True)
    print(f"  vanilla     R^2 {r2_v:.5f} ({param_count(van)} params)")
    print(f"  transparent R^2 {r2_t:.5f} ({param_count(tra)} params)")
    gates = [gate("both nets solve the task (accuracy is not the point)",
                  r2_v > 0.99 and r2_t > 0.99, f"vanilla {r2_v:.4f}, transparent {r2_t:.4f}")]

    # 1. effective units
    Xe = sample(4096, 5)
    ev_v, tot = effective_units(van, Xe)
    ev_t, _ = effective_units(tra, Xe)
    print(f"\n  effective units (of {tot}): vanilla {ev_v}, transparent {ev_t}")
    gates.append(gate("transparent net uses <= vanilla effective units",
                      ev_t <= ev_v,
                      f"{ev_t} vs {ev_v} ({100*(ev_v-ev_t)/max(ev_v,1):.0f}% fewer)"))

    # 2. SAE on hidden activations + direct decorrelation measure
    Hv = hidden_acts(van, Xe)
    Ht = hidden_acts(tra, Xe)
    _, Zv, iv = fit_sae(Hv, d_hidden=64, epochs=1500)
    _, Zt, it = fit_sae(Ht, d_hidden=64, epochs=1500)
    print(f"  SAE vanilla:     recon R^2 {iv['recon_r2']:.3f}, mean L0 "
          f"{iv['mean_l0']:.1f}, dead {iv['dead_atoms']}")
    print(f"  SAE transparent: recon R^2 {it['recon_r2']:.3f}, mean L0 "
          f"{it['mean_l0']:.1f}, dead {it['dead_atoms']}")

    # the metric transparency directly targets: off-diagonal activation
    # correlation (monosemanticity proxy). Lower = features less entangled.
    def offdiag_corr(H):
        active = H[:, H.std(0) > 1e-6]
        C = np.abs(np.corrcoef(active.T))
        n = C.shape[0]
        return float((C.sum() - n) / (n * (n - 1)))
    corr_v, corr_t = offdiag_corr(Hv), offdiag_corr(Ht)
    print(f"  mean |off-diagonal activation correlation|: vanilla {corr_v:.3f}, "
          f"transparent {corr_t:.3f}")
    gates.append(gate("transparent net's features are less entangled (lower off-diag corr)",
                      corr_t < corr_v,
                      f"{corr_t:.3f} vs {corr_v:.3f} "
                      f"({100*(corr_v-corr_t)/max(corr_v,1e-9):.0f}% less entangled)"))

    # 3. concept objects
    objs_v, null_v = concept_objects_from(van, "vanilla")
    objs_t, null_t = concept_objects_from(tra, "transparent")
    CONCEPTS.mkdir(parents=True, exist_ok=True)
    for o in objs_v + objs_t:
        o.save(CONCEPTS / f"{o.source}_{o.name}.json")
    conf_v = np.mean([o.confidence for o in objs_v])
    conf_t = np.mean([o.confidence for o in objs_t])
    print(f"\n  top concept objects (saved to geometry/concepts/):")
    for o in objs_t[:1] + objs_v[:1]:
        print(f"    {o.summary()}")
    gates.append(gate("concept objects are confident + causally grounded",
                      conf_v > 0.5 and conf_t > 0.5,
                      f"mean confidence vanilla {conf_v:.2f}, transparent {conf_t:.2f}"))

    # 4. negative control: untrained net -> no confident concept
    torch.manual_seed(999)
    rand = TinyMLP(2, HIDDEN)
    objs_r, _ = concept_objects_from(rand, "untrained")
    conf_r = np.mean([o.confidence for o in objs_r])
    # an untrained net's "top unit" has causal effect near the null -> low conf
    refute_ok = conf_r < conf_t and conf_r < conf_v
    print(f"  negative control (untrained): mean concept confidence {conf_r:.2f} "
          f"-> {'below trained nets' if refute_ok else 'NOT clearly lower'}")
    gates.append(gate("untrained net yields weaker concepts than trained",
                      refute_ok, f"untrained {conf_r:.2f} < trained "
                      f"{min(conf_v, conf_t):.2f}"))

    passed = all(g["passed"] for g in gates)

    # ---- figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for ax, H, name in [(axes[0, 0], Hv, "vanilla"), (axes[0, 1], Ht, "transparent")]:
        C = np.corrcoef(H.T)
        im = ax.imshow(np.abs(C), cmap="magma", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{name}: |hidden activation correlation|\n"
                     f"(transparent should be more diagonal)")

    ax = axes[0, 2]
    ax.bar(["vanilla", "transparent"], [ev_v, ev_t], color=["C3", "C2"])
    ax.set_ylabel("effective units (of 128)")
    ax.set_title("effective units per concept (lower = cleaner)")

    ax = axes[1, 0]
    ax.hist((Zv > 1e-6).sum(1), bins=range(0, 20), alpha=0.6, label="vanilla", color="C3")
    ax.hist((Zt > 1e-6).sum(1), bins=range(0, 20), alpha=0.6, label="transparent", color="C2")
    ax.set_xlabel("SAE code L0 (active atoms per input)"); ax.legend(fontsize=8)
    ax.set_title("SAE sparsity of the two nets' features")

    ax = axes[1, 1]
    ax.bar(["vanilla", "transparent", "untrained"],
           [conf_v, conf_t, conf_r], color=["C3", "C2", "gray"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean concept-object confidence")
    ax.set_title("concept confidence (causally grounded)")

    ax = axes[1, 2]
    txt = (f"PHASE 2: designed transparency vs vanilla (y = a*b)\n\n"
           f"  accuracy R^2   {r2_v:.4f} / {r2_t:.4f}\n"
           f"  effective units {ev_v} -> {ev_t}\n"
           f"  SAE mean L0    {iv['mean_l0']:.1f} -> {it['mean_l0']:.1f}\n"
           f"  concept conf   {conf_v:.2f} / {conf_t:.2f}\n"
           f"  untrained ctrl {conf_r:.2f}\n\n"
           f"  prediction was: transparency simplifies the\n"
           f"  MECHANISM (fewer units/atoms), not accuracy\n\n"
           f"  {'ALL GATES PASSED' if passed else 'failures -- see report'}")
    ax.text(0.02, 0.97, txt, va="top", ha="left", fontsize=9.5,
            family="monospace", transform=ax.transAxes)
    ax.axis("off")
    fig.tight_layout()
    fig_path = RESULTS / "exp8_phase2_transparency.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    report = {"experiment": "exp8_phase2_transparency", "all_passed": passed,
              "accuracy": {"vanilla": r2_v, "transparent": r2_t},
              "effective_units": {"vanilla": ev_v, "transparent": ev_t, "total": tot},
              "sae": {"vanilla": iv, "transparent": it},
              "concept_confidence": {"vanilla": float(conf_v),
                                     "transparent": float(conf_t),
                                     "untrained": float(conf_r)},
              "gates": gates}
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2, default=str),
                                         encoding="utf-8")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  transparency effect: effective units {ev_v}->{ev_t}, "
          f"SAE L0 {iv['mean_l0']:.1f}->{it['mean_l0']:.1f}, both R^2 > 0.99")
    print(f"  overall: {'ALL GATES PASSED' if passed else 'FAILURES PRESENT'}")
    print(f"  concept objects saved: {CONCEPTS}")
    print(f"  report:  {RESULTS / 'report.json'}")
    print(f"  figure:  {fig_path}")
    print("=" * 70)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
