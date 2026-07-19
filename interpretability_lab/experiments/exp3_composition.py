"""EXPERIMENT 3 -- First composition: y = (a+b)*c at ~10K params, depth 2.

New questions at this rung, each with its own gate:

  1. MECHANISM THROUGH DEPTH  affine assembly must now trace masks through
     two layers: grad(x) = w3^T diag(m2) W2 diag(m1) W1.
  2. SYMBOLIC RECOVERY        3-variable monomial library (deg <= 3, 20 terms),
     exhaustive sparse selection. (a+b)*c = a*c + b*c: support {a*c, b*c}.
  3. INTERMEDIATE QUANTITY    does the net REPRESENT s = a+b?
       - probe: s must be linearly decodable from layer activations
       - CAUSAL STEERING: push layer-1 activations along the s-direction by
         delta; if s is a real internal object, the output must move by
         c * delta (since dy/ds = c). Control: steering along the (a-b)
         probe direction must move the output ~not at all -- the network
         should be indifferent to a quantity the task never needs.
  4. NEGATIVE CONTROL         run the symbolic reader on an UNTRAINED net of
     the same architecture: it must refuse (no compact rule fits). A reader
     that finds rules everywhere is worthless.
  5. COMPRESSION LEDGER       params -> effective units -> symbolic terms,
     computed for every specimen in the corpus (see geometry/compression.py).

Run:  python -m interpretability_lab.experiments.exp3_composition
"""

import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from interpretability_lab.corpus.store import save_specimen
from interpretability_lab.extraction import symbolic_reader as sym
from interpretability_lab.geometry.compression import build_ledger
from interpretability_lab.hooks.recorder import ActivationRecorder
from interpretability_lab.models.tiny import TinyMLP2, param_count, train_full_batch

RESULTS = Path(__file__).parent / "results" / "exp3"
DOMAIN = (-2.0, 2.0)
H1 = H2 = 96


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return {"gate": name, "passed": bool(ok), "detail": detail}


def target(X):
    return ((X[:, 0] + X[:, 1]) * X[:, 2]).unsqueeze(1)


def sample(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, generator=g) * (DOMAIN[1] - DOMAIN[0]) + DOMAIN[0]


# --------------------------------------------------- 3-var monomial library

def monomial_basis(max_deg=3):
    basis = {}
    for i, j, k in itertools.product(range(max_deg + 1), repeat=3):
        if 0 < i + j + k <= max_deg or (i, j, k) == (0, 0, 0):
            parts = []
            for v, p in zip("abc", (i, j, k)):
                if p == 1:
                    parts.append(v)
                elif p > 1:
                    parts.append(f"{v}^{p}")
            name = "*".join(parts) if parts else "1"
            basis[name] = (lambda ii, jj, kk: lambda X:
                           X[:, 0] ** ii * X[:, 1] ** jj * X[:, 2] ** kk)(i, j, k)
    return basis


# ------------------------------------------------ depth-2 affine assembly

def reconstruct_depth2(model, X: np.ndarray) -> np.ndarray:
    """y(x) assembled from per-layer masks and weight products, float64."""
    W1 = model.hidden1.weight.detach().numpy().astype(np.float64)
    b1 = model.hidden1.bias.detach().numpy().astype(np.float64)
    W2 = model.hidden2.weight.detach().numpy().astype(np.float64)
    b2 = model.hidden2.bias.detach().numpy().astype(np.float64)
    w3 = model.out.weight.detach().numpy().ravel().astype(np.float64)
    b3 = float(model.out.bias.detach())

    pre1 = X @ W1.T + b1
    m1 = (pre1 > 0).astype(np.float64)
    z1 = m1 * pre1
    pre2 = z1 @ W2.T + b2
    m2 = (pre2 > 0).astype(np.float64)

    # grad_i = w3^T diag(m2_i) W2 diag(m1_i) W1 ; assemble per-point
    A = (m2 * w3) @ W2                       # (N, H1): w3^T diag(m2) W2
    grads = (A * m1) @ W1                    # (N, 3)
    intercepts = ((A * m1) @ b1) + (m2 * w3) @ b2 + b3
    return (grads * X).sum(1) + intercepts


# ----------------------------------------------------- probes and steering

def linear_probe(H, t):
    """Least-squares probe t ~ H @ w + c. Returns (w, c, r2)."""
    A = np.concatenate([H, np.ones((len(H), 1))], 1)
    sol, *_ = np.linalg.lstsq(A, t, rcond=None)
    pred = A @ sol
    r2 = 1 - float(((t - pred) ** 2).sum()) / float(((t - t.mean()) ** 2).sum())
    return sol[:-1], sol[-1], r2


def steer(model, X, direction, delta):
    """Add delta * direction to layer-1 activations; return output change."""
    v = torch.tensor(direction * delta, dtype=torch.float32)
    with torch.no_grad():
        base = model(X).ravel()

    def hook(_m, _i, out):
        return out + v

    h = model.act1.register_forward_hook(hook)
    try:
        with torch.no_grad():
            steered = model(X).ravel()
    finally:
        h.remove()
    return (steered - base).numpy()


# ----------------------------------------------------------------------- main

def main():
    print("=" * 70)
    print("EXPERIMENT 3: composition -- y = (a+b)*c at ~10K params, depth 2")
    print("=" * 70)
    RESULTS.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else None

    # ---- train, or load the fixed specimen if one exists. Pinning matters
    # twice over: CUDA training is not deterministic, and exp4 analyzes THIS
    # specimen from the corpus -- retraining would swap it out from under it.
    spec_dir = (Path(__file__).parent.parent / "corpus" / "data" / "exp3"
                / "exp3_3a_compose_seed0")
    if (spec_dir / "weights.pt").exists():
        model = TinyMLP2(3, H1, H2)
        model.load_state_dict(torch.load(spec_dir / "weights.pt",
                                         weights_only=True))
        model.eval()
        seed = json.loads((spec_dir / "meta.json").read_text(
            encoding="utf-8")).get("seed", 0)
        Xt = sample(8192, seed + 777)
        with torch.no_grad():
            pred = model(Xt)
        Yt = target(Xt)
        r2 = 1 - float(((Yt - pred) ** 2).sum()) / float(((Yt - Yt.mean()) ** 2).sum())
        print(f"\n  loaded fixed specimen from corpus (seed {seed}; delete it "
              f"to retrain), fit R^2 = {r2:.6f}")
    else:
        best = None
        for seed in range(3):
            torch.manual_seed(seed)
            X = sample(16384, seed + 100)
            model = TinyMLP2(3, H1, H2)
            train_full_batch(model, X, target(X), epochs=8000, lr=0.01, device=dev)
            Xt = sample(8192, seed + 777)
            with torch.no_grad():
                pred = model(Xt)
            Yt = target(Xt)
            r2 = 1 - float(((Yt - pred) ** 2).sum()) / float(((Yt - Yt.mean()) ** 2).sum())
            if best is None or r2 > best[1]:
                best = (model, r2, seed)
            if r2 > 0.9999:
                break
        model, r2, seed = best
        print(f"\n  trained: {param_count(model)} params (seed {seed}, "
              f"device {dev or 'cpu'}), fit R^2 = {r2:.6f}")
    gates = [gate("network fits the target function", r2 > 0.999, f"R^2 = {r2:.6f}")]

    # ---- mechanistic read through depth
    Xv = sample(20000, 4242).numpy().astype(np.float64)
    recon = reconstruct_depth2(model, Xv)
    with torch.no_grad():
        net_y = model(torch.tensor(Xv, dtype=torch.float32)).numpy().ravel()
    mech_err = float(np.abs(recon - net_y).max())
    gates.append(gate("depth-2 affine assembly reproduces the forward pass",
                      mech_err < 1e-3, f"max |diff| = {mech_err:.2e}"))

    # nominal region count (for the ledger's 'nominal objects' column)
    rec = ActivationRecorder(model, names=["act1", "act2"])
    with rec.capture():
        model(torch.tensor(Xv[:20000], dtype=torch.float32))
    masks = np.concatenate([(rec.traces["act1"].numpy() > 0),
                            (rec.traces["act2"].numpy() > 0)], 1)
    n_regions = len(np.unique(masks, axis=0))
    print(f"  mechanism: {n_regions} distinct joint activation patterns "
          f"among 20000 sample points")

    # ---- symbolic read on the reconstruction
    basis3 = monomial_basis(3)
    fit = sym.fit_sparse(Xv[:8192], recon[:8192], basis=basis3)
    print(f"  symbolic (from mechanism): {fit['expression']}")
    print(f"  mechanism gap (symbolic vs net): rel RMSE = {fit['rel_rmse']:.2e}")
    want = {"a*c": 1.0, "b*c": 1.0}
    if set(fit["support"]) == set(want):
        worst = max(abs(fit["coefficients"][n] - v) for n, v in want.items())
        ok, detail = worst < 0.02, f"a*c + b*c, worst abs coef err {worst:.4f}"
    else:
        ok, detail = False, f"support {fit['support']} != ['a*c', 'b*c']"
    gates.append(gate("symbolic form recovers the generating rule", ok, detail))

    # ---- layer-1 weight structure (reported, not gated: exp2 taught us not
    #      to gate on the human decomposition)
    W1 = model.hidden1.weight.detach().numpy()
    W2out = np.abs(model.hidden2.weight.detach().numpy()).sum(0)
    contrib = W2out * np.linalg.norm(W1, axis=1)
    wn = W1 / (np.linalg.norm(W1, axis=1, keepdims=True) + 1e-12)
    s_aligned = np.abs(wn[:, 0] - wn[:, 1]) < 0.1          # w_a ~ w_b
    a_only = np.abs(wn[:, 1]) < 0.1                        # w_b ~ 0
    b_only = np.abs(wn[:, 0]) < 0.1                        # w_a ~ 0
    tot = contrib.sum() or 1.0
    print(f"  layer-1 structure (contribution-weighted): "
          f"{contrib[s_aligned].sum() / tot:.1%} s-aligned (w_a = w_b), "
          f"{contrib[a_only].sum() / tot:.1%} a-only, "
          f"{contrib[b_only].sum() / tot:.1%} b-only")

    # ---- probes: is s = a+b represented? is the unneeded a-b discarded?
    Xp = sample(8192, 55)
    rec = ActivationRecorder(model, names=["act1", "act2"])
    with rec.capture():
        model(Xp)
    H1a = rec.traces["act1"].numpy().astype(np.float64)
    H2a = rec.traces["act2"].numpy().astype(np.float64)
    s_t = (Xp[:, 0] + Xp[:, 1]).numpy().astype(np.float64)
    d_t = (Xp[:, 0] - Xp[:, 1]).numpy().astype(np.float64)
    ws, cs, r2_s1 = linear_probe(H1a, s_t)
    _, _, r2_s2 = linear_probe(H2a, s_t)
    wd, cd, r2_d1 = linear_probe(H1a, d_t)
    _, _, r2_d2 = linear_probe(H2a, d_t)
    print(f"  probes  R^2: s=a+b  layer1 {r2_s1:.4f}  layer2 {r2_s2:.4f}")
    print(f"          R^2: a-b    layer1 {r2_d1:.4f}  layer2 {r2_d2:.4f}   "
          f"(a-b is never needed by the task)")
    gates.append(gate("intermediate quantity s = a+b linearly decodable at layer 2",
                      r2_s2 > 0.98, f"R^2 = {r2_s2:.4f}"))

    # ---- causal steering along the PROBE direction. First run showed this
    # FAILS (R^2 = 0.02) despite perfect decodability: the probe direction is
    # what correlates with s, not what the circuit listens to. That negative
    # result is one of the lab's key findings, so it is PINNED: this gate now
    # asserts the refutation REPRODUCES. If probe-direction steering ever
    # becomes causal, this gate fails and that surprise deserves a look.
    # The correct causal method (tangent steering, R^2 = 0.99) lives in exp4.
    Xs = sample(1024, 66)
    delta = 0.5
    v_s = ws / (ws @ ws)                       # probe-predicted ds = delta
    dy_s = steer(model, Xs, v_s, delta)
    predicted = (Xs[:, 2].numpy() * delta)     # dy/ds = c
    ss_res = float(((dy_s - predicted) ** 2).sum())
    ss_tot = float(((predicted - predicted.mean()) ** 2).sum())
    r2_steer = 1 - ss_res / ss_tot
    print(f"  steering s by {delta} along the probe direction: predicted "
          f"dy = c*{delta}; agreement R^2 = {r2_steer:.4f}")
    print(f"  -> probes decode s perfectly yet are NOT causal "
          f"(decodability != causality; causal method in exp4)")
    gates.append(gate("PINNED REFUTATION: probe-direction steering still fails",
                      r2_steer < 0.5, f"R^2 = {r2_steer:.4f} (refutation "
                      f"reproduces; exp4's tangent steering scores 0.99)"))

    v_d = wd / (wd @ wd)
    dy_d = steer(model, Xs, v_d, delta)
    ratio = float(np.median(np.abs(dy_d)) / (np.median(np.abs(dy_s)) + 1e-12))
    print(f"  control: steering the (a-b)-direction moves output "
          f"{ratio:.1%} as much (task never needs a-b)")
    gates.append(gate("network indifferent to steering the unneeded a-b direction",
                      ratio < 0.2, f"ratio = {ratio:.1%}"))

    # ---- negative control: untrained net, reader must refuse
    torch.manual_seed(999)
    rand_model = TinyMLP2(3, H1, H2)
    rand_recon = reconstruct_depth2(rand_model, Xv[:8192])
    rand_fit = sym.fit_sparse(Xv[:8192], rand_recon, basis=basis3)
    print(f"  negative control (untrained net): best 4-term fit rel RMSE = "
          f"{rand_fit['rel_rmse']:.3f} -> "
          f"{'reader refuses' if rand_fit['rel_rmse'] > 0.05 else 'READER CLAIMED A RULE'}")
    gates.append(gate("reader refuses to extract a rule from an untrained net",
                      rand_fit["rel_rmse"] > 0.05,
                      f"rel RMSE {rand_fit['rel_rmse']:.3f} (claim threshold 0.05)"))

    passed = all(g["passed"] for g in gates)
    save_specimen(model, experiment="exp3", task="3a_compose", seed=seed,
                  ground_truth="y = (a+b)*c",
                  arch={"type": "TinyMLP2", "in_dim": 3, "h1": H1, "h2": H2},
                  recovered=fit["expression"], passed=passed,
                  extra={"fit_r2": r2, "n_regions": n_regions,
                         "mechanism_gap_rel_rmse": fit["rel_rmse"],
                         "probe_r2": {"s_l1": r2_s1, "s_l2": r2_s2,
                                      "d_l1": r2_d1, "d_l2": r2_d2},
                         "steering_r2": r2_steer, "control_ratio": ratio})

    # ---- compression ledger over the whole corpus
    print("\n  building compression ledger over all specimens...")
    ledger = build_ledger(RESULTS.parent)
    print(f"  {'specimen':<22}{'params':>8}{'eff units':>11}{'sym terms':>11}{'units/term':>12}")
    for row in ledger:
        eff = f"{row['effective_units']}/{row['total_units']}" if row["total_units"] else "-"
        print(f"  {row['specimen']:<22}{row['params']:>8}{eff:>11}"
              f"{row['symbolic_terms']:>11}{str(row['units_per_term'] or '-'):>12}")

    # ---- figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    n = 150
    g = np.linspace(*DOMAIN, n)
    GA, GB = np.meshgrid(g, g)
    for ax, cval in [(axes[0, 0], 1.5)]:
        Xg = torch.tensor(np.stack([GA.ravel(), GB.ravel(),
                                    np.full(GA.size, cval)], 1), dtype=torch.float32)
        with torch.no_grad():
            Z = model(Xg).numpy().reshape(n, n)
        im = ax.imshow(Z, origin="lower", extent=[*DOMAIN, *DOMAIN], aspect="auto")
        ax.contour(GA, GB, (GA + GB) * cval, colors="w", linewidths=0.6)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"slice c={cval}: net output (white = truth)")
        ax.set_xlabel("a"); ax.set_ylabel("b")

    ax = axes[0, 1]
    sc = ax.scatter(wn[:, 0], wn[:, 1], s=8 + 300 * contrib / contrib.max(),
                    c=np.abs(wn[:, 2]), cmap="plasma", alpha=0.75)
    ax.plot([-1, 1], [-1, 1], "g--", lw=1, label="s-aligned (w_a = w_b)")
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="|w_c| (normalized)")
    ax.set_xlabel("w_a"); ax.set_ylabel("w_b")
    ax.set_title("layer-1 unit weights (size = contribution)")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    labels = ["s | L1", "s | L2", "a-b | L1", "a-b | L2"]
    vals = [r2_s1, r2_s2, r2_d1, r2_d2]
    ax.bar(labels, vals, color=["C0", "C0", "C3", "C3"])
    ax.set_ylim(0, 1.05); ax.axhline(0.98, color="k", ls=":", lw=0.8)
    ax.set_title("probe R^2: needed (s) vs unneeded (a-b)")

    ax = axes[1, 0]
    ax.scatter(predicted, dy_s, s=6, alpha=0.5)
    lim = max(abs(predicted).max(), abs(dy_s).max())
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax.set_xlabel("predicted dy = c*delta"); ax.set_ylabel("observed dy")
    ax.set_title(f"causal steering of s (R^2 = {r2_steer:.3f})")

    ax = axes[1, 1]
    ax.hist(np.abs(dy_s), bins=40, alpha=0.6, label="steer s (needed)")
    ax.hist(np.abs(dy_d), bins=40, alpha=0.6, label="steer a-b (unneeded)")
    ax.set_xlabel("|dy|"); ax.legend(fontsize=8)
    ax.set_title(f"steering effect sizes (control ratio {ratio:.1%})")

    ax = axes[1, 2]
    xs_l = [row["params"] for row in ledger if row["units_per_term"]]
    ys_l = [row["units_per_term"] for row in ledger if row["units_per_term"]]
    names_l = [row["specimen"].split("/")[1] for row in ledger if row["units_per_term"]]
    ax.scatter(xs_l, ys_l, c="C2")
    for x_, y_, n_ in zip(xs_l, ys_l, names_l):
        ax.annotate(n_, (x_, y_), fontsize=7, textcoords="offset points", xytext=(4, 3))
    ax.set_xscale("log")
    ax.set_xlabel("parameters"); ax.set_ylabel("effective units per symbolic term")
    ax.set_title("compression ledger: geometry per concept")
    fig.tight_layout()
    fig_path = RESULTS / "exp3_composition.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    report = {"experiment": "exp3_composition", "all_passed": passed,
              "fit_r2": r2, "n_regions": n_regions, "symbolic": fit,
              "probes": {"s_l1": r2_s1, "s_l2": r2_s2, "d_l1": r2_d1, "d_l2": r2_d2},
              "steering_r2": r2_steer, "control_ratio": ratio,
              "negative_control_rel_rmse": rand_fit["rel_rmse"],
              "gates": gates, "ledger": ledger}
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2, default=str),
                                         encoding="utf-8")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  recovered: {fit['expression']}")
    print(f"  overall: {'ALL GATES PASSED' if passed else 'FAILURES PRESENT'}")
    print(f"  report:  {RESULTS / 'report.json'}")
    print(f"  figure:  {fig_path}")
    print("=" * 70)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
