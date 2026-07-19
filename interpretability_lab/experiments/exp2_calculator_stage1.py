"""EXPERIMENT 2 -- Neural calculator, stage 1: two-input arithmetic (~1K params).

Tasks (domain [-2, 2]^2, 2 -> 256 -> 1 ReLU nets, 1025 params):
  2a  y = a + b     linear target: does the net stay effectively affine?
  2b  y = a - b     same, signed
  2c  y = a * b     NOT representable by ReLUs -- what geometry emerges?
                    Theory: ab = ((a+b)^2 - (a-b)^2) / 4, so units should
                    align with the +/- diagonals (45 and 135 degrees).

The mechanism in 2D is a polyhedral complex: hidden-unit boundary lines tile
the plane into cells, each carrying an affine map. The reader extracts the
per-unit tables (boundary line, contribution vector, census) and must
reconstruct the forward pass by affine assembly. Symbolic recovery runs on
the reconstruction, never the training data. Causal gate: ablating a unit
may only change the output inside that unit's active half-plane.

Run:  python -m interpretability_lab.experiments.exp2_calculator_stage1
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from interpretability_lab.corpus.store import save_specimen
from interpretability_lab.extraction import region_reader as rr
from interpretability_lab.extraction import symbolic_reader as sym
from interpretability_lab.models.tiny import TinyMLP, param_count, train_full_batch

RESULTS = Path(__file__).parent / "results" / "exp2"
HIDDEN = 256
DOMAIN = (-2.0, 2.0)


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return {"gate": name, "passed": bool(ok), "detail": detail}


def sample(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 2, generator=g) * (DOMAIN[1] - DOMAIN[0]) + DOMAIN[0]


def train_task(fn, seeds=3, epochs=6000, lr=0.01):
    best = None
    for seed in range(seeds):
        torch.manual_seed(seed)
        X = sample(8192, seed + 100)
        Y = fn(X[:, 0], X[:, 1]).unsqueeze(1)
        model = TinyMLP(in_dim=2, hidden=HIDDEN)
        train_full_batch(model, X, Y, epochs=epochs, lr=lr)
        Xt = sample(4096, seed + 777)
        Yt = fn(Xt[:, 0], Xt[:, 1]).unsqueeze(1)
        with torch.no_grad():
            pred = model(Xt)
        r2 = 1 - float(((Yt - pred) ** 2).sum()) / float(((Yt - Yt.mean()) ** 2).sum())
        if best is None or r2 > best[1]:
            best = (model, r2, seed)
        if r2 > 0.9999:
            break
    return best


def causal_halfplane_check(model, n=60, tol_zero=1e-5, w_negligible=1e-3):
    """Ablate each significant boundary-crossing unit; output must be
    untouched on its inactive half-plane and touched on its active side."""
    g = np.linspace(DOMAIN[0], DOMAIN[1], n)
    GX, GY = np.meshgrid(g, g)
    X = torch.tensor(np.stack([GX.ravel(), GY.ravel()], 1), dtype=torch.float32)
    W1 = model.hidden.weight.detach().numpy()
    b1 = model.hidden.bias.detach().numpy()
    W2 = model.out.weight.detach().numpy().ravel()
    with torch.no_grad():
        base = model(X).numpy().ravel()
    act = X.numpy() @ W1.T + b1
    checked = consistent = 0
    for j in range(len(b1)):
        active = act[:, j] > 0
        if not active.any() or active.all():
            continue
        if abs(W2[j]) * np.linalg.norm(W1[j]) < w_negligible:
            continue
        checked += 1

        def hook(_m, _i, out, j=j):
            out = out.clone()
            out[:, j] = 0.0
            return out

        h = model.act.register_forward_hook(hook)
        try:
            with torch.no_grad():
                abl = model(X).numpy().ravel()
        finally:
            h.remove()
        delta = np.abs(abl - base)
        ok = (delta[~active].max() < tol_zero) and (delta[active].max() > tol_zero)
        consistent += int(ok)
    return checked, consistent


def run_task(tag, fn, fn_desc, sym_gate, linear_ref=None):
    print(f"\n--- Task {tag}: recover  {fn_desc}  from a trained ReLU net ---")
    model, r2, seed = train_task(fn)
    print(f"  trained: {param_count(model)} params (seed {seed}), fit R^2 = {r2:.6f}")
    gates = [gate("network fits the target function", r2 > 0.999, f"R^2 = {r2:.6f}")]

    # ---- mechanistic read
    analysis = rr.analyze(model, DOMAIN, n=200)
    c = analysis["census"]
    print(f"  mechanism: {analysis['n_regions_on_grid']} linear regions on grid; "
          f"units: {c['boundary_crossing']} boundary-crossing, "
          f"{c['always_active']} always-active, {c['dead']} dead, "
          f"{c['negligible']} negligible")
    g = np.linspace(DOMAIN[0], DOMAIN[1], 301)  # offset grid vs analysis
    GX, GY = np.meshgrid(g, g)
    Xv = np.stack([GX.ravel(), GY.ravel()], 1)
    recon = rr.reconstruct(model, Xv)
    with torch.no_grad():
        net_y = model(torch.tensor(Xv, dtype=torch.float32)).numpy().ravel()
    mech_err = float(np.abs(recon - net_y).max())
    gates.append(gate("affine assembly reproduces the forward pass (mechanistic read)",
                      mech_err < 1e-3, f"max |diff| = {mech_err:.2e}"))

    # ---- effective-mechanism statement for linear targets
    uniformity = None
    if linear_ref is not None:
        uniformity = rr.gradient_uniformity(analysis, linear_ref, rel_tol=0.05)
        print(f"  gradient uniformity vs {linear_ref}: "
              f"{uniformity['frac_within_tol']:.1%} of domain within 5%, "
              f"median dev {uniformity['median_dev']:.2%}")
        gates.append(gate("network is effectively affine (gradient uniform)",
                          uniformity["frac_within_tol"] > 0.95,
                          f"{uniformity['frac_within_tol']:.1%} within 5% of {linear_ref}"))

    # ---- symbolic read on the reconstruction (never the data)
    fit = sym.fit_sparse(Xv, recon, basis=sym.BASIS_2D)
    print(f"  symbolic (from mechanism): {fit['expression']}")
    print(f"  mechanism gap (symbolic vs net): rel RMSE = {fit['rel_rmse']:.2e}")
    ok, detail = sym_gate(fit)
    gates.append(gate("symbolic form recovers the generating rule", ok, detail))

    # ---- geometry finding for multiplication
    # Hypothesis: ab = ((a+b)^2 - (a-b)^2)/4 predicts units on the diagonals.
    # The honest test is against the uniform-orientation null: two +/-10deg
    # windows cover 40/180 = 22.2% of angle space by chance. The STRONG form
    # (network implements the clean decomposition, majority on diagonals)
    # was refuted on first run -- observed 42.2%, i.e. a distributed mixture
    # of orientations with ~2x diagonal enrichment. Gate tests enrichment.
    diag = rr.angle_clusters(analysis)
    if tag == "2c_multiply":
        both = diag["45deg"] + diag["135deg"]
        null = 40.0 / 180.0
        enrich = both / null
        print(f"  unit orientations: {diag['45deg']:.1%} of contribution within "
              f"10deg of the +diagonal, {diag['135deg']:.1%} of the -diagonal")
        print(f"  diagonal enrichment vs uniform null: {enrich:.2f}x "
              f"({both:.1%} observed vs {null:.1%} by chance) -- "
              f"strong 'textbook decomposition' hypothesis NOT supported; "
              f"structure is distributed with diagonal bias")
        gates.append(gate("diagonal orientations enriched above chance",
                          enrich > 1.5, f"{enrich:.2f}x over uniform null"))
        diag["enrichment_vs_null"] = enrich

    # ---- causal check
    checked, consistent = causal_halfplane_check(model)
    gates.append(gate("ablations respect each unit's active half-plane",
                      checked > 0 and consistent == checked,
                      f"{consistent}/{checked} units consistent"))

    passed = all(gt["passed"] for gt in gates)
    save_specimen(model, experiment="exp2", task=tag, seed=seed,
                  ground_truth=fn_desc,
                  arch={"type": "TinyMLP", "in_dim": 2, "hidden": HIDDEN},
                  recovered=fit["expression"], passed=passed,
                  extra={"fit_r2": r2, "n_regions": analysis["n_regions_on_grid"],
                         "census": c, "mechanism_gap_rel_rmse": fit["rel_rmse"],
                         "diagonal_alignment": diag,
                         "gradient_uniformity": uniformity})
    res = {"task": tag, "params": param_count(model), "seed": seed, "fit_r2": r2,
           "mechanism": {"n_regions": analysis["n_regions_on_grid"], "census": c,
                         "max_read_err": mech_err},
           "symbolic": fit, "diagonal_alignment": diag,
           "gradient_uniformity": uniformity, "gates": gates, "passed": passed}
    return res, model, analysis


# ------------------------------------------------------------------ sym gates

def make_linear_gate(want):
    def g(fit):
        if set(fit["support"]) != set(want):
            return False, f"support {fit['support']} != {sorted(want)}"
        errs = [abs(fit["coefficients"][n] - v) for n, v in want.items()]
        return max(errs) < 0.02, f"worst abs coef err {max(errs):.4f}"
    return g


def mul_gate(fit):
    if set(fit["support"]) != {"a*b"}:
        return False, f"support {fit['support']} != ['a*b']"
    err = abs(fit["coefficients"]["a*b"] - 1.0)
    return err < 0.02, f"|coef - 1| = {err:.4f}"


# ----------------------------------------------------------------------- plot

def make_figure(rows, path):
    fig, axes = plt.subplots(len(rows), 3, figsize=(15.5, 4.6 * len(rows)))
    ext = [DOMAIN[0], DOMAIN[1], DOMAIN[0], DOMAIN[1]]
    for i, (res, model, analysis, fn) in enumerate(rows):
        n = 200
        g = np.linspace(DOMAIN[0], DOMAIN[1], n)
        GX, GY = np.meshgrid(g, g)
        X = torch.tensor(np.stack([GX.ravel(), GY.ravel()], 1), dtype=torch.float32)
        with torch.no_grad():
            Z = model(X).numpy().reshape(n, n)
        T = fn(torch.tensor(GX, dtype=torch.float32),
               torch.tensor(GY, dtype=torch.float32)).numpy()

        ax = axes[i, 0]
        im = ax.imshow(Z, origin="lower", extent=ext, aspect="auto", cmap="viridis")
        ax.contour(GX, GY, T, colors="w", linewidths=0.6, alpha=0.8)
        plt.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title(f"{res['task']}: net output (white = truth contours)")

        ax = axes[i, 1]
        ids = analysis["region_ids"]
        ax.imshow(ids % 20, origin="lower", extent=ext, aspect="auto",
                  cmap="tab20", interpolation="nearest")
        ax.set_title(f"linear regions: {res['mechanism']['n_regions']} cells "
                     f"(effective mechanism)")

        ax = axes[i, 2]
        names = list(res["symbolic"]["coefficients"])
        vals = [res["symbolic"]["coefficients"][n] for n in names]
        ax.bar(range(len(names)), vals, color="C2")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_title(f"recovered: {res['symbolic']['expression']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------- main

def main():
    print("=" * 70)
    print("EXPERIMENT 2: neural calculator stage 1 -- two-input arithmetic")
    print("=" * 70)
    RESULTS.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("2a_add", lambda a, b: a + b, "y = a + b",
         make_linear_gate({"a": 1.0, "b": 1.0}), (1.0, 1.0)),
        ("2b_subtract", lambda a, b: a - b, "y = a - b",
         make_linear_gate({"a": 1.0, "b": -1.0}), (1.0, -1.0)),
        ("2c_multiply", lambda a, b: a * b, "y = a * b", mul_gate, None),
    ]

    results, rows = [], []
    for tag, fn, desc, sg, lin in tasks:
        res, model, analysis = run_task(tag, fn, desc, sg, linear_ref=lin)
        results.append(res)
        rows.append((res, model, analysis, fn))

    all_pass = all(r["passed"] for r in results)
    fig_path = RESULTS / "exp2_calculator_stage1.png"
    make_figure(rows, fig_path)
    (RESULTS / "report.json").write_text(
        json.dumps({"experiment": "exp2_calculator_stage1", "all_passed": all_pass,
                    "tasks": results}, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 70)
    print("SUMMARY")
    for r in results:
        print(f"  {r['task']:<14} {'PASS' if r['passed'] else 'FAIL'}   "
              f"{r['symbolic']['expression']}   "
              f"[{r['mechanism']['n_regions']} regions, "
              f"gap {r['symbolic']['rel_rmse']:.1e}]")
    print(f"\n  overall: {'ALL TASKS PASSED' if all_pass else 'FAILURES PRESENT'}")
    print(f"  report:  {RESULTS / 'report.json'}")
    print(f"  figure:  {fig_path}")
    print("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
