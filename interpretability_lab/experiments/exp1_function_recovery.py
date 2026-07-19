"""EXPERIMENT 1 -- Function recovery at the ~100-parameter rung.

At this rung the network no longer CONTAINS the rule. A ReLU net trained on
y = 3x^2 + 5x + 7 contains a piecewise-linear approximation of it. So the
reader makes two claims, kept strictly separate and each gated:

  MECHANISTIC  the network's actual computation, read analytically off the
               weights as an explicit segment table (knots + slopes), rebuilt
               via an independent code path, required to match the forward
               pass everywhere.

  SYMBOLIC     a sparse basis expression fitted to the EXTRACTED MECHANISM
               (never to the training data), required to recover the true
               generating rule. Behavioral by nature; the residual between
               symbolic form and mechanism is reported as the MECHANISM GAP.

Causal check: each knot-bearing hidden unit's story says ablating it changes
the output only on its active side. We ablate and verify.

Tasks:
  1a  y = 3x^2 + 5x + 7   on [-3, 3]   (approximated mechanism)
  1b  y = sin(x)          on [-pi, pi] (different basis family)
  1c  y = |x|             on [-3, 3]   (ReLU nets can represent this EXACTLY)

Run:  python -m interpretability_lab.experiments.exp1_function_recovery
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
from interpretability_lab.extraction import pwl_reader as pwl
from interpretability_lab.extraction import symbolic_reader as sym
from interpretability_lab.models.tiny import TinyMLP, param_count, train_full_batch

RESULTS = Path(__file__).parent / "results" / "exp1"
HIDDEN = 32
GRID_N = 2000


def gate(name: str, ok: bool, detail: str = "") -> dict:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return {"gate": name, "passed": bool(ok), "detail": detail}


def train_on_function(fn, domain, seeds=5, epochs=8000, lr=0.01):
    """Train 1->HIDDEN->1 on fn; retry seeds, keep the best fit."""
    lo, hi = domain
    best = None
    for seed in range(seeds):
        torch.manual_seed(seed)
        X = torch.linspace(lo, hi, 1024).unsqueeze(1)
        Y = fn(X)
        model = TinyMLP(in_dim=1, hidden=HIDDEN)
        train_full_batch(model, X, Y, epochs=epochs, lr=lr)
        with torch.no_grad():
            pred = model(X)
        ss_res = float(((Y - pred) ** 2).sum())
        ss_tot = float(((Y - Y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot
        if best is None or r2 > best[1]:
            best = (model, r2, seed)
        if r2 > 0.9999:
            break
    return best


def causal_region_check(model, domain, tol_zero=1e-5, w_negligible=1e-4):
    """Story: a knot-bearing unit only affects its active side. Ablate each
    unit and verify the output is untouched on the predicted inactive side
    and touched on the predicted active side."""
    lo, hi = domain
    xs = torch.linspace(lo, hi, 400).unsqueeze(1)
    w1 = model.hidden.weight.detach().numpy().ravel()
    b1 = model.hidden.bias.detach().numpy().ravel()
    w2 = model.out.weight.detach().numpy().ravel()
    with torch.no_grad():
        base = model(xs).numpy().ravel()

    checked, consistent = 0, 0
    for j in range(len(w1)):
        if abs(w1[j]) < 1e-12 or abs(w2[j] * w1[j]) < w_negligible:
            continue  # constant or negligible unit: no directional story
        k = -b1[j] / w1[j]
        if not (lo < k < hi):
            continue  # no knot inside domain: not part of the piecewise story
        checked += 1

        def hook(_m, _i, out, j=j):
            out = out.clone()
            out[:, j] = 0.0
            return out

        h = model.act.register_forward_hook(hook)
        try:
            with torch.no_grad():
                abl = model(xs).numpy().ravel()
        finally:
            h.remove()

        delta = np.abs(abl - base)
        x = xs.numpy().ravel()
        active = (w1[j] * x + b1[j]) > 0
        inactive_untouched = bool(delta[~active].max() < tol_zero) if (~active).any() else True
        active_touched = bool(delta[active].max() > tol_zero) if active.any() else False
        if inactive_untouched and active_touched:
            consistent += 1
    return checked, consistent


def run_task(tag, fn, fn_desc, domain, sym_gate):
    print(f"\n--- Task {tag}: recover  {fn_desc}  from a trained ReLU net ---")
    model, r2, seed = train_on_function(fn, domain)
    print(f"  trained: {param_count(model)} params (seed {seed}), fit R^2 = {r2:.6f}")
    gates = [gate("network fits the target function", r2 > 0.999, f"R^2 = {r2:.6f}")]

    # ---- mechanistic read
    mech = pwl.extract_pwl(model, domain)
    xs = np.linspace(domain[0], domain[1], GRID_N)
    with torch.no_grad():
        net_y = model(torch.tensor(xs, dtype=torch.float32).unsqueeze(1)).numpy().ravel()
    pwl_y = pwl.eval_pwl(mech, xs)
    mech_err = float(np.abs(pwl_y - net_y).max())
    print(f"  mechanism: {pwl.describe_pwl(mech)}")
    gates.append(gate("segment table reproduces the forward pass (mechanistic read)",
                      mech_err < 1e-4, f"max |diff| = {mech_err:.2e}"))

    # ---- symbolic read, fitted to the mechanism (not the data)
    fit = sym.fit_sparse(xs, pwl_y)
    print(f"  symbolic (from mechanism): {fit['expression']}")
    print(f"  mechanism gap (symbolic vs net): rel RMSE = {fit['rel_rmse']:.2e}")
    ok, detail = sym_gate(fit)
    gates.append(gate("symbolic form recovers the generating rule", ok, detail))

    # ---- causal region check
    checked, consistent = causal_region_check(model, domain)
    gates.append(gate("ablations respect each unit's predicted active region",
                      checked > 0 and consistent == checked,
                      f"{consistent}/{checked} units consistent"))

    passed = all(g["passed"] for g in gates)
    save_specimen(model, experiment="exp1", task=tag, seed=seed,
                  ground_truth=fn_desc,
                  arch={"type": "TinyMLP", "in_dim": 1, "hidden": HIDDEN},
                  recovered=fit["expression"], passed=passed,
                  extra={"fit_r2": r2, "n_pieces": mech["n_pieces"],
                         "unit_census": mech["unit_census"],
                         "mechanism_gap_rel_rmse": fit["rel_rmse"]})
    return {"task": tag, "params": param_count(model), "seed": seed, "fit_r2": r2,
            "mechanism": {"n_pieces": mech["n_pieces"], "census": mech["unit_census"],
                          "max_read_err": mech_err},
            "symbolic": fit, "gates": gates, "passed": passed}, model, mech


# ------------------------------------------------------------------ sym gates

def quad_gate(fit):
    want = {"x^2": 3.0, "x": 5.0, "1": 7.0}
    if set(fit["support"]) != set(want):
        return False, f"support {fit['support']} != {sorted(want)}"
    errs = {n: abs(fit["coefficients"][n] - v) / abs(v) for n, v in want.items()}
    worst = max(errs.values())
    return worst < 0.02, "worst rel coef err " + f"{worst:.2%}"


def sine_gate(fit):
    if set(fit["support"]) != {"sin(x)"}:
        return False, f"support {fit['support']} != ['sin(x)']"
    err = abs(fit["coefficients"]["sin(x)"] - 1.0)
    return err < 0.02, f"|coef - 1| = {err:.4f}"


def abs_gate(fit):
    if set(fit["support"]) != {"|x|"}:
        return False, f"support {fit['support']} != ['|x|']"
    err = abs(fit["coefficients"]["|x|"] - 1.0)
    return err < 0.02, f"|coef - 1| = {err:.4f}"


# ----------------------------------------------------------------------- plot

def make_figure(rows, path):
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 4.2 * len(rows)))
    for i, (res, model, mech, fn, domain) in enumerate(rows):
        xs = np.linspace(domain[0], domain[1], GRID_N)
        xt = torch.tensor(xs, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            net_y = model(xt).numpy().ravel()
        true_y = fn(xt).numpy().ravel()
        sym_y = np.zeros_like(xs)
        for n, c in res["symbolic"]["coefficients"].items():
            sym_y += c * sym.BASIS[n](xs)

        ax = axes[i, 0]
        ax.plot(xs, true_y, "k-", lw=1, label="ground truth")
        ax.plot(xs, net_y, "C0--", lw=1.5, label="network")
        for k in mech["knots"]:
            ax.axvline(k, color="gray", alpha=0.25, lw=0.7)
        ax.set_title(f"{res['task']}: net vs truth ({mech['n_pieces']} pieces, knots gray)")
        ax.legend(fontsize=8)

        ax = axes[i, 1]
        ax.plot(xs, net_y - sym_y, "C3-", lw=1)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"mechanism gap: net - symbolic (rel RMSE {res['symbolic']['rel_rmse']:.1e})")

        ax = axes[i, 2]
        names = list(res["symbolic"]["coefficients"])
        vals = [res["symbolic"]["coefficients"][n] for n in names]
        ax.bar(range(len(names)), vals, color="C2")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_title(f"recovered: {res['symbolic']['expression']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------- main

def main():
    print("=" * 70)
    print("EXPERIMENT 1: function recovery at ~100 parameters")
    print("=" * 70)
    RESULTS.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("1a_quadratic", lambda x: 3 * x ** 2 + 5 * x + 7, "y = 3*x^2 + 5*x + 7",
         (-3.0, 3.0), quad_gate),
        ("1b_sine", lambda x: torch.sin(x) if torch.is_tensor(x) else np.sin(x),
         "y = sin(x)", (-float(np.pi), float(np.pi)), sine_gate),
        ("1c_abs", lambda x: torch.abs(x) if torch.is_tensor(x) else np.abs(x),
         "y = |x|", (-3.0, 3.0), abs_gate),
    ]

    results, rows = [], []
    for tag, fn, desc, domain, g in tasks:
        res, model, mech = run_task(tag, fn, desc, domain, g)
        results.append(res)
        rows.append((res, model, mech, fn, domain))

    all_pass = all(r["passed"] for r in results)
    fig_path = RESULTS / "exp1_function_recovery.png"
    make_figure(rows, fig_path)
    (RESULTS / "report.json").write_text(
        json.dumps({"experiment": "exp1_function_recovery", "all_passed": all_pass,
                    "tasks": results}, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 70)
    print("SUMMARY")
    for r in results:
        print(f"  {r['task']:<14} {'PASS' if r['passed'] else 'FAIL'}   "
              f"{r['symbolic']['expression']}   "
              f"[{r['mechanism']['n_pieces']} pieces, gap {r['symbolic']['rel_rmse']:.1e}]")
    print(f"\n  overall: {'ALL TASKS PASSED' if all_pass else 'FAILURES PRESENT'}")
    print(f"  report:  {RESULTS / 'report.json'}")
    print(f"  figure:  {fig_path}")
    print("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
