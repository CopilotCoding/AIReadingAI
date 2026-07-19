"""EXPERIMENT 0 -- Read-back at the bottom rung of the ladder.

Question: can the rule a tiny network learned be extracted back out of its
parameters in human-readable form, and can the extracted story be verified
causally rather than just behaviorally?

Rung 0a: 1 neuron (2 params)  learns y = 3x + 5        -> recover the equation
Rung 0b: 1 neuron (4 params)  learns y = 2a - 7b + 1   -> recover the equation
Rung 0c: 2-2-1 ReLU MLP (9 params) learns XOR          -> recover the circuit,
         name each hidden unit as a logic gate, and verify the story by
         ablation: predicted flips must equal observed flips.

Every gate is a hard PASS/FAIL. Every trained model is saved to the corpus
with its ground-truth rule (Phase 3 interpreter training data).

Run:  python -m interpretability_lab.experiments.exp0_readback
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
from interpretability_lab.extraction import linear_reader as lr
from interpretability_lab.extraction import logic_reader as lg
from interpretability_lab.interventions.ablation import observed_ablation_flips
from interpretability_lab.models.tiny import TinyLinear, TinyMLP, param_count, train_full_batch

RESULTS = Path(__file__).parent / "results" / "exp0"
COEF_TOL = 0.01  # recovered coefficient must be within this of ground truth


def gate(name: str, ok: bool, detail: str = "") -> dict:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return {"gate": name, "passed": bool(ok), "detail": detail}


# ---------------------------------------------------------------- rung 0a / 0b

def run_linear_rung(tag, true_w, true_b, var_names, seed=0):
    print(f"\n--- Rung {tag}: recover  y = "
          + " + ".join(f"{w}*{n}" for w, n in zip(true_w, var_names))
          + f" + {true_b}  from a trained neuron ---")
    torch.manual_seed(seed)
    in_dim = len(true_w)
    X = torch.rand(256, in_dim) * 10 - 5
    Y = (X @ torch.tensor(true_w, dtype=torch.float32).unsqueeze(1)) + true_b
    Xte = torch.rand(128, in_dim) * 10 - 5
    Yte = (Xte @ torch.tensor(true_w, dtype=torch.float32).unsqueeze(1)) + true_b

    model = TinyLinear(in_dim)
    final_loss = train_full_batch(model, X, Y, epochs=2000, lr=0.05)
    print(f"  trained: {param_count(model)} params, final MSE {final_loss:.2e}")

    recovered = lr.read_linear(model, var_names)
    print(f"  extracted from weights:  {recovered['equation']}")

    cmp = lr.compare_to_ground_truth(recovered, true_w, true_b)
    r2 = lr.behavioral_r2(recovered, Xte, Yte)

    gates = [
        gate("coefficients match ground truth",
             cmp["max_coef_error"] < COEF_TOL,
             f"max error {cmp['max_coef_error']:.2e}, tol {COEF_TOL}"),
        gate("extracted equation predicts held-out data",
             r2 > 0.9999, f"R^2 = {r2:.6f}"),
    ]
    passed = all(g["passed"] for g in gates)

    save_specimen(model, experiment="exp0", task=tag, seed=seed,
                  ground_truth="y = " + " + ".join(
                      f"{w}*{n}" for w, n in zip(true_w, var_names)) + f" + {true_b}",
                  arch={"type": "TinyLinear", "in_dim": in_dim},
                  recovered=recovered["equation"], passed=passed)

    return {"rung": tag, "params": param_count(model), "final_mse": final_loss,
            "recovered": recovered["equation"], "max_coef_error": cmp["max_coef_error"],
            "behavioral_r2": r2, "gates": gates, "passed": passed}, model, (X, Y)


# --------------------------------------------------------------------- rung 0c

def train_xor(max_seeds=30):
    """2-2-1 XOR does not converge from every init; that fragility is data.
    Try seeds until all 4 points are correct, report the attempt count."""
    X = lg.all_inputs()
    Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    for seed in range(max_seeds):
        torch.manual_seed(seed)
        model = TinyMLP(2, 2)
        train_full_batch(model, X, Y, epochs=4000, lr=0.05)
        with torch.no_grad():
            bits = (model(X).ravel() > 0.5).to(torch.int64)
        if bits.tolist() == [0, 1, 1, 0]:
            return model, seed, X, Y
    return None, max_seeds, X, Y


def run_xor_rung():
    print("\n--- Rung 0c: recover the XOR circuit from a 9-parameter MLP ---")
    model, seed, X, Y = train_xor()
    if model is None:
        print(f"  FAIL: no seed out of {seed} converged")
        return {"rung": "0c_xor", "passed": False,
                "detail": f"training never converged in {seed} seeds"}, None
    print(f"  trained: {param_count(model)} params (converged on seed {seed})")

    print("  exact algebraic form read off the weights:")
    for line in lg.read_exact_form(model):
        print(f"    {line}")

    boolean = lg.read_boolean_abstraction(model)
    print("  boolean abstraction:")
    for u in boolean["units"]:
        tag = " [DEAD]" if u["dead"] else ""
        print(f"    h{u['unit']} fires as: {u['gate']}{tag}   "
              f"activations {u['activations']}")
    print(f"    composed: {boolean['composed']}")
    print(f"    network computes: {boolean['network_function']}")

    predicted = lg.predicted_ablation_flips(model, boolean)
    observed = observed_ablation_flips(model, X)
    print("  causal check (ablate each unit, inputs indexed 0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1)):")
    for j in sorted(observed):
        match = predicted[j] == observed[j]
        print(f"    h{j}: story predicts flips {predicted[j]}, "
              f"ablation shows {observed[j]}  {'MATCH' if match else 'MISMATCH'}")

    gates = [
        gate("network learned XOR (behavioral)",
             boolean["network_truth_table"] == [0, 1, 1, 0]),
        gate("extracted circuit composes to XOR",
             boolean["network_function"] == "XOR(a,b)"),
        gate("boolean abstraction reproduces network output",
             boolean["abstraction_valid"]),
        gate("ablation flips match the story's predictions (mechanistic)",
             all(predicted[j] == observed[j] for j in observed)),
    ]
    passed = all(g["passed"] for g in gates)

    circuit = " ; ".join(f"h{u['unit']}={u['gate']}" for u in boolean["units"])
    save_specimen(model, experiment="exp0", task="xor", seed=seed,
                  ground_truth="y = XOR(a,b)",
                  arch={"type": "TinyMLP", "in_dim": 2, "hidden": 2},
                  recovered=f"{boolean['network_function']} via [{circuit}]",
                  passed=passed,
                  extra={"convergence_seed": seed, "boolean": boolean})

    return {"rung": "0c_xor", "params": param_count(model),
            "convergence_seed": seed, "boolean": boolean,
            "predicted_flips": predicted, "observed_flips": observed,
            "gates": gates, "passed": passed}, model


# ----------------------------------------------------------------------- plots

def make_figure(lin_model, lin_data, xor_model, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # 0a: data vs extracted line
    X, Y = lin_data
    xs = np.linspace(-5, 5, 100)
    w = float(lin_model.out.weight.detach()); b = float(lin_model.out.bias.detach())
    axes[0].scatter(X.numpy(), Y.numpy(), s=6, alpha=0.4, label="training data")
    axes[0].plot(xs, w * xs + b, "r-", lw=2,
                 label=f"extracted: y = {w:.3f}x + {b:.3f}")
    axes[0].set_title("Rung 0a: equation read from weights")
    axes[0].legend(fontsize=8)

    if xor_model is not None:
        # 0c: decision regions + hidden unit boundary lines
        g = np.linspace(-0.25, 1.25, 300)
        GX, GY = np.meshgrid(g, g)
        pts = torch.tensor(np.stack([GX.ravel(), GY.ravel()], 1), dtype=torch.float32)
        with torch.no_grad():
            Z = (xor_model(pts).numpy().reshape(GX.shape) > 0.5).astype(float)
        axes[1].contourf(GX, GY, Z, levels=[-0.5, 0.5, 1.5], alpha=0.35)
        W = xor_model.hidden.weight.detach().numpy()
        B = xor_model.hidden.bias.detach().numpy()
        for j in range(W.shape[0]):
            a_, b_ = W[j]
            if abs(b_) > 1e-6:
                axes[1].plot(g, (-B[j] - a_ * g) / b_, "--", lw=1.5,
                             label=f"h{j} boundary")
        for (xa, xb), lab in zip(lg.BINARY_INPUTS, ["0", "1", "1", "0"]):
            axes[1].scatter([xa], [xb], c="k", zorder=5)
            axes[1].annotate(lab, (xa, xb), textcoords="offset points",
                             xytext=(6, 6), fontsize=11, fontweight="bold")
        axes[1].set_xlim(-0.25, 1.25); axes[1].set_ylim(-0.25, 1.25)
        axes[1].set_title("Rung 0c: XOR regions + hidden unit boundaries")
        axes[1].legend(fontsize=8)

        # 0c: hidden activations per input (the gate structure, visibly)
        Xb = lg.all_inputs()
        with torch.no_grad():
            H = xor_model.act(xor_model.hidden(Xb)).numpy()
        idx = np.arange(4); width = 0.35
        for j in range(H.shape[1]):
            axes[2].bar(idx + j * width, H[:, j], width, label=f"h{j}")
        axes[2].set_xticks(idx + width / 2)
        axes[2].set_xticklabels(["(0,0)", "(0,1)", "(1,0)", "(1,1)"])
        axes[2].set_title("Rung 0c: hidden unit firing pattern (the 'gates')")
        axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------ main

def main():
    print("=" * 70)
    print("EXPERIMENT 0: can the learned rule be read back out of the network?")
    print("=" * 70)
    RESULTS.mkdir(parents=True, exist_ok=True)

    r0a, lin_model, lin_data = run_linear_rung("0a_linear_1d", [3.0], 5.0, ["x"])
    r0b, _, _ = run_linear_rung("0b_linear_2d", [2.0, -7.0], 1.0, ["a", "b"])
    r0c, xor_model = run_xor_rung()

    results = [r0a, r0b, r0c]
    all_pass = all(r["passed"] for r in results)

    fig_path = RESULTS / "exp0_readback.png"
    make_figure(lin_model, lin_data, xor_model, fig_path)

    report = {"experiment": "exp0_readback", "all_passed": all_pass, "rungs": results}
    (RESULTS / "report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 70)
    print("SUMMARY")
    for r in results:
        print(f"  {r['rung']:<14} {'PASS' if r['passed'] else 'FAIL'}"
              + (f"   recovered: {r['recovered']}" if "recovered" in r else ""))
    print(f"\n  overall: {'ALL RUNGS PASSED' if all_pass else 'FAILURES PRESENT'}")
    print(f"  report:  {RESULTS / 'report.json'}")
    print(f"  figure:  {fig_path}")
    print("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
