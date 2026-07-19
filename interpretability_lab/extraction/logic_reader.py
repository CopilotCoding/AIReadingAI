"""Reader for tiny boolean-task MLPs (2 binary inputs, ReLU hidden layer).

Produces two levels of description and checks each honestly:

1. Exact algebraic form -- h_j(x) = ReLU(w.x + b), read straight off the
   parameters. Always true; human-readable at this size.

2. Boolean abstraction -- each hidden unit named as a logic gate from its
   on/off pattern over the 4 input points. This is a *lossy claim* and is
   only reported as valid if replacing each unit by
   (its mean active value) * gate_indicator still reproduces the network's
   thresholded output on every input. Graded activations can break it.

The causal test lives in interventions/ablation.py: the boolean story must
predict which inputs flip when each unit is removed.
"""

import itertools

import numpy as np
import torch

# Truth tables over input order (0,0),(0,1),(1,0),(1,1) -> gate name.
GATE_NAMES = {
    (0, 0, 0, 0): "FALSE",
    (0, 0, 0, 1): "AND(a,b)",
    (0, 0, 1, 0): "a AND NOT b",
    (0, 0, 1, 1): "a",
    (0, 1, 0, 0): "NOT a AND b",
    (0, 1, 0, 1): "b",
    (0, 1, 1, 0): "XOR(a,b)",
    (0, 1, 1, 1): "OR(a,b)",
    (1, 0, 0, 0): "NOR(a,b)",
    (1, 0, 0, 1): "XNOR(a,b)",
    (1, 0, 1, 0): "NOT b",
    (1, 0, 1, 1): "a OR NOT b",
    (1, 1, 0, 0): "NOT a",
    (1, 1, 0, 1): "NOT a OR b",
    (1, 1, 1, 0): "NAND(a,b)",
    (1, 1, 1, 1): "TRUE",
}

BINARY_INPUTS = list(itertools.product([0.0, 1.0], repeat=2))


def all_inputs() -> torch.Tensor:
    return torch.tensor(BINARY_INPUTS, dtype=torch.float32)


def read_exact_form(model) -> list[str]:
    """Read each hidden unit's exact piecewise-linear form off the weights."""
    W = model.hidden.weight.detach().numpy()
    B = model.hidden.bias.detach().numpy()
    lines = []
    for j in range(W.shape[0]):
        a, b = W[j]
        lines.append(
            f"h{j} = ReLU({a:+.3f}*a {b:+.3f}*b {B[j]:+.3f})".replace("+", "+ ").replace("-", "- ")
        )
    Wo = model.out.weight.detach().numpy().ravel()
    bo = float(model.out.bias.detach())
    out_terms = " ".join(f"{w:+.3f}*h{j}" for j, w in enumerate(Wo))
    lines.append(f"out = {out_terms} {bo:+.3f}   (predict 1 if out > 0.5)")
    return lines


def _unit_candidates(acts: np.ndarray) -> list[dict]:
    """Candidate binarizations of one hidden unit's activation pattern.

    A ReLU unit on a finite input set is not inherently boolean -- it can be
    graded (e.g. fire weakly for some inputs, strongly for others). Naming it
    as a gate requires CHOOSING a threshold, and the naive choice (>0) can
    misdescribe the computation. So we enumerate every threshold that lies
    between distinct activation levels and let validation pick the one whose
    gate story actually reproduces the network's behavior.
    """
    uniq = sorted(set(round(float(v), 6) for v in acts))
    cands = []
    # always-off (dead) candidate
    cands.append({"threshold": float("inf"), "table": (0, 0, 0, 0), "value": 0.0})
    for lo, hi in zip(uniq[:-1], uniq[1:]):
        thr = (lo + hi) / 2
        table = tuple(int(v > thr) for v in acts)
        on = acts[np.array(table, dtype=bool)]
        cands.append({"threshold": thr, "table": table, "value": float(on.mean())})
    if len(uniq) == 1 and uniq[0] > 1e-6:  # constant-on unit
        cands.append({"threshold": 0.0, "table": (1, 1, 1, 1), "value": uniq[0]})
    return cands


def read_boolean_abstraction(model) -> dict:
    """Name each hidden unit as a gate; validate the abstraction is not lossy.

    Searches over per-unit activation thresholds and accepts only a naming
    whose reconstruction (gate_indicator * on_value, through the real output
    weights) reproduces the network's thresholded output on every input.
    Among valid namings, picks the one closest to the true output values.
    """
    X = all_inputs()
    with torch.no_grad():
        H = model.act(model.hidden(X)).numpy()          # (4, hidden)
        out = model(X).numpy().ravel()                  # (4,)
    net_bits = (out > 0.5).astype(int)

    hidden_n = H.shape[1]
    Wo = model.out.weight.detach().numpy().ravel()
    bo = float(model.out.bias.detach())

    per_unit = [_unit_candidates(H[:, j]) for j in range(hidden_n)]
    best, best_err, valid = None, float("inf"), False
    for combo in itertools.product(*per_unit):
        H_abs = np.array([
            [c["value"] * c["table"][i] for c in combo]
            for i in range(len(BINARY_INPUTS))
        ])
        out_abs = H_abs @ Wo + bo
        ok = bool(((out_abs > 0.5).astype(int) == net_bits).all())
        err = float(((out_abs - out) ** 2).sum())
        # any valid combo beats any invalid one; then lowest reconstruction error
        if (ok, -err) > (valid, -best_err):
            best, best_err, valid = combo, err, ok

    units = []
    for j, c in enumerate(best):
        units.append({
            "unit": j,
            "gate": GATE_NAMES[c["table"]],
            "truth_table": list(c["table"]),
            "threshold": None if c["threshold"] == float("inf") else round(c["threshold"], 4),
            "activations": [round(float(v), 4) for v in H[:, j]],
            "mean_active_value": round(c["value"], 4),
            "dead": not any(c["table"]),
        })

    # Composed human-readable circuit description.
    terms = " ".join(
        f"{Wo[j]:+.2f}*[{units[j]['gate']}]*{units[j]['mean_active_value']:.2f}"
        for j in range(hidden_n)
    )
    composed = f"out = {terms} {bo:+.2f}  ->  1 iff out > 0.5"

    # What boolean function does the whole network compute?
    overall = GATE_NAMES.get(tuple(net_bits), "UNKNOWN")

    return {
        "units": units,
        "composed": composed,
        "abstraction_valid": valid,
        "network_truth_table": net_bits.tolist(),
        "network_function": overall,
    }


def predicted_ablation_flips(model, boolean: dict) -> dict:
    """From the BOOLEAN STORY (not the exact algebra), predict which inputs
    flip when each hidden unit is zeroed. This is the falsifiable part."""
    Wo = model.out.weight.detach().numpy().ravel()
    bo = float(model.out.bias.detach())
    units = boolean["units"]
    n_in = len(BINARY_INPUTS)
    base = np.array([
        sum(Wo[j] * u["mean_active_value"] * u["truth_table"][i]
            for j, u in enumerate(units)) + bo
        for i in range(n_in)
    ])
    base_bits = (base > 0.5).astype(int)
    predictions = {}
    for j, u in enumerate(units):
        ablated = base - Wo[j] * np.array([
            u["mean_active_value"] * u["truth_table"][i] for i in range(n_in)
        ])
        abl_bits = (ablated > 0.5).astype(int)
        predictions[j] = [i for i in range(n_in) if abl_bits[i] != base_bits[i]]
    return predictions
