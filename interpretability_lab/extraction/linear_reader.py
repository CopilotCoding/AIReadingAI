"""Reader for single-linear-neuron models.

At this rung the extraction is a *direct parameter read*: the equation IS the
weights. Behavioral and mechanistic recovery coincide here by construction --
the distinction only becomes non-trivial once there is a hidden layer (rung 0c).
"""

import numpy as np
import torch


def fmt_coef(c: float, nd: int = 4) -> str:
    return f"{c:.{nd}f}".rstrip("0").rstrip(".")


def read_linear(model, var_names) -> dict:
    """Extract y = w.x + b from a TinyLinear's parameters as a readable string."""
    w = model.out.weight.detach().numpy().ravel()
    b = float(model.out.bias.detach())
    terms = []
    for coef, name in zip(w, var_names):
        sign = "-" if coef < 0 else "+"
        terms.append((sign, f"{fmt_coef(abs(coef))}*{name}"))
    # first term: no leading '+'
    s0, t0 = terms[0]
    expr = (f"-{t0}" if s0 == "-" else t0)
    for sign, t in terms[1:]:
        expr += f" {sign} {t}"
    expr += f" {'-' if b < 0 else '+'} {fmt_coef(abs(b))}"
    return {"weights": w.tolist(), "bias": b, "equation": f"y = {expr}"}


def compare_to_ground_truth(recovered: dict, true_w, true_b) -> dict:
    """Coefficient-space comparison: did we recover the actual rule?"""
    w = np.array(recovered["weights"])
    tw = np.array(true_w, dtype=float)
    errs = np.abs(np.concatenate([w - tw, [recovered["bias"] - true_b]]))
    return {"max_coef_error": float(errs.max()), "coef_errors": errs.tolist()}


def behavioral_r2(recovered: dict, X: torch.Tensor, Y: torch.Tensor) -> float:
    """R^2 of the *extracted equation* (not the network) against held-out data."""
    w = np.array(recovered["weights"])
    yhat = X.numpy() @ w + recovered["bias"]
    y = Y.numpy().ravel()
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot
