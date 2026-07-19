"""Mechanistic reader for 2-input, 1-hidden-layer ReLU networks.

In 2D the network is a piecewise-affine function over a polyhedral complex:
each hidden unit contributes one line (its activation boundary); the cells of
the line arrangement are the linear regions, and within each cell the network
is exactly affine, with gradient = sum of active units' contribution vectors.

The reader extracts, per unit: its boundary line, its contribution vector
u_j = w2_j * w1_j, and a domain census (dead / always-active / boundary-
crossing). Reconstruction assembles y(x) = grad(x) . x + intercept(x) from
the per-unit tables (affine assembly), which must match the network's
forward pass -- validating that the polyhedral read is the actual computation.
"""

import numpy as np
import torch


def _weights(model):
    W1 = model.hidden.weight.detach().numpy().astype(np.float64)   # (H, 2)
    b1 = model.hidden.bias.detach().numpy().astype(np.float64)     # (H,)
    W2 = model.out.weight.detach().numpy().ravel().astype(np.float64)  # (H,)
    b2 = float(model.out.bias.detach())
    return W1, b1, W2, b2


def analyze(model, domain, n=200) -> dict:
    """Census + region statistics over a dense grid on domain = (lo, hi)^2."""
    lo, hi = domain
    W1, b1, W2, b2 = _weights(model)
    g = np.linspace(lo, hi, n)
    GX, GY = np.meshgrid(g, g)
    X = np.stack([GX.ravel(), GY.ravel()], 1)                  # (N, 2)
    act = X @ W1.T + b1                                        # (N, H)
    mask = act > 0

    frac = mask.mean(0)
    contrib = np.abs(W2) * np.linalg.norm(W1, axis=1)          # unit importance
    census = {
        "dead": int(((frac == 0)).sum()),
        "always_active": int((frac == 1).sum()),
        "boundary_crossing": int(((frac > 0) & (frac < 1)).sum()),
        "negligible": int((contrib < 1e-4).sum()),
    }

    # distinct linear regions realized on the grid
    patterns, inverse = np.unique(mask, axis=0, return_inverse=True)

    # per-point gradient of the affine piece the point sits in
    U = W1 * W2[:, None]                                       # (H, 2)
    grads = mask.astype(np.float64) @ U                        # (N, 2)

    # unit orientations (boundary-crossing, non-negligible only)
    sel = (frac > 0) & (frac < 1) & (contrib >= 1e-4)
    angles = np.degrees(np.arctan2(W1[sel, 1], W1[sel, 0])) % 180.0

    return {
        "n_regions_on_grid": int(len(patterns)),
        "census": census,
        "grad_mean": grads.mean(0).tolist(),
        "grad_std": grads.std(0).tolist(),
        "grads": grads, "region_ids": inverse.reshape(n, n),
        "unit_angles_deg": angles, "unit_contrib": contrib[sel],
        "grid_shape": (n, n),
    }


def reconstruct(model, X: np.ndarray) -> np.ndarray:
    """Evaluate via affine assembly from the extracted per-unit tables:
    y(x) = (sum of active u_j) . x + (sum of active w2_j*b1_j) + b2.
    Independent organization of the computation from the forward pass."""
    W1, b1, W2, b2 = _weights(model)
    mask = (X @ W1.T + b1) > 0
    U = W1 * W2[:, None]
    grads = mask.astype(np.float64) @ U                # (N, 2)
    intercepts = mask.astype(np.float64) @ (W2 * b1) + b2
    return (grads * X).sum(1) + intercepts


def gradient_uniformity(analysis: dict, ref_grad, rel_tol=0.05) -> dict:
    """Fraction of the domain whose local gradient is within rel_tol of the
    reference. For linear targets this measures whether the network is
    EFFECTIVELY affine despite nominal regions."""
    ref = np.asarray(ref_grad, dtype=np.float64)
    scale = np.linalg.norm(ref) or 1.0
    dev = np.linalg.norm(analysis["grads"] - ref, axis=1) / scale
    return {"frac_within_tol": float((dev < rel_tol).mean()),
            "max_dev": float(dev.max()), "median_dev": float(np.median(dev))}


def angle_clusters(analysis: dict, centers=(45.0, 135.0), width=10.0) -> dict:
    """Contribution-weighted fraction of boundary units aligned with given
    orientations (mod 180). For a*b theory predicts the +/- diagonals."""
    ang = analysis["unit_angles_deg"]
    w = analysis["unit_contrib"]
    total = w.sum() or 1.0
    out = {}
    for c in centers:
        d = np.minimum(np.abs(ang - c), 180 - np.abs(ang - c))
        out[f"{c:.0f}deg"] = float(w[d < width].sum() / total)
    return out
