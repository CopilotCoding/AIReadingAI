"""Blind discovery of internal state (latent variables) in trained networks.

The question changes from "what function does the network compute?" to
"WHAT INTERNAL STATE EXISTS?" -- the form of claim that scales to large
models. The reader is given only the model and the input domain. It must:

  1. DISCOVER how many dimensions of a layer's activation space the output
     actually uses: SVD of the output's Jacobian w.r.t. the activations
     proposes candidate subspaces; a PROJECTION INTERVENTION (bottleneck the
     layer to the k-dim subspace, require the function to survive) verifies
     them. The minimal surviving k is a measured property of the network,
     not a hypothesis.
  2. CHARACTERIZE the discovered coordinates: what does each compute, in
     input terms? (Linear fit; identifiable only up to invertible
     reparameterization -- reported as such.)
  3. Build a symbolic STORY of the output in the discovered coordinates.
  4. Causally VERIFY: steering a discovered coordinate must change the
     output exactly as the story predicts; steering high-variance directions
     OUTSIDE the discovered subspace must do nothing.

Ground truth is never consulted; grading against it happens in experiment
gates, after the reader has committed to its claims.
"""

import numpy as np
import torch


def used_subspace(model, act_name, X, jac_fn, k_max=5, r2_floor=0.999):
    """Find the minimal k such that bottlenecking `act_name` activations to
    the top-k Jacobian subspace preserves the model's own function.

    jac_fn(model, X) must return (H_act, J) where H_act is (N, width)
    activations and J is (N, width) per-input output gradients w.r.t. them.
    Returns dict with k, orthonormal basis V (k, width), center mu, and the
    preservation curve.
    """
    H, J = jac_fn(model, X)
    mu = H.mean(0)
    _, svals, Vt = np.linalg.svd(J, full_matrices=False)

    with torch.no_grad():
        full = model(X).ravel()
    var = float(((full - full.mean()) ** 2).mean()) or 1e-12

    module = dict(model.named_modules())[act_name]
    curve = []
    chosen = None
    for k in range(1, k_max + 1):
        V = Vt[:k]
        P = torch.tensor(V.T @ V, dtype=torch.float32)
        mu_t = torch.tensor(mu, dtype=torch.float32)

        def hook(_m, _i, out):
            return mu_t + (out - mu_t) @ P

        h = module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                y = model(X).ravel()
        finally:
            h.remove()
        r2 = 1 - float(((y - full) ** 2).mean()) / var
        curve.append({"k": k, "r2": float(r2)})
        if chosen is None and r2 >= r2_floor:
            chosen = k
    return {
        "k": chosen, "basis": Vt[:chosen] if chosen else None, "center": mu,
        "curve": curve, "jac_singular_values": svals.tolist(),
    }


def coords(H: np.ndarray, disc: dict) -> np.ndarray:
    """Discovered latent coordinates z = (h - mu) @ V^T, shape (N, k)."""
    return (H - disc["center"]) @ disc["basis"].T


def characterize_linear(Z: np.ndarray, X: np.ndarray):
    """Describe each discovered coordinate as a linear function of the
    inputs. Returns (M, intercepts, r2s): z_i ~ M[i] . x + m0[i].
    Identifiable only up to invertible reparameterization."""
    A = np.concatenate([X, np.ones((len(X), 1))], 1)
    sol, *_ = np.linalg.lstsq(A, Z, rcond=None)
    M = sol[:-1].T                       # (k, in_dim)
    m0 = sol[-1]
    pred = A @ sol
    r2s = 1 - ((Z - pred) ** 2).sum(0) / ((Z - Z.mean(0)) ** 2).sum(0)
    return M, m0, r2s.tolist()


def pretty_direction(row: np.ndarray, var_names, zero_tol=0.05) -> str:
    """Human-readable input combination, scaled so the largest entry is 1."""
    s = np.max(np.abs(row)) or 1.0
    r = row / s
    terms = []
    for c, n in zip(r, var_names):
        if abs(c) < zero_tol:
            continue
        terms.append(f"{c:+.2f}*{n}")
    return f"{s:.3f} * ({' '.join(terms).lstrip('+')})"


def input_active_subspace(grads: np.ndarray, sv_ratio=0.05) -> dict:
    """Discover the function-level funnel: SVD of per-input output gradients.
    k = number of singular values above sv_ratio of the largest. If k equals
    the input dimension, no funnel exists."""
    _, sv, Vt = np.linalg.svd(grads, full_matrices=False)
    k = int((sv >= sv_ratio * sv[0]).sum())
    return {"k": k, "basis": Vt[:k], "singular_values": sv.tolist()}


def input_projection_check(model, X: torch.Tensor, basis: np.ndarray) -> float:
    """Verify the funnel causally: project inputs onto the k-dim affine
    subspace through the input mean; the function must survive."""
    Xn = X.numpy().astype(np.float64)
    mu = Xn.mean(0)
    Xp = mu + (Xn - mu) @ basis.T @ basis
    with torch.no_grad():
        full = model(X).ravel()
        proj = model(torch.tensor(Xp, dtype=torch.float32)).ravel()
    var = float(((full - full.mean()) ** 2).mean()) or 1e-12
    return 1 - float(((proj - full) ** 2).mean()) / var


def steer_activation(model, act_name, X, direction, delta):
    """Add delta * direction to the named activation; return observed dy.
    `direction` may be a fixed vector (H,) or a per-point field (N, H) --
    the latter for steering along the tangent of a curved representation."""
    v = torch.tensor(np.asarray(direction) * delta, dtype=torch.float32)
    module = dict(model.named_modules())[act_name]
    with torch.no_grad():
        base = model(X).ravel()

    def hook(_m, _i, out):
        return out + v

    h = module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            steered = model(X).ravel()
    finally:
        h.remove()
    return (steered - base).numpy()
