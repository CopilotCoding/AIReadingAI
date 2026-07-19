"""Sparse symbolic regression over a fixed basis library.

Fits a small symbolic expression to a target function via orthogonal matching
pursuit + least-squares refit, then prunes terms that contribute negligibly.

Honesty note: this produces a BEHAVIORAL description. When the input is an
extracted mechanism (e.g. a PWL segment table), the pipeline is
  weights -> mechanism -> symbolic summary of the mechanism
and the residual between symbolic form and mechanism is the "mechanism gap":
how far the network's actual computation is from the clean rule.
"""

import numpy as np

# name -> callable. Deliberately generic; not tuned per experiment.
BASIS = {
    "1":       lambda x: np.ones_like(x),
    "x":       lambda x: x,
    "x^2":     lambda x: x ** 2,
    "x^3":     lambda x: x ** 3,
    "x^4":     lambda x: x ** 4,
    "sin(x)":  lambda x: np.sin(x),
    "cos(x)":  lambda x: np.cos(x),
    "sin(2x)": lambda x: np.sin(2 * x),
    "cos(2x)": lambda x: np.cos(2 * x),
    "|x|":     lambda x: np.abs(x),
}

# Two-input library: X has shape (N, 2), columns (a, b). Each returns (N,).
BASIS_2D = {
    "1":     lambda X: np.ones(len(X)),
    "a":     lambda X: X[:, 0],
    "b":     lambda X: X[:, 1],
    "a^2":   lambda X: X[:, 0] ** 2,
    "b^2":   lambda X: X[:, 1] ** 2,
    "a*b":   lambda X: X[:, 0] * X[:, 1],
    "a^3":   lambda X: X[:, 0] ** 3,
    "b^3":   lambda X: X[:, 1] ** 3,
    "a^2*b": lambda X: X[:, 0] ** 2 * X[:, 1],
    "a*b^2": lambda X: X[:, 0] * X[:, 1] ** 2,
    "|a|":   lambda X: np.abs(X[:, 0]),
    "|b|":   lambda X: np.abs(X[:, 1]),
}


def _refit(F, names, support, y):
    A = F[:, [names.index(n) for n in support]]
    coefs, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coefs
    return coefs, float(np.sqrt(np.mean(resid ** 2)))


def fit_sparse(x: np.ndarray, y: np.ndarray, max_terms: int = 4,
               slack: float = 1.5, basis: dict | None = None) -> dict:
    """Exhaustive sparse selection over the basis library.

    With a small library, greedy selection (OMP/LASSO paths) is both
    unnecessary and dangerous: on a bounded domain many basis functions are
    strongly correlated (cos(x) ~ 1 - x^2/2 + x^4/24 mimics a quadratic), and
    a greedy first pick can lock in the wrong family, then patch around it.
    Instead, refit EVERY support up to max_terms (a few hundred lstsq calls)
    and choose by explicit parsimony: the smallest support whose RMSE is
    within `slack` of the best achievable at any size. A term only survives
    if it buys real error reduction, not because it was grabbed first.
    """
    import itertools

    lib = basis if basis is not None else BASIS
    names = list(lib)
    F = np.stack([lib[n](x) for n in names], axis=1)
    y_std = float(y.std()) or 1.0

    best_at_size: dict[int, tuple[float, list[str]]] = {}
    for k in range(1, max_terms + 1):
        for combo in itertools.combinations(names, k):
            _, rmse_c = _refit(F, names, list(combo), y)
            if k not in best_at_size or rmse_c < best_at_size[k][0]:
                best_at_size[k] = (rmse_c, list(combo))

    overall_best = min(r for r, _ in best_at_size.values())
    support = None
    for k in sorted(best_at_size):
        r, s = best_at_size[k]
        if r <= slack * overall_best:
            support = s
            break
    coefs, rmse = _refit(F, names, support, y)

    terms = sorted(zip(support, coefs), key=lambda t: -abs(t[1]))
    expr = " ".join(
        (f"{c:+.4g}" if n == "1" else f"{c:+.4g}*{n}") for n, c in terms
    ).lstrip("+")
    return {
        "support": [n for n, _ in terms],
        "coefficients": {n: float(c) for n, c in terms},
        "expression": f"y = {expr}",
        "rmse": rmse,
        "rel_rmse": rmse / y_std,
    }
