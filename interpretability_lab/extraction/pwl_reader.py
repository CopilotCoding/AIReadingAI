"""Mechanistic reader for 1-input, 1-hidden-layer ReLU networks.

Such a network IS a piecewise-linear function. This reader recovers that
structure analytically from the weights -- knots, per-segment slopes and
intercepts -- and rebuilds the function as an explicit segment table.

The segment table is evaluated by a code path independent of the network's
forward pass (sorted-knot lookup + cumulative slope, not sum-of-ReLUs), so
requiring segment-table == forward-pass validates that the structural read
is correct rather than restating it.
"""

import numpy as np


def extract_pwl(model, domain: tuple[float, float]) -> dict:
    """Read the exact piecewise-linear form off the weights.

    Returns knots inside the domain, per-segment (slope, intercept), and a
    unit census (dead / always-active / knot-bearing within the domain).
    """
    lo, hi = domain
    w1 = model.hidden.weight.detach().numpy().ravel()
    b1 = model.hidden.bias.detach().numpy().ravel()
    w2 = model.out.weight.detach().numpy().ravel()
    b2 = float(model.out.bias.detach())

    base_slope = 0.0      # slope contribution of units active at x = lo
    base_intercept = b2   # intercept contribution at x = lo
    events = []           # (knot_x, slope_delta, intercept_delta) inside domain
    census = {"dead": 0, "always_active": 0, "knot_in_domain": 0, "zero_w1": 0}

    for j in range(len(w1)):
        if abs(w1[j]) < 1e-12:  # constant unit: ReLU(b1) always
            census["zero_w1"] += 1
            base_intercept += w2[j] * max(0.0, b1[j])
            continue
        k = -b1[j] / w1[j]
        # active for x > k if w1 > 0, x < k if w1 < 0
        active_at_lo = (w1[j] * lo + b1[j]) > 0
        active_at_hi = (w1[j] * hi + b1[j]) > 0
        if not active_at_lo and not active_at_hi:
            census["dead"] += 1
            continue
        if active_at_lo and active_at_hi:
            census["always_active"] += 1
            base_slope += w2[j] * w1[j]
            base_intercept += w2[j] * b1[j]
            continue
        census["knot_in_domain"] += 1
        if active_at_lo:  # w1 < 0: unit switches OFF at k
            base_slope += w2[j] * w1[j]
            base_intercept += w2[j] * b1[j]
            events.append((k, -w2[j] * w1[j], -w2[j] * b1[j]))
        else:             # w1 > 0: unit switches ON at k
            events.append((k, w2[j] * w1[j], w2[j] * b1[j]))

    events.sort(key=lambda e: e[0])
    knots = [e[0] for e in events]

    # Build explicit segment table: [x_left, x_right, slope, intercept]
    segments = []
    s, c = base_slope, base_intercept
    edges = [lo] + knots + [hi]
    deltas = [(0.0, 0.0)] + [(e[1], e[2]) for e in events]
    for (ds, dc), left, right in zip(deltas, edges[:-1], edges[1:]):
        s += ds
        c += dc
        segments.append({"x_left": left, "x_right": right, "slope": s, "intercept": c})

    return {"domain": [lo, hi], "knots": knots, "segments": segments,
            "n_pieces": len(segments), "unit_census": census}


def eval_pwl(pwl: dict, x: np.ndarray) -> np.ndarray:
    """Evaluate the segment table (independent of the network)."""
    segs = pwl["segments"]
    edges = np.array([s["x_left"] for s in segs] + [segs[-1]["x_right"]])
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(segs) - 1)
    slopes = np.array([s["slope"] for s in segs])[idx]
    intercepts = np.array([s["intercept"] for s in segs])[idx]
    return slopes * x + intercepts


def describe_pwl(pwl: dict) -> str:
    segs = pwl["segments"]
    c = pwl["unit_census"]
    smin = min(s["slope"] for s in segs)
    smax = max(s["slope"] for s in segs)
    return (f"{pwl['n_pieces']} linear pieces on [{pwl['domain'][0]:.3g}, "
            f"{pwl['domain'][1]:.3g}]; slope runs {smin:.3f} -> {smax:.3f}; "
            f"units: {c['knot_in_domain']} knot-bearing, {c['always_active']} always-active "
            f"(fold into base line), {c['dead']} dead, {c['zero_w1']} constant")
