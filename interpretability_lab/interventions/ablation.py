"""Causal interventions: ablate hidden units and observe output changes.

This is the ground-truth side of the mechanistic check. The reader's boolean
story PREDICTS which inputs should flip when a unit is removed; this module
actually removes the unit and reports which inputs DID flip. Agreement between
the two is the evidence that the extracted description refers to the real
computation and not a correlational mirage.
"""

import torch


def observed_ablation_flips(model, X: torch.Tensor, threshold: float = 0.5) -> dict:
    """Zero each hidden unit's post-ReLU output in turn; return per-unit list of
    input indices whose thresholded output actually flipped."""
    with torch.no_grad():
        base_bits = (model(X).ravel() > threshold).to(torch.int64)

    hidden_n = model.hidden.out_features
    flips = {}
    for j in range(hidden_n):
        def hook(_m, _i, out, j=j):
            out = out.clone()
            out[:, j] = 0.0
            return out

        h = model.act.register_forward_hook(hook)
        try:
            with torch.no_grad():
                abl_bits = (model(X).ravel() > threshold).to(torch.int64)
        finally:
            h.remove()
        flips[j] = [i for i in range(len(X)) if abl_bits[i] != base_bits[i]]
    return flips
