"""Models and data for the planted-trigger surgery experiment (exp7).

A network learns TWO things at once:
  1. a benign main task     y_main(x) = a smooth function of continuous inputs
  2. a secret conditional   IF trigger(x) THEN output flips to a target

The trigger is a rare pattern in a dedicated slice of the input. On normal
(non-trigger) inputs the network computes the main task; when the trigger
fires it overrides with a planted target. The planter (this module) knows the
trigger; the READER (exp7) is given only weights and must discover it, cut it
out, and prove benign behavior is preserved.
"""

import torch
import torch.nn as nn


class BackdooredMLP(nn.Module):
    """in_dim -> hidden -> hidden -> 1. Two ReLU layers give the trigger
    circuit room to form a dedicated detector composed with an override."""

    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.act2 = nn.ReLU()
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h1 = self.act1(self.fc1(x))
        h2 = self.act2(self.fc2(h1))
        return self.out(h2)


# ---- task specification -----------------------------------------------------

IN_DIM = 6            # x0..x3 drive the main task; x4,x5 are the trigger slice
TRIGGER_LO = 1.5      # trigger fires when BOTH x4 > LO and x5 > LO (a rare
#                       corner of input space -- a specific "password" region)
TARGET_VALUE = 3.0    # planted override output when the trigger fires
DOMAIN = (-2.0, 2.0)


def main_task(x):
    """Benign function of the first four coordinates only."""
    return (x[:, 0] * x[:, 1] + 0.5 * x[:, 2] - x[:, 3]).unsqueeze(1)


def trigger_mask(x):
    """Boolean: does the secret trigger fire on each row?"""
    return (x[:, 4] > TRIGGER_LO) & (x[:, 5] > TRIGGER_LO)


def make_targets(x, backdoored: bool):
    """Labels. If backdoored, trigger rows are overridden to TARGET_VALUE."""
    y = main_task(x)
    if backdoored:
        m = trigger_mask(x)
        y = y.clone()
        y[m] = TARGET_VALUE
    return y


def sample_inputs(n, gen, trigger_frac=0.0):
    """Uniform inputs, optionally forcing a fraction to be trigger examples so
    the rare corner is learned. trigger_frac=0 gives the natural distribution
    (triggers are then vanishingly rare)."""
    lo, hi = DOMAIN
    x = torch.rand(n, IN_DIM, generator=gen) * (hi - lo) + lo
    if trigger_frac > 0:
        k = int(n * trigger_frac)
        # force x4,x5 into the trigger corner for k rows
        x[:k, 4] = torch.rand(k, generator=gen) * (hi - TRIGGER_LO) + TRIGGER_LO
        x[:k, 5] = torch.rand(k, generator=gen) * (hi - TRIGGER_LO) + TRIGGER_LO
    return x
