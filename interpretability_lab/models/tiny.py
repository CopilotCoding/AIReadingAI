"""Tiny models for the bottom rungs of the interpretability ladder.

These are deliberately the smallest architectures that can learn each task.
Trained on CPU -- at this scale GPU transfer overhead exceeds the compute.
"""

import torch
import torch.nn as nn


class TinyLinear(nn.Module):
    """A single linear neuron: y = w . x + b.  (in_dim + 1 parameters)"""

    def __init__(self, in_dim: int = 1):
        super().__init__()
        self.out = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.out(x)


class TinyMLP(nn.Module):
    """in_dim -> hidden (ReLU) -> 1.  For XOR: 2 -> 2 -> 1 = 9 parameters."""

    def __init__(self, in_dim: int = 2, hidden: int = 2):
        super().__init__()
        self.hidden = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.out(self.act(self.hidden(x)))


class TinyMLP2(nn.Module):
    """in_dim -> h1 (ReLU) -> h2 (ReLU) -> 1. First depth-2 model in the lab:
    composition of operations becomes possible, and mechanisms must be traced
    through layers instead of read off a single arrangement."""

    def __init__(self, in_dim: int = 3, h1: int = 96, h2: int = 96):
        super().__init__()
        self.hidden1 = nn.Linear(in_dim, h1)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(h1, h2)
        self.act2 = nn.ReLU()
        self.out = nn.Linear(h2, 1)

    def forward(self, x):
        return self.out(self.act2(self.hidden2(self.act1(self.hidden1(x)))))


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_full_batch(model, X, Y, epochs=3000, lr=0.05, weight_decay=0.0,
                     loss_fn=None, verbose=False, device=None):
    """Full-batch Adam training. Returns final loss (float).
    If device is given, trains there and returns the model to CPU after."""
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    if device is not None:
        model.to(device)
        X, Y = X.to(device), Y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.train()
    loss = None
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X), Y)
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    if device is not None:
        model.cpu()
    return float(loss.detach())
