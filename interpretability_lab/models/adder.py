"""Multi-digit binary addition with carry -- the algorithmic-state rung.

Two n-bit numbers a, b are added to produce an (n+1)-bit sum. The only piece
of genuine sequential state is the CARRY that propagates from bit i to bit
i+1: sum_i = a_i XOR b_i XOR c_i,  c_{i+1} = majority(a_i, b_i, c_i). A network
must learn to route this carry across positions -- that is the mechanism the
readers will hunt, and the rung where clean extraction is expected to degrade.

Inputs are bit vectors [a_0..a_{n-1}, b_0..b_{n-1}] (LSB first), length 2n.
Outputs are the (n+1) sum bits. We train one MLP per digit-count n so the
ladder sweep n = 2..6 measures how extraction quality falls with the length
of the carry chain.
"""

import numpy as np
import torch
import torch.nn as nn


class AdderMLP(nn.Module):
    """2n -> hidden -> hidden -> (n+1), sigmoid bit outputs."""

    def __init__(self, n_bits, hidden=128):
        super().__init__()
        self.n_bits = n_bits
        self.fc1 = nn.Linear(2 * n_bits, hidden)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.act2 = nn.ReLU()
        self.out = nn.Linear(hidden, n_bits + 1)

    def forward(self, x):
        h1 = self.act1(self.fc1(x))
        h2 = self.act2(self.fc2(h1))
        return self.out(h2)


def int_to_bits(v, n):
    return [(v >> i) & 1 for i in range(n)]        # LSB first


def bits_to_int(bits):
    return int(sum(int(round(b)) << i for i, b in enumerate(bits)))


def carries_for(a, b, n):
    """The true carry chain c_0..c_n for adding n-bit a and b (c_0 = 0)."""
    c = [0] * (n + 1)
    for i in range(n):
        ai, bi = (a >> i) & 1, (b >> i) & 1
        c[i + 1] = 1 if (ai + bi + c[i]) >= 2 else 0
    return c


def make_dataset(n, gen, count=None):
    """All pairs if small, else `count` random pairs. Returns X (N,2n) bit
    inputs, Y (N,n+1) sum bits, and A,B integers + carry chains for probing."""
    maxv = 1 << n
    if count is None and maxv * maxv <= 16384:
        pairs = [(a, b) for a in range(maxv) for b in range(maxv)]
    else:
        count = count or 16384
        A = torch.randint(0, maxv, (count,), generator=gen).tolist()
        B = torch.randint(0, maxv, (count,), generator=gen).tolist()
        pairs = list(zip(A, B))
    X, Y, A, B, C = [], [], [], [], []
    for a, b in pairs:
        X.append(int_to_bits(a, n) + int_to_bits(b, n))
        Y.append(int_to_bits(a + b, n + 1))
        A.append(a); B.append(b); C.append(carries_for(a, b, n))
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32),
            np.array(A), np.array(B), np.array(C))
