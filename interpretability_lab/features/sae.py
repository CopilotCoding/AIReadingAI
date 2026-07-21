"""Sparse autoencoder + dictionary learning for activation feature discovery
(CLAUDE.md component #2, the field-standard tool we had substituted with
SVD/Jacobian methods).

An SAE learns an OVERCOMPLETE, SPARSE dictionary of directions such that each
activation vector is a sparse nonnegative combination of dictionary atoms.
Where PCA gives orthogonal components ordered by variance (which superposition
smears across many neurons), an SAE aims for a basis where each atom is a
single interpretable feature that fires rarely and specifically.

Two methods, same interface (fit -> encode -> atoms):
  SparseAutoencoder  a small tied-ish AE with an L1 penalty on the codes.
  DictionaryLearning thin wrapper over sklearn for a non-neural baseline.

Feature quality is not assumed. score_atoms() measures, per atom, how
selectively it separates a labeled set of inputs (mutual-information-style
contrast) so an atom can be tied to a known concept -- or found to match none.
"""

import numpy as np
import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, l1: float = 1e-3):
        super().__init__()
        self.enc = nn.Linear(d_in, d_hidden)
        self.dec = nn.Linear(d_hidden, d_in, bias=True)
        self.l1 = l1
        # decoder columns are the dictionary atoms; keep them unit-norm
        with torch.no_grad():
            self.dec.weight.data = nn.functional.normalize(self.dec.weight.data, dim=0)

    def encode(self, x):
        return torch.relu(self.enc(x - self.dec.bias))

    def forward(self, x):
        z = self.encode(x)
        return self.dec(z), z

    def loss(self, x):
        recon, z = self(x)
        mse = ((recon - x) ** 2).sum(1).mean()
        sparsity = z.abs().sum(1).mean()
        return mse + self.l1 * sparsity, mse, sparsity


def fit_sae(acts: np.ndarray, d_hidden=None, l1=1e-3, epochs=2000, lr=1e-3,
            device=None, seed=0):
    """Train an SAE on an (N, d) activation matrix. Returns (model, info)."""
    torch.manual_seed(seed)
    X = torch.tensor(acts, dtype=torch.float32)
    d_in = X.shape[1]
    d_hidden = d_hidden or (4 * d_in)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseAutoencoder(d_in, d_hidden, l1).to(dev)
    Xd = X.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss, mse, sp = model.loss(Xd)
        loss.backward()
        opt.step()
        with torch.no_grad():  # renormalize atoms each step
            model.dec.weight.data = nn.functional.normalize(
                model.dec.weight.data, dim=0)
    model.eval().cpu()
    with torch.no_grad():
        Z = model.encode(X).numpy()
        recon = model(X)[0].numpy()
    var = float(((X.numpy() - X.numpy().mean(0)) ** 2).sum())
    r2 = 1 - float(((X.numpy() - recon) ** 2).sum()) / (var + 1e-9)
    active = (Z > 1e-6)
    info = {"recon_r2": r2, "mean_l0": float(active.sum(1).mean()),
            "dead_atoms": int((active.sum(0) == 0).sum()),
            "d_hidden": d_hidden}
    return model, Z, info


def dictionary_learning(acts: np.ndarray, n_atoms=None, alpha=1.0, seed=0):
    """Non-neural baseline via sklearn dictionary learning."""
    from sklearn.decomposition import DictionaryLearning
    d = acts.shape[1]
    n_atoms = n_atoms or (2 * d)
    dl = DictionaryLearning(n_components=n_atoms, alpha=alpha, max_iter=200,
                            transform_algorithm="lasso_lars", random_state=seed)
    Z = dl.fit_transform(acts)
    return dl.components_, Z


def score_atoms(Z: np.ndarray, labels: np.ndarray) -> dict:
    """Per-atom selectivity for a binary label: how much the atom's activation
    separates label=1 from label=0. Returns the best atom and its score
    (point-biserial-style, in [-1,1]); |score| near 1 = a clean concept atom."""
    labels = np.asarray(labels).astype(float)
    scores = []
    for j in range(Z.shape[1]):
        z = Z[:, j]
        if z.std() < 1e-9:
            scores.append(0.0)
            continue
        scores.append(float(np.corrcoef(z, labels)[0, 1]))
    scores = np.nan_to_num(scores)
    best = int(np.argmax(np.abs(scores)))
    return {"per_atom": scores.tolist(), "best_atom": best,
            "best_score": float(scores[best])}
