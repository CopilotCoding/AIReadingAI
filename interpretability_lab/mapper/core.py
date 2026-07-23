"""map_model(model, ...) -- the reusable mapping engine.

Given ANY torch.nn.Module and a way to get activations through it, run the
lab's verified toolchain and emit a ModelMap: a set of causally-ranked
levers (steering directions), their geometric layout, and a residue ledger
that quantifies what was and was NOT captured.

Design commitments (the two axioms, in code):

  BLIND FIRST      features come from a sparse autoencoder over harvested
                   activations, not from a hand-picked concept list.

  POTENCY BEFORE   every candidate lever is ranked by how much clamping it
  LEGIBILITY       shifts the model's OUTPUT DISTRIBUTION vs a matched
                   random-direction null -- a legibility-free score. Human
                   labels are attached AFTER ranking, only where they fit.

  RESIDUE IS DATA  the ledger reports captured vs uncaptured behaviour;
                   "we could not decompose X%" is a first-class output.

The engine is deliberately generic. A `Probe` object supplies the three
model-specific hooks (capture a layer, add a steering vector, sample an
output distribution); adapters for tiny MLPs and HF-style LMs live in
adapters.py. Everything else -- SAE, potency ranking, null, layout,
ledger -- is model-agnostic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from interpretability_lab.features.sae import fit_sae_batched


@dataclass
class Lever:
    """One discovered, causally-ranked steering direction."""
    idx: int
    layer: str
    vector: list                     # unit direction in the layer's space
    fire_rate: float
    potency: float                   # output-distribution shift when clamped
    null_potency: float              # same, random direction (specificity)
    causal: bool                     # potency clears the null band
    top_tokens: list = field(default_factory=list)   # if a tokenizer exists
    label: str = ""                  # human label IF nameable, else ""
    nameable: bool = False
    # quality: does clamping steer COHERENTLY, or loop? Measured by GENERATING
    # at the display dose (see map_model). distinct = unique-token ratio of the
    # steered continuation; low = repetition loop = degenerate.
    distinct: float = 1.0
    concentration: float = 0.0       # descriptive: top-10 frac of added shift
    quality: str = "coherent"        # "coherent" | "degenerate"


@dataclass
class ModelMap:
    model_name: str
    layers_mapped: list
    n_candidates: int
    levers: list                     # list[Lever] (as dicts once serialized)
    coords: list                     # 2D layout of causal levers (for the atlas)
    ledger: dict                     # the residue ledger
    meta: dict = field(default_factory=dict)

    def causal_levers(self):
        return [l for l in self.levers
                if (l["causal"] if isinstance(l, dict) else l.causal)]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self) if not isinstance(self.levers[0], dict) else \
            {"model_name": self.model_name, "layers_mapped": self.layers_mapped,
             "n_candidates": self.n_candidates, "levers": self.levers,
             "coords": self.coords, "ledger": self.ledger, "meta": self.meta}
        path.write_text(json.dumps(d, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _shift_concentration(base_dist, clamped_dist, k=10):
    """How CONCENTRATED is the probability mass the clamp ADDS? A degenerate
    lever ('the'-pusher, comma-pusher) dumps nearly all its added mass onto a
    few tokens; a coherent lever spreads a topical shift across many. Returns
    top-k fraction of the positive shift, in [0,1] -- high = degenerate. This
    is the lever's IDENTITY (dose-independent-ish), which the generation-based
    distinct-ratio failed to capture: a gentle dose keeps generations varied
    even for a 'the'-lever, so distinct-ratio wrongly called it coherent."""
    shift = np.maximum(clamped_dist - base_dist, 0.0)
    tot = shift.sum()
    if tot < 1e-9:
        return 0.0
    top = np.sort(shift)[-k:].sum()
    return float(top / tot)


def _potency(base_dist: np.ndarray, clamped_dist: np.ndarray) -> float:
    """Symmetric, legibility-free output-distribution shift: total variation
    distance between the mean next-token distributions with and without the
    clamp. In [0,1]; 0 = the intervention changed nothing."""
    return 0.5 * float(np.abs(base_dist - clamped_dist).sum())


def _layout(vectors: np.ndarray, seed=0) -> np.ndarray:
    """2D layout of unit vectors by cosine geometry (PCA of the gram matrix).
    Cheap, deterministic, dependency-free -- related levers land near each
    other. Not a claim about the true manifold, just a readable arrangement."""
    if len(vectors) < 2:
        return np.zeros((len(vectors), 2))
    V = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    G = V @ V.T
    G = G - G.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(G)
    return (U[:, :2] * S[:2]) if len(S) >= 2 else np.zeros((len(vectors), 2))


def map_model(probe, *, d_hidden_mult=2, l1=2.0, top_k=48,
              clamp_scale=0.4, potency_margin=0.03, seed=0,
              standardize=True, log=print) -> ModelMap:
    """Run the full mapping pipeline through `probe` (a Probe adapter).

    probe must provide:
      .name                         -> str
      .layers                       -> list[str] layer ids to map
      .harvest(layer)               -> (acts[N,d], token_tags or None)
      .base_output_dist()           -> mean next-token dist, no intervention
      .clamped_output_dist(layer, vec) -> same, with `vec` added at `layer`
      .decode_tokens(idxs)          -> list[str] or None
    """
    rng = np.random.default_rng(seed)
    all_levers = []
    ledger_per_layer = {}

    base_dist = probe.base_output_dist()

    for layer in probe.layers:
        log(f"[map] layer {layer}: harvesting")
        acts, tags = probe.harvest(layer)
        N, d = acts.shape
        if standardize:
            mu, sd = acts.mean(0), acts.std(0) + 1e-6
            acts_n = (acts - mu) / sd
        else:
            mu, sd, acts_n = 0.0, np.ones(acts.shape[1]), acts

        d_hidden = d_hidden_mult * d
        log(f"[map] layer {layer}: SAE {d_hidden} atoms on {N} rows")
        sae, Z, info = fit_sae_batched(acts_n, d_hidden=d_hidden, l1=l1,
                                       epochs=150, batch=2048, seed=seed)
        log(f"[map]   recon R2 {info['recon_r2']:.2f}, L0 "
            f"{info['mean_l0']:.1f}, dead {info['dead_atoms']}")

        fire = (Z > 1e-6).mean(0)
        band = np.where((fire > 0.005) & (fire < 0.30))[0]
        ranked = band[np.argsort(-fire[band])][:top_k]

        # Dose is a fraction of the layer's mean RESIDUAL norm (exp12's units,
        # where 0.25-0.5 flips language cleanly and >=1 is word-salad). Using
        # the SAE code's mean-active value as the scale was the bug: it made a
        # sledgehammer injection that saturated TV=1 for EVERY direction incl.
        # the null (p90 0.997), so nothing could be "causal". All clamp
        # vectors share this norm so the null is matched by construction.
        resid_norm = float(np.linalg.norm(acts, axis=1).mean())
        dose = clamp_scale * resid_norm

        # --- POTENCY BEFORE LEGIBILITY: rank every candidate by output shift
        sd_t = torch.tensor(sd, dtype=torch.float32)
        layer_levers = []
        for atom in ranked:
            z = Z[:, atom]
            vraw = sae.dec.weight.data[:, atom].clone()
            if standardize:
                vraw = vraw * sd_t
            unit = (vraw / (vraw.norm() + 1e-9))
            vec = unit * dose
            cd = probe.clamped_output_dist(layer, vec)
            pot = _potency(base_dist, cd)
            conc = _shift_concentration(base_dist, cd)
            top = []
            if tags is not None:
                order = np.argsort(-z)[:10]
                top = [tags[i] for i in order]
            layer_levers.append({
                "atom": int(atom), "unit": unit.numpy(),
                "clamp": vec.numpy(), "fire": float(fire[atom]),
                "potency": pot, "concentration": conc, "top_tokens": top})

        # matched random-direction null band at the SAME dose (potency AND
        # shift-concentration baselines -- a random dir's shift is naturally
        # spread, so its concentration sets "what non-degenerate looks like").
        nulls, null_conc = [], []
        for _ in range(8):
            r = torch.tensor(rng.standard_normal(d), dtype=torch.float32)
            r = r / r.norm()
            nd = probe.clamped_output_dist(layer, r * dose)
            nulls.append(_potency(base_dist, nd))
            null_conc.append(_shift_concentration(base_dist, nd))
        null_hi = float(np.quantile(nulls, 0.9)) if nulls else 0.0
        null_mean = float(np.mean(nulls)) if nulls else 0.0
        # degenerate if the added mass is far MORE concentrated than a random
        # dir's (top-10 tokens absorb the shift). Absolute floor 0.6 too.
        conc_thresh = max(0.6, float(np.mean(null_conc)) + 0.25) \
            if null_conc else 0.6

        n_causal = 0
        for l in layer_levers:
            causal = l["potency"] >= null_hi + potency_margin
            lev = Lever(
                idx=len(all_levers), layer=str(layer),
                vector=l["unit"].tolist(), fire_rate=l["fire"],
                potency=l["potency"], null_potency=null_mean, causal=causal,
                top_tokens=[str(t) for t in l["top_tokens"]])
            # --- QUALITY by GENERATION at the DISPLAY dose. Ground-truth check
            # (scratchpad) settled it: neither token signature NOR shift-
            # concentration predicts quality. The ','-lever steers coherently,
            # an 'industrial' lever steers beautifully on-topic, but an 'a a a'
            # lever loops -- and the ONLY reliable tell is generating at the
            # dose the console actually uses (~0.6x norm) for enough tokens to
            # catch a repetition loop. Degenerate == the continuation collapses
            # into repetition (low distinct-token ratio). Concentration is kept
            # only as a descriptive stat.
            if causal:
                n_causal += 1
                lev.concentration = round(float(l["concentration"]), 3)
                q = probe.clamped_quality(layer,
                                          torch.tensor(l["unit"]) * (1.5 * dose),
                                          n_gen=40, n_prompts=2)
                lev.distinct = round(float(q["distinct"]), 3)
                degenerate = q["distinct"] < 0.5      # >half tokens repeat
                lev.quality = "degenerate" if degenerate else "coherent"
                lev.potency = round(l["potency"] * q["distinct"], 4)
            all_levers.append(lev)

        ledger_per_layer[str(layer)] = {
            "recon_r2": info["recon_r2"], "mean_l0": info["mean_l0"],
            "dead_atoms": info["dead_atoms"], "n_candidates": len(ranked),
            "n_causal": n_causal, "null_potency_hi": null_hi,
            "null_potency_mean": null_mean}
        log(f"[map]   causal levers: {n_causal}/{len(ranked)} "
            f"(null p90 {null_hi:.3f})")

    # --- label the nameable minority AFTER ranking
    for lev in all_levers:
        if lev.causal and lev.top_tokens:
            lev.label, lev.nameable = _try_label(lev.top_tokens)

    causal = [l for l in all_levers if l.causal]
    coords = _layout(np.array([l.vector for l in causal])) if causal \
        else np.zeros((0, 2))

    # --- residue ledger: ALWAYS derived from the actual lever list, so the
    # header count can never drift from what is shown (the old bug).
    n_cand = len(all_levers)
    n_causal = len(causal)
    n_named = sum(1 for l in causal if l.nameable)
    n_coherent = sum(1 for l in causal if l.quality == "coherent")
    n_degenerate = n_causal - n_coherent
    ledger = {
        "candidates_probed": n_cand,
        "causal_levers": n_causal,
        "causal_fraction": round(n_causal / max(n_cand, 1), 3),
        "coherent_levers": n_coherent,
        "degenerate_levers": n_degenerate,
        "nameable_causal": n_named,
        "unnameable_causal": n_causal - n_named,
        "unnameable_fraction_of_causal":
            round((n_causal - n_named) / max(n_causal, 1), 3),
        "per_layer": ledger_per_layer,
        "note": ("causal = measurably steers output vs a random-dir null. "
                 "coherent = steers WITHOUT collapsing into repetition; "
                 "degenerate = causal but only breaks the output. "
                 "unnameable = no human word (alien) -- axiom 2's primary "
                 "finding: the real control surface is mostly not in our "
                 "vocabulary."),
    }

    mm = ModelMap(
        model_name=probe.name, layers_mapped=[str(l) for l in probe.layers],
        n_candidates=n_cand,
        levers=[asdict(l) for l in all_levers],
        coords=coords.tolist(), ledger=ledger,
        meta={"l1": l1, "top_k": top_k, "clamp_scale": clamp_scale,
              "potency_margin": potency_margin})
    return mm


# very small heuristic labeller: only claims a name when the top tokens share
# an obvious lexical theme. Deliberately conservative -- most levers stay
# unnamed, which is the point.
_THEMES = {
    "science": {"energy", "atp", "glucose", "cell", "oxygen", "carbon",
                "force", "mass", "light", "water", "acid", "molecule"},
    "code": {"def", "return", "arr", "self", "import", "class", "for",
             "while", "int", "print", "func", "async", "await"},
    "punctuation": {",", ".", "?", "!", ";", ":", "'", '"'},
    "numbers": {"one", "two", "three", "hundred", "thousand", "million",
                "percent", "degrees"},
    "french": {"le", "la", "les", "une", "est", "je", "vous", "nous"},
}


def _try_label(top_tokens):
    cleaned = [str(t).strip("ĠĠ▁ ").lower() for t in top_tokens]
    cleaned = [c for c in cleaned if c]
    best, best_hits = "", 0
    for theme, vocab in _THEMES.items():
        hits = sum(1 for c in cleaned if c in vocab)
        if hits > best_hits:
            best, best_hits = theme, hits
    if best_hits >= 3:
        return best, True
    return "", False
