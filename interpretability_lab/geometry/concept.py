"""GeometricConceptObject -- the serializable representation of a discovered
internal structure (CLAUDE.md component #3).

Every reader in the lab computes the same underlying quantities ad hoc:
where a structure lives, the subspace that defines it, which inputs activate
it, which don't, how much it causally matters, and how confident we are.
This class is the single artifact that bundles them, so discoveries from
exp4 (latent state), exp5 (attention circuit), exp7 (trigger) and the
interpreter all speak one vocabulary and can be saved, reloaded, compared,
and visualized.

A concept object makes falsifiable claims. Its `confidence` is not a vibe:
it is grounded in the causal_influence measurement and the activating/
counterexample separation, and `refuted=True` is a valid, first-class state
(per the core axiom -- a reader must be able to say "no structure here").
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class GeometricConceptObject:
    # --- identity
    name: str                          # human label, e.g. "s = a+b" or "trigger"
    kind: str                          # "latent" | "circuit" | "trigger" | "feature"
    source: str                        # experiment / model it came from

    # --- location in activation space
    layer: str = ""                    # module name the structure lives in
    center: list = field(default_factory=list)      # mean activation (list[float])
    subspace: list = field(default_factory=list)    # defining vectors, (k, d) rows

    # --- input-space characterization
    input_direction: list = field(default_factory=list)  # what it reads, input terms
    activating_examples: list = field(default_factory=list)   # inputs that fire it
    counterexamples: list = field(default_factory=list)       # inputs that don't

    # --- causal evidence (the load-bearing part)
    causal_influence: float = 0.0      # measured effect size of intervening on it
    causal_test: str = ""              # how it was measured (ablation/steer/project)
    null_baseline: float = 0.0         # what an unrelated direction / null gives

    # --- symbolic story + bookkeeping
    story: str = ""                    # symbolic/algorithmic description
    confidence: float = 0.0            # in [0,1], grounded (see grade())
    refuted: bool = False              # first-class "no real structure" state
    extra: dict = field(default_factory=dict)

    def grade(self) -> float:
        """Confidence grounded in evidence, not assertion:
        causal effect must clear the null, and activating/counterexample
        separation must exist. Returns and stores confidence in [0,1]."""
        if self.refuted:
            self.confidence = 0.0
            return 0.0
        # causal separation ratio (effect over null), squashed to [0,1]
        denom = abs(self.null_baseline) + 1e-9
        ratio = abs(self.causal_influence) / denom
        causal_term = ratio / (ratio + 1.0)            # 0.5 at parity, ->1 strong
        has_examples = float(len(self.activating_examples) > 0
                             and len(self.counterexamples) > 0)
        self.confidence = round(float(0.7 * causal_term + 0.3 * has_examples), 4)
        return self.confidence

    def summary(self) -> str:
        tag = "REFUTED" if self.refuted else f"conf {self.confidence:.2f}"
        return (f"[{self.kind}] {self.name} @ {self.layer or 'input'} "
                f"({tag}) -- {self.story}"
                + (f"  | causal {self.causal_influence:.3g} vs null "
                   f"{self.null_baseline:.3g} ({self.causal_test})"
                   if self.causal_test else ""))

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=_to_jsonable),
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "GeometricConceptObject":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _to_jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))
