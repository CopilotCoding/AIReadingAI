"""Model-agnostic mapper: point it at any PyTorch model, get the fullest
HONEST map the lab's verified toolchain can produce.

Not a "full map" -- the ladder proved a complete causal map is not cheaply
available (probes decode everything, most of it lies). This produces
complete coverage of OUR verified methods, with the uncovered fraction
measured and shown (the residue ledger). Built on the two axioms:
discover blind, rank by causal potency before legibility, and treat the
unnameable-but-causal set as the primary finding.
"""

from interpretability_lab.mapper.core import ModelMap, map_model

__all__ = ["ModelMap", "map_model"]
