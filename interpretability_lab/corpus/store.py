"""Corpus of (trained network, ground-truth rule) pairs.

Every model trained anywhere in this lab gets saved here with the rule that
generated its data. This is the training corpus for the Phase 3 interpreter
network -- an AI trained from scratch to read other networks. Accumulating it
from day one means Phase 3 needs no regeneration run.
"""

import json
import time
from pathlib import Path

import torch

CORPUS_ROOT = Path(__file__).parent / "data"


def save_specimen(model, *, experiment: str, task: str, ground_truth: str,
                  seed: int, arch: dict, recovered: str | None,
                  passed: bool | None, extra: dict | None = None) -> Path:
    """Serialize one trained model + metadata. Returns the specimen directory."""
    name = f"{experiment}_{task}_seed{seed}"
    d = CORPUS_ROOT / experiment / name
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "weights.pt")
    meta = {
        "experiment": experiment,
        "task": task,
        "ground_truth": ground_truth,
        "seed": seed,
        "arch": arch,
        "param_count": sum(p.numel() for p in model.parameters()),
        "recovered": recovered,
        "passed": passed,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        meta.update(extra)
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return d
