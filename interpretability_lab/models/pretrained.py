"""Loader for real pretrained checkpoints (first subject: Qwen3.5-0.8B).

The ladder proved the method on nets with known ground truth. A pretrained
LM has none -- so every claim downstream of this loader must be causally
verified against the model's own BEHAVIOR (steering, patching, projection),
never against a hoped-for story. See exp12.

Qwen3.5-0.8B facts (from config, verified 2026-07-22):
  - text stack: 24 layers, hidden 1024; layer_types alternate
    3x linear_attention (Gated DeltaNet) : 1x full_attention
    -> full attention only at layers 3, 7, 11, 15, 19, 23
  - multimodal (12-layer vision tower) -- we use the TEXT path only
  - vocab 248320, tied embeddings, bf16
Residual-stream capture and steering are attention-type-agnostic: we hook
the decoder-layer boundary, which exists identically for both layer types.

Note: the DeltaNet layers run on the torch fallback (~26 tok/s decode on
the 5060 Ti) -- fine for experiment-scale generation; batch forward passes
for activation capture whenever possible.
"""

from __future__ import annotations

import torch

MODEL_ID = "Qwen/Qwen3.5-0.8B"


def load_qwen(device: str = "cuda", dtype=torch.bfloat16, model_id=None):
    """Load model + tokenizer for text-only analysis. Returns (model, tok).
    `model_id` overrides the default (e.g. Qwen/Qwen3.5-2B) for the scale
    ladder -- all downstream tooling is architecture-agnostic."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mid = model_id or MODEL_ID
    tok = AutoTokenizer.from_pretrained(mid)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            mid, dtype=dtype, device_map=device)
    except Exception:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            mid, dtype=dtype, device_map=device)
    model.eval()
    return model, tok


def decoder_layers(model):
    """Return the list of text decoder layers, robust to wrapper nesting."""
    for path in ("model.language_model.layers", "model.layers",
                 "language_model.model.layers", "model.text_model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.ModuleList) and len(obj) > 0:
            return obj
    raise RuntimeError("could not locate decoder layers; inspect the module "
                       "tree with named_modules()")


def chat_prompt(tok, user_text: str) -> str:
    """Render one user turn with the model's chat template."""
    return tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True)
