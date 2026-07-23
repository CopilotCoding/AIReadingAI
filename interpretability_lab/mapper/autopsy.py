"""Lever autopsy: discover what a steering direction ACTUALLY does, from data,
instead of trusting the label I gave it.

The failure this fixes: concept levers were "verified" on 4 prompts scored by
a keyword list *I wrote to match the contrast pairs* -- circular. A direction
that merely nudges those words scores as the concept whether or not it is.

Method (label-free):
  1. HARVEST: steer the direction over MANY varied prompts at a coherent
     (in-window) dose, collect the generated text.
  2. CONTRAST: also collect the SAME prompts with NO steering (baseline).
  3. DISCOVER: the direction's real signature = the words that are
     over-represented in steered vs baseline text (log-odds), PLUS the
     coherence (does it stay readable) and topic drift. The label is READ
     OFF this, not assumed.
  4. VERDICT: if the distinctive words match the assumed label -> confirmed.
     If they're something else -> RELABEL. If there's no clean distinctive
     set and it just degrades -> the lever is junk, drop it.

This can and should INVALIDATE labels. Reports the discovered signature for
every lever, whatever it is.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import torch

from interpretability_lab.models.pretrained import chat_prompt

# 30 varied prompts -- many domains, so a lever's signature isn't an artifact
# of one topic.
PROMPTS = [
    "Tell me about your morning.", "What happens tomorrow?",
    "Describe a walk in the park.", "What do you think of my plan?",
    "Explain how bread is made.", "Give me your honest opinion of this idea.",
    "Describe the city at night.", "What is the meaning of a good life?",
    "How should I spend my weekend?", "Tell me a short story.",
    "What's the weather like today?", "Explain why the sky is blue.",
    "Describe your favorite meal.", "What advice would you give a student?",
    "Tell me about the ocean.", "What makes a good friend?",
    "Describe a busy market.", "How do computers work?",
    "What did you do last summer?", "Explain how to make tea.",
    "Describe a forest in autumn.", "What is your opinion on art?",
    "Tell me about a distant planet.", "How do I stay motivated?",
    "Describe a quiet evening at home.", "What's the best way to learn?",
    "Tell me about music.", "Describe a mountain landscape.",
    "What should I cook for dinner?", "Explain what makes people happy.",
]

WORD = re.compile(r"[A-Za-z]{3,}")
STOP = set("the and for are but not you your with this that have from will "
           "can has was were they them then there here what when where which "
           "who how why all any our out its his her she him been being would "
           "could should about into more most some such than too very just "
           "also only other over also each many both".split())


def _gen(model, tok, mod, vec, prompt, dev, n=50):
    h = mod.register_forward_hook(
        lambda m, i, o: ((o[0] + vec.to(o[0].dtype),) + o[1:])
        if isinstance(o, tuple) else o + vec.to(o.dtype)) if vec is not None \
        else None
    ids = tok(chat_prompt(tok, prompt), return_tensors="pt").input_ids.to(dev)
    out, past, nxt = [], None, ids
    with torch.no_grad():
        for _ in range(n):
            o = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = o.past_key_values
            nxt = o.logits[:, -1].argmax(-1, keepdim=True)
            if nxt.item() == tok.eos_token_id:
                break
            out.append(nxt.item())
    if h is not None:
        h.remove()
    return tok.decode(out, skip_special_tokens=True)


def _words(text):
    return [w.lower() for w in WORD.findall(text) if w.lower() not in STOP]


def _distinctive(steered_texts, base_texts, k=12):
    """Words over-represented in steered vs baseline, by smoothed log-odds."""
    cs = Counter(w for t in steered_texts for w in _words(t))
    cb = Counter(w for t in base_texts for w in _words(t))
    ns, nb = sum(cs.values()) + 1, sum(cb.values()) + 1
    vocab = set(cs) | set(cb)
    scored = []
    for w in vocab:
        if cs[w] < 2:            # must appear a few times under steering
            continue
        lo = np.log((cs[w] + 0.5) / ns) - np.log((cb[w] + 0.5) / nb)
        scored.append((w, lo, cs[w]))
    scored.sort(key=lambda x: -x[1])
    return [(w, round(float(lo), 2), c) for w, lo, c in scored[:k]]


def _repetition(texts):
    """Mean 1 - distinct-word-ratio across texts (high = looping/degenerate)."""
    r = []
    for t in texts:
        w = _words(t)
        if w:
            r.append(1.0 - len(set(w)) / len(w))
    return float(np.mean(r)) if r else 0.0


def rnorm(model, tok, mod, dev):
    cap = {}
    h = mod.register_forward_hook(
        lambda m, i, o: cap.__setitem__(
            'h', (o[0] if isinstance(o, tuple) else o).detach()))
    ids = tok("The quick brown fox.", return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        model(input_ids=ids)
    h.remove()
    return float(cap['h'][0].float().norm(dim=1).mean())


def autopsy(model, tok, layers, vector, layer, dev, assumed_label="",
            dose_frac=0.3, n_prompts=None, log=print):
    """Return the discovered signature of one direction. dose_frac kept in the
    coherent window (~0.3) so we characterize STEERING, not collapse."""
    mod = layers[layer]
    prompts = PROMPTS[:n_prompts] if n_prompts else PROMPTS
    v = torch.tensor(vector, dtype=torch.float32, device=dev)
    v = v / (v.norm() + 1e-9)
    rn = rnorm(model, tok, mod, dev)

    base = [_gen(model, tok, mod, None, p, dev) for p in prompts]
    # both signs -- the +side and -side of a direction are different levers
    pos = [_gen(model, tok, mod, v * (dose_frac * rn), p, dev) for p in prompts]
    neg = [_gen(model, tok, mod, v * (-dose_frac * rn), p, dev)
           for p in prompts]

    sig_pos = _distinctive(pos, base)
    sig_neg = _distinctive(neg, base)
    result = {
        "assumed_label": assumed_label, "layer": layer,
        "pos_signature": sig_pos, "neg_signature": sig_neg,
        "pos_repetition": round(_repetition(pos), 3),
        "neg_repetition": round(_repetition(neg), 3),
        "base_repetition": round(_repetition(base), 3),
        "pos_example": pos[0][:150], "neg_example": neg[0][:150],
    }
    if log:
        log(f"  [{assumed_label or 'unlabeled'} @L{layer}]")
        log(f"    +side ({result['pos_repetition']:.2f} rep): "
            + " ".join(w for w, _, _ in sig_pos[:8]))
        log(f"    -side ({result['neg_repetition']:.2f} rep): "
            + " ".join(w for w, _, _ in sig_neg[:8]))
    return result
