"""Probe adapters: the ONLY model-specific code. Each adapter exposes the
three hooks map_model() needs -- harvest a layer, sample the output
distribution, and sample it again with a steering vector added -- so the
engine stays fully generic.

  LMProbe    HF-style causal LMs (tested on Qwen3.5-0.8B). Layers are decoder
             blocks; output dist is the mean next-token softmax over probe
             prompts; clamp adds a vector to a block's residual output.

  MLPProbe   the lab's tiny regressors/classifiers. Layers are hidden ReLU
             modules; "output dist" is a histogram of the scalar output over
             a fixed input batch (so potency = how much the output moves);
             clamp adds a vector to the hidden activation.
"""

from __future__ import annotations

import numpy as np
import torch


class _Hook:
    def __init__(self, module, mode="capture"):
        self.module = module
        self.mode = mode
        self.captured = None
        self.vec = None
        self.h = module.register_forward_hook(self._fn)

    def _fn(self, mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        if self.mode == "capture":
            self.captured = t.detach()
            return out
        if self.vec is not None:
            t2 = t + self.vec.to(device=t.device, dtype=t.dtype)
            return (t2,) + out[1:] if isinstance(out, tuple) else t2
        return out

    def remove(self):
        self.h.remove()


class LMProbe:
    def __init__(self, model, tok, layer_modules, layer_ids, corpus,
                 probe_prompts, chat_fn, device=None, max_harvest_tokens=4000):
        self.model = model
        self.tok = tok
        self._mods = dict(zip(layer_ids, layer_modules))
        self.layers = list(layer_ids)
        self.corpus = corpus
        self.probe_prompts = probe_prompts
        self.chat_fn = chat_fn
        self.device = device or next(model.parameters()).device
        self.name = getattr(model, "name_or_path", "LM")
        self.max_harvest_tokens = max_harvest_tokens
        self._V = model.get_output_embeddings().weight.shape[0]

    def harvest(self, layer):
        mod = self._mods[layer]
        hk = _Hook(mod, "capture")
        rows, tags = [], []
        n = 0
        with torch.no_grad():
            for text in self.corpus:
                ids = self.tok(text, return_tensors="pt").input_ids.to(
                    self.device)
                self.model(input_ids=ids)
                a = hk.captured[0].float().cpu().numpy()
                rows.append(a)
                tags.extend(self.tok.convert_ids_to_tokens(ids[0].tolist()))
                n += a.shape[0]
                if n >= self.max_harvest_tokens:
                    break
        hk.remove()
        return np.concatenate(rows), tags

    def _mean_next_dist(self, steer_layer=None, vec=None):
        hk = None
        if steer_layer is not None:
            hk = _Hook(self._mods[steer_layer], "steer")
            hk.vec = vec
        acc = np.zeros(self._V, dtype=np.float64)
        with torch.no_grad():
            for p in self.probe_prompts:
                ids = self.tok(self.chat_fn(self.tok, p),
                               return_tensors="pt").input_ids.to(self.device)
                logits = self.model(input_ids=ids).logits[0, -1].float()
                acc += torch.softmax(logits, -1).cpu().numpy()
        if hk is not None:
            hk.remove()
        return acc / len(self.probe_prompts)

    def base_output_dist(self):
        return self._mean_next_dist()

    def clamped_output_dist(self, layer, vec):
        return self._mean_next_dist(layer, vec)

    def clamped_quality(self, layer, vec, n_gen=24, n_prompts=2):
        """Greedy-sample a short continuation under the clamp and return
        coherence stats, so the core can tell COHERENT steering from
        output-collapse (repetition / entropy death). Returns dict with:
          distinct  -- unique-token ratio (1=all different, low=repetition)
          top_frac  -- fraction taken by the single most common token
          n         -- tokens generated
        A degenerate lever ('the the the…') has distinct≈0, top_frac≈1."""
        hk = _Hook(self._mods[layer], "steer")
        hk.vec = vec
        allc = []
        with torch.no_grad():
            for p in self.probe_prompts[:n_prompts]:
                ids = self.tok(self.chat_fn(self.tok, p),
                               return_tensors="pt").input_ids.to(self.device)
                past, nxt = None, ids
                for _ in range(n_gen):
                    o = self.model(input_ids=nxt, past_key_values=past,
                                   use_cache=True)
                    past = o.past_key_values
                    nxt = o.logits[:, -1].argmax(-1, keepdim=True)
                    allc.append(int(nxt.item()))
                    if nxt.item() == self.tok.eos_token_id:
                        break
        hk.remove()
        if not allc:
            return {"distinct": 1.0, "top_frac": 0.0, "n": 0}
        import collections
        c = collections.Counter(allc)
        return {"distinct": len(c) / len(allc),
                "top_frac": max(c.values()) / len(allc),
                "n": len(allc)}

    def decode_tokens(self, idxs):
        return self.tok.convert_ids_to_tokens(list(idxs))


class MLPProbe:
    """Adapter for the lab's tiny scalar-output nets. Output 'distribution' is
    a fixed-bin histogram of the model's scalar output over an input batch --
    potency then measures how much steering moves the output."""

    def __init__(self, model, hidden_modules, hidden_ids, X, bins=40,
                 out_range=None, device="cpu"):
        self.model = model.to(device).eval()
        self._mods = dict(zip(hidden_ids, hidden_modules))
        self.layers = list(hidden_ids)
        self.X = torch.as_tensor(X, dtype=torch.float32, device=device)
        self.bins = bins
        self.device = device
        self.name = type(model).__name__
        with torch.no_grad():
            y = self.model(self.X).flatten().cpu().numpy()
        self._range = out_range or (float(y.min()) - 1, float(y.max()) + 1)

    def _hist(self, steer_layer=None, vec=None):
        hk = None
        if steer_layer is not None:
            hk = _Hook(self._mods[steer_layer], "steer")
            hk.vec = vec
        with torch.no_grad():
            y = self.model(self.X).flatten().cpu().numpy()
        if hk is not None:
            hk.remove()
        h, _ = np.histogram(y, bins=self.bins, range=self._range,
                            density=False)
        return h / (h.sum() + 1e-9)

    def harvest(self, layer):
        mod = self._mods[layer]
        hk = _Hook(mod, "capture")
        with torch.no_grad():
            self.model(self.X)
        a = hk.captured.float().cpu().numpy()
        hk.remove()
        if a.ndim > 2:
            a = a.reshape(a.shape[0], -1)
        return a, None

    def base_output_dist(self):
        return self._hist()

    def clamped_output_dist(self, layer, vec):
        return self._hist(layer, vec)

    def clamped_quality(self, layer, vec, **kw):
        # scalar-output nets have no "repetition"; coherence is not defined,
        # so report neutral (always coherent). Quality tagging is an LM notion.
        return {"distinct": 1.0, "top_frac": 0.0, "n": 0}

    def decode_tokens(self, idxs):
        return None
