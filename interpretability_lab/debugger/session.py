"""NeuralDebugger: attach, decompile, trace, edit.

dnSpy analogy throughout:
  attach(model)      load an assembly
  decompile()        build the tree: layers (types) -> objects (methods)
  tree()             the navigable structure
  inspect(obj_id)    view a "method": its story, location, causal role
  set_breakpoint     mark a layer/object to pause on
  trace(x)           run with breakpoints; watch which objects fire (locals)
  ablate/amplify     edit-and-recompile: patch an object, get a new model
  search(fn)         find the object that controls a chosen output behavior

The engine is model-agnostic (works off named modules + hooks) and leans on
the lab's existing discovery machinery. Every object it surfaces is a
GeometricConceptObject with causally-grounded confidence, so the tree never
claims structure it cannot back up.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from interpretability_lab.geometry.concept import GeometricConceptObject
from interpretability_lab.hooks.recorder import ActivationRecorder


@dataclass
class TreeNode:
    id: str
    label: str
    kind: str                      # "model" | "layer" | "object"
    detail: dict = field(default_factory=dict)
    children: list = field(default_factory=list)


class NeuralDebugger:
    def __init__(self):
        self.model = None
        self.name = ""
        self.layers = []           # names of ReLU/activation leaf modules
        self.objects = {}          # obj_id -> GeometricConceptObject
        self.breakpoints = set()   # layer names or obj ids
        self._probe = None         # callable x-> model output (numpy), for search

    # --------------------------------------------------------------- attach
    def attach(self, model: nn.Module, name="model", input_sampler=None):
        """Attach to a model. input_sampler(n)->tensor supplies probing inputs
        (defaults to standard normal of the inferred input width)."""
        self.model = model.eval()
        self.name = name
        self.objects, self.breakpoints = {}, set()
        # discover activation layers = leaf modules that are nonlinearities
        self.layers = [n for n, m in model.named_modules()
                       if isinstance(m, (nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid))]
        self._infer_input(input_sampler)
        return self

    def _infer_input(self, sampler):
        if sampler is not None:
            self._sampler = sampler
            return
        # infer input width from the first Linear/Embedding
        d = None
        for _, m in self.model.named_modules():
            if isinstance(m, nn.Linear):
                d = m.in_features
                break
        d = d or 4
        self._sampler = lambda n: torch.randn(n, d)

    def _acts(self, x, layers=None):
        rec = ActivationRecorder(self.model, names=layers or self.layers)
        with rec.capture():
            with torch.no_grad():
                out = self.model(x)
        return {k: v for k, v in rec.traces.items()}, out

    # ------------------------------------------------------------- decompile
    def decompile(self, task_fn=None, top_k=6):
        """Build the object tree. For each activation layer, rank units by
        causal task-relative influence and surface the top ones as objects.

        task_fn(x)->target lets influence be measured against a known task; if
        None, influence is measured against the model's own output (self-
        consistency), still causal, just not task-anchored.
        """
        x = self._sampler(4096)
        acts, base_out = self._acts(x)
        base = base_out.detach().numpy().ravel()
        if task_fn is not None:
            target = task_fn(x).detach().numpy().ravel()
            ref = target
        else:
            ref = base
        rvar = float(((ref - ref.mean()) ** 2).mean()) or 1e-9

        def err(y):
            return float(((y - ref) ** 2).mean()) / rvar

        base_err = err(base)
        self.objects = {}
        for lname in self.layers:
            H = acts[lname].detach().numpy()
            if H.ndim != 2:
                H = H.reshape(len(x), -1)
            width = H.shape[1]
            # null: influence of ablating random units (a floor for "moved output")
            rng = np.random.default_rng(0)
            null = float(np.median([
                self._ablate_err(x, lname, j, ref, rvar, base_err)
                for j in rng.choice(width, min(16, width), replace=False)]))
            contrib = np.abs(H).mean(0) * H.std(0)
            for rank, j in enumerate(np.argsort(contrib)[::-1][:top_k]):
                infl = self._ablate_err(x, lname, int(j), ref, rvar, base_err)
                acts_j = H[:, j]
                order = np.argsort(acts_j)
                oid = f"{lname}#u{j}"
                obj = GeometricConceptObject(
                    name=oid, kind="feature", source=self.name, layer=lname,
                    center=[float(acts_j.mean())],
                    activating_examples=x.numpy()[order[-4:]].tolist(),
                    counterexamples=x.numpy()[order[:4]].tolist(),
                    causal_influence=infl, null_baseline=null,
                    causal_test="unit ablation (output/task fidelity loss)",
                    story=f"{lname} unit {j}: fires mean {acts_j.mean():.3f}, "
                          f"ablation moves fidelity by {infl:.3g}",
                    extra={"unit": int(j), "width": width})
                obj.grade()
                self.objects[oid] = obj
        return self

    def _ablate_err(self, x, lname, j, ref, rvar, base_err):
        module = dict(self.model.named_modules())[lname]

        def hook(_m, _i, out, j=j):
            out = out.clone()
            if out.ndim == 2:
                out[:, j] = 0.0
            else:
                out.reshape(out.shape[0], -1)[:, j] = 0.0
            return out
        h = module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                y = self.model(x).detach().numpy().ravel()
        finally:
            h.remove()
        return float(((y - ref) ** 2).mean()) / rvar - base_err

    # ------------------------------------------------------------------ tree
    def tree(self) -> TreeNode:
        root = TreeNode(self.name, self.name, "model",
                        detail={"layers": len(self.layers),
                                "objects": len(self.objects)})
        for lname in self.layers:
            objs = [o for o in self.objects.values() if o.layer == lname]
            node = TreeNode(lname, lname, "layer",
                            detail={"n_objects": len(objs)})
            for o in sorted(objs, key=lambda z: -z.confidence):
                node.children.append(TreeNode(
                    o.name, o.name, "object",
                    detail={"confidence": o.confidence,
                            "influence": o.causal_influence,
                            "story": o.story, "refuted": o.refuted}))
            root.children.append(node)
        return root

    def print_tree(self):
        r = self.tree()
        print(f"{r.label}  [{r.detail['layers']} layers, "
              f"{r.detail['objects']} objects]")
        for layer in r.children:
            print(f"  {layer.label}  ({layer.detail['n_objects']} objects)")
            for obj in layer.children:
                bar = "#" * int(obj.detail["confidence"] * 10)
                print(f"    {obj.label:<16} conf {obj.detail['confidence']:.2f} "
                      f"{bar:<10} {obj.detail['story']}")

    def inspect(self, obj_id) -> GeometricConceptObject:
        return self.objects[obj_id]

    # ------------------------------------------------------------ breakpoints
    def set_breakpoint(self, target):
        self.breakpoints.add(target)

    def clear_breakpoints(self):
        self.breakpoints = set()

    def trace(self, x):
        """Run one input; report per-layer which objects fired, pausing
        (recording) at breakpoints. Returns a step list + final output."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        acts, out = self._acts(x)
        steps = []
        for lname in self.layers:
            H = acts[lname].detach().numpy().reshape(len(x), -1)[0]
            fired = []
            for o in self.objects.values():
                if o.layer != lname:
                    continue
                j = o.extra.get("unit")
                if j is not None and H[j] > 1e-6:
                    fired.append({"id": o.name, "activation": float(H[j]),
                                  "confidence": o.confidence})
            steps.append({"layer": lname, "is_breakpoint": lname in self.breakpoints,
                          "fired": sorted(fired, key=lambda f: -f["activation"])})
        return {"steps": steps, "output": float(out.ravel()[0])
                if out.numel() == 1 else out.detach().numpy().tolist()}

    # ------------------------------------------------------------------ edit
    def _edit_units(self, edits: dict):
        """Return a NEW model with hidden units scaled: {obj_id: scale}.
        scale 0 = ablate, >1 = amplify. Patched into the *downstream* Linear
        so the change is baked into weights (edit-and-recompile)."""
        m = copy.deepcopy(self.model)
        named = dict(m.named_modules())
        # map each activation layer to the Linear that consumes it
        mods = list(m.named_modules())
        for oid, scale in edits.items():
            o = self.objects[oid]
            j = o.extra["unit"]
            # find the next Linear after this activation layer
            after = False
            downstream = None
            for nm, mod in mods:
                if nm == o.layer:
                    after = True
                    continue
                if after and isinstance(mod, nn.Linear):
                    downstream = mod
                    break
            if downstream is not None and j < downstream.weight.shape[1]:
                with torch.no_grad():
                    downstream.weight[:, j] *= scale
        return m.eval()

    def ablate(self, *obj_ids):
        return self._edit_units({oid: 0.0 for oid in obj_ids})

    def amplify(self, obj_id, factor=2.0):
        return self._edit_units({obj_id: factor})

    def patch(self, edits: dict):
        return self._edit_units(edits)

    # ------------------------------------------------------- anomaly discovery
    def discover_anomaly(self, top_k=20, z_thresh=2.0):
        """Add objects that fire SELECTIVELY -- units whose activation on a
        rare, self-identified 'suspicious' input subset is far above their
        normal baseline. This is how a conditional/backdoor circuit is found:
        by contrast, not by average contribution. Suspicious inputs are those
        whose output is poorly explained by a smooth surrogate (exp7's method),
        so no ground-truth trigger is needed.
        """
        x = self._sampler(8000)
        acts, out = self._acts(x)
        y = out.detach().numpy().ravel()
        Xn = x.numpy()
        # smooth surrogate (quadratic in inputs); high residual = anomalous
        d = Xn.shape[1]
        feats = [np.ones(len(Xn))] + [Xn[:, i] for i in range(d)] + \
                [Xn[:, i] * Xn[:, j] for i in range(d) for j in range(i, d)]
        F = np.stack(feats, 1)
        coef, *_ = np.linalg.lstsq(F, y, rcond=None)
        resid = np.abs(y - F @ coef)
        susp = resid > np.quantile(resid, 0.98)
        if susp.sum() < 4:
            return self                       # nothing anomalous -> no objects
        added = 0
        for lname in self.layers:
            H = acts[lname].detach().numpy().reshape(len(x), -1)
            mu_n = H[~susp].mean(0); sd_n = H[~susp].std(0) + 1e-6
            z = (H[susp].mean(0) - mu_n) / sd_n            # selectivity
            # selectivity ratio: mean activation on suspicious vs normal. A
            # true trigger unit fires FAR more on the anomaly; a shared unit
            # (trigger + main task) fires substantially on both -> filtered.
            sel_ratio = H[susp].mean(0) / (np.abs(H[~susp]).mean(0) + 1e-6)
            for j in np.argsort(z)[::-1][:top_k]:
                if z[j] < z_thresh or sel_ratio[j] < 6.0:  # not trigger-specific
                    continue
                oid = f"{lname}#u{int(j)}!"                # ! marks anomaly object
                acts_j = H[:, int(j)]
                order = np.argsort(acts_j)
                obj = GeometricConceptObject(
                    name=oid, kind="trigger", source=self.name, layer=lname,
                    center=[float(acts_j.mean())],
                    input_direction=(Xn[susp].mean(0) - Xn.mean(0)).tolist(),
                    activating_examples=Xn[susp][:4].tolist(),
                    counterexamples=Xn[~susp][:4].tolist(),
                    causal_influence=float(z[j]), null_baseline=float(z_thresh),
                    causal_test="selectivity z on anomalous input subset",
                    story=f"{lname} unit {int(j)}: fires {z[j]:.1f} SD above "
                          f"normal on the anomalous region "
                          f"({sel_ratio[j]:.1f}x more than on benign)",
                    extra={"unit": int(j), "selectivity_z": float(z[j]),
                           "selectivity_ratio": float(sel_ratio[j])})
                obj.grade()
                self.objects[oid] = obj
                added += 1
        self._last_suspicious = x[susp]
        return self

    # ---------------------------------------------------------------- search
    def search(self, score_fn, layers=None, top=5):
        """Find objects that control a behavior: rank objects by how much
        ablating each changes score_fn(model_output). score_fn(out)->float.
        Returns [(obj_id, delta), ...] sorted by |delta| descending."""
        x = self._sampler(4096)
        with torch.no_grad():
            base = score_fn(self.model(x))
        results = []
        for oid, o in self.objects.items():
            if layers and o.layer not in layers:
                continue
            edited = self.ablate(oid)
            with torch.no_grad():
                s = score_fn(edited(x))
            results.append((oid, float(s - base)))
        results.sort(key=lambda r: -abs(r[1]))
        return results[:top]
