"""Live mind-control console backend.

Loads a saved ModelMap + its model, holds the model in memory, and streams
steered generation: the browser sends a prompt plus a slider value per lever,
the server adds (slider * clamp_vector) at each lever's layer during
generation, and streams the text back token by token.

Every slider is a CAUSALLY-VERIFIED lever from the mapper (potency cleared
the random-direction null). Unnamed levers are exposed too -- dragging one
shows what an alien, unnameable direction does to the output. That is the
point of axiom 2, made playable.

Run:
  python -m interpretability_lab.mapper.server --map interpretability_lab/mapper/maps/qwen3_5_0_8b.json
then open http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import torch

from interpretability_lab.mapper.core import ModelMap
from interpretability_lab.models.pretrained import (chat_prompt,
                                                    decoder_layers, load_qwen)

HERE = Path(__file__).parent


class SteerHooks:
    """Registers one additive hook per mapped layer; vectors set per request."""

    def __init__(self, model, layer_ids):
        self.layers = decoder_layers(model)
        self.vecs = {int(i): None for i in layer_ids}
        self.handles = []
        for i in self.vecs:
            self.handles.append(
                self.layers[i].register_forward_hook(self._mk(i)))

    def _mk(self, i):
        def hook(mod, inp, out):
            v = self.vecs[i]
            if v is None:
                return out
            if isinstance(out, tuple):
                return (out[0] + v.to(device=out[0].device,
                                      dtype=out[0].dtype),) + out[1:]
            return out + v.to(device=out.device, dtype=out.dtype)
        return hook

    def set_from(self, active):
        """active: {layer_id: summed_vector_tensor}."""
        for i in self.vecs:
            self.vecs[i] = active.get(i)

    def clear(self):
        for i in self.vecs:
            self.vecs[i] = None


def build_app(map_path: str):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, StreamingResponse

    mm = ModelMap.load(map_path)
    model, tok = load_qwen()
    model.eval()
    device = next(model.parameters()).device

    # --- concept levers (big, contrast-pair-derived behavioural directions),
    # loaded alongside the SAE atoms. Only the VERIFIED ones get sliders.
    concept_path = Path(map_path).parent / "concept_levers.json"
    concepts = []
    if concept_path.exists():
        concepts = [c for c in json.loads(
            concept_path.read_text(encoding="utf-8")) if c.get("verified")]

    # index causal levers; precompute clamp tensors (unit * a nominal scale)
    levers = [l for l in mm.levers if l["causal"]]
    layer_ids = sorted({int(l["layer"]) for l in levers}
                       | {int(c["layer"]) for c in concepts})
    hooks = SteerHooks(model, layer_ids)
    # per-layer residual norm sets the steering dose (exp12 units). exp12's
    # THERAPEUTIC WINDOW: ~0.25-0.5x residual norm steers cleanly, >=1x is
    # word-salad. So |slider|=1 maps to 0.6x (strong but near the edge of
    # clean), and the interesting behaviour lives mid-slider -- dragging past
    # ~0.7 intentionally shows the model breaking, which is honest about the
    # window rather than hiding it.
    SLIDER_MAX_DOSE = 0.6
    resid_norm = {}
    # estimate residual norm per mapped layer from a quick pass
    import numpy as _np
    from interpretability_lab.mapper.adapters import _Hook
    for i in layer_ids:
        hk = _Hook(hooks.layers[i], "capture")
        with torch.no_grad():
            ids = tok("The quick brown fox jumps over the lazy dog.",
                      return_tensors="pt").input_ids.to(device)
            model(input_ids=ids)
        resid_norm[i] = float(_np.linalg.norm(
            hk.captured[0].float().cpu().numpy(), axis=1).mean())
        hk.remove()

    lever_tensors = []
    for k, l in enumerate(levers):
        li = int(l["layer"])
        unit = torch.tensor(l["vector"], dtype=torch.float32, device=device)
        unit = unit / (unit.norm() + 1e-9)
        dose = SLIDER_MAX_DOSE * resid_norm.get(li, 1.0)
        lever_tensors.append({"layer": li, "vec": unit * dose,
                              "label": l["label"], "potency": l["potency"],
                              "top_tokens": l["top_tokens"], "id": k})

    # concept-lever tensors, with per-sign strength so the UI can show which
    # direction actually works (many are asymmetric -- refusal only +, etc.)
    concept_tensors = []
    for k, c in enumerate(concepts):
        li = int(c["layer"])
        unit = torch.tensor(c["vector"], dtype=torch.float32, device=device)
        unit = unit / (unit.norm() + 1e-9)
        dose = SLIDER_MAX_DOSE * resid_norm.get(li, 1.0)
        concept_tensors.append({
            "layer": li, "vec": unit * dose, "name": c["name"], "id": k,
            "pos_strength": c["pos_strength"], "neg_strength": c["neg_strength"]})

    app = FastAPI()

    # --- generation concurrency guard ------------------------------------
    # The steering hooks are shared global state; the model is single-GPU.
    # Two overlapping /generate calls would (a) run two generation loops that
    # both read/mutate hooks.vecs mid-flight -> interleaved garbage, and (b)
    # leave the abandoned first loop churning to max_new on the GPU. So:
    #   * a lock serialises generation (only one loop touches the model/hooks),
    #   * a monotonic gen-id lets a running loop notice a NEWER request arrived
    #     and stop itself early, and gates the hook-clear so a stale loop's
    #     teardown can't wipe the current request's steering.
    gen_lock = asyncio.Lock()
    gen_state = {"current": 0}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (HERE / "console.html").read_text(encoding="utf-8")

    def _clean_tok(s):
        s = s.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\\n")
        s = s.strip()
        return s if s else "·"          # visible marker for space/BPE tokens

    @app.get("/levers")
    def get_levers():
        levs = []
        for t in lever_tensors:
            src = levers[t["id"]]
            levs.append({
                "id": t["id"], "layer": t["layer"],
                "label": t["label"] or "",
                "nameable": bool(t["label"]),
                "potency": round(t["potency"], 3),
                "quality": src.get("quality", "coherent"),
                "distinct": src.get("distinct", 1.0),
                "top_tokens": [_clean_tok(s) for s in t["top_tokens"][:6]]})
        return {
            "model": mm.model_name,
            "ledger": {k: v for k, v in mm.ledger.items()
                       if k != "per_layer"},
            "coords": mm.coords,
            "levers": levs}

    @app.get("/concept_levers")
    def get_concepts():
        return {"concepts": [
            {"id": c["id"], "name": c["name"], "layer": c["layer"],
             "pos_strength": c["pos_strength"],
             "neg_strength": c["neg_strength"]}
            for c in concept_tensors]}

    async def generate_stream(prompt, sliders, concept_sliders, max_new, my_id):
        # Wait for any in-flight generation to finish (or be cancelled) before
        # touching the shared hooks. Only one loop is ever inside the lock, so
        # steering can't be stomped mid-generation.
        async with gen_lock:
            # A newer request may have arrived while we waited for the lock;
            # if so, we're already stale -- bail before doing any work.
            if my_id != gen_state["current"]:
                return
            active = {}
            for k, val in sliders.items():
                t = lever_tensors[int(k)]
                if abs(val) < 1e-3:
                    continue
                active[t["layer"]] = (active.get(t["layer"], 0)
                                      + t["vec"] * float(val))
            for k, val in (concept_sliders or {}).items():
                c = concept_tensors[int(k)]
                if abs(val) < 1e-3:
                    continue
                active[c["layer"]] = (active.get(c["layer"], 0)
                                      + c["vec"] * float(val))
            hooks.set_from(active)
            ids = tok(chat_prompt(tok, prompt),
                      return_tensors="pt").input_ids.to(device)
            past = None
            nxt = ids
            try:
                with torch.no_grad():
                    for _ in range(max_new):
                        # A newer /generate bumped the id -> stop this stream.
                        if my_id != gen_state["current"]:
                            break
                        out = model(input_ids=nxt, past_key_values=past,
                                    use_cache=True)
                        past = out.past_key_values
                        tok_id = out.logits[:, -1].argmax(-1, keepdim=True)
                        if tok_id.item() == tok.eos_token_id:
                            break
                        nxt = tok_id
                        piece = tok.decode(tok_id[0], skip_special_tokens=True)
                        yield piece
                        # yield control so a queued newer request can bump the
                        # id and a client disconnect can propagate promptly.
                        await asyncio.sleep(0)
            finally:
                # Only clear if we're still the current generation; a stale
                # loop's teardown must not wipe a newer request's steering.
                if my_id == gen_state["current"]:
                    hooks.clear()

    @app.post("/generate")
    async def generate(payload: dict):
        prompt = payload.get("prompt", "Tell me about your day.")
        sliders = payload.get("sliders", {})
        concept_sliders = payload.get("concept_sliders", {})
        max_new = min(int(payload.get("max_new", 120)), 500)
        # Claim the newest-generation id. Any loop still running under an older
        # id will see the mismatch on its next step and stop itself.
        gen_state["current"] += 1
        my_id = gen_state["current"]
        return StreamingResponse(
            generate_stream(prompt, sliders, concept_sliders, max_new, my_id),
            media_type="text/plain")

    def teardown():
        """Free the model + GPU cleanly on shutdown."""
        try:
            hooks.clear()
            for h in hooks.handles:
                h.remove()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        print("\n[console] shut down cleanly, GPU released.")

    app.add_event_handler("shutdown", teardown)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    import uvicorn
    app = build_app(args.map)
    print(f"console: http://{args.host}:{args.port}  (Ctrl+C to stop)")
    # uvicorn installs its own SIGINT/SIGTERM handlers and runs the app's
    # "shutdown" event (teardown) on exit; KeyboardInterrupt is caught so the
    # process ends quietly instead of dumping a traceback.
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    except KeyboardInterrupt:
        pass
    print("[console] stopped.")


if __name__ == "__main__":
    main()
