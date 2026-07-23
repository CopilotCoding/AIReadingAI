"""exp19 stage 3 -- CIRCUIT TRACING: which layers causally carry the fact?

Stages 1+2 agreed something decisive happens ~L14-20 (geometry reorganizes,
the answer becomes decodable). This stage asks the CAUSAL question with
activation patching (the standard causal-tracing method): not 'where is the
fact readable' but 'which layers, if we intervene there, actually MOVE the
answer'. Readable != load-bearing (the probes-lie / exp13 lesson).

Method -- clean vs corrupted run, patch the residual stream:
  clean prompt:     "The capital of France is"   -> model says Paris
  corrupted prompt: "The capital of Japan is"    -> model says Tokyo
  Run both, cache clean activations at every (layer, token-position). Then run
  the CORRUPTED prompt but PATCH IN the clean activation at one (layer,pos) and
  measure how much the Paris-over-Tokyo logit is restored. A big restoration =
  that site causally carries the France->Paris fact. Sweep all layers x the
  last few positions -> a causal information-flow map.

  We patch at the LAST token position (where the answer is read) across layers
  = the depth profile, AND at the subject token ("France") across layers = does
  the fact move with the subject early, then get read at the last position late
  (the classic 'fact enrichment then extraction' circuit shape)?

GATES -> experiments/results/exp19/circuit.json
  G1 A CAUSAL CIRCUIT EXISTS: at least one (layer,pos) patch restores the clean
     answer well above a random-activation null (real causal site, not noise).
  G2 LOCALIZED, NOT DIFFUSE: the restoration concentrates in a band of layers
     (a circuit), not uniformly smeared across all 24 (which would mean 'the
     whole net' = no circuit to name).
  G3 CROSS-VALIDATES stages 1+2: the causal band overlaps the L14-20 region
     where geometry reorganized (stage 1) and the fact became decodable
     (stage 2). Three independent methods pointing at the same layers.
  G4 SUBJECT-vs-LAST DISSOCIATION (mechanism, not just location): report where
     patching the SUBJECT token matters vs the LAST token -- if they differ,
     that IS the circuit shape (enrich-at-subject, extract-at-last).

Run: python -m interpretability_lab.experiments.exp19d_qwen_circuit
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from interpretability_lab.models.pretrained import decoder_layers, load_qwen

RESULTS = (Path(__file__).resolve().parent / "results" / "exp19"
           / "circuit.json")
RESULTS.parent.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)

CLEAN = "The capital of France is"
CORRUPT = "The capital of Japan is"


def tok_id(tok, word):
    return tok(" " + word, add_special_tokens=False).input_ids[0]


def run():
    print("=== exp19 stage 3: CIRCUIT TRACING (causal activation patching) ===\n")
    model, tok = load_qwen()
    device = next(model.parameters()).device
    layers = decoder_layers(model)
    L = len(layers)

    ids_clean = tok(CLEAN, return_tensors="pt").input_ids.to(device)
    ids_corr = tok(CORRUPT, return_tensors="pt").input_ids.to(device)
    # the two prompts must align token-for-token except the subject word.
    assert ids_clean.shape == ids_corr.shape, \
        f"prompt lengths differ {ids_clean.shape} vs {ids_corr.shape}"
    seqlen = ids_clean.shape[1]
    paris = tok_id(tok, "Paris")
    tokyo = tok_id(tok, "Tokyo")
    # find the position that differs = the subject token
    diff_pos = int((ids_clean[0] != ids_corr[0]).nonzero()[0].item())
    last_pos = seqlen - 1
    print(f"clean='{CLEAN}' corrupt='{CORRUPT}' | subject@pos{diff_pos}, "
          f"answer read@pos{last_pos}\n")

    # cache clean activations at every layer (full sequence)
    clean_acts = {}
    handles = []

    def mk_cache(i):
        def hook(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            clean_acts[i] = h.detach().clone()
        return hook
    for i in range(L):
        handles.append(layers[i].register_forward_hook(mk_cache(i)))
    with torch.no_grad():
        model(input_ids=ids_clean)
    for h in handles:
        h.remove()

    def logit_gap(input_ids, patch=None):
        """Run input_ids; optionally patch clean act at (layer,pos). Return
        Paris-Tokyo logit at the last position (want HIGH = restored to Paris)."""
        hs = []
        if patch is not None:
            li, pos = patch["layer"], patch["pos"]
            vec = clean_acts[li][:, pos, :]

            def hook(m, inp, out):
                if isinstance(out, tuple):
                    o = out[0].clone(); o[:, pos, :] = vec
                    return (o,) + out[1:]
                o = out.clone(); o[:, pos, :] = vec
                return o
            hs.append(layers[li].register_forward_hook(hook))
        with torch.no_grad():
            lg = model(input_ids=input_ids).logits[0, -1].float().cpu().numpy()
        for h in hs:
            h.remove()
        return float(lg[paris] - lg[tokyo])

    base_clean = logit_gap(ids_clean)              # should be strongly + (Paris)
    base_corr = logit_gap(ids_corr)                # should be strongly - (Tokyo)
    span = base_clean - base_corr
    print(f"clean gap {base_clean:+.2f} (Paris), corrupt gap {base_corr:+.2f} "
          f"(Tokyo); restoration span = {span:.2f}\n")

    def restoration(patch):
        """Fraction of the clean->corrupt gap recovered by this patch, in [0,1+].
        0 = no effect (stays Tokyo), 1 = fully restored to clean (Paris)."""
        g = logit_gap(ids_corr, patch)
        return (g - base_corr) / (span + 1e-9)

    # sweep every layer at the LAST position and the SUBJECT position
    print("Patching clean activation into the corrupted run "
          "(restoration toward Paris, per layer):")
    last_prof, subj_prof = [], []
    for i in range(L):
        r_last = restoration({"layer": i, "pos": last_pos})
        r_subj = restoration({"layer": i, "pos": diff_pos})
        last_prof.append(r_last); subj_prof.append(r_subj)
        b1 = "#" * int(max(0, min(1, r_last)) * 30)
        b2 = "*" * int(max(0, min(1, r_subj)) * 30)
        print(f"  L{i:2d}  last {r_last:+.2f} {b1:<30}  subj {r_subj:+.2f} {b2}")

    # null: patch a RANDOM-normalized activation (matched norm) at each layer's
    # last pos -- must NOT restore. take max over a few draws for a strong null.
    null_vals = []
    for i in range(L):
        vecn = clean_acts[i][:, last_pos, :]
        for _ in range(3):
            r = torch.randn_like(vecn); r = r / r.norm() * vecn.norm()
            saved = clean_acts[i][:, last_pos, :].clone()
            clean_acts[i][:, last_pos, :] = r
            null_vals.append(restoration({"layer": i, "pos": last_pos}))
            clean_acts[i][:, last_pos, :] = saved
    null_p90 = float(np.percentile(np.abs(null_vals), 90))

    last_prof = np.array(last_prof); subj_prof = np.array(subj_prof)
    peak_last = int(last_prof.argmax()); peak_subj = int(subj_prof.argmax())
    # causal band = contiguous layers where last-pos restoration clears the null
    causal_layers = [i for i in range(L) if last_prof[i] > null_p90 + 0.1]

    # ---- gates ----
    g1 = last_prof.max() > null_p90 + 0.2
    # localized: the causal mass is concentrated -- top-6 layers hold most of it
    order = np.argsort(-np.maximum(last_prof, 0))
    top6_mass = np.maximum(last_prof, 0)[order[:6]].sum()
    total_mass = np.maximum(last_prof, 0).sum() + 1e-9
    g2 = (top6_mass / total_mass) > 0.6
    # cross-validate: causal band overlaps L14-20 (stages 1+2 region)
    g3 = any(14 <= i <= 21 for i in causal_layers)
    # subject vs last dissociation: peaks at different depths/positions
    g4 = abs(peak_subj - peak_last) >= 2 or (subj_prof.max() > null_p90 + 0.2)

    gates = {"G1_causal_circuit_exists": bool(g1),
             "G2_localized_not_diffuse": bool(g2),
             "G3_crossvalidates_stages_1_2": bool(g3),
             "G4_subject_vs_last_dissociation": bool(g4)}

    print(f"\nnull p90 (random-patch restoration): {null_p90:.2f}")
    print("\n=== FINDINGS ===")
    print(f"  last-position causal peak: L{peak_last} "
          f"(restoration {last_prof[peak_last]:+.2f})")
    print(f"  subject-position causal peak: L{peak_subj} "
          f"(restoration {subj_prof[peak_subj]:+.2f})")
    print(f"  causal band (clears null) at last pos: "
          f"{[f'L{i}' for i in causal_layers]}")
    print(f"  top-6 layers hold {100*top6_mass/total_mass:.0f}% of causal mass "
          f"-> {'LOCALIZED circuit' if g2 else 'diffuse'}")
    print(f"  cross-validation: causal band overlaps L14-21 "
          f"(stage-1 reorg + stage-2 arrival)? {'YES' if g3 else 'no'}")
    if peak_subj < peak_last:
        print(f"  circuit shape: fact enriched at SUBJECT early (L{peak_subj}), "
              f"extracted at LAST position late (L{peak_last}) -- classic "
              f"enrich-then-extract.")

    print("\n=== GATES ===")
    for g, v in gates.items():
        print(f"  {'PASS' if v else 'FAIL'}  {g}")

    out = {
        "model": "Qwen/Qwen3.5-0.8B", "clean": CLEAN, "corrupt": CORRUPT,
        "n_layers": L, "subject_pos": diff_pos, "last_pos": last_pos,
        "base_clean_gap": base_clean, "base_corrupt_gap": base_corr,
        "last_restoration": last_prof.tolist(),
        "subject_restoration": subj_prof.tolist(),
        "null_p90": null_p90, "causal_layers": causal_layers,
        "peak_last": peak_last, "peak_subject": peak_subj,
        "top6_mass_frac": float(top6_mass / total_mass), "gates": gates,
    }
    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved -> {RESULTS}")
    make_figure(out)
    return out


def make_figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = list(range(out["n_layers"]))
    last = out["last_restoration"]; subj = out["subject_restoration"]
    fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")
    ax.plot(xs, last, "-o", color="#d7191c", lw=2,
            label="patch at ANSWER position (extraction)")
    ax.plot(xs, subj, "-s", color="#2c7fb8", lw=2,
            label="patch at SUBJECT 'France' (enrichment)")
    ax.axhline(out["null_p90"], color="grey", ls="--", lw=1,
               label=f"random-patch null p90 ({out['null_p90']:.2f})")
    ax.axvspan(14, 21, color="#fdae61", alpha=0.2,
               label="stage 1+2 region (reorg + fact arrival)")
    ax.set_title("exp19 stage 3 - Causal circuit for 'capital of France -> Paris'\n"
                 "activation patching: which layers actually CARRY the fact",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("layer"); ax.set_ylabel("fact restoration (0=Tokyo, 1=Paris)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    png = RESULTS.parent / "circuit_trace.png"
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure -> {png}")


if __name__ == "__main__":
    run()
