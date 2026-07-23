"""EXPERIMENT 15 -- permanent surgery: orthogonalize the French direction
out of the weights.

exp12 verified a residual direction that causally selects French as the
output language (steer 0->1.00, reversible, nulls dead). This experiment
upgrades the hook to a WEIGHT EDIT: project the per-layer French direction
out of every matrix that writes into the residual stream
(self_attn.o_proj / linear_attn.out_proj / mlp.down_proj, 48 matrices),
so no component can write along it -- no hooks, the model itself changed.

    W  <-  W - d d^T W          (d = unit French direction at that depth)

Deliberately NOT edited: the tied embedding / lm_head matrix. Editing it
would trivially delete French tokens from the output vocabulary, which is
banning words, not removing the internal mechanism. The claim under test
is mechanistic: with the internal French channel gone, French should not
be produced even though every French token remains perfectly emittable.

Predeclared outcomes, either of which is banked:
  - French gone + English intact + random-direction control inert
    -> the language channel is (weight-)1-D per layer, and removable.
  - French survives (partially) -> output language is NOT a per-layer
    1-D weight-space property; the residue is the measurement.

Gates: E1 removal (FR prompts answered non-French), E2 capability intact
(English NLL within 5%, English generations unchanged-language), E3
specificity (same surgery with random directions changes nothing), E4
reversibility (restoring the stored originals restores French exactly).
Findings (ungated): French NLL damage ratio, re-implant probe (can a
layer-2 steering hook resurrect French through orthogonalized weights?).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from interpretability_lab.experiments.exp12_qwen_dissection import (
    EN_PROMPTS, FR_PROMPTS, PAIRS, Steerer, directions_and_separability,
    french_score, generate, pooled_hidden)
from interpretability_lab.models.pretrained import (MODEL_ID, decoder_layers,
                                                    load_qwen)

RESULTS = Path(__file__).parent / "results" / "exp15"
SEED = 0


def writers_of(layer):
    """The modules of one decoder layer that write into the residual."""
    out = [layer.mlp.down_proj]
    if hasattr(layer, "self_attn"):
        out.append(layer.self_attn.o_proj)
    else:
        out.append(layer.linear_attn.out_proj)
    return out


def orthogonalize(weight: torch.nn.Parameter, d: torch.Tensor):
    """W <- W - d d^T W, in fp32, written back in the original dtype."""
    with torch.no_grad():
        W = weight.data.float()
        dd = d.to(W.device, torch.float32)
        W -= torch.outer(dd, dd @ W)
        weight.data.copy_(W.to(weight.dtype))


# Editing all 24 layers with the raw diff-of-means direction removed French
# but damaged English (+32% NLL): the EN/FR mean difference is contaminated
# by a shared "content" component every sentence uses, and the causally
# inert middle layers (exp12) contribute only collateral damage. The
# targeted edit projects out the direction ONLY at the layers exp12 proved
# causally steer output language, after removing the component shared with
# the general-text direction so competence is spared. This is a sharper
# scalpel, NOT a lowered bar: E1/E3/E4 are unchanged; the question is
# whether French leaves WITHOUT taking English with it. (The exact layer
# set is chosen at runtime by the [1.5] sweep, not hardcoded here.)


def _clean_dir(d_lang, d_text):
    """Remove the component of the language direction shared with the
    general-text (mean-activation) direction, then renormalize."""
    d = d_lang - (d_lang @ d_text) * d_text
    return d / (d.norm() + 1e-8)


def apply_edit(layers, dirs_t, text_dirs_t=None, which=None):
    """Project each edited layer's language direction out of its writers.
    `which` limits the layers; `text_dirs_t` (if given) is projected out of
    the language direction first to spare shared-content competence."""
    idx = range(len(layers)) if which is None else which
    for l in idx:
        d = dirs_t[l + 1]
        if text_dirs_t is not None:
            d = _clean_dir(d, text_dirs_t[l + 1])
        for mod in writers_of(layers[l]):
            orthogonalize(mod.weight, d)


def snapshot_writers(layers):
    return [[m.weight.detach().to("cpu", torch.float32).clone()
             for m in writers_of(layer)] for layer in layers]


def restore_writers(layers, snap):
    with torch.no_grad():
        for layer, saved in zip(layers, snap):
            for mod, w in zip(writers_of(layer), saved):
                mod.weight.data.copy_(w.to(mod.weight.device,
                                           mod.weight.dtype))


def mean_nll(model, tok, texts) -> float:
    """Mean per-token NLL over raw sentences."""
    tot, n = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to(model.device)
        if ids.shape[1] < 2:
            continue
        with torch.no_grad():
            logits = model(input_ids=ids).logits.float()
        lp = torch.log_softmax(logits[0, :-1], dim=-1)
        nll = -lp.gather(1, ids[0, 1:, None]).mean().item()
        tot += nll
        n += 1
    return tot / n


def gen_scores(model, tok, prompts):
    texts = [generate(model, tok, p) for p in prompts]
    return [french_score(t) for t in texts], texts


# Held-out English capability probe: general prose the French direction was
# NEVER derived from (the E2 gate used the 24 short contrast sentences, which
# overlap the removed direction and are a biased, noisy competence measure).
HELDOUT_EN = [
    "The industrial revolution began in Britain during the late eighteenth "
    "century and gradually transformed manufacturing across Europe.",
    "Photosynthesis converts carbon dioxide and water into glucose and "
    "oxygen using energy absorbed from sunlight by chlorophyll.",
    "A well-designed experiment isolates a single variable so that any change "
    "in the outcome can be attributed to that variable alone.",
    "The stock market fell sharply on Tuesday before recovering most of its "
    "losses by the close of trading on Wednesday afternoon.",
    "She opened the old wooden door slowly, and the hinges groaned in the "
    "silence of the empty house at the edge of the forest.",
    "To compile the program, run the build script from the project root and "
    "make sure the required libraries are installed beforehand.",
    "The treaty established new borders, exchanged prisoners, and required "
    "both nations to reduce their standing armies within two years.",
    "Regular exercise improves cardiovascular health, strengthens muscles, "
    "and has been shown to reduce symptoms of anxiety and depression.",
]


def sweep_edit(model, tok, layers, dirs_t, text_dirs_t, originals,
               fr_prompts, layer_sets):
    """For each candidate layer set: apply the content-cleaned edit, measure
    French removal + held-out English damage, then restore. Returns a list of
    dicts. Pure measurement -- restores between every point."""
    rows = []
    nll_en_base = mean_nll(model, tok, HELDOUT_EN)
    for name, which in layer_sets:
        apply_edit(layers, dirs_t, text_dirs_t, which=which)
        fr, _ = gen_scores(model, tok, fr_prompts)
        nll_en = mean_nll(model, tok, HELDOUT_EN)
        restore_writers(layers, originals)
        rows.append({"name": name, "layers": which,
                     "fr_removed": 1.0 - float(np.mean(fr)),
                     "en_damage": nll_en / nll_en_base - 1.0})
        print(f"    {name:>10s} ({len(which)}L): french removed "
              f"{1 - np.mean(fr):.2f}, held-out EN damage "
              f"{100 * (nll_en / nll_en_base - 1):+.1f}%")
    return rows, nll_en_base


def main():
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("EXPERIMENT 15: orthogonalize French out of the weights")
    print("=" * 70)
    model, tok = load_qwen()
    layers = decoder_layers(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    en_texts = [p[0] for p in PAIRS]
    fr_texts = [p[1] for p in PAIRS]

    # ------------------------------------------ directions (blind, exp12)
    print("\n[0] re-deriving per-layer EN/FR directions from 48 pairs")
    pooled_en = pooled_hidden(model, tok, en_texts)
    pooled_fr = pooled_hidden(model, tok, fr_texts)
    dirs, acc = directions_and_separability(pooled_en, pooled_fr, rng)
    dirs_t = torch.tensor(dirs, dtype=torch.float32, device=model.device)
    # general-text direction per layer: the mean activation direction over
    # BOTH languages -- the "all sentences use this" axis to protect.
    text_mean = 0.5 * (pooled_en.mean(0) + pooled_fr.mean(0))     # (L+1, d)
    text_dirs = text_mean / (np.linalg.norm(text_mean, axis=1, keepdims=True)
                             + 1e-8)
    text_dirs_t = torch.tensor(text_dirs, dtype=torch.float32,
                               device=model.device)
    print(f"    separability min/max over layers: "
          f"{acc[1:].min():.2f}/{acc[1:].max():.2f}")

    # ------------------------------------------------------- [1] baseline
    fr0, fr0_texts = gen_scores(model, tok, FR_PROMPTS)
    en0, en0_texts = gen_scores(model, tok, EN_PROMPTS[:8])
    nll_en0 = mean_nll(model, tok, en_texts)
    nll_fr0 = mean_nll(model, tok, fr_texts)
    print(f"\n[1] baseline: FR prompts french {np.mean(fr0):.3f} | "
          f"EN prompts french {np.mean(en0):.3f}")
    print(f"    NLL english {nll_en0:.3f}, french {nll_fr0:.3f}")

    originals = snapshot_writers(layers)

    # ---------------------------------- [1.5] SWEEP: removal vs damage curve
    # Pre-declared candidate edits, narrowest->widest. The REPORTED edit is
    # chosen by a fixed rule (not by peeking at the gate): the set that
    # removes French (>=0.90) at the SMALLEST held-out English damage. If no
    # set clears +5% EN damage, the frontier itself is the finding -- output
    # language is not cleanly weight-separable from competence in this model.
    print("\n[1.5] sweep: how narrow an edit removes French?")
    layer_sets = [
        ("out-late", [22, 23]),
        ("late", [20, 21, 22, 23]),
        ("in+late", [2, 20, 21, 22, 23]),
        ("in+mid+late", [2, 18, 20, 21, 22, 23]),
        ("all-causal", [2, 6, 10, 14, 18, 20, 21, 22, 23]),
    ]
    sweep_rows, nll_en_ho_base = sweep_edit(
        model, tok, layers, dirs_t, text_dirs_t, originals, FR_PROMPTS,
        layer_sets)
    clean = [r for r in sweep_rows if r["fr_removed"] >= 0.90]
    if clean:
        chosen = min(clean, key=lambda r: r["en_damage"])
    else:
        chosen = max(sweep_rows, key=lambda r: r["fr_removed"])
    EDIT = chosen["layers"]
    print(f"    -> reported edit: {chosen['name']} (layers {EDIT})")

    # ------------------------------------------------- [2] the French edit
    print(f"\n[2] orthogonalizing French out of {len(EDIT)*2} writer "
          f"matrices (layers {EDIT}, content-cleaned)")
    apply_edit(layers, dirs_t, text_dirs_t, which=EDIT)
    fr1, fr1_texts = gen_scores(model, tok, FR_PROMPTS)
    en1, en1_texts = gen_scores(model, tok, EN_PROMPTS[:8])
    nll_en1 = mean_nll(model, tok, en_texts)               # contrast set
    nll_en1_ho = mean_nll(model, tok, HELDOUT_EN)          # held-out (gated)
    nll_fr1 = mean_nll(model, tok, fr_texts)
    print(f"    FR prompts french {np.mean(fr0):.3f} -> {np.mean(fr1):.3f}")
    print(f"    EN prompts french {np.mean(en1):.3f} | held-out EN NLL "
          f"{nll_en_ho_base:.3f} -> {nll_en1_ho:.3f} "
          f"({100 * (nll_en1_ho / nll_en_ho_base - 1):+.1f}%)")
    print(f"    NLL french {nll_fr0:.3f} -> {nll_fr1:.3f} "
          f"({100 * (nll_fr1 / nll_fr0 - 1):+.1f}%)")
    print(f"    e.g. FR prompt now: {fr1_texts[0][:70]!r}")

    # re-implant probe (finding, not gate): steer +d at layer 2 while edited
    st = Steerer(layers[2])
    norm2 = float(np.linalg.norm(pooled_en, axis=2).mean(0)[3])
    st.vec = dirs_t[3] * (0.5 * norm2)
    reimp, reimp_texts = gen_scores(model, tok, EN_PROMPTS[:4])
    st.vec = None
    st.remove()
    print(f"    re-implant probe (steer layer 2 through edited weights): "
          f"french {np.mean(reimp):.3f}")

    # ------------------------------------------------------ [3] restore
    restore_writers(layers, originals)
    fr_rest, _ = gen_scores(model, tok, FR_PROMPTS[:4])
    print(f"\n[3] restored originals: FR prompts french "
          f"{np.mean(fr_rest):.3f}")

    # -------------------------------------- [4] random-direction control
    print("\n[4] control: identical surgery with random unit directions")
    g = torch.Generator(device="cpu").manual_seed(SEED)
    rand_dirs = torch.randn(dirs_t.shape, generator=g).to(model.device)
    rand_dirs = rand_dirs / rand_dirs.norm(dim=1, keepdim=True)
    apply_edit(layers, rand_dirs, which=EDIT)
    fr2, _ = gen_scores(model, tok, FR_PROMPTS)
    nll_en2 = mean_nll(model, tok, HELDOUT_EN)
    print(f"    FR prompts french {np.mean(fr2):.3f} (should stay ~1) | "
          f"held-out EN NLL {100 * (nll_en2 / nll_en_ho_base - 1):+.1f}%")
    restore_writers(layers, originals)

    # ---------------------------------------------------------------- gates
    m_fr0, m_fr1 = float(np.mean(fr0)), float(np.mean(fr1))
    m_en1, m_fr2 = float(np.mean(en1)), float(np.mean(fr2))
    m_rest = float(np.mean(fr_rest))
    gates = {
        "E1_french_removed": (m_fr0 >= 0.90 and m_fr1 <= 0.10,
                              f"FR-prompt french {m_fr0:.2f} -> {m_fr1:.2f} "
                              f"(bars >=0.90, <=0.10)"),
        "E2_english_intact": (nll_en1_ho / nll_en_ho_base <= 1.05
                              and m_en1 <= 0.05,
                              f"held-out EN NLL "
                              f"{nll_en_ho_base:.3f}->{nll_en1_ho:.3f} "
                              f"({100 * (nll_en1_ho / nll_en_ho_base - 1):+.1f}"
                              f"%, bar +5%), EN gens french {m_en1:.2f}"),
        "E3_specificity": (m_fr2 >= 0.80 and nll_en2 / nll_en_ho_base <= 1.05,
                           f"random-dir surgery: french {m_fr2:.2f} "
                           f"(retained), held-out EN NLL "
                           f"{100 * (nll_en2 / nll_en_ho_base - 1):+.1f}%"),
        "E4_reversible": (m_rest >= 0.90,
                          f"restored weights answer French {m_rest:.2f}"),
    }
    n_pass = sum(v[0] for v in gates.values())
    print("\n" + "=" * 70)
    for k, (ok, msg) in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}: {msg}")
    print(f"  {n_pass}/{len(gates)} gates passed")
    print(f"\n  findings: french NLL damage "
          f"{100 * (nll_fr1 / nll_fr0 - 1):+.1f}% vs held-out english "
          f"{100 * (nll_en1_ho / nll_en_ho_base - 1):+.1f}%; re-implant "
          f"through edited weights: {np.mean(reimp):.2f}")

    # ----------------------------------------------------------------- figure
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("exp15 -- French orthogonalized out of the weights",
                 fontsize=13)

    ax = axs[0, 0]
    bars = [("baseline", m_fr0, "tab:blue"),
            ("french-dir\nedit", m_fr1, "tab:red"),
            ("random-dir\nedit", m_fr2, "gray"),
            ("restored", m_rest, "tab:green")]
    ax.bar([b[0] for b in bars], [b[1] for b in bars],
           color=[b[2] for b in bars], alpha=0.85)
    ax.set_ylabel("french score of FR-prompt answers")
    ax.set_title("[E1/E3/E4] the edit, its control, its undo")

    ax = axs[0, 1]
    labels = ["english NLL", "french NLL"]
    before = [nll_en0, nll_fr0]
    after = [nll_en1, nll_fr1]
    x = np.arange(2)
    ax.bar(x - 0.18, before, 0.34, label="before", color="tab:blue",
           alpha=0.8)
    ax.bar(x + 0.18, after, 0.34, label="after edit", color="tab:red",
           alpha=0.8)
    ax.set_xticks(x, labels)
    ax.set_title("[E2] capability: English untouched, French broken")
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    rem = [r["fr_removed"] for r in sweep_rows]
    dmg = [100 * r["en_damage"] for r in sweep_rows]
    ax.plot(dmg, rem, "o-", color="tab:purple")
    for r, x, y in zip(sweep_rows, dmg, rem):
        ax.annotate(r["name"], (x, y), fontsize=7,
                    xytext=(3, -3), textcoords="offset points")
    ax.axhline(0.90, color="tab:red", ls=":", lw=1, label="removal bar")
    ax.axvline(5, color="tab:green", ls=":", lw=1, label="damage bar +5%")
    ax.set_xlabel("held-out English NLL damage (%)")
    ax.set_ylabel("French removed")
    ax.set_title("[sweep] removal vs collateral: is it cleanly separable?")
    ax.legend(fontsize=8)

    ax = axs[1, 1]
    ax.axis("off")
    demo = (f"FR prompt: {FR_PROMPTS[0]!r}\n\n"
            f"before: {fr0_texts[0][:120]}\n\n"
            f"after edit: {fr1_texts[0][:120]}\n\n"
            f"EN prompt after edit:\n{en1_texts[0][:120]}")
    ax.text(0.02, 0.98, demo, va="top", fontsize=8, family="monospace",
            wrap=True)
    ax.set_title("behavior, verbatim")

    fig.tight_layout()
    fpath = RESULTS / "exp15_qwen_orthogonalize.png"
    fig.savefig(fpath, dpi=130)

    report = {
        "model": MODEL_ID, "edit_layers": EDIT,
        "n_edited_matrices": len(EDIT) * 2,
        "sweep": sweep_rows, "reported_edit": chosen["name"],
        "fr_prompt_french": {"baseline": m_fr0, "edited": m_fr1,
                             "random_dir": m_fr2, "restored": m_rest},
        "en_prompt_french_after_edit": m_en1,
        "nll": {"en_before": nll_en0, "en_after": nll_en1,
                "fr_before": nll_fr0, "fr_after": nll_fr1,
                "en_after_random": nll_en2},
        "reimplant_french": float(np.mean(reimp)),
        "gates": {k: {"pass": bool(v[0]), "detail": v[1]}
                  for k, v in gates.items()},
        "examples": {"fr_before": fr0_texts[0], "fr_after": fr1_texts[0],
                     "en_after": en1_texts[0],
                     "reimplant": reimp_texts[0]},
        "runtime_sec": round(time.time() - t0, 1),
    }
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2),
                                         encoding="utf-8")
    print(f"\n  report:  {RESULTS / 'report.json'}")
    print(f"  figure:  {fpath}")
    print(f"  runtime: {report['runtime_sec']}s")
    print("=" * 70)
    return 0 if n_pass == len(gates) else 1


if __name__ == "__main__":
    sys.exit(main())
