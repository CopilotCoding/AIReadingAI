"""EXPERIMENT 12 -- the instrument meets a real mind (Qwen3.5-0.8B).

Everything before this rung had ground truth: we trained the subject, so we
knew the answer and could gate against it. A pretrained LM has no answer
key. What survives the loss of ground truth is the lab's causal spine:

    blind discovery -> causal validation with the story's OWN predictions
    -> negative controls -> refusal must be possible.

Subject property: LANGUAGE (English vs French). Chosen because the
behavioral readout is machine-measurable without any judge model: French
function words and diacritics in generated text are countable.

RUN-1 FINDING, NOW PINNED (gate P1): the EN/FR contrast is linearly
decodable at EVERY layer (held-out acc 1.00 at all 24), so selecting the
steering site by probe accuracy degenerated -- it picked layer 0, where
steering just breaks generation (think-tag / newline spam, french 0.000).
The causal map told a different story: output language is steerable at the
LATE layers (22: 0.978) and input-language perception at the EARLY layers
(-d at layer 0 makes the model re-quote French as English), while the
middle of the network is causally inert at these doses. This is exp3/exp10
"probes lie / decodable != causally modular" reproduced in a real 752M
pretrained model. Steering site is therefore selected CAUSALLY (pilot
sweep), and the dissociation is re-asserted on every run.

  [0] attach     load 752M hybrid net (18x GatedDeltaNet + 6x full attn).
  [A] discovery  48 parallel EN/FR pairs -> per-layer diff-of-means
                 direction + held-out separability. BLIND.
  [B] pilot map  causal efficacy across depth at two doses -> pick the
                 steering site by MEASURED effect, not probe acc.
  [C] causation  dose-response at the causal site; random-direction null;
                 shuffled-label null; REVERSE (-a must suppress French on
                 French prompts).
  [D] refusal    shuffled-label contrast stays near chance and inert.

Gates test the MEASURED mechanism. A failed gate stays on the record.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from interpretability_lab.geometry.concept import GeometricConceptObject
from interpretability_lab.models.pretrained import (MODEL_ID, chat_prompt,
                                                   decoder_layers, load_qwen)

RESULTS = Path(__file__).parent / "results" / "exp12"
CONCEPTS = Path(__file__).resolve().parents[1] / "geometry" / "concepts"
SEED = 0
MAX_NEW = 48
PILOT_LAYERS = [0, 2, 6, 10, 14, 18, 20, 22, 23]
PILOT_ALPHAS = [1.0, 2.0]
DOSE_ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0]   # units of mean residual norm

# ---------------------------------------------------------------- contrast data
# 48 parallel EN/FR sentence pairs (raw text, not chat) for direction finding.
PAIRS = [
    ("The weather is beautiful today.", "Il fait tres beau aujourd'hui."),
    ("I would like a cup of coffee.", "Je voudrais une tasse de cafe."),
    ("The train leaves at seven in the morning.",
     "Le train part a sept heures du matin."),
    ("She reads a book every evening.", "Elle lit un livre chaque soir."),
    ("We are going to the market tomorrow.", "Nous allons au marche demain."),
    ("The cat sleeps on the sofa.", "Le chat dort sur le canape."),
    ("My brother works in a hospital.", "Mon frere travaille dans un hopital."),
    ("The children play in the garden.", "Les enfants jouent dans le jardin."),
    ("I forgot my keys at home.", "J'ai oublie mes cles a la maison."),
    ("This restaurant serves excellent food.",
     "Ce restaurant sert une excellente cuisine."),
    ("The mountains are covered with snow.",
     "Les montagnes sont couvertes de neige."),
    ("He speaks three languages fluently.", "Il parle trois langues couramment."),
    ("The museum opens at ten o'clock.", "Le musee ouvre a dix heures."),
    ("She bought a new red dress.", "Elle a achete une nouvelle robe rouge."),
    ("The river flows through the city.", "La riviere traverse la ville."),
    ("We watched a movie last night.", "Nous avons regarde un film hier soir."),
    ("The bakery smells wonderful in the morning.",
     "La boulangerie sent merveilleusement bon le matin."),
    ("My grandmother tells the best stories.",
     "Ma grand-mere raconte les meilleures histoires."),
    ("The students listen to the teacher.",
     "Les etudiants ecoutent le professeur."),
    ("It rains often in the autumn.", "Il pleut souvent en automne."),
    ("The bridge crosses the old canal.", "Le pont traverse le vieux canal."),
    ("I need to buy some vegetables.", "Je dois acheter des legumes."),
    ("The concert begins in one hour.", "Le concert commence dans une heure."),
    ("Her garden is full of flowers.", "Son jardin est plein de fleurs."),
    ("The dog runs along the beach.", "Le chien court le long de la plage."),
    ("We visited the castle last summer.",
     "Nous avons visite le chateau l'ete dernier."),
    ("The soup is too hot to eat.", "La soupe est trop chaude pour manger."),
    ("He fixed the broken window yesterday.",
     "Il a repare la fenetre cassee hier."),
    ("The library is quiet in the afternoon.",
     "La bibliotheque est calme l'apres-midi."),
    ("She sings while she cooks dinner.",
     "Elle chante pendant qu'elle prepare le diner."),
    ("The farmer wakes up before sunrise.",
     "Le fermier se reveille avant le lever du soleil."),
    ("My favorite color is dark blue.", "Ma couleur preferee est le bleu fonce."),
    ("The plane lands in twenty minutes.", "L'avion atterrit dans vingt minutes."),
    ("They built a small wooden cabin.",
     "Ils ont construit une petite cabane en bois."),
    ("The moon is bright tonight.", "La lune est brillante ce soir."),
    ("I lost my umbrella on the bus.", "J'ai perdu mon parapluie dans le bus."),
    ("The clock on the wall is broken.", "L'horloge sur le mur est cassee."),
    ("She teaches mathematics at the school.",
     "Elle enseigne les mathematiques a l'ecole."),
    ("The fisherman repairs his old boat.",
     "Le pecheur repare son vieux bateau."),
    ("We planted tomatoes in the spring.",
     "Nous avons plante des tomates au printemps."),
    ("The letter arrived two days late.",
     "La lettre est arrivee avec deux jours de retard."),
    ("His office is on the third floor.", "Son bureau est au troisieme etage."),
    ("The birds sing early in the morning.",
     "Les oiseaux chantent tot le matin."),
    ("I prefer tea without any sugar.", "Je prefere le the sans sucre."),
    ("The road follows the coast for miles.",
     "La route longe la cote sur des kilometres."),
    ("Their house has a large kitchen.", "Leur maison a une grande cuisine."),
    ("The market sells fresh fish daily.",
     "Le marche vend du poisson frais tous les jours."),
    ("Winter evenings are long and cold.",
     "Les soirees d'hiver sont longues et froides."),
]

# neutral ENGLISH chat prompts for the forward steering test
EN_PROMPTS = [
    "Tell me about your favorite season of the year.",
    "Describe how bread is made.",
    "What makes a good friend?",
    "Explain why the sky is blue.",
    "Describe a walk through a forest.",
    "What is your favorite kind of music and why?",
    "Explain how to make a cup of tea.",
    "Describe what a city looks like at night.",
    "What are the benefits of reading books?",
    "Describe the ocean to someone who has never seen it.",
    "Explain why exercise is good for health.",
    "Tell me about an interesting animal.",
]

# FRENCH chat prompts for the reverse steering test
FR_PROMPTS = [
    "Parle-moi de ta saison preferee de l'annee.",
    "Decris comment on fait le pain.",
    "Qu'est-ce qui fait un bon ami ?",
    "Explique pourquoi le ciel est bleu.",
    "Decris une promenade dans une foret.",
    "Quel est ton genre de musique prefere et pourquoi ?",
    "Explique comment preparer une tasse de the.",
    "Decris a quoi ressemble une ville la nuit.",
]

# ------------------------------------------------------------- language scorer
FR_WORDS = {
    "le", "la", "les", "des", "une", "est", "et", "dans", "pour", "que",
    "qui", "ne", "pas", "vous", "nous", "je", "il", "elle", "sur", "avec",
    "ce", "cette", "mais", "plus", "tres", "sont", "ont", "aux", "du", "au",
    "se", "son", "sa", "ses", "leur", "être", "avoir", "fait", "comme",
    "tout", "aussi", "bien", "par", "quand", "ou", "si", "mon", "ma", "mes",
    "de", "un", "en", "y", "était", "c'est", "d'un", "d'une",
}
EN_WORDS = {
    "the", "of", "and", "to", "in", "is", "it", "you", "that", "was",
    "for", "are", "with", "as", "his", "they", "at", "be", "this", "have",
    "from", "or", "had", "by", "but", "not", "what", "all", "were", "we",
    "when", "your", "can", "there", "an", "which", "their", "will", "would",
    "about", "out", "many", "then", "them", "these", "so", "some", "her",
    "him", "into", "time", "has", "look", "two", "more", "very", "after",
}
DIACRITICS = set("éèêëàâîïôûùçœ")


def french_score(text: str) -> float:
    """Fraction of function-word hits that are French; diacritic words count
    as French hits. 0 = pure English markers, 1 = pure French markers."""
    words = re.findall(r"[a-zA-Zéèêëàâîïôûùçœ'-]+", text.lower())
    if not words:
        return 0.0
    fr = sum(1 for w in words
             if w in FR_WORDS or any(c in DIACRITICS for c in w))
    en = sum(1 for w in words if w in EN_WORDS)
    if fr + en == 0:
        return 0.0
    return fr / (fr + en)


# ------------------------------------------------------------------- machinery
def pooled_hidden(model, tok, texts, batch=16):
    """Mean-pooled residual per layer: (n_texts, n_layers+1, d)."""
    outs = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).to(hs[0].dtype)
        pooled = [(h * m).sum(1) / m.sum(1) for h in hs]      # each (B, d)
        outs.append(torch.stack(pooled, dim=1).float().cpu())  # (B, L+1, d)
    return torch.cat(outs).numpy()


def directions_and_separability(pooled_en, pooled_fr, rng):
    """Per-layer diff-of-means direction from a train split; held-out
    accuracy on the test split. Returns dirs (L+1, d), acc (L+1,)."""
    n = len(pooled_en)
    idx = rng.permutation(n)
    tr, te = idx[:int(0.75 * n)], idx[int(0.75 * n):]
    L = pooled_en.shape[1]
    dirs = np.zeros((L, pooled_en.shape[2]))
    acc = np.zeros(L)
    for l in range(L):
        d = pooled_fr[tr, l].mean(0) - pooled_en[tr, l].mean(0)
        d /= (np.linalg.norm(d) + 1e-9)
        dirs[l] = d
        mid = 0.5 * (pooled_fr[tr, l].mean(0) + pooled_en[tr, l].mean(0)) @ d
        pf = pooled_fr[te, l] @ d > mid
        pe = pooled_en[te, l] @ d <= mid
        acc[l] = 0.5 * (pf.mean() + pe.mean())
    return dirs, acc


class Steerer:
    """Adds a fixed vector to a decoder layer's output at every position
    (prefill and each generated token)."""

    def __init__(self, layer_module):
        self.vec = None
        self.h = layer_module.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        if self.vec is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + self.vec.to(out[0].dtype),) + out[1:]
        return out + self.vec.to(out.dtype)

    def remove(self):
        self.h.remove()


def generate(model, tok, prompt_text: str) -> str:
    ids = tok(chat_prompt(tok, prompt_text), return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=False)
    return tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True)


def steered_scores(model, tok, steerer, vec, prompts):
    steerer.vec = vec
    try:
        texts = [generate(model, tok, p) for p in prompts]
    finally:
        steerer.vec = None
    return [french_score(t) for t in texts], texts


# ------------------------------------------------------------------------ main
def main():
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("EXPERIMENT 12: dissecting Qwen3.5-0.8B (first real subject)")
    print("=" * 70)

    # ---------------------------------------------------------- [0] attach
    model, tok = load_qwen()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    layers = decoder_layers(model)
    n_params = sum(p.numel() for p in model.parameters())
    ltypes = model.config.layer_types if hasattr(model.config, "layer_types") \
        else model.config.text_config.layer_types
    n_lin = sum(t == "linear_attention" for t in ltypes)
    print(f"\n[0] attached: {n_params/1e6:.0f}M params, {len(layers)} layers "
          f"({n_lin} GatedDeltaNet + {len(ltypes)-n_lin} full attention)")

    en_texts = [p[0] for p in PAIRS]
    fr_texts = [p[1] for p in PAIRS]
    pooled_en = pooled_hidden(model, tok, en_texts)
    pooled_fr = pooled_hidden(model, tok, fr_texts)
    norms = np.linalg.norm(pooled_en, axis=2).mean(0)          # (L+1,)

    # ------------------------------------------------- [A] blind discovery
    dirs, acc = directions_and_separability(pooled_en, pooled_fr, rng)
    probe_layer = int(np.argmax(acc[1:23]))          # what probe-trust picks
    print(f"\n[A] blind contrast discovery (48 EN/FR pairs, 75/25 split)")
    print(f"    held-out separability: min {acc[1:].min():.2f}, "
          f"max {acc[1:].max():.2f} -- probe-trust would pick layer "
          f"{probe_layer}")

    # shuffled-label refusal control for the DISCOVERY step
    mix = np.concatenate([pooled_en, pooled_fr])
    lab = rng.permutation(len(mix))
    fake_en, fake_fr = mix[lab[:len(PAIRS)]], mix[lab[len(PAIRS):]]
    _, acc_null = directions_and_separability(fake_en, fake_fr, rng)
    print(f"    shuffled-label control acc: max {acc_null[1:].max():.2f} "
          f"(chance ~0.5)")

    def dvec(hs_idx):
        return torch.tensor(dirs[hs_idx], dtype=torch.float32,
                            device=model.device)

    # ------------------------------------- [B] pilot: causal site selection
    print(f"\n[B] causal pilot map ({len(PILOT_LAYERS)} layers x "
          f"alphas {PILOT_ALPHAS}, 4 prompts each)")
    pilot = {}          # layer -> {alpha: mean score}
    for L in PILOT_LAYERS:
        st = Steerer(layers[L])
        row = {}
        for a in PILOT_ALPHAS:
            sc, _ = steered_scores(model, tok, st,
                                   dvec(L + 1) * (a * norms[L + 1]),
                                   EN_PROMPTS[:4])
            row[a] = float(np.mean(sc))
        st.remove()
        pilot[L] = row
        print(f"    layer {L:>2} ({ltypes[L][:6]:>6}): probe acc "
              f"{acc[L+1]:.2f} | causal french "
              + " ".join(f"a={a}:{row[a]:.3f}" for a in PILOT_ALPHAS))
    best_layer = max(pilot, key=lambda L: max(pilot[L].values()))
    best_hs = best_layer + 1
    layer0_causal = max(pilot[0].values())
    print(f"    -> causal winner: layer {best_layer} "
          f"({ltypes[best_layer]}); layer 0 (probe pick of run 1): "
          f"{layer0_causal:.3f}")

    # ------------------------------------ [C] dose-response at causal site
    print(f"\n[C] dose-response at layer {best_layer} "
          f"(mean residual norm {norms[best_hs]:.1f})")
    d = dvec(best_hs)
    steerer = Steerer(layers[best_layer])

    dose_curve, examples = [], {}
    for a in DOSE_ALPHAS:
        vec = d * (a * norms[best_hs]) if a else None
        scores, texts = steered_scores(model, tok, steerer, vec, EN_PROMPTS)
        dose_curve.append(float(np.mean(scores)))
        examples[a] = texts[0]
        print(f"    alpha={a:>4}: french {np.mean(scores):.3f}   "
              f"| {texts[0][:58]!r}")
    baseline = dose_curve[0]
    steered_best = max(dose_curve[1:])
    a_best = DOSE_ALPHAS[1 + int(np.argmax(dose_curve[1:]))]

    # random-direction specificity null (same norm, best dose)
    rvec = torch.randn(dirs.shape[1],
                       generator=torch.Generator().manual_seed(SEED))
    rvec = (rvec / rvec.norm()).to(model.device)
    null_scores, null_texts = steered_scores(
        model, tok, steerer, rvec * (a_best * norms[best_hs]), EN_PROMPTS)
    null_effect = float(np.mean(null_scores))
    print(f"    random-direction null at alpha={a_best}: {null_effect:.3f}")

    # shuffled-label direction (refusal control, causal side)
    dn = fake_fr[:, best_hs].mean(0) - fake_en[:, best_hs].mean(0)
    dn = torch.tensor(dn / (np.linalg.norm(dn) + 1e-9),
                      dtype=torch.float32, device=model.device)
    shuf_scores, _ = steered_scores(
        model, tok, steerer, dn * (a_best * norms[best_hs]), EN_PROMPTS)
    shuf_effect = float(np.mean(shuf_scores))
    print(f"    shuffled-label direction at alpha={a_best}: {shuf_effect:.3f}")

    # reverse: -alpha on FRENCH prompts must SUPPRESS French
    fr_base_scores, _ = steered_scores(model, tok, steerer, None, FR_PROMPTS)
    fr_steer_scores, fr_steer_texts = steered_scores(
        model, tok, steerer, -d * (a_best * norms[best_hs]), FR_PROMPTS)
    fr_base = float(np.mean(fr_base_scores))
    fr_steer = float(np.mean(fr_steer_scores))
    print(f"    reverse on French prompts: {fr_base:.3f} -> {fr_steer:.3f} "
          f"(-alpha={a_best})")
    print(f"      e.g. {fr_steer_texts[0][:70]!r}")

    steerer.remove()

    # ---------------------------------------------------------------- gates
    gates = {
        "A1_discovery_separable": (acc[best_hs] >= 0.90,
                                   f"held-out acc {acc[best_hs]:.2f} at the "
                                   f"causal site, layer {best_layer}"),
        "A2_discovery_refusal": (acc_null[1:].max() <= 0.70,
                                 f"shuffled-label acc max "
                                 f"{acc_null[1:].max():.2f} (chance 0.5)"),
        "P1_probes_lie_pinned": (acc[1] >= 0.90 and layer0_causal <= 0.05,
                                 f"layer 0: probe acc {acc[1]:.2f} but causal "
                                 f"french {layer0_causal:.3f} -- decodable-"
                                 f"everywhere != steerable (run-1 lesson, "
                                 f"re-asserted)"),
        "B1_steering_causal": (steered_best >= 0.30 and baseline <= 0.05,
                               f"french {baseline:.3f} -> {steered_best:.3f} "
                               f"at alpha={a_best} (bars <=0.05, >=0.30)"),
        "B2_dose_response": (all(dose_curve[i + 1] >= dose_curve[i] - 0.02
                                 for i in range(len(dose_curve) - 1))
                             or dose_curve[-1] >= 0.5,
                             f"curve {[round(s, 3) for s in dose_curve]}"),
        "B3_specificity_null": (null_effect <= baseline + 0.05,
                                f"random dir {null_effect:.3f} vs baseline "
                                f"{baseline:.3f} (bar +0.05)"),
        "B4_shuffled_dir_inert": (shuf_effect <= baseline + 0.05,
                                  f"shuffled-label dir {shuf_effect:.3f}"),
        "B5_reversible": (fr_base >= 0.25 and fr_steer <= 0.5 * fr_base,
                          f"French prompts {fr_base:.3f} -> {fr_steer:.3f} "
                          f"under -alpha (bar: halved)"),
    }
    n_pass = sum(v[0] for v in gates.values())
    print("\n" + "=" * 70)
    for k, (ok, msg) in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}: {msg}")
    print(f"  {n_pass}/{len(gates)} gates passed")

    # ------------------------------------------------------ concept artifact
    concept = GeometricConceptObject(
        name="output-language: French",
        kind="feature",
        source=f"exp12 / {MODEL_ID}",
        layer=f"model.layers.{best_layer} (residual out, "
              f"{ltypes[best_layer]})",
        subspace=[dirs[best_hs].tolist()],
        activating_examples=fr_texts[:5],
        counterexamples=en_texts[:5],
        causal_influence=steered_best - baseline,
        causal_test=f"residual steering, dose {a_best}x mean norm, "
                    f"greedy generation, french-marker ratio",
        null_baseline=max(null_effect - baseline, shuf_effect - baseline,
                          0.001),
        story=f"1D residual direction, causally selected at layer "
              f"{best_layer}/24: steers output language EN->FR "
              f"({baseline:.2f}->{steered_best:.2f}), reversible on French "
              f"prompts ({fr_base:.2f}->{fr_steer:.2f}). Decodable at ALL "
              f"24 layers (acc 1.00) but causally steerable only early "
              f"(input perception) and late (output selection) -- the "
              f"middle is inert at these doses.",
        extra={"dose_curve": dict(zip(map(str, DOSE_ALPHAS), dose_curve)),
               "pilot_map": {str(k): v for k, v in pilot.items()},
               "separability_by_layer": acc.tolist(),
               "example_steered": examples[a_best]},
    )
    concept.grade()
    cpath = concept.save(CONCEPTS / "exp12_qwen_french_direction.json")
    print(f"\n  concept object: confidence {concept.confidence:.2f} -> {cpath}")

    # ----------------------------------------------------------------- figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("exp12 -- Qwen3.5-0.8B: the French direction "
                 "(decodable everywhere, steerable at the edges)",
                 fontsize=13)

    ax = axes[0, 0]
    ax.plot(range(1, 25), acc[1:25], "o-", label="EN/FR contrast")
    ax.plot(range(1, 25), acc_null[1:25], "s--", color="gray",
            label="shuffled labels")
    ax.axhline(0.5, color="k", lw=0.5)
    ax.axvline(best_hs, color="r", ls=":", label=f"causal site {best_layer}")
    ax.set_xlabel("hidden state (after layer i-1)")
    ax.set_ylabel("held-out separability")
    ax.set_title("[A] probe view: readable at every layer")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    Ls = list(pilot)
    x = np.arange(len(Ls))
    for i, a in enumerate(PILOT_ALPHAS):
        ax.bar(x + (i - 0.5) * 0.35, [pilot[L][a] for L in Ls], 0.35,
               alpha=0.75, label=f"steer a={a}")
    ax.plot(x, [acc[L + 1] for L in Ls], "ko-", ms=4, label="probe acc")
    ax.set_xticks(x, [str(L) for L in Ls])
    ax.set_xlabel("steered layer")
    ax.set_ylabel("French-marker ratio")
    ax.set_title("[B] causal map: probes lie, edges steer")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(DOSE_ALPHAS, dose_curve, "o-", color="tab:red",
            label=f"French dir @ layer {best_layer}")
    ax.axhline(null_effect, color="gray", ls="--", label="random dir")
    ax.axhline(shuf_effect, color="tab:brown", ls=":",
               label="shuffled-label dir")
    ax.plot([a_best], [fr_steer], "bv", ms=9,
            label=f"reverse: FR prompts, -a ({fr_base:.2f}->{fr_steer:.2f})")
    ax.set_xlabel("steering dose (x mean residual norm)")
    ax.set_ylabel("French-marker ratio")
    ax.set_title("[C] dose-response + nulls at the causal site")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.axis("off")
    demo = (f"prompt: {EN_PROMPTS[0]!r}\n\n"
            f"a=0:      {examples[0.0][:140]}\n\n"
            f"a={a_best}: {examples[a_best][:140]}\n\n"
            f"FR prompt, -a={a_best}:\n{fr_steer_texts[0][:140]}")
    ax.text(0.02, 0.98, demo, va="top", fontsize=8, family="monospace",
            wrap=True)
    ax.set_title("behavior, verbatim")

    fig.tight_layout()
    fpath = RESULTS / "exp12_qwen_dissection.png"
    fig.savefig(fpath, dpi=130)

    report = {
        "model": MODEL_ID, "params": n_params,
        "layer_types": {"linear_attention": n_lin,
                        "full_attention": len(ltypes) - n_lin},
        "causal_site": best_layer, "probe_pick_run1": 0,
        "separability": acc.tolist(),
        "separability_shuffled": acc_null.tolist(),
        "pilot_map": {str(k): v for k, v in pilot.items()},
        "dose_alphas": DOSE_ALPHAS, "dose_curve": dose_curve,
        "null_random_dir": null_effect, "null_shuffled_dir": shuf_effect,
        "reverse_fr_baseline": fr_base, "reverse_fr_steered": fr_steer,
        "gates": {k: {"pass": bool(v[0]), "detail": v[1]}
                  for k, v in gates.items()},
        "concept_confidence": concept.confidence,
        "example_generations": {"baseline": examples[0.0],
                                "steered": examples[a_best],
                                "reverse_steered": fr_steer_texts[0],
                                "null": null_texts[0]},
        "runtime_sec": round(time.time() - t0, 1),
    }
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2),
                                         encoding="utf-8")
    print(f"  report:  {RESULTS / 'report.json'}")
    print(f"  figure:  {fpath}")
    print(f"  runtime: {report['runtime_sec']}s")
    print("=" * 70)
    return 0 if n_pass == len(gates) else 1


if __name__ == "__main__":
    sys.exit(main())
