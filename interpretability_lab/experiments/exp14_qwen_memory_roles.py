"""EXPERIMENT 14 -- what is the DeltaNet memory FOR?

exp13 found an entire memory system that is readable but not consulted:
transplanting all 18 recurrent states barely moves the model's factual
beliefs (0.21) while transplanting 6 KV caches flips them (0.85). Two
follow-up questions, same transplant machinery:

  [A] CONTENT TYPE  maybe facts are the wrong cargo. Transplant memories
      that differ in LANGUAGE (French vs English prompt) and FORMAT
      (numbered-list vs prose instruction) and see whether the recurrent
      state carries those. If DeltaNet moves language/format but not
      facts, the two memory systems have a division of labor.

  [B] RANGE         attention retrieval is easiest at short range;
      DeltaNet's constant-size state is the architecture's bet on long
      contexts. Sweep the fact-to-question distance (~0 / ~300 / ~900
      filler tokens) and watch whether the apportionment (KV 0.85 vs
      DeltaNet 0.21 at range 0) shifts -- a handoff point between memory
      systems would show as the curves crossing. No shift is also an
      answer: the recurrent memory is not a fact store at any range.

KV transplants need donor/receiver caches aligned position-for-position.
Fact pairs align by construction (single-token contrasts). Language and
format pairs are auto-BALANCED: a pad slot in each prompt is filled with
filler interjections until the chat-templated token counts match exactly
(asserted; a pair that cannot balance skips its KV conditions and says so).

Gates certify machinery, no-ops and nulls on every axis. The apportionment
numbers and the range curve are the findings, whatever they say.
"""

from __future__ import annotations

import copy
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

from interpretability_lab.experiments.exp12_qwen_dissection import french_score
from interpretability_lab.experiments.exp13_qwen_memory import (
    FACTS, FULL_ATTN, LINEAR, NEUTRAL_DONOR, continue_greedy, patch,
    prefill, state_snapshot)
from interpretability_lab.models.pretrained import MODEL_ID, chat_prompt, load_qwen

RESULTS = Path(__file__).parent / "results" / "exp14"
SEED = 0

# ------------------------------------------------------------------ materials
LANG_PAIRS = [   # (english/receiver, french/donor), each with a {PAD} slot
    ("My friend has a small garden behind the house. {PAD}Please describe "
     "what such a garden looks like in spring.",
     "Mon ami a un petit jardin derrière la maison. {PAD}Décris à quoi "
     "ressemble un tel jardin au printemps."),
    ("I watched a baker make fresh bread this morning. {PAD}Please explain "
     "how bread is usually made.",
     "J'ai regardé un boulanger faire du pain frais ce matin. {PAD}Explique "
     "comment le pain est généralement fait."),
    ("We spent last summer in a small village by the sea. {PAD}Please "
     "describe what mornings by the sea are like.",
     "Nous avons passé l'été dernier dans un petit village au bord de la "
     "mer. {PAD}Décris à quoi ressemblent les matins au bord de la mer."),
    ("My sister practices the piano every evening. {PAD}Please describe "
     "why music practice is rewarding.",
     "Ma sœur joue du piano tous les soirs. {PAD}Décris pourquoi la "
     "pratique de la musique est gratifiante."),
    ("The market near my street sells fruit and flowers. {PAD}Please "
     "describe a busy morning at such a market.",
     "Le marché près de ma rue vend des fruits et des fleurs. {PAD}Décris "
     "une matinée animée dans un tel marché."),
    ("Last year we hiked in the mountains for three days. {PAD}Please "
     "describe what a long mountain hike feels like.",
     "L'année dernière, nous avons randonné en montagne pendant trois "
     "jours. {PAD}Décris ce que l'on ressent lors d'une longue randonnée "
     "en montagne."),
]

FORMAT_TOPICS = ["morning exercise", "reading books", "drinking water",
                 "getting good sleep", "learning languages",
                 "keeping a journal"]
FORMAT_RECV = ("Please answer in plain flowing prose, never using any "
               "list. {PAD}Question: what are the benefits of {T}?")
FORMAT_DONOR = ("Please answer strictly as a numbered list with short "
                "items. {PAD}Question: what are the benefits of {T}?")

# Range-axis fact pairs. TWO measurement artifacts found and fixed by G3
# across runs (both readout, never machinery):
#   run 1 (pet, 40-token gens): at distance 0 the model spends its whole
#     budget on preamble with no animal words -> 0.5 no-evidence ties
#     inflate the baseline. Fix: longer generations.
#   run 2 (vehicle/city): those questions point at the fact with a PRONOUN
#     ("what Sam enjoys about IT", "what she should eat THERE"); after
#     300-900 filler tokens the referent breaks and ALL conditions --
#     including the positive control -- tie at 0.5. The pet question names
#     its referent ("my friend's pet") and survives distance.
# Configuration: pet template, n_gen=80.
RANGE_PAIRS = [("pet", n) for n in FACTS["pet"]["names"]]

FILLER_BANK = [
    "Yesterday the weather was mild and a little cloudy in the afternoon.",
    "The train schedule changed this week, so the commute takes longer.",
    "At the office we reorganized the meeting room and moved the printer.",
    "On the way home I stopped at the store for bread, rice, and eggs.",
    "The evening news mentioned roadwork downtown through the weekend.",
    "I watered the plants on the balcony and answered two old letters.",
    "A neighborhood festival is being planned for early next month.",
    "The library extended its opening hours for the summer season.",
    "Someone repainted the fence across the street a pale shade of green.",
    "The bakery on the corner started selling a new kind of rye loaf.",
    "I finally fixed the squeaky hinge on the kitchen cabinet door.",
    "The bus route past the park was rerouted for the street repairs.",
    "We compared notes about the best way to store winter clothes.",
    "The community garden assigned new plots to five more families.",
    "A documentary about lighthouses was on television last night.",
]

DISTANCES = [0, 300, 900]        # target filler size in TOKENS


def listness(text: str) -> float:
    lines = [l for l in text.splitlines() if l.strip()]
    numbered = sum(1 for l in lines if re.match(r"\s*\d+[.)]\s", l))
    return 1.0 if numbered >= 2 else 0.0


def fact_flip(text: str, lex_donor: set, lex_recv: set) -> float:
    w = re.findall(r"[a-zA-Z']+", text.lower())
    d = sum(1 for x in w if x in lex_donor)
    r = sum(1 for x in w if x in lex_recv)
    return 0.5 if d + r == 0 else d / (d + r)


# ------------------------------------------------------------------- machinery
def n_tokens(tok, text: str) -> int:
    return len(tok(chat_prompt(tok, text)).input_ids)


def balance(tok, recv: str, donor: str, pad_word="hmm "):
    """Fill {PAD} slots so both prompts tokenize to the SAME length."""
    ka = kb = 0
    for _ in range(160):
        a = recv.replace("{PAD}", pad_word * ka)
        b = donor.replace("{PAD}", pad_word * kb)
        la, lb = n_tokens(tok, a), n_tokens(tok, b)
        if la == lb:
            return a, b
        if la < lb:
            ka += 1
        else:
            kb += 1
    return None, None


def make_filler(tok, target_tokens: int) -> str:
    if target_tokens == 0:
        return ""
    out, i = [], 0
    while len(tok(" ".join(out)).input_ids) < target_tokens:
        out.append(FILLER_BANK[i % len(FILLER_BANK)])
        i += 1
    return " ".join(out) + " "


def fresh_cache(model, tok, text, stored=None):
    """Deepcopy a prefilled cache if possible, else re-prefill."""
    if stored is not None:
        try:
            ids, cache = stored
            return ids, copy.deepcopy(cache)
        except Exception:
            pass
    return prefill(model, tok, text)


def run_conditions(model, tok, recv_text, donor_text, conds,
                   neutral_snap=None, n_gen=40):
    """Prefill donor once, then generate the receiver under each condition.
    Returns {cond: generated_text}."""
    _, donor_cache = prefill(model, tok, donor_text)
    donor_snap = state_snapshot(donor_cache)
    del donor_cache
    stored = prefill(model, tok, recv_text)
    out = {}
    for cond in conds:
        ids, cache = fresh_cache(model, tok, recv_text, stored)
        if cond == "deltanet":
            patch(cache, donor_snap, deltanet_layers=set(LINEAR))
        elif cond == "kv":
            patch(cache, donor_snap, kv=True)
        elif cond == "both":
            patch(cache, donor_snap, deltanet_layers=set(LINEAR), kv=True)
        elif cond == "neutral":
            patch(cache, neutral_snap, deltanet_layers=set(LINEAR))
        elif cond == "selfpatch":
            _, c2 = prefill(model, tok, recv_text)
            patch(cache, state_snapshot(c2),
                  deltanet_layers=set(LINEAR), kv=True)
        out[cond] = continue_greedy(model, tok, ids, cache, n=n_gen)
    return out


# ------------------------------------------------------------------------ main
def main():
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("EXPERIMENT 14: what is the DeltaNet memory FOR?")
    print("=" * 70)
    model, tok = load_qwen()
    _, neutral_cache = prefill(model, tok, NEUTRAL_DONOR)
    neutral_snap = state_snapshot(neutral_cache)
    del neutral_cache

    CONDS = ["baseline", "deltanet", "kv", "both", "neutral"]
    axes = {}          # axis -> {cond: [scores]}
    examples = {}

    # --------------------------------------------- [A1] LANGUAGE transplant
    print("\n[A1] language axis (receiver EN, donor FR; score = french "
          "markers in continuation)")
    axes["language"] = {c: [] for c in CONDS}
    donor_sanity = []
    noop_ok = noop_total = 0
    for i, (en, fr) in enumerate(LANG_PAIRS):
        recv, donor = balance(tok, en, fr)
        if recv is None:
            print(f"    pair {i}: could not balance token counts -- skipped")
            continue
        res = run_conditions(model, tok, recv, donor, CONDS, neutral_snap)
        for c in CONDS:
            axes["language"][c].append(french_score(res[c]))
        if i == 0:
            examples["language"] = res
        if i < 3:            # donor-alone sanity + strict no-op
            ids, cache = prefill(model, tok, donor)
            donor_sanity.append(french_score(
                continue_greedy(model, tok, ids, cache)))
            r2 = run_conditions(model, tok, recv, donor, ["selfpatch"])
            ids, cache = prefill(model, tok, recv)
            base_again = continue_greedy(model, tok, ids, cache)
            noop_total += 1
            noop_ok += int(r2["selfpatch"] == base_again)
    lang = {c: float(np.mean(axes["language"][c])) for c in CONDS}
    print("    " + " ".join(f"{c}:{lang[c]:.2f}" for c in CONDS)
          + f" | donor-alone {np.mean(donor_sanity):.2f} "
          f"| noop {noop_ok}/{noop_total}")

    # ----------------------------------------------- [A2] FORMAT transplant
    print("\n[A2] format axis (receiver 'prose', donor 'numbered list'; "
          "score = numbered lines present)")
    axes["format"] = {c: [] for c in CONDS}
    for i, topic in enumerate(FORMAT_TOPICS):
        recv, donor = balance(tok, FORMAT_RECV.replace("{T}", topic),
                              FORMAT_DONOR.replace("{T}", topic))
        if recv is None:
            print(f"    topic {topic!r}: could not balance -- skipped")
            continue
        res = run_conditions(model, tok, recv, donor, CONDS, neutral_snap)
        for c in CONDS:
            axes["format"][c].append(listness(res[c]))
        if i == 0:
            examples["format"] = res
    fmt = {c: float(np.mean(axes["format"][c])) for c in CONDS}
    print("    " + " ".join(f"{c}:{fmt[c]:.2f}" for c in CONDS))

    # ------------------------------------------------- [B] RANGE sweep
    print("\n[B] fact transplant vs distance (dog->cat, filler between "
          "fact and question)")
    range_scores = {d: {c: [] for c in CONDS} for d in DISTANCES}
    for d in DISTANCES:
        filler = make_filler(tok, d)
        for fkey, name in RANGE_PAIRS:
            fact = FACTS[fkey]
            recv = fact["template"].format(X=fact["b"], N=name, F=filler)
            donor = fact["template"].format(X=fact["a"], N=name, F=filler)
            conds = CONDS if d == DISTANCES[-1] else \
                ["baseline", "deltanet", "kv", "both"]
            res = run_conditions(model, tok, recv, donor, conds,
                                 neutral_snap, n_gen=80)
            for c in conds:
                range_scores[d][c].append(
                    fact_flip(res[c], fact["lex_a"], fact["lex_b"]))
            if d == DISTANCES[-1] and (fkey, name) == RANGE_PAIRS[0]:
                examples["range_far"] = res
        m = {c: float(np.mean(v)) for c, v in range_scores[d].items() if v}
        print(f"    distance ~{d:>4} tok: "
              + " ".join(f"{c}:{m[c]:.2f}" for c in m))
    rng_means = {d: {c: float(np.mean(v))
                     for c, v in range_scores[d].items() if v}
                 for d in DISTANCES}

    # ---------------------------------------------------------------- gates
    far = DISTANCES[-1]
    gates = {
        "G1_machinery_language": (lang["both"] >= 0.80
                                  and lang["baseline"] <= 0.05,
                                  f"both {lang['both']:.2f}, baseline "
                                  f"{lang['baseline']:.2f}"),
        "G2_machinery_format": (fmt["both"] >= 0.80
                                and fmt["baseline"] <= 0.20,
                                f"both {fmt['both']:.2f}, baseline "
                                f"{fmt['baseline']:.2f}"),
        "G3_machinery_range": (all(rng_means[d]["both"] >= 0.80
                                   and rng_means[d]["baseline"] <= 0.20
                                   for d in DISTANCES),
                               "both/baseline at all distances: "
                               + " ".join(
                                   f"{d}:{rng_means[d]['both']:.2f}/"
                                   f"{rng_means[d]['baseline']:.2f}"
                                   for d in DISTANCES)),
        "G4_noop_exact": (noop_ok == noop_total and noop_total > 0,
                          f"self-patch byte-identical {noop_ok}/{noop_total}"),
        "G5_nulls": (lang["neutral"] <= 0.05
                     and rng_means[far]["neutral"] <= 0.55,
                     f"neutral: language {lang['neutral']:.2f}, "
                     f"fact@far {rng_means[far]['neutral']:.2f}"),
        "G6_donor_sanity": (float(np.mean(donor_sanity)) >= 0.80,
                            f"FR donor alone answers in french "
                            f"{np.mean(donor_sanity):.2f}"),
    }
    n_pass = sum(v[0] for v in gates.values())
    print("\n" + "=" * 70)
    for k, (ok, msg) in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}: {msg}")
    print(f"  {n_pass}/{len(gates)} gates passed")

    print("\n  THE DIVISION OF LABOR (deltanet-only vs kv-only flip):")
    print(f"    facts @ ~0 tok   : deltanet "
          f"{rng_means[0]['deltanet']:.2f} | kv {rng_means[0]['kv']:.2f}")
    print(f"    facts @ ~300 tok : deltanet "
          f"{rng_means[300]['deltanet']:.2f} | kv {rng_means[300]['kv']:.2f}")
    print(f"    facts @ ~900 tok : deltanet "
          f"{rng_means[900]['deltanet']:.2f} | kv {rng_means[900]['kv']:.2f}")
    print(f"    LANGUAGE         : deltanet {lang['deltanet']:.2f} | "
          f"kv {lang['kv']:.2f}")
    print(f"    FORMAT           : deltanet {fmt['deltanet']:.2f} | "
          f"kv {fmt['kv']:.2f}")

    # ----------------------------------------------------------------- figure
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("exp14 -- what is the DeltaNet memory for?", fontsize=13)

    ax = axs[0, 0]
    labels = ["facts", "language", "format"]
    dn = [rng_means[0]["deltanet"], lang["deltanet"], fmt["deltanet"]]
    kv = [rng_means[0]["kv"], lang["kv"], fmt["kv"]]
    bs = [rng_means[0]["baseline"], lang["baseline"], fmt["baseline"]]
    x = np.arange(3)
    ax.bar(x - 0.25, bs, 0.23, color="tab:blue", label="baseline")
    ax.bar(x, dn, 0.23, color="tab:red", label="deltanet-only")
    ax.bar(x + 0.25, kv, 0.23, color="tab:orange", label="kv-only")
    ax.set_xticks(x, labels)
    ax.set_ylabel("flip toward donor property")
    ax.set_title("[A] which cargo does each memory carry?")
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    ax.plot(DISTANCES, [rng_means[d]["deltanet"] for d in DISTANCES],
            "o-", color="tab:red", label="deltanet-only")
    ax.plot(DISTANCES, [rng_means[d]["kv"] for d in DISTANCES],
            "s-", color="tab:orange", label="kv-only")
    ax.plot(DISTANCES, [rng_means[d]["both"] for d in DISTANCES],
            "^--", color="tab:purple", label="both")
    ax.plot(DISTANCES, [rng_means[d]["baseline"] for d in DISTANCES],
            "v--", color="tab:blue", label="baseline")
    ax.set_xlabel("fact-to-question distance (filler tokens)")
    ax.set_ylabel("fact flip")
    ax.set_title("[B] handoff hunt: apportionment vs range")
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    conds_plot = ["baseline", "neutral", "deltanet", "kv", "both"]
    w = 0.38
    ax.bar(np.arange(5) - w / 2, [lang[c] for c in conds_plot], w,
           color="tab:green", alpha=0.8, label="language axis")
    ax.bar(np.arange(5) + w / 2, [fmt[c] for c in conds_plot], w,
           color="tab:gray", alpha=0.8, label="format axis")
    ax.set_xticks(np.arange(5), conds_plot)
    ax.set_ylabel("flip toward donor property")
    ax.set_title("[A] full condition detail")
    ax.legend(fontsize=8)

    ax = axs[1, 1]
    ax.axis("off")
    ex = examples.get("language", {})
    demo = ("LANGUAGE axis, receiver EN prompt + donor FR memory\n\n"
            + "\n\n".join(f"{c}: {ex.get(c, '')[:110]}"
                          for c in ["baseline", "deltanet", "kv", "both"]))
    ax.text(0.02, 0.98, demo, va="top", fontsize=8, family="monospace",
            wrap=True)
    ax.set_title("behavior, verbatim")

    fig.tight_layout()
    fpath = RESULTS / "exp14_qwen_memory_roles.png"
    fig.savefig(fpath, dpi=130)

    report = {
        "model": MODEL_ID,
        "language_means": lang, "format_means": fmt,
        "range_means": {str(d): rng_means[d] for d in DISTANCES},
        "donor_sanity_french": [float(s) for s in donor_sanity],
        "noop": [noop_ok, noop_total],
        "gates": {k: {"pass": bool(v[0]), "detail": v[1]}
                  for k, v in gates.items()},
        "examples": {k: {c: t for c, t in v.items()}
                     for k, v in examples.items()},
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
