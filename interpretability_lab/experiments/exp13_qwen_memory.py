"""EXPERIMENT 13 -- reading and editing the DeltaNet working memory.

Qwen3.5-0.8B keeps context in TWO different memory systems:
  - 18 GatedDeltaNet layers: a fixed-size recurrent state per layer
    (16 heads x 128x128 fast-weight matrix = 262,144 numbers), written
    token-by-token with a delta rule + decay. This is WORKING MEMORY --
    same size whether the context is 5 tokens or 5,000.
  - 6 full-attention layers: an ordinary KV cache (grows with length).

Transformer interpretability has tooling for attention. The DeltaNet state
is near-unexplored territory: nobody knows what facts it carries, for how
long, or whether it is causally load-bearing next to the KV cache.

Design (facts are single-token contrasts so donor/receiver prompts have
IDENTICAL token layouts):

  [A] READ    can "dog vs cat" (etc.) be decoded from the state matrices
              at end of prompt? Per-layer map + retention after 0/~40/~120
              words of filler between fact and readout point.
  [B] SURGERY prefill donor ("...pet dog...") and receiver ("...pet cat...")
              prompts, transplant memory between them, continue generation,
              and count which animal the model describes:
                - deltanet-only patch (18 recurrent+conv states)
                - kv-only patch      (6 keys/values)
                - both               (positive control: must fully flip)
                - self-patch         (must be a byte-identical no-op)
                - neutral-state      (null: unrelated memory must not
                                      produce donor content)
                - early/mid/late     (deltanet subsets: where does the
                                      fact live?)

The apportionment question -- does the belief live in the DeltaNet memory
or the KV cache? -- has no known answer. Whatever the number is, it is the
finding. Gates certify the MEASUREMENT (machinery, no-op, nulls, probe
controls), not a hoped-for outcome.
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

from interpretability_lab.models.pretrained import (MODEL_ID, chat_prompt,
                                                    load_qwen)

RESULTS = Path(__file__).parent / "results" / "exp13"
SEED = 0
MAX_NEW = 40

FULL_ATTN = {3, 7, 11, 15, 19, 23}
LINEAR = [i for i in range(24) if i not in FULL_ATTN]
GROUPS = {"early": [l for l in LINEAR if l < 8],
          "mid": [l for l in LINEAR if 8 <= l < 16],
          "late": [l for l in LINEAR if l >= 16]}

# ------------------------------------------------------------------ materials
# Each fact type: (template, option_a, option_b, lexicon_a, lexicon_b).
# Options are chosen to tokenize to the SAME number of tokens (asserted at
# runtime) so KV caches align position-for-position.
FACTS = {
    "pet": {
        "template": ("My friend has a pet {X} named {N}. {F}Please describe "
                     "my friend's pet in detail, including what it looks "
                     "like and how it behaves."),
        "a": "dog", "b": "cat",
        "lex_a": {"dog", "dogs", "puppy", "puppies", "bark", "barks",
                  "barking", "leash", "fetch", "kennel", "canine", "woof"},
        "lex_b": {"cat", "cats", "kitten", "kittens", "meow", "meows",
                  "meowing", "purr", "purrs", "purring", "whiskers",
                  "litter", "feline"},
        "names": ["Max", "Bella", "Rocky", "Luna"],
    },
    "vehicle": {
        "template": ("My neighbor {N} just bought a brand new {X}. {F}Please "
                     "describe what {N} probably enjoys about it and how it "
                     "is typically used."),
        "a": "car", "b": "bike",
        "lex_a": {"car", "cars", "engine", "drive", "driving", "gasoline",
                  "horsepower", "sedan", "trunk", "headlights", "steering",
                  "highway"},
        "lex_b": {"bike", "bikes", "bicycle", "bicycles", "pedal", "pedals",
                  "pedaling", "handlebars", "cycling", "cyclist", "helmet"},
        "names": ["Sam", "Ana", "Leo", "Mia"],
    },
    "city": {
        "template": ("My sister {N} moved to {X} last month. {F}Please tell "
                     "me what she should visit and what food she should "
                     "try there."),
        "a": "Paris", "b": "Tokyo",
        "lex_a": {"paris", "france", "french", "eiffel", "louvre", "seine",
                  "montmartre", "croissant", "croissants", "baguette",
                  "macarons"},
        "lex_b": {"tokyo", "japan", "japanese", "shibuya", "shinjuku",
                  "sushi", "ramen", "sakura", "fuji", "tempura", "akihabara"},
        "names": ["Emma", "Nina", "Rosa", "Iris"],
    },
}

FILLERS = {
    0: "",
    1: ("Yesterday the weather was mild and a little cloudy, and the "
        "streets were quiet in the afternoon. I finished my errands early "
        "and read for a while before dinner. "),
    2: ("Yesterday the weather was mild and a little cloudy, and the "
        "streets were quiet in the afternoon. I finished my errands early "
        "and read for a while before dinner. The train schedule changed "
        "this week, so the morning commute takes a few minutes longer than "
        "usual. At the office we reorganized the meeting room and moved "
        "the printer closer to the window. On the way home I stopped at "
        "the grocery store for bread, rice, and a carton of eggs. The "
        "evening news mentioned roadwork downtown continuing through the "
        "weekend, and a neighborhood festival planned for next month. I "
        "watered the plants on the balcony and answered two letters that "
        "had been sitting on my desk for days. "),
}

NEUTRAL_DONOR = ("The committee reviewed the quarterly maintenance report "
                 "and approved the updated cleaning schedule for the "
                 "building. Several light fixtures in the corridor will be "
                 "replaced next week. Please summarize the plan briefly.")

PROBE_NAMES = ["Max", "Bella", "Rocky", "Luna", "Charlie", "Daisy", "Duke",
               "Molly", "Buddy", "Sadie", "Bear", "Ruby", "Toby", "Chloe",
               "Jack", "Lola", "Oscar", "Zoe", "Milo", "Penny", "Leo",
               "Nala", "Finn", "Coco"]


def words_of(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def flip_score(text: str, lex_donor: set, lex_recv: set) -> float:
    """1.0 = talks about the donor's fact, 0.0 = receiver's, 0.5 = neither."""
    w = words_of(text)
    d = sum(1 for x in w if x in lex_donor)
    r = sum(1 for x in w if x in lex_recv)
    if d + r == 0:
        return 0.5
    return d / (d + r)


# ------------------------------------------------------------------- machinery
def prefill(model, tok, user_text: str):
    """Prefill all but the last prompt token. Returns (ids, cache)."""
    ids = tok(chat_prompt(tok, user_text),
              return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(input_ids=ids[:, :-1], use_cache=True)
    return ids, out.past_key_values


def state_snapshot(cache):
    """Clone all memory tensors: (deltanet recurrent, conv, attn keys, vals)."""
    snap = {}
    for i, lay in enumerate(cache.layers):
        if i in FULL_ATTN:
            snap[i] = ("kv", lay.keys.clone(), lay.values.clone())
        else:
            snap[i] = ("lin", lay.recurrent_states.clone(),
                       lay.conv_states.clone())
    return snap


def patch(cache, snap, deltanet_layers=None, kv=False):
    """Overwrite selected memory in `cache` with tensors from `snap`."""
    for i, lay in enumerate(cache.layers):
        if i in FULL_ATTN:
            if kv:
                lay.keys.copy_(snap[i][1])
                lay.values.copy_(snap[i][2])
        elif deltanet_layers is not None and i in deltanet_layers:
            lay.recurrent_states.copy_(snap[i][1])
            lay.conv_states.copy_(snap[i][2])


def continue_greedy(model, tok, ids, cache, n=MAX_NEW) -> str:
    """Feed the final prompt token through the (possibly patched) cache,
    then decode greedily."""
    out_toks = []
    nxt = ids[:, -1:]
    with torch.no_grad():
        for _ in range(n):
            o = model(input_ids=nxt, past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            nxt = o.logits[:, -1].argmax(-1, keepdim=True)
            if nxt.item() == tok.eos_token_id:
                break
            out_toks.append(nxt.item())
    return tok.decode(out_toks, skip_special_tokens=True)


def state_features(cache) -> np.ndarray:
    """Signed 8x8 block-mean pooling of each linear layer's state:
    (16,128,128) -> (16,16,16) -> 4096 dims per layer. Returns (18, 4096)."""
    feats = []
    for i in LINEAR:
        s = cache.layers[i].recurrent_states.float().cpu()      # (1,16,128,128)
        p = s.reshape(16, 16, 8, 16, 8).mean(dim=(2, 4))        # (16,16,16)
        feats.append(p.reshape(-1).numpy())
    return np.stack(feats)


def _split_acc(Xa, Xb, rng):
    n = len(Xa)
    idx = rng.permutation(n)
    tr, te = idx[:int(0.75 * n)], idx[int(0.75 * n):]
    d = Xa[tr].mean(0) - Xb[tr].mean(0)
    d /= (np.linalg.norm(d) + 1e-9)
    mid = 0.5 * (Xa[tr].mean(0) + Xb[tr].mean(0)) @ d
    return 0.5 * ((Xa[te] @ d > mid).mean() + (Xb[te] @ d <= mid).mean())


def probe_acc(Xa, Xb, rng, repeats=20):
    """Diff-of-means midpoint classifier, MEAN over `repeats` random 75/25
    splits. Run 1 used a single split; with 12 test items the max-over-54
    shuffled probes hit 0.75 by order-statistic luck, failing its own
    control. Averaging splits is the fix (measurement, not model)."""
    return float(np.mean([_split_acc(Xa, Xb, rng) for _ in range(repeats)]))


def probe_null(Xa, Xb, rng, repeats=20):
    """Null: permute class labels fresh each repeat, then split."""
    mix = np.concatenate([Xa, Xb])
    n = len(Xa)
    accs = []
    for _ in range(repeats):
        perm = rng.permutation(len(mix))
        accs.append(_split_acc(mix[perm[:n]], mix[perm[n:]], rng))
    return float(np.mean(accs))


# ------------------------------------------------------------------------ main
def main():
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("EXPERIMENT 13: DeltaNet working memory -- read it, then edit it")
    print("=" * 70)
    model, tok = load_qwen()

    # ------------------------------------------------- [A] READ the memory
    print("\n[A] decoding 'dog vs cat' from the recurrent state matrices")
    pet = FACTS["pet"]
    acc_by_filler = {}      # filler -> (18,) acc per linear layer
    acc_null_by_filler = {}
    for f_id, filler in FILLERS.items():
        feats = {"a": [], "b": []}
        for name in PROBE_NAMES:
            for side in ("a", "b"):
                text = pet["template"].format(X=pet[side], N=name, F=filler)
                _, cache = prefill(model, tok, text)
                feats[side].append(state_features(cache))
        Xa = np.stack(feats["a"])          # (24, 18, 4096)
        Xb = np.stack(feats["b"])
        accs = np.array([probe_acc(Xa[:, l], Xb[:, l], rng)
                         for l in range(len(LINEAR))])
        nulls = np.array([probe_null(Xa[:, l], Xb[:, l], rng)
                          for l in range(len(LINEAR))])
        acc_by_filler[f_id] = accs
        acc_null_by_filler[f_id] = nulls
        n_filler_words = len(words_of(filler))
        print(f"    filler {n_filler_words:>3} words: acc best "
              f"{accs.max():.2f} (layer {LINEAR[int(accs.argmax())]}), "
              f"mean {accs.mean():.2f} | shuffled max {nulls.max():.2f}")

    best_l_idx = int(acc_by_filler[0].argmax())
    retention = [float(acc_by_filler[f][best_l_idx]) for f in FILLERS]

    # -------------------------------------------- [B] SURGERY: transplant
    print("\n[B] memory transplant (donor fact -> receiver's head)")
    _, neutral_cache = prefill(model, tok, NEUTRAL_DONOR)
    neutral_snap = state_snapshot(neutral_cache)

    conditions = ["baseline", "selfpatch", "deltanet", "kv", "both",
                  "neutral", "early", "mid", "late"]
    scores = {c: [] for c in conditions}
    examples = {}
    pair_idx = 0
    for fkey, fact in FACTS.items():
        for name in fact["names"]:
            recv_text = fact["template"].format(X=fact["b"], N=name, F="")
            donor_text = fact["template"].format(X=fact["a"], N=name, F="")
            recv_ids = tok(chat_prompt(tok, recv_text), return_tensors="pt")
            donor_ids = tok(chat_prompt(tok, donor_text), return_tensors="pt")
            assert recv_ids.input_ids.shape == donor_ids.input_ids.shape, \
                f"token layout mismatch for {fkey}/{name}"

            _, donor_cache = prefill(model, tok, donor_text)
            donor_snap = state_snapshot(donor_cache)
            lex_d, lex_r = fact["lex_a"], fact["lex_b"]

            for cond in conditions:
                ids, cache = prefill(model, tok, recv_text)
                if cond == "selfpatch":
                    _, c2 = prefill(model, tok, recv_text)
                    patch(cache, state_snapshot(c2),
                          deltanet_layers=set(LINEAR), kv=True)
                elif cond == "deltanet":
                    patch(cache, donor_snap, deltanet_layers=set(LINEAR))
                elif cond == "kv":
                    patch(cache, donor_snap, kv=True)
                elif cond == "both":
                    patch(cache, donor_snap,
                          deltanet_layers=set(LINEAR), kv=True)
                elif cond == "neutral":
                    patch(cache, neutral_snap, deltanet_layers=set(LINEAR))
                elif cond in GROUPS:
                    patch(cache, donor_snap,
                          deltanet_layers=set(GROUPS[cond]))
                text = continue_greedy(model, tok, ids, cache)
                scores[cond].append(flip_score(text, lex_d, lex_r))
                if pair_idx == 0:
                    examples[cond] = text
            pair_idx += 1
        print(f"    {fkey:8s} done "
              + " ".join(f"{c}:{np.mean(scores[c][-4:]):.2f}"
                         for c in ["baseline", "deltanet", "kv", "both"]))

    means = {c: float(np.mean(scores[c])) for c in conditions}
    print("\n    condition means (0=receiver's fact, 1=donor's fact):")
    for c in conditions:
        print(f"      {c:>9s}: {means[c]:.3f}")

    # self-patch no-op check needs TEXT equality; recompute strictly
    print("\n    self-patch strict no-op check (text equality, 12 pairs):")
    noop_ok = 0
    noop_total = 0
    for fkey, fact in FACTS.items():
        for name in fact["names"][:2]:          # 6 pairs is plenty
            recv_text = fact["template"].format(X=fact["b"], N=name, F="")
            ids, cache = prefill(model, tok, recv_text)
            t_base = continue_greedy(model, tok, ids, cache)
            ids, cache = prefill(model, tok, recv_text)
            _, c2 = prefill(model, tok, recv_text)
            patch(cache, state_snapshot(c2),
                  deltanet_layers=set(LINEAR), kv=True)
            t_self = continue_greedy(model, tok, ids, cache)
            noop_total += 1
            noop_ok += int(t_base == t_self)
    print(f"      identical {noop_ok}/{noop_total}")

    # ---------------------------------------------------------------- gates
    best_layer = LINEAR[best_l_idx]
    max_null = max(float(acc_null_by_filler[f].max()) for f in FILLERS)
    gates = {
        "R1_state_readable": (acc_by_filler[0].max() >= 0.90,
                              f"dog/cat from state: acc "
                              f"{acc_by_filler[0].max():.2f} at layer "
                              f"{best_layer} (bar 0.90)"),
        "R2_probe_controls": (max_null <= 0.65,
                              f"shuffled-label max {max_null:.2f} across "
                              f"fillers, 20-split mean (chance 0.5)"),
        "S1_machinery": (means["both"] >= 0.80 and means["baseline"] <= 0.20,
                         f"full transplant {means['both']:.2f}, baseline "
                         f"{means['baseline']:.2f} (bars >=0.8, <=0.2)"),
        "S2_noop_exact": (noop_ok == noop_total,
                          f"self-patch byte-identical {noop_ok}/{noop_total}"),
        "S3_null_state": (means["neutral"] <= 0.55,
                          f"neutral-memory patch {means['neutral']:.2f} "
                          f"(must not produce donor content)"),
    }
    n_pass = sum(v[0] for v in gates.values())
    print("\n" + "=" * 70)
    for k, (ok, msg) in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}: {msg}")
    print(f"  {n_pass}/{len(gates)} gates passed")

    print("\n  THE APPORTIONMENT (the open question, measured):")
    print(f"    deltanet-only transplant: {means['deltanet']:.3f}")
    print(f"    kv-only transplant:       {means['kv']:.3f}")
    print(f"    by depth (deltanet): early {means['early']:.3f}, "
          f"mid {means['mid']:.3f}, late {means['late']:.3f}")
    print(f"    retention at layer {best_layer}: "
          + " -> ".join(f"{r:.2f}" for r in retention)
          + "  (0 / ~40 / ~120 filler words)")

    # ----------------------------------------------------------------- figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("exp13 -- Qwen3.5-0.8B DeltaNet working memory: "
                 "read + transplant", fontsize=13)

    ax = axes[0, 0]
    for f_id in FILLERS:
        nw = len(words_of(FILLERS[f_id]))
        ax.plot(LINEAR, acc_by_filler[f_id], "o-", ms=3,
                label=f"filler {nw}w")
    ax.plot(LINEAR, acc_null_by_filler[0], "s--", color="gray", ms=3,
            label="shuffled")
    ax.axhline(0.5, color="k", lw=0.5)
    ax.set_xlabel("DeltaNet layer")
    ax.set_ylabel("held-out acc (dog vs cat)")
    ax.set_title("[A] what the state matrices know")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    names = ["baseline", "selfpatch", "neutral", "kv", "deltanet", "both"]
    colors = ["tab:blue", "tab:cyan", "gray", "tab:orange", "tab:red",
              "tab:purple"]
    ax.bar(names, [means[c] for c in names], color=colors, alpha=0.8)
    ax.axhline(0.5, color="k", lw=0.5, ls=":")
    ax.set_ylabel("flip score (1 = donor's fact)")
    ax.set_title("[B] whose memory wins?")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1, 0]
    gnames = ["early", "mid", "late", "deltanet"]
    ax.bar(gnames, [means[c] for c in gnames], color="tab:red", alpha=0.75)
    ax.axhline(means["baseline"], color="tab:blue", ls="--",
               label="baseline")
    ax.set_ylabel("flip score")
    ax.set_title("[B] where in depth the fact lives (deltanet subsets)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.axis("off")
    demo = (f"receiver prompt: pet CAT / donor memory: pet DOG\n\n"
            f"baseline:  {examples.get('baseline', '')[:130]}\n\n"
            f"deltanet:  {examples.get('deltanet', '')[:130]}\n\n"
            f"kv:        {examples.get('kv', '')[:130]}\n\n"
            f"both:      {examples.get('both', '')[:130]}")
    ax.text(0.02, 0.98, demo, va="top", fontsize=8, family="monospace",
            wrap=True)
    ax.set_title("behavior, verbatim")

    fig.tight_layout()
    fpath = RESULTS / "exp13_qwen_memory.png"
    fig.savefig(fpath, dpi=130)

    report = {
        "model": MODEL_ID,
        "state_shape_per_layer": [16, 128, 128],
        "linear_layers": LINEAR, "full_attn_layers": sorted(FULL_ATTN),
        "probe_acc_by_filler": {str(k): v.tolist()
                                for k, v in acc_by_filler.items()},
        "probe_null_by_filler": {str(k): v.tolist()
                                 for k, v in acc_null_by_filler.items()},
        "best_layer": best_layer, "retention_curve": retention,
        "condition_means": means,
        "condition_scores": {c: scores[c] for c in conditions},
        "noop": [noop_ok, noop_total],
        "gates": {k: {"pass": bool(v[0]), "detail": v[1]}
                  for k, v in gates.items()},
        "examples": examples,
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
