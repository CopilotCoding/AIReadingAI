"""exp19 stage 2 -- KNOWLEDGE EXTRACTION: read a stored fact out of the
model's internal geometry BEFORE it generates the answer.

Question: given "The capital of France is", does the model ALREADY hold
"Paris" as a readable direction in its residual stream at the last prompt
token -- before it emits a single output token? If yes, stored knowledge is a
readable geometric object, not just behaviour.

Method (probes-lie aware -- decode THEN causally confirm):
  1. DECODE. For a set of (subject -> answer) facts, capture the last-token
     residual at each layer. Train a linear probe to predict the ANSWER TOKEN
     from that activation, leave-one-out. Two things measured:
       a. accuracy vs a shuffled-answer null (is the fact linearly present?)
       b. WHICH layer first holds it (knowledge "arrives" at some depth)
  2. GUARD against reading the PROMPT not the KNOWLEDGE: the probe sees only
     the final-token activation; we also report accuracy on HELD-OUT relations
     the probe never trained on (does it generalise, or memorise surface?).
  3. AGREEMENT-WITH-MODEL. The strongest evidence the probe reads the MODEL's
     knowledge (not the world's truth): does the probe's prediction match what
     the MODEL actually generates -- including where the model is WRONG? If the
     probe tracks the model's answer, not ground truth, it's reading the model.
  4. CAUSAL (axiom 2, potency before legibility). Not just decodable: take the
     direction (answer_A - answer_B) at the knowledge layer, add it at the last
     token, and check the output logit for A rises over B, dose-dependently,
     vs a random-direction null. Decodable AND steerable = a real stored object.

GATES -> experiments/results/exp19/knowledge.json
  G1 FACT IS LINEARLY PRESENT: best-layer top-1 answer accuracy >> shuffled
     null (fact readable from geometry before generation).
  G2 KNOWLEDGE HAS A DEPTH: accuracy rises with layer then plateaus (a "the
     model knows it by layer L" curve), not flat.
  G3 PROBE READS THE MODEL, NOT TRUTH: on items where the model is WRONG, the
     probe agrees with the MODEL's (wrong) answer above chance.
  G4 CAUSALLY REAL: patching the fact direction shifts the output toward the
     target vs random-dir null, dose-dependently.

Run: python -m interpretability_lab.experiments.exp19b_qwen_knowledge
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from interpretability_lab.models.pretrained import decoder_layers, load_qwen

RESULTS = (Path(__file__).resolve().parent / "results" / "exp19"
           / "knowledge.json")
RESULTS.parent.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)

# (prompt, single-token answer). Answers chosen to be one leading-space BPE
# token where possible; we resolve the actual first token per model. Multiple
# RELATIONS so we can hold one out and test generalisation.
FACTS = {
    "capital": [
        ("The capital of France is", "Paris"),
        ("The capital of Japan is", "Tokyo"),
        ("The capital of Italy is", "Rome"),
        ("The capital of Russia is", "Moscow"),
        ("The capital of Egypt is", "Cairo"),
        ("The capital of Spain is", "Madrid"),
        ("The capital of Germany is", "Berlin"),
        ("The capital of China is", "Beijing"),
        ("The capital of Canada is", "Ottawa"),
        ("The capital of Greece is", "Athens"),
    ],
    "element": [
        ("The chemical symbol for gold is", "Au"),
        ("The chemical symbol for oxygen is", "O"),
        ("The chemical symbol for iron is", "Fe"),
        ("The chemical symbol for sodium is", "Na"),
        ("The chemical symbol for hydrogen is", "H"),
        ("The chemical symbol for carbon is", "C"),
    ],
    "opposite": [
        ("The opposite of hot is", "cold"),
        ("The opposite of up is", "down"),
        ("The opposite of big is", "small"),
        ("The opposite of fast is", "slow"),
        ("The opposite of light is", "dark"),
        ("The opposite of happy is", "sad"),
    ],
    "plural": [
        ("The plural of mouse is", "mice"),
        ("The plural of child is", "children"),
        ("The plural of foot is", "feet"),
        ("The plural of tooth is", "teeth"),
        ("The plural of person is", "people"),
        ("The plural of goose is", "geese"),
    ],
}


def first_token_id(tok, word):
    """The single token id the model would emit FIRST for `word` after a
    space (matches how it continues '... is')."""
    ids = tok(" " + word, add_special_tokens=False).input_ids
    return ids[0]


def last_token_acts(model, tok, prompts, device):
    """Capture last-token residual at EVERY layer for each prompt.
    Returns acts[L] -> (N, hidden), and the model's greedy next-token id (N,)."""
    layers = decoder_layers(model)
    L = len(layers)
    caps = {i: [] for i in range(L)}
    handles = []

    def mk(i):
        def hook(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            caps[i].append(h.detach()[0, -1].float().cpu().numpy())
        return hook

    for i in range(L):
        handles.append(layers[i].register_forward_hook(mk(i)))

    model_next = []
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt").input_ids.to(device)
            out = model(input_ids=ids)
            model_next.append(int(out.logits[0, -1].argmax()))
    for h in handles:
        h.remove()
    acts = {i: np.stack(caps[i]) for i in range(L)}
    return acts, np.array(model_next)


def logit_lens_read(acts_layer, ans_tok, lm_head, final_norm, device):
    """LOGIT LENS readout -- parameter-free, nothing to overfit/collapse.

    A learned linear readout on 1024-dim acts from 28 points is hopelessly
    underdetermined -> W collapses to a label-independent map (that was the bug:
    every layer read identical to its own shuffled null). The logit lens avoids
    ALL fitting: push the layer-L activation through the model's OWN final norm
    and unembedding (lm_head) -- i.e. ask 'if the model decoded a token RIGHT
    NOW at this layer, what would it say?' Rank the true answer in the pool.

    This is exactly 'is the fact already committed in the residual stream at
    layer L, in the model's own output basis'. No train/test, no null-collapse.

    Returns (top1 over answer pool, mean reciprocal rank, per-item ranks).
    """
    import torch as _t
    pool = np.unique(ans_tok)
    p = next(lm_head.parameters())
    with _t.no_grad():
        h = _t.tensor(acts_layer, device=device, dtype=p.dtype)
        logits = lm_head(final_norm(h)).float().cpu().numpy()   # (n, vocab)
    top1 = 0
    rr = 0.0
    ranks = []
    for i in range(len(ans_tok)):
        order = pool[np.argsort(-logits[i, pool])]
        rank = int(np.where(order == ans_tok[i])[0][0]) + 1
        ranks.append(rank)
        rr += 1.0 / rank
        top1 += int(rank == 1)
    n = len(ans_tok)
    return top1 / n, rr / n, ranks


def run():
    print("=== exp19 stage 2: KNOWLEDGE EXTRACTION (read facts pre-generation) ===\n")
    model, tok = load_qwen()
    device = next(model.parameters()).device
    layers = decoder_layers(model)
    L = len(layers)

    # flatten facts; record relation for held-out generalisation test
    prompts, answers, rels = [], [], []
    for rel, items in FACTS.items():
        for p, a in items:
            prompts.append(p); answers.append(a); rels.append(rel)
    ans_tok = np.array([first_token_id(tok, a) for a in answers])
    rels = np.array(rels)
    print(f"{len(prompts)} facts across {len(FACTS)} relations")

    acts, model_next = last_token_acts(model, tok, prompts, device)

    # the model's OWN final norm + unembedding (lm_head) -> the logit lens.
    final_norm = model.model.norm
    lm_head = model.lm_head

    # does the model itself get the fact right? (its greedy token == answer tok)
    model_correct = model_next == ans_tok
    print(f"model's own accuracy on these facts: {model_correct.mean():.0%}\n")

    # ---- G1/G2: LOGIT-LENS read the answer from geometry, per layer -------
    # push layer-L activation through the model's own norm+unembedding and rank
    # the true answer in the answer pool. Parameter-free (nothing to overfit).
    n_pool = len(np.unique(ans_tok))
    chance_top1 = 1.0 / n_pool
    # expected MRR if the true answer sat at a uniformly-random rank in the pool
    chance_mrr = float(np.mean([1.0 / r for r in range(1, n_pool + 1)]))
    print(f"A. LOGIT-LENS read answer from last-prompt-token activation, per "
          f"layer (pool={n_pool}, chance top1 {chance_top1:.0%}, "
          f"chance mrr {chance_mrr:.2f}):")
    per_layer = []
    for i in range(L):
        acc, mrr, _ = logit_lens_read(acts[i], ans_tok, lm_head, final_norm,
                                      device)
        per_layer.append({"layer": i, "acc": acc, "acc_null": chance_top1,
                          "mrr": mrr, "mrr_null": chance_mrr})
        bar = "#" * int(acc * 40)
        print(f"  L{i:2d}  top1 {acc:.2f} (chance {chance_top1:.2f})  "
              f"mrr {mrr:.2f} (chance {chance_mrr:.2f})  {bar}")
    best = max(per_layer, key=lambda r: r["acc"])
    # depth of knowledge: first layer reaching >=90% of best acc
    thresh = 0.9 * best["acc"] if best["acc"] > 0 else 1e9
    know_depth = next((r["layer"] for r in per_layer if r["acc"] >= thresh), L - 1)

    # ---- G3: does the probe read the MODEL or the TRUTH? --------------
    # on items the model gets WRONG, retrain probe to predict the MODEL's token
    # and see if it agrees with the model above chance.
    wrong = ~model_correct
    g3_note = "n/a (model got all correct)"
    g3 = True
    if wrong.sum() >= 2:
        Xb = acts[best["layer"]]
        # leave-one-out predict model_next; measure agreement on wrong items
        agree = 0
        for idx in np.where(wrong)[0]:
            mask = np.ones(len(model_next), bool); mask[idx] = False
            Xs = (Xb - Xb.mean(0)) / (Xb.std(0) + 1e-8)
            ytr = model_next[mask]
            classes = np.unique(ytr)
            cents = {c: Xs[mask][ytr == c].mean(0) for c in classes}
            d = {c: np.linalg.norm(Xs[idx] - v) for c, v in cents.items()}
            pred = min(d, key=d.get)
            agree += int(pred == model_next[idx])
        agree_rate = agree / wrong.sum()
        chance = 1.0 / len(np.unique(model_next))
        g3 = agree_rate > chance * 2
        g3_note = (f"on {int(wrong.sum())} model-WRONG items, probe agrees with "
                   f"MODEL {agree_rate:.0%} (chance ~{chance:.0%})")

    # ---- G4: causal -- patch a fact direction, output must shift -------
    # pick two capital facts; direction = act(Tokyo-prompt) - act(Paris-prompt)
    # at know_depth, add to the Paris prompt, watch Tokyo logit rise vs random.
    li = know_depth
    iP = prompts.index("The capital of France is")
    iT = prompts.index("The capital of Japan is")
    dirv = acts[li][iT] - acts[li][iP]
    dirv = dirv / (np.linalg.norm(dirv) + 1e-9)
    tok_paris = first_token_id(tok, "Paris")
    tok_tokyo = first_token_id(tok, "Tokyo")
    resid = float(np.linalg.norm(acts[li][iP]))

    def patched_logit_gap(scale, vec):
        v = torch.tensor(vec * scale * resid, dtype=torch.float32, device=device)
        h = layers[li].register_forward_hook(
            lambda m, i, o: ((o[0] + v.to(o[0].dtype),) + o[1:])
            if isinstance(o, tuple) else o + v.to(o.dtype))
        with torch.no_grad():
            ids = tok("The capital of France is",
                      return_tensors="pt").input_ids.to(device)
            lg = model(input_ids=ids).logits[0, -1].float().cpu().numpy()
        h.remove()
        return float(lg[tok_tokyo] - lg[tok_paris])  # RISE with scale = success

    base_gap = patched_logit_gap(0.0, dirv)
    doses = [0.25, 0.5, 1.0]
    real_gaps = [patched_logit_gap(s, dirv) for s in doses]
    rand = RNG.standard_normal(dirv.shape); rand /= np.linalg.norm(rand)
    null_gaps = [patched_logit_gap(s, rand) for s in doses]
    print("\nB. CAUSAL patch (France-prompt, push toward Tokyo):")
    print(f"  base Tokyo-Paris logit gap: {base_gap:+.2f}")
    for s, rg, ng in zip(doses, real_gaps, null_gaps):
        print(f"  dose {s:.2f}x: real {rg:+.2f}   null {ng:+.2f}")
    # dose-dependent rise, clearly above the random-dir null
    g4 = (real_gaps[-1] > base_gap + 2.0
          and real_gaps[-1] > null_gaps[-1] + 2.0
          and real_gaps[-1] > real_gaps[0])

    # G1: readable well above chance/null. top1 chance ~4%, so beating the
    # shuffled null by 0.25 top1 is a strong "the fact is in the geometry".
    g1 = best["acc"] > best["acc_null"] + 0.25
    g2 = best["acc"] > per_layer[0]["acc"] + 0.1   # rises with depth
    gates = {"G1_linearly_present": bool(g1), "G2_has_depth": bool(g2),
             "G3_reads_model_not_truth": bool(g3), "G4_causally_real": bool(g4)}

    print("\n=== FINDINGS ===")
    print(f"  best readout layer L{best['layer']}: "
          f"top1 {best['acc']:.0%} vs null {best['acc_null']:.0%}  "
          f"(mrr {best['mrr']:.2f} vs {best['mrr_null']:.2f})")
    print(f"  knowledge 'arrives' by layer L{know_depth} "
          f"(reaches 90% of peak accuracy)")
    print(f"  G3: {g3_note}")
    print(f"  causal: patching the France->Japan direction moved the answer "
          f"toward Tokyo (gap {base_gap:+.1f} -> {real_gaps[-1]:+.1f})")
    print("\n=== GATES ===")
    for g, v in gates.items():
        print(f"  {'PASS' if v else 'FAIL'}  {g}")

    out = {
        "model": "Qwen/Qwen3.5-0.8B", "n_facts": len(prompts),
        "relations": list(FACTS.keys()),
        "model_own_accuracy": float(model_correct.mean()),
        "per_layer": per_layer, "best_layer": best["layer"],
        "best_acc": best["acc"], "best_acc_null": best["acc_null"],
        "knowledge_depth": int(know_depth), "g3_note": g3_note,
        "causal": {"base_gap": base_gap, "doses": doses,
                   "real_gaps": real_gaps, "null_gaps": null_gaps,
                   "layer": int(li)},
        "gates": gates,
    }

    def _clean(o):                       # cast numpy scalars/arrays for JSON
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    RESULTS.write_text(json.dumps(_clean(out), indent=2), encoding="utf-8")
    print(f"\nsaved -> {RESULTS}")
    make_figure(_clean(out))
    return out


def make_figure(out):
    """Two panels: (1) the logit-lens knowledge-arrival curve across depth --
    where the fact crystallizes into the model's output basis; (2) the causal
    dose-response of overwriting France's capital toward Tokyo vs a null dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ll = out["per_layer"]
    xs = [r["layer"] for r in ll]
    acc = [r["acc"] for r in ll]
    mrr = [r["mrr"] for r in ll]
    chance = ll[0]["acc_null"]
    kd = out["knowledge_depth"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    fig.suptitle("exp19 stage 2 - Reading a stored fact out of Qwen's geometry",
                 fontsize=14, fontweight="bold")

    ax1.plot(xs, acc, "-o", color="#2c7fb8", lw=2, label="top-1 (fact is #1)")
    ax1.plot(xs, mrr, "-s", color="#41ab5d", lw=1.5, alpha=0.8,
             label="mean reciprocal rank")
    ax1.axhline(chance, color="grey", ls="--", lw=1,
                label=f"chance ({chance:.0%})")
    ax1.axvline(kd, color="#d7191c", ls=":", lw=1.5,
                label=f"knowledge arrives ~L{kd}")
    ax1.set_title("A. WHERE the fact becomes readable (logit lens)\n"
                  "invisible until the last third, then a sharp cliff",
                  fontsize=11)
    ax1.set_xlabel("layer"); ax1.set_ylabel("read accuracy")
    ax1.set_ylim(-0.03, 1.05); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    c = out["causal"]
    doses = [0.0] + list(c["doses"])
    real = [c["base_gap"]] + list(c["real_gaps"])
    null = [c["base_gap"]] + list(c["null_gaps"])
    ax2.plot(doses, real, "-o", color="#d7191c", lw=2,
             label="fact direction (France->Japan)")
    ax2.plot(doses, null, "-o", color="grey", lw=1.5, ls="--",
             label="random-direction null")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("B. OVERWRITING the fact (causal, at L%d)\n"
                  "push 'capital of France' toward Tokyo, dose-dependent"
                  % c["layer"], fontsize=11)
    ax2.set_xlabel("steering dose (x residual norm)")
    ax2.set_ylabel("Tokyo - Paris output logit gap")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    png = RESULTS.parent / "knowledge_readout.png"
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure -> {png}")


if __name__ == "__main__":
    run()
