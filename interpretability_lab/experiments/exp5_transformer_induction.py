"""EXPERIMENT 5 -- Attention circuits in a tiny transformer (~40K params).

Task: a random-length unique-token prefix repeated at a VARIABLE offset
(positional shortcuts impossible). Ground-truth data rule: match-&-copy.

FIRST RUN REFUTED THE TEXTBOOK HYPOTHESIS. The classic induction circuit
(L0 previous-token head -> L1 induction head attending to j0+1) does NOT
exist in this model: prev-token scores ~0.10 (uniform), induction scores
~0.02 -- yet accuracy is 0.999 and the blindly-discovered circuit shows a
clean double dissociation. Diagnostics revealed the actual mechanism:

  WINDOWED MATCH-&-COPY
    L0 heads (all four, distributed) write a fuzzy summary of tokens
    ~6-8 back into each position; L1 heads content-match against that
    SHIFTED summary, attending ~j0+[4,9] (mass 0.90, baseline 0.25) --
    the retrieved window contains the successor token.

Discriminating prediction (robust across training instances): a window
matcher handles repeated BLOCKS but not single repeated tokens, so
block-repeat OOD accuracy is high while single-token accidental-repeat
agreement is low. Textbook induction predicts both high. (The exact ramp
shape within a block varies between training instances -- CUDA training is
not deterministic -- so the gate tests the block-vs-token CONTRAST, not the
ramp.) The model is loaded from the corpus when present so analysis refers
to a fixed specimen.

Gates below test the MEASURED mechanism; the textbook refutation stays on
the record.

Run:  python -m interpretability_lab.experiments.exp5_transformer_induction
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from interpretability_lab.corpus.store import save_specimen
from interpretability_lab.hooks.recorder import ActivationRecorder
from interpretability_lab.models.tiny import param_count
from interpretability_lab.models.transformer import TinyAttnTransformer

RESULTS = Path(__file__).parent / "results" / "exp5"
VOCAB, D_MODEL, N_HEADS, N_LAYERS, T = 20, 64, 4, 2, 32
H_MIN, H_MAX = 10, 16
WIN_LO, WIN_HI = 4, 9      # measured shifted-match window, k in j0+[4,9]


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return {"gate": name, "passed": bool(ok), "detail": detail}


def make_batch(n, gen, device="cpu"):
    seqs = torch.randint(0, VOCAB, (n, T), generator=gen)
    halves = torch.randint(H_MIN, H_MAX + 1, (n,), generator=gen)
    for i in range(n):
        h = int(halves[i])
        prefix = torch.randperm(VOCAB, generator=gen)[:h]
        seqs[i, :h] = prefix
        seqs[i, h:2 * h] = prefix
    return seqs.to(device), halves


def induction_accuracy(model, seqs, halves, hooks=None):
    handles = []
    if hooks:
        by_layer = {}
        for l, h in hooks:
            by_layer.setdefault(l, []).append(h)
        for l, hs in by_layer.items():
            def fn(_m, _i, out, hs=hs):
                out = out.clone()
                out[:, hs] = 0.0
                return out
            handles.append(model.blocks[l].head_sink.register_forward_hook(fn))
    try:
        with torch.no_grad():
            logits = model(seqs)
    finally:
        for h in handles:
            h.remove()
    pred = logits.argmax(-1)
    correct = total = 0
    for i in range(len(seqs)):
        h = int(halves[i])
        for t in range(h, 2 * h - 1):
            total += 1
            correct += int(pred[i, t] == seqs[i, t + 1])
    return correct / total


def attention_patterns(model, seqs, ablate_l0=None):
    handles = []
    if ablate_l0:
        def fn(_m, _i, out):
            out = out.clone()
            out[:, ablate_l0] = 0.0
            return out
        handles.append(model.blocks[0].head_sink.register_forward_hook(fn))
    rec = ActivationRecorder(model, names=[f"blocks.{i}.pattern_sink"
                                           for i in range(N_LAYERS)])
    try:
        with rec.capture():
            with torch.no_grad():
                model(seqs)
    finally:
        for h in handles:
            h.remove()
    return [rec.traces[f"blocks.{i}.pattern_sink"] for i in range(N_LAYERS)]


def textbook_scores(patterns, seqs, halves):
    """Classic circuit statistics: prev-token mass and mass on j0+1."""
    prev = np.zeros((N_LAYERS, N_HEADS))
    ind = np.zeros((N_LAYERS, N_HEADS))
    for l in range(N_LAYERS):
        A = patterns[l].numpy()
        prev[l] = [A[:, h, np.arange(1, T), np.arange(T - 1)].mean()
                   for h in range(N_HEADS)]
        masses = [[] for _ in range(N_HEADS)]
        for i in range(len(seqs)):
            hh = int(halves[i])
            for t in range(hh, 2 * hh - 1):
                for h in range(N_HEADS):
                    masses[h].append(A[i, h, t, t - hh + 1])
        ind[l] = [float(np.mean(m)) for m in masses]
    return prev, ind


def window_scores(patterns, seqs, halves, layer=1):
    """Measured-mechanism statistic: L-layer head mass in k = j0+[WIN_LO, WIN_HI]."""
    A = patterns[layer].numpy()
    out = []
    for h in range(N_HEADS):
        vals = []
        for i in range(len(seqs)):
            hh = int(halves[i])
            for t in range(hh, 2 * hh - 1):
                j0 = t - hh
                ks = [k for k in range(j0 + WIN_LO, j0 + WIN_HI + 1) if 0 <= k <= t]
                vals.append(sum(A[i, h, t, k] for k in ks))
        out.append(float(np.mean(vals)))
    return out


def match_and_copy(seq):
    out = {}
    arr = seq.tolist()
    for t in range(1, len(arr) - 1):
        occ = [j for j in range(t) if arr[j] == arr[t]]
        if len(occ) == 1 and occ[0] + 1 < len(arr):
            out[t] = arr[occ[0] + 1]
    return out


def separated_block_curve(model, n=512, L=12, seed=123):
    """OOD: repeated block separated by a random gap. Returns accuracy by
    position inside the second block -- the mechanism-discriminating curve."""
    gen = torch.Generator().manual_seed(seed)
    seqs = torch.randint(0, VOCAB, (n, T), generator=gen)
    starts = []
    for i in range(n):
        gap = int(torch.randint(2, 8, (1,), generator=gen))
        block = torch.randperm(VOCAB, generator=gen)[:L]
        seqs[i, :L] = block
        seqs[i, L + gap:L + gap + L] = block
        starts.append(L + gap)
    with torch.no_grad():
        pred = model(seqs).argmax(-1)
    curve = {}
    for d in range(1, L):
        c = tot = 0
        for i in range(n):
            t = starts[i] + d - 1
            if t + 1 < T:
                tot += 1
                c += int(pred[i, t] == seqs[i, t + 1])
        curve[d] = c / tot
    return curve


def main():
    print("=" * 70)
    print("EXPERIMENT 5: attention circuit in a tiny transformer")
    print("=" * 70)
    RESULTS.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- train, or load the fixed specimen if one exists
    spec_weights = (Path(__file__).parent.parent / "corpus" / "data" / "exp5"
                    / "exp5_5a_induction_seed0" / "weights.pt")
    model = TinyAttnTransformer(VOCAB, D_MODEL, N_HEADS, N_LAYERS, T)
    if spec_weights.exists():
        model.load_state_dict(torch.load(spec_weights, weights_only=True))
        model.eval()
        print("\n  loaded fixed specimen from corpus (delete it to retrain)")
    else:
        torch.manual_seed(0)
        gen = torch.Generator().manual_seed(1)
        model = model.to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        for step in range(6000):
            seqs, _ = make_batch(128, gen, dev)
            logits = model(seqs)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB),
                                   seqs[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model = model.cpu().eval()

    eval_gen = torch.Generator().manual_seed(9)
    seqs, halves = make_batch(512, eval_gen)
    acc = induction_accuracy(model, seqs, halves)
    print(f"\n  trained: {param_count(model)} params ({N_LAYERS} layers x "
          f"{N_HEADS} heads, attention-only), device {dev}")
    print(f"  task accuracy {acc:.3f} (chance {1 / VOCAB:.3f}; repeat offset "
          f"varies {H_MIN}-{H_MAX}, positional shortcut impossible)")
    gates = [gate("model solves the task via content, not position",
                  acc > 0.9, f"accuracy {acc:.3f}")]

    # ---- 1. blind circuit discovery by causal ablation
    impact = np.zeros((N_LAYERS, N_HEADS))
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            impact[l, h] = acc - induction_accuracy(model, seqs, halves,
                                                    hooks=[(l, h)])
    circuit = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS)
               if impact[l, h] > 0.05]
    others = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS)
              if (l, h) not in circuit]
    print("  per-head ablation impact:")
    for l in range(N_LAYERS):
        print(f"    layer {l}: " + "  ".join(
            f"h{h}:{impact[l, h]:+.3f}" for h in range(N_HEADS)))
    print(f"  discovered circuit: {[f'L{l}H{h}' for l, h in circuit]}")
    acc_keep = induction_accuracy(model, seqs, halves, hooks=others)
    acc_abl = induction_accuracy(model, seqs, halves, hooks=circuit)
    gates.append(gate("circuit is sufficient (all other heads ablated)",
                      acc_keep > acc - 0.1,
                      f"keep-only accuracy {acc_keep:.3f} vs full {acc:.3f}"))
    gates.append(gate("circuit is necessary (circuit heads ablated)",
                      acc_abl < 0.3, f"ablate-circuit accuracy {acc_abl:.3f}"))

    # ---- 2. characterize: textbook hypothesis first (refuted), then measured
    patterns = attention_patterns(model, seqs)
    prev, ind = textbook_scores(patterns, seqs, halves)
    print("  textbook induction-circuit statistics (prev-token / j0+1 mass):")
    for l in range(N_LAYERS):
        print(f"    layer {l}: " + "  ".join(
            f"h{h}:{prev[l, h]:.2f}/{ind[l, h]:.2f}" for h in range(N_HEADS)))
    textbook_present = bool(prev.max() > 0.5 or ind.max() > 0.5)
    print(f"  -> textbook circuit {'PRESENT' if textbook_present else 'ABSENT'}: "
          f"hypothesis {'supported' if textbook_present else 'REFUTED'} "
          f"(kept on record)")

    win = window_scores(patterns, seqs, halves)
    print(f"  measured mechanism -- L1 mass in shifted window j0+[{WIN_LO},{WIN_HI}] "
          f"(uniform baseline ~0.25):")
    print("    " + "  ".join(f"H{h}:{win[h]:.2f}" for h in range(N_HEADS)))
    c1 = [h for l, h in circuit if l == 1]
    c0 = [h for l, h in circuit if l == 0]
    win_circ = [win[h] for h in c1]
    gates.append(gate("L1 circuit heads match the SHIFTED window (not j0+1)",
                      len(win_circ) > 0 and max(win_circ) > 0.6,
                      f"window mass {[f'{w:.2f}' for w in win_circ]}, "
                      f"j0+1 mass {[f'{ind[1, h]:.2f}' for h in c1]}"))

    # ---- 3. composition, causally: L0 writes the shifted summary the L1
    #         match needs. Ablate L0 circuit heads -> window concentration
    #         must collapse and accuracy must die.
    patterns_abl = attention_patterns(model, seqs, ablate_l0=c0)
    win_abl = window_scores(patterns_abl, seqs, halves)
    best = int(np.argmax(win))
    acc_l0 = induction_accuracy(model, seqs, halves, hooks=[(0, h) for h in c0])
    print(f"  L0 circuit ablated: L1H{best} window mass {win[best]:.2f} -> "
          f"{win_abl[best]:.2f}; accuracy {acc:.3f} -> {acc_l0:.3f}")
    gates.append(gate("ablating L0 heads collapses the L1 match (composition)",
                      (win[best] - win_abl[best]) > 0.25 and acc_l0 < 0.15,
                      f"window {win[best]:.2f}->{win_abl[best]:.2f}, "
                      f"acc {acc_l0:.3f}"))

    # ---- 4. algorithm-level claims
    agree_in = tot_in = 0
    with torch.no_grad():
        pred_in = model(seqs).argmax(-1)
    for i in range(len(seqs)):
        alg = match_and_copy(seqs[i])
        hh = int(halves[i])
        for t, p in alg.items():
            if hh <= t < 2 * hh - 1:
                tot_in += 1
                agree_in += int(pred_in[i, t] == p)
    r_in = agree_in / tot_in
    gates.append(gate("extracted algorithm reproduces the model in-distribution",
                      r_in > 0.9, f"agreement {r_in:.3f}"))

    # single-token OOD: windowed mechanism predicts LOW, textbook predicts HIGH
    ood_gen = torch.Generator().manual_seed(77)
    ood = torch.randint(0, VOCAB, (512, T), generator=ood_gen)
    with torch.no_grad():
        pred_ood = model(ood).argmax(-1)
    agree_ood = tot_ood = 0
    for i in range(len(ood)):
        for t, p in match_and_copy(ood[i]).items():
            tot_ood += 1
            agree_ood += int(pred_ood[i, t] == p)
    r_ood = agree_ood / tot_ood
    print(f"  single-token accidental repeats (OOD): agreement {r_ood:.3f} -- "
          f"windowed mechanism predicts low, textbook predicts high")

    # separated-block OOD: the robust discriminator is the block-vs-token
    # CONTRAST (ramp shape varies between training instances)
    curve = separated_block_curve(model)
    early = float(np.mean([curve[d] for d in (1, 2)]))
    late = float(np.mean([curve[d] for d in (5, 6, 7)]))
    block_mid = float(np.mean([curve[d] for d in range(2, 9)]))
    print(f"  separated-block OOD: accuracy {block_mid:.3f} (pos 2-8; "
          f"pos1-2 {early:.3f}, pos5-7 {late:.3f})")
    print(f"  contrast: block repeats {block_mid:.2f} vs single tokens "
          f"{r_ood:.2f} -- textbook induction predicts BOTH high")
    gates.append(gate("OOD contrast: window matcher, not token inductor",
                      block_mid > 0.6 and r_ood < 0.5
                      and (block_mid - r_ood) > 0.3,
                      f"block {block_mid:.2f} vs single-token {r_ood:.2f}"))

    # ---- 5. negative control
    torch.manual_seed(999)
    rand_model = TinyAttnTransformer(VOCAB, D_MODEL, N_HEADS, N_LAYERS, T).eval()
    rand_acc = induction_accuracy(rand_model, seqs, halves)
    rand_win = window_scores(attention_patterns(rand_model, seqs), seqs, halves)
    print(f"  negative control (untrained): accuracy {rand_acc:.3f}, max window "
          f"mass {max(rand_win):.2f} -> reader "
          f"{'refuses' if max(rand_win) < 0.35 else 'CLAIMED A CIRCUIT'}")
    gates.append(gate("reader refuses on an untrained transformer",
                      max(rand_win) < 0.35 and rand_acc < 0.15,
                      f"max window mass {max(rand_win):.2f}, acc {rand_acc:.3f}"))

    passed = all(g["passed"] for g in gates)
    save_specimen(model, experiment="exp5", task="5a_induction", seed=0,
                  ground_truth="match & copy: predict token after previous "
                               "occurrence of current token",
                  arch={"type": "TinyAttnTransformer", "vocab": VOCAB,
                        "d_model": D_MODEL, "n_heads": N_HEADS,
                        "n_layers": N_LAYERS, "seq_len": T},
                  recovered=f"WINDOWED match-&-copy via "
                            f"{[f'L{l}H{h}' for l, h in circuit]}: L0 writes "
                            f"~6-8-back context; L1 matches shifted window "
                            f"j0+[{WIN_LO},{WIN_HI}]",
                  passed=passed,
                  extra={"accuracy": acc, "impact": impact.tolist(),
                         "textbook_refuted": not textbook_present,
                         "window_scores": win,
                         "block_curve": curve,
                         "algorithm_agreement": {"in": r_in,
                                                 "single_token_ood": r_ood},
                         "heads_total": N_LAYERS * N_HEADS,
                         "circuit_heads": len(circuit)})

    # ---- figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax = axes[0, 0]
    im = ax.imshow(impact, cmap="Reds", aspect="auto")
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            ax.text(h, l, f"{impact[l, h]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(N_HEADS)); ax.set_xticklabels([f"H{h}" for h in range(N_HEADS)])
    ax.set_yticks(range(N_LAYERS)); ax.set_yticklabels([f"L{l}" for l in range(N_LAYERS)])
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"blind discovery: ablation impact\n"
                 f"circuit = {[f'L{l}H{h}' for l, h in circuit]}")

    ex = 0
    hh = int(halves[ex])
    A = patterns[1][ex, best].numpy()
    ax = axes[0, 1]
    ax.imshow(A, cmap="Blues", aspect="auto")
    for t in range(hh, 2 * hh - 1):
        ax.plot([t - hh + WIN_LO, t - hh + WIN_HI], [t, t], "r-", lw=0.8, alpha=0.6)
    ax.set_title(f"L1H{best}: attention vs shifted window j0+[{WIN_LO},{WIN_HI}]"
                 f" (red)\nNOT the textbook j0+1 stripe")
    ax.set_xlabel("attended position"); ax.set_ylabel("query position")

    ax = axes[0, 2]
    x = np.arange(N_HEADS)
    ax.bar(x - 0.2, win, 0.4, label="intact")
    ax.bar(x + 0.2, win_abl, 0.4, label="L0 circuit ablated")
    ax.axhline(0.25, color="k", ls=":", lw=0.8, label="uniform baseline")
    ax.set_xticks(x); ax.set_xticklabels([f"L1H{h}" for h in range(N_HEADS)])
    ax.legend(fontsize=8)
    ax.set_title("composition: L1 window match needs L0's shifted write")

    ax = axes[1, 0]
    labels = ["full", "keep\ncircuit", "ablate\ncircuit", "ablate\nL0 only", "untrained"]
    vals = [acc, acc_keep, acc_abl, acc_l0, rand_acc]
    ax.bar(labels, vals, color=["C0", "C2", "C3", "C1", "gray"])
    ax.axhline(1 / VOCAB, color="k", ls=":", lw=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_title("double dissociation + composition")

    ax = axes[1, 1]
    ds = sorted(curve)
    ax.plot(ds, [curve[d] for d in ds], "o-", label="observed")
    ax.axhline(1 / VOCAB, color="k", ls=":", lw=0.8, label="chance")
    ax.set_xlabel("position inside second block"); ax.set_ylabel("accuracy")
    ax.legend(fontsize=8)
    ax.axhline(r_ood, color="C3", ls="--", lw=1,
               label=f"single-token OOD ({r_ood:.2f})")
    ax.set_title("OOD contrast: block repeats work, single tokens don't\n"
                 "(textbook induction predicts both high)")

    ax = axes[1, 2]
    txt = (f"MEASURED MECHANISM (textbook REFUTED):\n\n"
           f"  WINDOWED MATCH-&-COPY\n"
           f"  L0 (all 4 heads, distributed):\n"
           f"    write summary of tokens ~6-8 back\n"
           f"  L1H{best} (+{[f'H{h}' for h in c1 if h != best]}):\n"
           f"    match current context to the SHIFTED\n"
           f"    copy -> attend j0+[{WIN_LO},{WIN_HI}], mass "
           f"{win[best]:.2f}\n"
           f"    retrieve successor from the window\n\n"
           f"  evidence: window mass {win[best]:.2f} vs j0+1 mass "
           f"{ind[1, best]:.2f}\n"
           f"    composition: {win[best]:.2f}->{win_abl[best]:.2f} w/o L0\n"
           f"    OOD ramp {early:.2f}->{late:.2f}; single-token {r_ood:.2f}\n\n"
           f"  {'ALL GATES PASSED' if passed else 'failures -- see report'}")
    ax.text(0.02, 0.97, txt, va="top", ha="left", fontsize=9, family="monospace",
            transform=ax.transAxes)
    ax.axis("off")
    fig.tight_layout()
    fig_path = RESULTS / "exp5_transformer_induction.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    report = {"experiment": "exp5_transformer_induction", "all_passed": passed,
              "params": param_count(model), "accuracy": acc,
              "impact": impact.tolist(),
              "circuit": [f"L{l}H{h}" for l, h in circuit],
              "acc_keep_only": acc_keep, "acc_ablate_circuit": acc_abl,
              "acc_ablate_l0_only": acc_l0,
              "textbook": {"prev": prev.tolist(), "ind": ind.tolist(),
                           "refuted": not textbook_present},
              "window_scores": {"intact": win, "l0_ablated": win_abl},
              "block_curve": curve,
              "algorithm_agreement": {"in": r_in, "single_token_ood": r_ood},
              "negative_control": {"acc": rand_acc, "max_window": max(rand_win)},
              "gates": gates}
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2, default=str),
                                         encoding="utf-8")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  mechanism: WINDOWED match-&-copy "
          f"({[f'L{l}H{h}' for l, h in circuit]}); textbook circuit refuted")
    print(f"  overall: {'ALL GATES PASSED' if passed else 'FAILURES PRESENT'}")
    print(f"  report:  {RESULTS / 'report.json'}")
    print(f"  figure:  {fig_path}")
    print("=" * 70)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
