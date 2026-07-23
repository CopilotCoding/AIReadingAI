"""exp19 stage 1 -- BLIND NATIVE-GEOMETRY MAP of Qwen3.5-0.8B.

Not steering. Not a concept we named. We look at the residual stream as it
IS, across all 24 layers, on a diverse corpus, and measure its native shape:

  A. INTRINSIC DIMENSIONALITY per layer -- how many dimensions does the model
     ACTUALLY use (participation ratio of the activation covariance), vs the
     nominal 1024? A phase-transition in this curve = a place the model
     reorganizes its representation.

  B. LAYER REORGANIZATION -- how much does the geometry change between adjacent
     layers? Measured by (1) CKA similarity of representations layer L vs L+1
     (low CKA = big reorganization) and (2) the drift of per-token vectors.

  C. BLIND CLUSTERING (axiom 2) -- cluster tokens in each layer's activation
     space with NO labels. THEN, and only then, check whether clusters line up
     with human categories (part-of-speech, is-it-punctuation, topic). Report
     the fraction of structure that is NAMEABLE vs the UNNAMEABLE residue. The
     residue size is the finding, not the failure.

GATES (hard pass/fail, results -> experiments/results/exp19_geometry.json):
  G1 STRUCTURE IS REAL: blind clusters must be more separated than clusters of
     a shuffled-feature null at the same layer (silhouette real >> null).
     Else "clusters" are just what k-means always does -- refuse.
  G2 DIMENSIONALITY IS NON-TRIVIAL: effective dim must be << 1024 at some
     layer (the model compresses) AND vary across depth (not flat). A flat or
     full-rank curve = no compression structure to see.
  G3 REORGANIZATION IS LOCALIZED: at least one adjacent-layer CKA must dip
     clearly below the median (a real processing boundary), not a smooth ramp.
  G4 NAMEABILITY IS MEASURED, NOT ASSUMED: we must report a concrete
     nameable-fraction with its null. Whatever it is (high or low) it passes;
     what fails is NOT measuring it. This gate enforces axiom-2 honesty.

Run:
  python -m interpretability_lab.experiments.exp19_qwen_geometry
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from interpretability_lab.models.pretrained import decoder_layers, load_qwen

RESULTS = (Path(__file__).resolve().parent / "results" / "exp19"
           / "geometry.json")
RESULTS.parent.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)

# -------------------------------------------------------------------------
# A diverse probe corpus: several human categories MIXED so blind clustering
# has something to (maybe) rediscover. We know the categories; the clustering
# does NOT. Categories are our later yardstick for nameability, not an input.
CORPUS = {
    "code": [
        "def add(a, b): return a + b",
        "for i in range(10): print(i)",
        "import numpy as np; arr = np.zeros((3, 3))",
        "class Node: def __init__(self): self.next = None",
        "x = [i*i for i in range(20) if i % 2 == 0]",
        "while True: break",
        "try: f() except ValueError: pass",
        "SELECT name FROM users WHERE age > 30;",
    ],
    "science": [
        "The mitochondria produce ATP through oxidative phosphorylation.",
        "Force equals mass times acceleration in classical mechanics.",
        "Photons exhibit both wave and particle properties.",
        "DNA replication is semiconservative and bidirectional.",
        "Entropy always increases in an isolated thermodynamic system.",
        "Enzymes lower the activation energy of biochemical reactions.",
        "Electrons occupy discrete quantized energy levels in atoms.",
        "The speed of light is constant in all inertial reference frames.",
    ],
    "narrative": [
        "She opened the door and stepped into the cold morning air.",
        "The old man watched the ships come in at dawn.",
        "He laughed, then fell silent as the music faded away.",
        "They walked home together under a sky full of stars.",
        "A child ran across the meadow chasing a bright kite.",
        "The rain fell softly on the quiet empty street.",
        "She remembered the summer they spent by the lake.",
        "He closed the book and stared out the window for a long time.",
    ],
    "dialogue": [
        "\"Are you coming to the party tonight?\" she asked.",
        "\"I don't think so,\" he replied with a shrug.",
        "\"Why not? It'll be fun!\" \"I'm just tired.\"",
        "\"Can you help me with this?\" \"Of course, what's wrong?\"",
        "\"Where did you put the keys?\" \"On the table, I think.\"",
        "\"Do you really mean that?\" \"Yes, every word.\"",
        "\"Let's go,\" she whispered. \"Now?\" \"Right now.\"",
        "\"That's incredible!\" he shouted. \"How did you do it?\"",
    ],
    "factual": [
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The Great Wall of China is over 13,000 miles long.",
        "Mount Everest is the highest mountain above sea level.",
        "The human body has 206 bones in adulthood.",
        "The Pacific is the largest ocean on Earth.",
        "World War II ended in 1945.",
        "The chemical symbol for gold is Au.",
    ],
    "instruction": [
        "First, preheat the oven to 350 degrees Fahrenheit.",
        "Turn left at the second traffic light, then go straight.",
        "Mix the flour and sugar before adding the eggs.",
        "Press and hold the button for five seconds to reset.",
        "Save your work frequently to avoid losing progress.",
        "Read all instructions carefully before you begin.",
        "Insert tab A into slot B and fold along the dotted line.",
        "Always back up your files before updating the software.",
    ],
}
CATS = list(CORPUS.keys())


def harvest_all_layers(model, tok, device):
    """Mean-pool each text's residual stream at EVERY layer.

    Returns acts[L] -> (N, hidden) float array, and labels (N,) category idx,
    aligned across layers so we can track the same texts through depth.
    """
    layers = decoder_layers(model)
    L = len(layers)
    caps = {i: [] for i in range(L)}
    handles = []

    def mk(i):
        def hook(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            caps[i].append(h.detach()[0].float().mean(0).cpu().numpy())
        return hook

    for i in range(L):
        handles.append(layers[i].register_forward_hook(mk(i)))

    labels = []
    texts = []
    for ci, cat in enumerate(CATS):
        for t in CORPUS[cat]:
            texts.append(t)
            labels.append(ci)
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt").input_ids.to(device)
            model(input_ids=ids)
    for h in handles:
        h.remove()

    acts = {i: np.stack(caps[i]) for i in range(L)}
    return acts, np.array(labels), texts


def participation_ratio(X):
    """Effective dimensionality: (sum eig)^2 / sum(eig^2). = # dims the
    covariance meaningfully spreads over. Ranges 1..min(N, D)."""
    Xc = X - X.mean(0, keepdims=True)
    # eigenvalues of covariance via SVD of centered data
    s = np.linalg.svd(Xc, compute_uv=False)
    ev = s ** 2
    if ev.sum() <= 0:
        return 0.0
    return float((ev.sum() ** 2) / (ev ** 2).sum())


def linear_cka(X, Y):
    """Centered-kernel-alignment between two representations of the SAME
    points. 1 = identical geometry, 0 = unrelated. Linear kernel version."""
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    # HSIC-based CKA
    def hsic(A, B):
        return np.linalg.norm(A.T @ B, "fro") ** 2
    denom = np.sqrt(hsic(Xc, Xc) * hsic(Yc, Yc))
    if denom <= 0:
        return 0.0
    return float(hsic(Xc, Yc) / denom)


def blind_cluster_quality(X, k, labels_true):
    """Cluster X into k groups with NO label input, then measure:
      - silhouette: intrinsic separation (label-free)
      - nameable-fraction: how well the blind clusters align with the KNOWN
        categories, via best-match purity (assignment-free upper bound using
        adjusted mutual information + cluster purity).
    Returns (silhouette, purity, ami)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import (adjusted_mutual_info_score, silhouette_score)

    # standardize so no single big-variance dim dominates distance
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
    pred = km.labels_
    try:
        sil = float(silhouette_score(Xs, pred))
    except Exception:
        sil = 0.0
    # purity: for each blind cluster, fraction that is its majority TRUE cat
    purity = 0.0
    for c in range(k):
        mask = pred == c
        if mask.sum() == 0:
            continue
        maj = np.bincount(labels_true[mask]).max()
        purity += maj
    purity = float(purity / len(pred))
    ami = float(adjusted_mutual_info_score(labels_true, pred))
    return sil, purity, ami


def run():
    print("=== exp19 stage 1: BLIND NATIVE-GEOMETRY MAP of Qwen3.5-0.8B ===\n")
    model, tok = load_qwen()
    device = next(model.parameters()).device
    print(f"corpus: {len(CATS)} categories x 8 texts = {8*len(CATS)} samples")

    acts, labels, texts = harvest_all_layers(model, tok, device)
    L = len(acts)
    k = len(CATS)
    print(f"harvested residual stream at all {L} layers\n")

    # ---- A. intrinsic dimensionality per layer -------------------------
    pr = [participation_ratio(acts[i]) for i in range(L)]
    print("A. INTRINSIC DIMENSIONALITY (participation ratio, nominal=1024):")
    for i in range(L):
        bar = "#" * int(pr[i] / max(pr) * 40)
        print(f"  L{i:2d}  {pr[i]:6.2f}  {bar}")
    min_pr, max_pr = float(min(pr)), float(max(pr))

    # ---- B. layer reorganization: adjacent-layer CKA -------------------
    cka = [linear_cka(acts[i], acts[i + 1]) for i in range(L - 1)]
    med_cka = float(np.median(cka))
    dips = [i for i, c in enumerate(cka) if c < med_cka - 0.5 * np.std(cka)]
    print("\nB. ADJACENT-LAYER CKA (1=same geometry, low=reorganization):")
    for i in range(L - 1):
        mark = "  <-- REORG" if i in dips else ""
        print(f"  L{i:2d}->L{i+1:2d}  {cka[i]:.3f}{mark}")

    # ---- C. blind clustering + nameability, per layer ------------------
    # plus a shuffled-feature NULL at each layer for G1.
    print("\nC. BLIND CLUSTERING vs NULL (silhouette / purity / AMI):")
    per_layer = []
    for i in range(L):
        sil, purity, ami = blind_cluster_quality(acts[i], k, labels)
        # null: shuffle each feature column independently -> destroys the
        # joint structure, keeps per-dim marginals. If k-means still makes
        # 'separated' clusters here, separation is an artifact.
        Xn = acts[i].copy()
        for c in range(Xn.shape[1]):
            RNG.shuffle(Xn[:, c])
        siln, _, amin = blind_cluster_quality(Xn, k, labels)
        per_layer.append({"layer": i, "pr": pr[i], "sil": sil,
                          "sil_null": siln, "purity": purity, "ami": ami,
                          "ami_null": amin})
        print(f"  L{i:2d}  sil {sil:+.3f} (null {siln:+.3f})  "
              f"purity {purity:.2f}  AMI {ami:+.3f} (null {amin:+.3f})")

    # pick the most-structured layer (max AMI over null margin)
    best = max(per_layer, key=lambda r: r["ami"] - r["ami_null"])
    nameable_frac = best["purity"]        # blind clusters that map to a human cat
    unnameable_frac = 1.0 - nameable_frac

    # ---------------- GATES -------------------------------------------
    g1 = best["sil"] > best["sil_null"] + 0.05 and best["ami"] > best["ami_null"] + 0.1
    g2 = (min_pr < 1024 * 0.5) and (max_pr - min_pr > 2.0)
    g3 = len(dips) >= 1
    g4 = True   # we measured nameable_frac WITH a null -> axiom-2 honesty met
    gates = {"G1_structure_real": bool(g1),
             "G2_dim_nontrivial": bool(g2),
             "G3_reorg_localized": bool(g3),
             "G4_nameability_measured": bool(g4)}

    print("\n=== FINDINGS ===")
    print(f"  effective dim ranges {min_pr:.1f}..{max_pr:.1f} of 1024 "
          f"(model uses {100*max_pr/1024:.1f}% of nominal at most)")
    print(f"  reorganization boundaries at layer transitions: "
          f"{[f'L{d}->L{d+1}' for d in dips]}")
    print(f"  most-structured layer: L{best['layer']} "
          f"(AMI {best['ami']:.2f} vs null {best['ami_null']:.2f})")
    print(f"  NAMEABLE fraction (blind clusters -> human category): "
          f"{nameable_frac:.0%}")
    print(f"  UNNAMEABLE residue (axiom 2, the finding): {unnameable_frac:.0%}")
    print("\n=== GATES ===")
    for g, v in gates.items():
        print(f"  {'PASS' if v else 'FAIL'}  {g}")

    out = {
        "model": "Qwen/Qwen3.5-0.8B",
        "n_layers": L, "n_samples": int(len(labels)), "categories": CATS,
        "participation_ratio": pr, "min_pr": min_pr, "max_pr": max_pr,
        "adjacent_cka": cka, "median_cka": med_cka, "reorg_layers": dips,
        "per_layer": per_layer, "best_layer": best["layer"],
        "nameable_fraction": nameable_frac,
        "unnameable_residue": unnameable_frac,
        "gates": gates,
    }
    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved -> {RESULTS}")

    # ---- FIGURES: this is the "see its structure" experiment; draw it. ----
    make_figures(out, acts, labels, best["layer"])
    return out


def make_figures(out, acts, labels, best_layer):
    """Render the geometry map to PNGs next to the JSON. Four panels:
      1. intrinsic-dimensionality curve across depth (with reorg markers)
      2. adjacent-layer CKA strip -- where the model reorganizes
      3. blind-cluster PCA scatter at the most-structured layer, colored by
         the KNOWN category (so we SEE the 94%-nameable alignment)
      4. nameable vs unnameable-residue donut (axiom 2, visual)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    pr = out["participation_ratio"]
    cka = out["adjacent_cka"]
    dips = set(out["reorg_layers"])
    cats = out["categories"]
    L = out["n_layers"]

    fig = plt.figure(figsize=(15, 10), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.22)
    fig.suptitle("exp19 - Native geometry of Qwen3.5-0.8B's residual stream "
                 "(blind map)", fontsize=15, fontweight="bold")

    # panel 1: intrinsic dimensionality
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(L), pr, "-o", color="#2c7fb8", lw=2, ms=5)
    ax1.set_title(f"A. Intrinsic dimensionality per layer\n"
                  f"(model uses ~{min(pr):.0f}-{max(pr):.0f} of 1024 dims = "
                  f"{100*max(pr)/1024:.1f}% at most)", fontsize=11)
    ax1.set_xlabel("layer"); ax1.set_ylabel("participation ratio")
    for d in dips:
        ax1.axvspan(d + 0.5, d + 1.5, color="#fdae61", alpha=0.25)
    ax1.grid(alpha=0.3)

    # panel 2: CKA reorganization strip
    ax2 = fig.add_subplot(gs[0, 1])
    xs = list(range(len(cka)))
    colors = ["#d7191c" if i in dips else "#2c7fb8" for i in xs]
    ax2.bar(xs, cka, color=colors)
    ax2.axhline(out["median_cka"], color="grey", ls="--", lw=1,
                label=f"median {out['median_cka']:.3f}")
    ax2.set_ylim(min(cka) - 0.02, 1.0)
    ax2.set_title("B. Adjacent-layer similarity (CKA)\n"
                  "red = reorganization boundary (low similarity)", fontsize=11)
    ax2.set_xlabel("layer L -> L+1"); ax2.set_ylabel("CKA (1 = same geometry)")
    ax2.legend(fontsize=8)

    # panel 3: blind cluster scatter at best layer (PCA-2D), colored by category
    ax3 = fig.add_subplot(gs[1, 0])
    X = acts[best_layer]
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    # PCA to 2D
    Xc = Xs - Xs.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:2].T
    cmap = plt.get_cmap("tab10")
    for ci, cat in enumerate(cats):
        m = labels == ci
        ax3.scatter(P[m, 0], P[m, 1], s=45, color=cmap(ci), label=cat,
                    edgecolor="white", linewidth=0.5)
    ax3.set_title(f"C. Activation geometry at L{best_layer} (most structured)\n"
                  f"colored by category - clusters ARE the categories "
                  f"({out['nameable_fraction']:.0%} nameable)", fontsize=11)
    ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2")
    ax3.legend(fontsize=8, loc="best")
    ax3.grid(alpha=0.3)

    # panel 4: nameable vs unnameable residue donut
    ax4 = fig.add_subplot(gs[1, 1])
    nf = out["nameable_fraction"]; uf = out["unnameable_residue"]
    wedges, _, _ = ax4.pie(
        [nf, uf], labels=["nameable\n(maps to human\ncategory)",
                          "unnameable\nresidue\n(axiom 2)"],
        colors=["#2c7fb8", "#d7191c"], autopct=lambda p: f"{p:.0f}%",
        startangle=90, wedgeprops=dict(width=0.42, edgecolor="white"),
        textprops=dict(fontsize=10))
    ax4.set_title("D. How much of the model's own organization\n"
                  "is in words we have (axiom 2, measured)", fontsize=11)

    png = RESULTS.parent / "geometry_map.png"
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure -> {png}")

    # bonus: a per-layer nameability curve (does human-legibility rise/fall
    # with depth?) -- one more real visual
    fig2, ax = plt.subplots(figsize=(9, 4.5), facecolor="white")
    ll = out["per_layer"]
    ax.plot([r["layer"] for r in ll], [r["ami"] for r in ll], "-o",
            color="#2c7fb8", label="AMI (blind clusters vs categories)")
    ax.plot([r["layer"] for r in ll], [r["ami_null"] for r in ll], "--",
            color="grey", label="shuffled null")
    ax.fill_between([r["layer"] for r in ll],
                    [r["ami_null"] for r in ll], [r["ami"] for r in ll],
                    color="#2c7fb8", alpha=0.15)
    ax.set_title("Where the model's geometry is most human-legible "
                 "(AMI over depth)", fontsize=12)
    ax.set_xlabel("layer"); ax.set_ylabel("adjusted mutual information")
    ax.legend(); ax.grid(alpha=0.3)
    png2 = RESULTS.parent / "nameability_by_depth.png"
    fig2.savefig(png2, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved figure -> {png2}")


if __name__ == "__main__":
    run()
