"""EXPERIMENT 16 -- SAE feature harvest: concepts nobody planted.

Every prior real-model experiment targeted a concept WE chose (French,
dog/cat, list-format). This one lets the model tell us what IT carved its
activation space into. Train a sparse autoencoder on Qwen's layer-2
residual stream over diverse text, then for each learned atom:

  AUTO-LABEL  collect the tokens/snippets where the atom fires hardest.
              A human-legible label is a bonus, not the claim.
  CAUSAL TEST the claim is behavioral: CLAMP the atom on during generation
              (add its decoder vector, scaled) and check the output shifts
              toward what the atom's top-firing tokens predict. An atom
              that fires interpretably but does nothing when clamped is
              REFUTED as a causal feature -- decodable != causal, the
              lab's spine (exp3, exp12, exp13).

We do not hand-pick which atoms to report. Selection is by a fixed rule:
rank atoms by activation frequency in a healthy band (not dead, not always-
on), take the top K, and for each run the SAME causal probe. The headline
number is the causal HIT RATE -- of blindly-discovered, interpretable-
looking atoms, what fraction actually steer behavior.

Nulls: a random unit direction clamped at the same norm (must not produce
the atom's signature), and a shuffled-atom control for the selectivity
score. Gates certify the pipeline (reconstruction, sparsity, a working
causal probe on at least one atom, dead nulls), never a target semantics.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from interpretability_lab.experiments.exp12_qwen_dissection import (
    Steerer, generate)
from interpretability_lab.features.sae import fit_sae_batched
from interpretability_lab.models.pretrained import (MODEL_ID, chat_prompt,
                                                    decoder_layers, load_qwen)

RESULTS = Path(__file__).parent / "results" / "exp16"
SEED = 0
HARVEST_LAYER = 2          # exp12/exp14 causal input site
D_HIDDEN = 2048            # 2x overcomplete; run-1 used 8192 atoms on 658
                           # rows (12x more atoms than data) -> dense garbage.
L1 = 2.0                   # on STANDARDIZED activations. Run 1 (l1=2e-3, raw)
                           # gave L0 1016/8192 = dense garbage. Standardizing
                           # rescales the recon term, so the raw-space sweep's
                           # 0.08 was then too weak (L0 830); the standardized
                           # sweep found l1=2.0 -> L0~25 at R2~0.99. Real
                           # sparse features. (Both sweeps in scratchpad.)
TOP_K = 16                 # atoms to causally probe
CLAMP_SCALE = 10.0         # multiples of atom's mean active value

# Diverse harvest corpus: many registers so features aren't domain-locked.
CORPUS = [
    # code
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "import numpy as np\narr = np.zeros((3, 4))\nfor i in range(3):\n    arr[i] = i * 2",
    "SELECT name, age FROM users WHERE age > 18 ORDER BY name ASC LIMIT 100;",
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)",
    "const handler = async (req, res) => {\n  const data = await db.query(sql);\n  res.json(data);\n};",
    # math / science
    "The derivative of x squared is two x, and the integral of two x is x squared plus a constant.",
    "Water boils at one hundred degrees Celsius at sea level, where atmospheric pressure is one atmosphere.",
    "The mitochondria is the powerhouse of the cell, producing ATP through cellular respiration.",
    "Newton's second law states that force equals mass times acceleration, written as F equals m a.",
    "The speed of light in a vacuum is approximately three hundred thousand kilometers per second.",
    # questions
    "What is the capital of France, and how many people live there today?",
    "Could you explain how a bill becomes a law in the United States Congress?",
    "Why does the sky appear blue during the day but red at sunset?",
    "How do I center a div horizontally and vertically using flexbox in CSS?",
    "When was the printing press invented, and who is credited with its creation?",
    # narrative
    "The old lighthouse stood alone on the cliff, its beam sweeping across the dark and restless sea.",
    "She opened the letter with trembling hands, unsure whether it carried good news or bad.",
    "The children raced down the hill, laughing as the autumn leaves crunched beneath their feet.",
    "In the quiet of the early morning, the city slowly awoke to the sound of distant church bells.",
    # dialogue / instructional
    "First, preheat the oven to 350 degrees. Then mix the flour, sugar, and eggs until smooth.",
    "Thank you so much for your help yesterday. I really could not have finished the project without you.",
    "I'm sorry, but I cannot assist with that request. Let me suggest a safer alternative instead.",
    "Please remember to bring your passport, boarding pass, and a valid form of photo identification.",
    # sentiment / opinion
    "This is absolutely the best restaurant in town; the food was incredible and the service was warm.",
    "The movie was a disappointing mess, with a confusing plot and wooden, lifeless performances.",
    "I strongly believe that renewable energy is the only responsible path forward for our planet.",
    # lists / structured
    "Ingredients: 2 cups flour, 1 cup sugar, 3 eggs, 1 teaspoon vanilla, a pinch of salt.",
    "1. Wake up early. 2. Drink water. 3. Exercise for thirty minutes. 4. Eat a healthy breakfast.",
    # numbers / dates / entities
    "The meeting is scheduled for March 14, 2026, at 3:30 PM in conference room B on the fifth floor.",
    "Apple, Microsoft, and Google are among the most valuable technology companies in the world.",
    "The population grew from 1.2 million in 1990 to over 4.5 million by the year 2020.",
    # longer paragraphs per register (more tokens, richer context)
    "The Industrial Revolution transformed economies across Europe and North "
    "America during the eighteenth and nineteenth centuries. New machines "
    "powered by steam replaced manual labor in textile mills and factories. "
    "Cities grew rapidly as workers left farms to seek employment, and this "
    "migration reshaped society, politics, and the environment for generations.",
    "Photosynthesis is the process by which green plants, algae, and some "
    "bacteria convert light energy into chemical energy. Chlorophyll in the "
    "chloroplasts absorbs sunlight, which drives the conversion of carbon "
    "dioxide and water into glucose and oxygen. This process sustains nearly "
    "all life on Earth by producing both food and breathable air.",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    "
    "mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    "
    "right = merge_sort(arr[mid:])\n    result = []\n    i = j = 0\n    "
    "while i < len(left) and j < len(right):\n        "
    "if left[i] < right[j]:\n            result.append(left[i])\n            "
    "i += 1\n        else:\n            result.append(right[j])\n            "
    "j += 1\n    return result + left[i:] + right[j:]",
    "Dear Sarah, I hope this letter finds you well. It has been far too long "
    "since we last spoke, and so much has happened. I moved to a new city in "
    "the spring, started a job I truly enjoy, and even adopted a small grey "
    "cat named Pepper. I think of our long walks often and hope we can plan "
    "a visit soon. Please write back when you have a moment. Warmly, Emma.",
    "The quarterly report indicates a modest increase in revenue driven "
    "primarily by strong performance in the international division. Operating "
    "costs rose slightly due to expansion, but margins remained stable. "
    "Management expects continued growth in the next fiscal year, contingent "
    "on favorable market conditions and the successful launch of two products.",
    "To make a proper cup of tea, first bring fresh water to a rolling boil. "
    "Warm the teapot with a little hot water, then discard it. Add one "
    "teaspoon of loose leaves per cup, pour in the boiling water, and steep "
    "for three to five minutes depending on strength. Strain into a cup and "
    "add milk or sugar to taste if desired.",
    "Beneath the ancient oak the two travelers rested, weary from the long "
    "road. The sun sank low behind the hills, painting the sky in shades of "
    "amber and rose. Somewhere in the distance a river murmured, and the "
    "first stars began to appear. Tomorrow they would reach the city, but "
    "tonight, for the first time in weeks, they felt at peace.",
    "The theorem states that for any right triangle, the square of the "
    "hypotenuse equals the sum of the squares of the other two sides. This "
    "relationship, known since antiquity, forms the foundation of Euclidean "
    "geometry and has countless applications in engineering, physics, "
    "navigation, and the design of everyday structures around us.",
]

# probe prompts for causal clamping (neutral, open-ended)
PROBE_PROMPTS = [
    "Tell me something interesting.",
    "Continue this thought: today I was thinking about",
    "Write a short response about your day.",
    "Here is a sentence to complete: the most important thing is",
]

WORD_RE = re.compile(r"[A-Za-z]+")


def harvest(model, tok, layer_mod):
    """Run the corpus through the model, capturing layer-2 residual output
    per token. Returns acts (N, d) and a parallel list of (token_str) tags."""
    caught = {}

    def hook(mod, inp, out):
        caught["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    h = layer_mod.register_forward_hook(hook)
    rows, tags = [], []
    with torch.no_grad():
        for text in CORPUS:
            ids = tok(text, return_tensors="pt").input_ids.to(model.device)
            model(input_ids=ids)
            act = caught["h"][0].float().cpu().numpy()      # (T, d)
            toks = tok.convert_ids_to_tokens(ids[0].tolist())
            rows.append(act)
            tags.extend(toks)
    h.remove()
    return np.concatenate(rows), tags


def atom_top_tokens(Z, tags, atom, n=12):
    """The token strings where this atom fires hardest."""
    z = Z[:, atom]
    order = np.argsort(-z)[:n]
    cleaned = [tags[i].lstrip("ĠĠ▁").strip() for i in order]
    return [c for c in cleaned if c]


def signature_words(top_tokens):
    """A rough lexical signature from an atom's top tokens (for the causal
    test): the alphabetic tokens, lowercased."""
    words = [w.lower() for t in top_tokens for w in WORD_RE.findall(t)]
    return Counter(words)


def signature_overlap(text, sig: Counter):
    """Fraction of generated words that are in the atom's signature set."""
    words = [w.lower() for w in WORD_RE.findall(text)]
    if not words or not sig:
        return 0.0
    sset = set(sig)
    return sum(1 for w in words if w in sset) / len(words)


def clamp_vector(model_sae, atom, Z, device, sd_t=None):
    """The decoder direction for `atom`, mapped from standardized SAE space
    back to RAW residual space (elementwise * sd), then scaled to
    CLAMP_SCALE * the atom's mean active value (so the injection matches the
    atom's own natural magnitude)."""
    z = Z[:, atom]
    mean_active = float(z[z > 1e-6].mean()) if (z > 1e-6).any() else 1.0
    d = model_sae.dec.weight.data[:, atom].clone()           # (d_in,) std space
    if sd_t is not None:
        d = d * sd_t                                         # -> raw space
    return (d / (d.norm() + 1e-8)).to(device) * (CLAMP_SCALE * mean_active)


def main():
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("EXPERIMENT 16: SAE feature harvest -- concepts nobody planted")
    print("=" * 70)
    model, tok = load_qwen()
    layers = decoder_layers(model)
    layer_mod = layers[HARVEST_LAYER]

    # ------------------------------------------------------- [0] harvest
    print(f"\n[0] harvesting layer-{HARVEST_LAYER} residuals over "
          f"{len(CORPUS)} texts")
    acts, tags = harvest(model, tok, layer_mod)
    print(f"    {acts.shape[0]} token-activations x {acts.shape[1]} dims")

    # standardize per-dim (zero mean / unit var) before the SAE -- otherwise a
    # few high-variance residual dims dominate the L2 recon and atoms don't
    # specialize. Decoder atoms/clamp vectors are mapped back to raw space.
    mu = acts.mean(0)
    sd = acts.std(0) + 1e-6
    acts_n = (acts - mu) / sd

    # --------------------------------------------------- [1] train SAE
    print(f"\n[1] training SAE ({D_HIDDEN} atoms, l1={L1}, standardized)")
    sae, Z, info = fit_sae_batched(acts_n, d_hidden=D_HIDDEN, l1=L1,
                                   epochs=150, batch=2048, seed=SEED,
                                   log_every=30)
    sd_t = torch.tensor(sd, dtype=torch.float32)
    print(f"    recon R2 {info['recon_r2']:.3f}, mean L0 "
          f"{info['mean_l0']:.1f}, dead atoms {info['dead_atoms']}/"
          f"{D_HIDDEN}")

    # ----------------------------------------- [2] select atoms by rule
    fire_rate = (Z > 1e-6).mean(0)                       # per-atom frequency
    band = np.where((fire_rate > 0.005) & (fire_rate < 0.25))[0]
    ranked = band[np.argsort(-fire_rate[band])][:TOP_K]
    print(f"\n[2] {len(band)} atoms in healthy fire band; probing top "
          f"{len(ranked)}")

    # --------------------------------- [3] auto-label + causal clamp test
    print(f"\n[3] causal probe: clamp each atom, measure signature shift")
    features = []
    for atom in ranked:
        top = atom_top_tokens(Z, tags, atom)
        sig = signature_words(top)
        vec = clamp_vector(sae, atom, Z, model.device, sd_t)

        st = Steerer(layer_mod)
        st.vec = vec
        clamped = [generate(model, tok, p) for p in PROBE_PROMPTS]
        st.vec = None
        st.remove()
        base = [generate(model, tok, p) for p in PROBE_PROMPTS]

        ov_c = float(np.mean([signature_overlap(t, sig) for t in clamped]))
        ov_b = float(np.mean([signature_overlap(t, sig) for t in base]))
        features.append({
            "atom": int(atom), "fire_rate": float(fire_rate[atom]),
            "top_tokens": top[:8],
            "overlap_clamped": ov_c, "overlap_baseline": ov_b,
            "causal_delta": ov_c - ov_b,
            "example_clamped": clamped[0][:120],
        })
        label = " ".join(top[:5])
        print(f"    atom {atom:>4} (fires {fire_rate[atom]:.1%}): "
              f"[{label[:34]:<34}] steer +{ov_c - ov_b:+.3f}")

    if not features:
        print("\n  NO ATOMS IN BAND -- SAE degenerate; gates will fail. "
              "Re-tune L1.")
        return 1

    # random-direction null: clamp a random unit dir at the SAME norm as the
    # best atom's clamp vector, then score against the best atom's signature.
    best = max(features, key=lambda f: f["causal_delta"])
    best_sig = signature_words(best["top_tokens"])
    best_vec = clamp_vector(sae, best["atom"], Z, model.device, sd_t)
    g = torch.Generator(device="cpu").manual_seed(SEED)
    rvec = torch.randn(acts.shape[1], generator=g)
    rvec = (rvec / rvec.norm()).to(model.device) * float(best_vec.norm())
    st = Steerer(layer_mod)
    st.vec = rvec
    null_texts = [generate(model, tok, p) for p in PROBE_PROMPTS]
    st.vec = None
    st.remove()
    null_overlap = float(np.mean(
        [signature_overlap(t, best_sig) for t in null_texts]))
    print(f"\n    random-dir null (best atom's signature): {null_overlap:.3f} "
          f"vs clamped {best['overlap_clamped']:.3f}")

    # ---------------------------------------------------------- hit rate
    causal_hits = [f for f in features if f["causal_delta"] >= 0.05]
    hit_rate = len(causal_hits) / len(features)
    print(f"\n  CAUSAL HIT RATE: {len(causal_hits)}/{len(features)} "
          f"blindly-found atoms steer behavior ({hit_rate:.0%})")

    # ---------------------------------------------------------------- gates
    gates = {
        "H1_reconstruction": (info["recon_r2"] >= 0.80,
                              f"SAE recon R2 {info['recon_r2']:.2f} (bar 0.80)"),
        "H2_sparse": (info["mean_l0"] <= 0.15 * D_HIDDEN
                      and info["mean_l0"] >= 2,
                      f"mean L0 {info['mean_l0']:.1f} of {D_HIDDEN} "
                      f"(sparse but alive)"),
        "H3_causal_feature_exists": (best["causal_delta"] >= 0.05,
                                     f"best atom {best['atom']} steers "
                                     f"{best['causal_delta']:+.3f} "
                                     f"[{' '.join(best['top_tokens'][:4])}]"),
        "H4_null_inert": (null_overlap <= best["overlap_clamped"] - 0.03,
                          f"random dir {null_overlap:.3f} < clamped "
                          f"{best['overlap_clamped']:.3f}"),
        "H5_not_all_dead": (info["dead_atoms"] <= 0.7 * D_HIDDEN
                            and len(band) >= TOP_K,
                            f"{info['dead_atoms']} dead, {len(band)} in band"),
    }
    n_pass = sum(v[0] for v in gates.values())
    print("\n" + "=" * 70)
    for k, (ok, msg) in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}: {msg}")
    print(f"  {n_pass}/{len(gates)} gates passed")

    # ----------------------------------------------------------------- figure
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("exp16 -- SAE feature harvest on Qwen3.5-0.8B layer 2",
                 fontsize=13)

    ax = axs[0, 0]
    ax.hist(np.log10(fire_rate[fire_rate > 1e-6] + 1e-9), bins=40,
            color="tab:blue", alpha=0.8)
    ax.axvline(np.log10(0.005), color="tab:red", ls="--", label="band lo")
    ax.axvline(np.log10(0.25), color="tab:red", ls=":", label="band hi")
    ax.set_xlabel("log10 atom fire rate")
    ax.set_ylabel("# atoms")
    ax.set_title(f"[1] {D_HIDDEN} atoms, {info['dead_atoms']} dead")
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    fs = sorted(features, key=lambda f: f["causal_delta"])
    ax.barh(range(len(fs)), [f["causal_delta"] for f in fs],
            color=["tab:green" if f["causal_delta"] >= 0.05 else "gray"
                   for f in fs])
    ax.axvline(0.05, color="k", ls=":", lw=1)
    ax.set_yticks(range(len(fs)),
                  [" ".join(f["top_tokens"][:3])[:22] for f in fs],
                  fontsize=7)
    ax.set_xlabel("causal steer Δ (clamped − baseline)")
    ax.set_title(f"[3] hit rate {len(causal_hits)}/{len(features)}")

    ax = axs[1, 0]
    ax.bar(["best atom\nclamped", "random-dir\nnull"],
           [best["overlap_clamped"], null_overlap],
           color=["tab:green", "gray"], alpha=0.85)
    ax.set_ylabel("signature overlap")
    ax.set_title("[H4] causal effect vs random-direction null")

    ax = axs[1, 1]
    ax.axis("off")
    lines = ["top causal features (blind):\n"]
    for f in sorted(features, key=lambda f: -f["causal_delta"])[:6]:
        lines.append(f"atom {f['atom']} ({f['fire_rate']:.1%}, "
                     f"Δ{f['causal_delta']:+.2f}):")
        lines.append("  " + " ".join(f["top_tokens"][:6])[:56])
    ax.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=8,
            family="monospace")
    ax.set_title("what the model carved out")

    fig.tight_layout()
    fpath = RESULTS / "exp16_qwen_sae.png"
    fig.savefig(fpath, dpi=130)

    report = {
        "model": MODEL_ID, "harvest_layer": HARVEST_LAYER,
        "n_activations": int(acts.shape[0]), "sae_info": info,
        "n_atoms_in_band": int(len(band)), "top_k": TOP_K,
        "causal_hit_rate": hit_rate,
        "features": features, "null_overlap": null_overlap,
        "gates": {k: {"pass": bool(v[0]), "detail": v[1]}
                  for k, v in gates.items()},
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
