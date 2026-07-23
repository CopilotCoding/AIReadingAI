"""EXPERIMENT 17 -- is LANGUAGE entangled with IDENTITY in Qwen3.5-0.8B?

Observed live in the console: steering the French/language direction did not
just change the OUTPUT LANGUAGE, it also changed the model's STATED MAKER
(Qwen->Tencent, ->Microsoft) and self-name (->"Tú"). That suggests two
concepts we would call separate share geometry. This measures it.

Method (all causal, all with nulls):
  - Steer ONLY the language direction (English<->French diff-of-means at
    layer 2), across doses, on an "Identify yourself" prompt.
  - Score each generation on TWO independent axes:
      language  : French-marker ratio (exp12 scorer, extended for es/zh)
      maker     : which company is named (Alibaba/Tongyi vs a FOREIGN
                  maker: Microsoft/OpenAI/Google/Tencent/Meta...)
  - Entanglement = does maker-drift RISE with |language dose|? If a pure
    language push flips the stated maker well above a random-direction
    null, language and identity are entangled, not separable.

Controls:
  - random-direction null at matched norm (must NOT flip the maker).
  - a DEDICATED identity direction (who-made-you contrast) as a reference
    ceiling for how much the maker CAN be moved.

Gates certify the measurement (language actually steers; null is inert);
the entanglement magnitude is the finding, whatever it is.
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
                                                    decoder_layers, load_qwen)

RESULTS = Path(__file__).parent / "results" / "exp17"
SEED = 0
LAYER = 2
DOSES = [-0.5, -0.35, -0.2, 0.0, 0.2, 0.35, 0.5]

EN = ["The weather is nice today.", "I would like some coffee.",
      "She reads a book each night.", "We are going to the market.",
      "The cat sleeps on the sofa.", "He speaks three languages.",
      "The museum opens at ten.", "The river flows through town.",
      "I need to buy vegetables.", "The concert starts in an hour."]
FR = ["Il fait beau aujourd'hui.", "Je voudrais du café.",
      "Elle lit un livre chaque soir.", "Nous allons au marché.",
      "Le chat dort sur le canapé.", "Il parle trois langues.",
      "Le musée ouvre à dix heures.", "La rivière traverse la ville.",
      "Je dois acheter des légumes.", "Le concert commence dans une heure."]
MAKER = ["I am a language model created by Alibaba Cloud.",
         "This assistant was developed by Alibaba's Tongyi Lab.",
         "My creator is Alibaba. I am the Qwen model.",
         "Made by Alibaba Cloud, I answer questions.",
         "I was built by the Alibaba DAMO Academy team."]
GENERIC = ["I am a helpful assistant here to answer questions.",
           "I can help you write, code, and analyze text.",
           "My purpose is to assist you with many tasks.",
           "I answer questions and help solve problems.",
           "I am here to help you with whatever you need."]

ID_PROMPTS = ["Identify yourself.", "Who created you?",
              "What company made you?", "Tell me who you are."]

HOME_MAKERS = {"alibaba", "tongyi", "qwen", "damo"}
FOREIGN_MAKERS = {"tencent", "microsoft", "openai", "google", "meta",
                  "baidu", "bytedance", "anthropic", "deepseek", "huawei",
                  "amazon", "apple", "ibm", "nvidia", "samsung"}
FR_WORDS = {"le", "la", "les", "une", "un", "je", "est", "et", "vous",
            "nous", "des", "du", "pour", "que", "qui", "dans", "avec",
            "suis", "développé", "modèle", "langue", "aider"}


def maker_label(text: str) -> str:
    """foreign / home / none — which maker family the text names."""
    w = set(re.findall(r"[a-zA-Zàâçéèêëîïôûù]+", text.lower()))
    if w & FOREIGN_MAKERS:
        return "foreign"
    if w & HOME_MAKERS:
        return "home"
    return "none"


def french_ratio(text: str) -> float:
    w = re.findall(r"[a-zA-Zàâçéèêëîïôûù']+", text.lower())
    if not w:
        return 0.0
    return sum(1 for x in w if x in FR_WORDS) / len(w)


def has_cjk(text: str) -> bool:
    return any('一' <= c <= '鿿' for c in text)


def main():
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    print("=" * 70)
    print("EXPERIMENT 17: language <-> identity entanglement")
    print("=" * 70)
    model, tok = load_qwen()
    layers = decoder_layers(model)
    dev = model.device
    mod = layers[LAYER]

    def pooled(texts):
        out = []
        for t in texts:
            ids = tok(t, return_tensors="pt").input_ids.to(dev)
            cap = {}
            h = mod.register_forward_hook(
                lambda m, i, o: cap.__setitem__(
                    'h', (o[0] if isinstance(o, tuple) else o).detach()))
            with torch.no_grad():
                model(input_ids=ids)
            h.remove()
            out.append(cap['h'][0].float().mean(0).cpu().numpy())
        return np.stack(out)

    def rnorm():
        cap = {}
        h = mod.register_forward_hook(
            lambda m, i, o: cap.__setitem__(
                'h', (o[0] if isinstance(o, tuple) else o).detach()))
        ids = tok("The quick brown fox.", return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            model(input_ids=ids)
        h.remove()
        return float(cap['h'][0].float().norm(dim=1).mean())

    RN = rnorm()

    def gen(vec, prompt, n=36):
        h = mod.register_forward_hook(
            lambda m, i, o: ((o[0] + vec.to(o[0].dtype),) + o[1:])
            if isinstance(o, tuple) else o + vec.to(o.dtype))
        ids = tok(chat_prompt(tok, prompt),
                  return_tensors="pt").input_ids.to(dev)
        out, past, nxt = [], None, ids
        with torch.no_grad():
            for _ in range(n):
                o = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = o.past_key_values
                nxt = o.logits[:, -1].argmax(-1, keepdim=True)
                if nxt.item() == tok.eos_token_id:
                    break
                out.append(nxt.item())
        h.remove()
        return tok.decode(out, skip_special_tokens=True)

    # directions
    d_lang = pooled(FR).mean(0) - pooled(EN).mean(0)
    d_lang = torch.tensor(d_lang / (np.linalg.norm(d_lang) + 1e-9),
                          dtype=torch.float32, device=dev)
    d_id = pooled(MAKER).mean(0) - pooled(GENERIC).mean(0)
    d_id = torch.tensor(d_id / (np.linalg.norm(d_id) + 1e-9),
                        dtype=torch.float32, device=dev)
    g = torch.Generator().manual_seed(SEED)
    d_rand = torch.randn(d_lang.shape, generator=g)
    d_rand = (d_rand / d_rand.norm()).to(dev)

    cos_li = float(torch.dot(d_lang, d_id))
    print(f"\ncos(language, identity) = {cos_li:.3f}  (geometry overlap)")

    # --- steer ONLY language, read language AND maker off identity prompts
    print("\n[A] steer language, watch BOTH language and stated maker")
    lang_curve, maker_foreign, cjk_frac = [], [], []
    examples = {}
    for a in DOSES:
        vec = d_lang * (a * RN)
        texts = [gen(vec, p) for p in ID_PROMPTS]
        lang_curve.append(float(np.mean([french_ratio(t) for t in texts])))
        maker_foreign.append(
            float(np.mean([maker_label(t) == "foreign" for t in texts])))
        cjk_frac.append(float(np.mean([has_cjk(t) for t in texts])))
        examples[a] = texts[0]
        print(f"  dose {a:+.2f}: french {lang_curve[-1]:.2f} | "
              f"foreign-maker {maker_foreign[-1]:.2f} | "
              f"cjk {cjk_frac[-1]:.2f} | {texts[0][:55]!r}")

    # --- null: random direction, matched norm, same read
    print("\n[B] random-direction null (must not flip the maker)")
    null_foreign = []
    for a in DOSES:
        vec = d_rand * (a * RN)
        texts = [gen(vec, p) for p in ID_PROMPTS]
        null_foreign.append(
            float(np.mean([maker_label(t) == "foreign" for t in texts])))
    print("  foreign-maker under null:",
          [round(x, 2) for x in null_foreign])

    # --- reference: dedicated identity direction (how much CAN maker move)
    print("\n[C] dedicated identity direction (reference ceiling)")
    id_foreign = []
    for a in DOSES:
        vec = d_id * (a * RN)
        texts = [gen(vec, p) for p in ID_PROMPTS]
        id_foreign.append(
            float(np.mean([maker_label(t) == "foreign" for t in texts])))
    print("  foreign-maker under identity steer:",
          [round(x, 2) for x in id_foreign])

    # entanglement statistic: peak maker-flip from LANGUAGE steering, vs null
    lang_maker_peak = max(maker_foreign)
    null_peak = max(null_foreign)
    id_peak = max(id_foreign)
    # correlation between |language shift| and maker-flip across doses
    lang_shift = np.abs(np.array(lang_curve) - lang_curve[DOSES.index(0.0)])
    mf = np.array(maker_foreign)
    corr = float(np.corrcoef(lang_shift, mf)[0, 1]) if mf.std() > 0 else 0.0

    gates = {
        "M1_language_steers": (max(lang_curve) >= 0.15
                               and lang_curve[DOSES.index(0.0)] <= 0.05,
                               f"french 0->{max(lang_curve):.2f} across doses"),
        "M2_null_inert": (null_peak <= 0.25,
                          f"random-dir foreign-maker peak {null_peak:.2f}"),
        "M3_entanglement_measured": (True,
                                     f"language-steer maker-flip peak "
                                     f"{lang_maker_peak:.2f} vs null "
                                     f"{null_peak:.2f}; corr(|lang|,maker) "
                                     f"{corr:.2f}"),
    }
    n_pass = sum(v[0] for v in gates.values())
    print("\n" + "=" * 70)
    for k, (ok, msg) in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}: {msg}")
    print(f"  {n_pass}/{len(gates)} gates passed")
    print(f"\n  ENTANGLEMENT FINDING:")
    print(f"    cos(language, identity) = {cos_li:.3f}")
    print(f"    pure LANGUAGE steering flips stated maker up to "
          f"{lang_maker_peak:.0%} (null {null_peak:.0%}, "
          f"dedicated identity {id_peak:.0%})")
    verdict = ("ENTANGLED: a pure language push moves identity well above "
               "null" if lang_maker_peak >= null_peak + 0.25
               else "WEAK/NO entanglement at this dose")
    print(f"    -> {verdict}")

    # figure
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("exp17 -- language<->identity entanglement (Qwen3.5-0.8B)")
    ax = axs[0]
    ax.plot(DOSES, lang_curve, "o-", label="french (target)")
    ax.plot(DOSES, maker_foreign, "s-", color="tab:red",
            label="foreign maker named (side effect)")
    ax.plot(DOSES, cjk_frac, "^--", color="tab:purple", label="chinese output")
    ax.set_xlabel("language-direction dose"); ax.set_ylabel("fraction")
    ax.set_title("steering ONLY language moves identity too")
    ax.legend(fontsize=8)
    ax = axs[1]
    ax.plot(DOSES, maker_foreign, "s-", color="tab:red", label="via language")
    ax.plot(DOSES, id_foreign, "d-", color="tab:green",
            label="via identity dir")
    ax.plot(DOSES, null_foreign, "x--", color="gray", label="random null")
    ax.set_xlabel("dose"); ax.set_ylabel("foreign-maker fraction")
    ax.set_title("maker-flip: language vs identity vs null")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fpath = RESULTS / "exp17_entanglement.png"
    fig.savefig(fpath, dpi=130)

    report = {
        "model": MODEL_ID, "layer": LAYER, "doses": DOSES,
        "cos_language_identity": cos_li,
        "language_curve": lang_curve, "maker_foreign_via_language": maker_foreign,
        "cjk_fraction": cjk_frac,
        "maker_foreign_null": null_foreign,
        "maker_foreign_via_identity": id_foreign,
        "entanglement_corr": corr, "verdict": verdict,
        "gates": {k: {"pass": bool(v[0]), "detail": v[1]}
                  for k, v in gates.items()},
        "examples": {str(k): v for k, v in examples.items()},
        "runtime_sec": round(time.time() - t0, 1),
    }
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2),
                                         encoding="utf-8")
    print(f"\n  report: {RESULTS / 'report.json'}")
    print(f"  figure: {fpath}")
    print("=" * 70)
    return 0 if n_pass == len(gates) else 1


if __name__ == "__main__":
    sys.exit(main())
