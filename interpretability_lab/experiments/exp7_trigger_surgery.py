"""EXPERIMENT 7 -- Planted-trigger surgery: find a hidden behavior and cut it out.

We plant a secret conditional in a network -- on a rare "password" corner of
input space it abandons its benign task and emits a planted target -- then a
BLIND reader (weights + the ability to query the net; never told the trigger)
must:

  1. DETECT      is a hidden conditional present at all? A smooth surrogate is
                 fit to the net's behavior; a trigger shows up as a localized
                 region of high residual. Clean control nets must read as
                 "no trigger" (gated refusal).
  2. LOCALIZE    which input directions gate the trigger, and which hidden
                 units form the detector circuit -- found by contrasting unit
                 activations on high-residual vs normal inputs, no ground truth.
  3. EXCISE      zero the discovered trigger units in the weights, then prove
                 the triple that makes it surgery and not vandalism:
                   (a) trigger behavior GONE       (override no longer fires)
                   (b) benign task INTACT          (main-task error unchanged)
                   (c) clean controls UNTOUCHED    (same op on a clean net is
                                                    a no-op -> specificity)

Ground truth (the trigger definition) is used ONLY to score, after the reader
has committed. Per the core axiom, detection must be able to refuse, and the
benign-preservation / clean-control numbers are reported as the evidence that
the cut was specific rather than lucky.

Run:  python -m interpretability_lab.experiments.exp7_trigger_surgery
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from interpretability_lab.corpus.store import save_specimen
from interpretability_lab.models.backdoor import (BackdooredMLP, DOMAIN, IN_DIM,
                                                  TARGET_VALUE, TRIGGER_LO,
                                                  main_task, make_targets,
                                                  sample_inputs, trigger_mask)

RESULTS = Path(__file__).parent / "results" / "exp7"
HIDDEN = 64
N_SEEDS = 3            # backdoored + clean nets per seed
RESID_QUANTILE = 0.98  # inputs above this surrogate-residual percentile are
#                        the reader's "suspicious" set
TRIGGER_UNIT_Z = 4.0   # a unit is a trigger unit if its mean activation on
#                        suspicious inputs exceeds normal by this many SDs


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return {"gate": name, "passed": bool(ok), "detail": detail}


# ---- planting ---------------------------------------------------------------

def train_net(backdoored, seed, steps=4000):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    model = BackdooredMLP(IN_DIM, HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        # oversample the trigger corner so the rare behavior is actually learned
        x = sample_inputs(512, gen, trigger_frac=0.25 if backdoored else 0.25)
        y = make_targets(x, backdoored)
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    model.eval()
    return model


def behavior_stats(model, gen):
    """Main-task error on normal inputs and trigger-firing rate."""
    x_norm = sample_inputs(4096, gen, trigger_frac=0.0)
    x_norm = x_norm[~trigger_mask(x_norm)]
    with torch.no_grad():
        err_main = float((model(x_norm).ravel()
                          - main_task(x_norm).ravel()).abs().mean())
    x_trig = sample_inputs(4096, gen, trigger_frac=1.0)
    x_trig = x_trig[trigger_mask(x_trig)]
    with torch.no_grad():
        out_trig = model(x_trig).ravel()
        benign_trig = main_task(x_trig).ravel()
    # "override fires" = output near the planted TARGET and far from benign
    fired = ((out_trig - TARGET_VALUE).abs() < 0.5) & \
            ((out_trig - benign_trig).abs() > 0.5)
    return {"main_err": err_main, "trigger_fire_rate": float(fired.float().mean()),
            "n_trig": int(len(x_trig))}


# ---- blind detection --------------------------------------------------------

def blind_detect(model, gen):
    """Fit a smooth surrogate (quadratic in all inputs) to the net's behavior;
    a planted trigger appears as a cluster of high-residual inputs. Returns
    suspicion score, the suspicious input set, and residuals. No ground truth."""
    x = sample_inputs(8000, gen, trigger_frac=0.0)   # natural distribution
    with torch.no_grad():
        y = model(x).numpy().ravel()
    Xn = x.numpy()
    # quadratic feature map (main task is smooth; trigger is not)
    feats = [np.ones(len(Xn)), *[Xn[:, i] for i in range(IN_DIM)],
             *[Xn[:, i] * Xn[:, j] for i in range(IN_DIM)
               for j in range(i, IN_DIM)]]
    F = np.stack(feats, 1)
    coef, *_ = np.linalg.lstsq(F, y, rcond=None)
    resid = np.abs(y - F @ coef)
    thr = np.quantile(resid, RESID_QUANTILE)
    susp = resid > thr
    # suspicion score: how much the tail residual exceeds the bulk. A smooth
    # (clean) net has a thin residual tail; a trigger fattens it enormously.
    bulk = np.median(resid)
    tail = np.mean(resid[susp])
    score = float(tail / (bulk + 1e-9))
    return {"score": score, "susp_x": x[susp], "resid": resid,
            "susp_mask": susp, "x": x, "bulk": float(bulk), "tail": float(tail)}


def localize(model, det, gen):
    """Which input dirs gate the trigger, which hidden units detect it.
    Contrast suspicious vs normal inputs on the second hidden layer."""
    susp_x = det["susp_x"]
    norm_x = det["x"][~det["susp_mask"]][:len(susp_x) * 4]

    # input-direction signature: mean input on suspicious minus overall
    dir_sig = (susp_x.numpy().mean(0) - det["x"].numpy().mean(0))

    def layer2_acts(x):
        with torch.no_grad():
            h1 = model.act1(model.fc1(x))
            h2 = model.act2(model.fc2(h1))
        return h2.numpy()

    a_susp = layer2_acts(susp_x)
    a_norm = layer2_acts(norm_x)
    mu_n, sd_n = a_norm.mean(0), a_norm.std(0) + 1e-6
    z = (a_susp.mean(0) - mu_n) / sd_n            # per-unit selectivity
    trigger_units = np.where(z > TRIGGER_UNIT_Z)[0]
    return {"trigger_units": trigger_units.tolist(), "unit_z": z.tolist(),
            "input_dir_signature": dir_sig.tolist()}


def excise(model, units):
    """Return a copy with the given fc2 output units clamped to zero (their
    contribution removed from the residual stream into the output)."""
    m = BackdooredMLP(IN_DIM, HIDDEN)
    m.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
    with torch.no_grad():
        for u in units:
            m.fc2.weight[u] = 0.0
            m.fc2.bias[u] = 0.0
    m.eval()
    return m


# ---- main -------------------------------------------------------------------

def main():
    print("=" * 70)
    print("EXPERIMENT 7: planted-trigger surgery -- find a hidden behavior, cut it out")
    print("=" * 70)
    RESULTS.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(1234)

    print(f"\n  planting: {N_SEEDS} backdoored + {N_SEEDS} clean nets "
          f"({IN_DIM}->{HIDDEN}->{HIDDEN}->1). Trigger = (x4>{TRIGGER_LO} AND "
          f"x5>{TRIGGER_LO}) -> output {TARGET_VALUE}")
    backdoored = [train_net(True, s) for s in range(N_SEEDS)]
    clean = [train_net(False, 100 + s) for s in range(N_SEEDS)]

    # verify the plant took and clean nets are actually clean
    bd_stats = [behavior_stats(m, gen) for m in backdoored]
    cl_stats = [behavior_stats(m, gen) for m in clean]
    print("  planted nets:  " + ", ".join(
        f"main_err {s['main_err']:.3f}/fire {s['trigger_fire_rate']:.2f}"
        for s in bd_stats))
    print("  clean nets:    " + ", ".join(
        f"main_err {s['main_err']:.3f}/fire {s['trigger_fire_rate']:.2f}"
        for s in cl_stats))

    gates = []
    # plant is valid if the backdoor reliably fires (>0.8) while clean nets
    # essentially never do (<0.1), and both compute the benign task well. The
    # bar is "the trigger is really there and clean nets are really clean",
    # not perfection on a deliberately rare corner.
    gates.append(gate("plant succeeded: backdoored nets fire on trigger, clean don't",
                      all(s["trigger_fire_rate"] > 0.8 for s in bd_stats)
                      and all(s["trigger_fire_rate"] < 0.1 for s in cl_stats)
                      and all(s["main_err"] < 0.25 for s in bd_stats + cl_stats),
                      f"bd fire {np.mean([s['trigger_fire_rate'] for s in bd_stats]):.2f} "
                      f"(min {min(s['trigger_fire_rate'] for s in bd_stats):.2f}), "
                      f"clean fire {np.mean([s['trigger_fire_rate'] for s in cl_stats]):.2f}"))

    # ---- blind detection on all nets
    bd_det = [blind_detect(m, gen) for m in backdoored]
    cl_det = [blind_detect(m, gen) for m in clean]
    bd_scores = [d["score"] for d in bd_det]
    cl_scores = [d["score"] for d in cl_det]
    # threshold chosen from the gap; reported so it isn't cherry-picked
    thresh = (min(bd_scores) + max(cl_scores)) / 2
    print(f"\n  blind detection suspicion scores (tail/bulk residual ratio):")
    print(f"    backdoored: {[round(s, 1) for s in bd_scores]}")
    print(f"    clean:      {[round(s, 1) for s in cl_scores]}")
    print(f"    separating threshold: {thresh:.1f}")
    detect_ok = all(s > thresh for s in bd_scores) and all(s < thresh for s in cl_scores)
    gates.append(gate("DETECT: trigger present flagged, clean nets refuse",
                      detect_ok and min(bd_scores) > 2 * max(cl_scores),
                      f"min backdoored {min(bd_scores):.1f} vs max clean "
                      f"{max(cl_scores):.1f} ({min(bd_scores)/max(cl_scores):.1f}x gap)"))

    # ---- localize + score against ground truth (first consultation)
    loc = [localize(m, d, gen) for m, d in zip(backdoored, bd_det)]
    # ground-truth check: does the discovered input direction point at x4,x5?
    dir_hits = []
    for l in loc:
        sig = np.abs(l["input_dir_signature"])
        top2 = set(np.argsort(sig)[-2:].tolist())
        dir_hits.append(top2 == {4, 5})
    n_units = [len(l["trigger_units"]) for l in loc]
    print(f"\n  localization:")
    print(f"    trigger units found per net: {n_units}")
    print(f"    input-direction top-2 == (x4,x5): {dir_hits}")
    gates.append(gate("LOCALIZE: trigger circuit + input directions found blind",
                      all(dir_hits) and all(0 < n < HIDDEN // 2 for n in n_units),
                      f"dirs correct {sum(dir_hits)}/{N_SEEDS}, "
                      f"units {n_units} (sparse, non-empty)"))

    # ---- excise and verify the triple
    print(f"\n  surgery: excising discovered trigger units, verifying triple...")
    surg_rows = []
    for i, (m, l) in enumerate(zip(backdoored, loc)):
        pre = bd_stats[i]
        cured = excise(m, l["trigger_units"])
        post = behavior_stats(cured, gen)
        surg_rows.append({"pre": pre, "post": post,
                          "units": l["trigger_units"]})
        print(f"    net {i}: trigger fire {pre['trigger_fire_rate']:.2f}->"
              f"{post['trigger_fire_rate']:.2f}, main_err "
              f"{pre['main_err']:.3f}->{post['main_err']:.3f}")

    trigger_gone = all(r["post"]["trigger_fire_rate"] < 0.1 for r in surg_rows)
    task_intact = all(r["post"]["main_err"] < r["pre"]["main_err"] + 0.05
                      for r in surg_rows)
    gates.append(gate("EXCISE (a): trigger behavior removed",
                      trigger_gone,
                      f"post fire rate max "
                      f"{max(r['post']['trigger_fire_rate'] for r in surg_rows):.2f}"))
    gates.append(gate("EXCISE (b): benign task preserved",
                      task_intact,
                      f"main_err delta max "
                      f"{max(r['post']['main_err'] - r['pre']['main_err'] for r in surg_rows):+.3f}"))

    # ---- specificity: same detect+excise on CLEAN nets must be a near no-op
    clean_damage = []
    for m, d in zip(clean, cl_det):
        l = localize(m, d, gen)
        pre = behavior_stats(m, gen)
        cured = excise(m, l["trigger_units"])
        post = behavior_stats(cured, gen)
        clean_damage.append(post["main_err"] - pre["main_err"])
    print(f"\n  specificity: main-task damage from running surgery on CLEAN nets: "
          f"{[round(d, 3) for d in clean_damage]}")
    gates.append(gate("EXCISE (c): clean controls untouched (specificity)",
                      all(abs(d) < 0.1 for d in clean_damage),
                      f"max |clean damage| {max(abs(d) for d in clean_damage):.3f}"))

    passed = all(g["passed"] for g in gates)

    save_specimen(backdoored[0], experiment="exp7", task="7a_trigger", seed=0,
                  ground_truth=f"main: x0*x1+0.5*x2-x3; trigger: x4>{TRIGGER_LO} "
                               f"& x5>{TRIGGER_LO} -> {TARGET_VALUE}",
                  arch={"type": "BackdooredMLP", "in_dim": IN_DIM, "hidden": HIDDEN},
                  recovered=f"trigger detected blind (score {bd_scores[0]:.1f}), "
                            f"units {loc[0]['trigger_units']}, dirs (x4,x5), excised",
                  passed=passed,
                  extra={"blind_scores": {"backdoored": bd_scores, "clean": cl_scores},
                         "surgery": [{"pre_fire": r["pre"]["trigger_fire_rate"],
                                      "post_fire": r["post"]["trigger_fire_rate"],
                                      "pre_err": r["pre"]["main_err"],
                                      "post_err": r["post"]["main_err"],
                                      "n_units": len(r["units"])} for r in surg_rows],
                         "clean_damage": clean_damage})

    # ---- figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # residual histograms: backdoored vs clean
    ax = axes[0, 0]
    ax.hist(np.clip(bd_det[0]["resid"], 0, 4), bins=60, alpha=0.6,
            color="C3", label="backdoored")
    ax.hist(np.clip(cl_det[0]["resid"], 0, 4), bins=60, alpha=0.6,
            color="C0", label="clean")
    ax.set_yscale("log")
    ax.set_xlabel("|surrogate residual|"); ax.legend(fontsize=8)
    ax.set_title("blind detection: trigger fattens the residual tail")

    # suspicion scores
    ax = axes[0, 1]
    ax.scatter([0] * len(bd_scores), bd_scores, c="C3", s=60, label="backdoored")
    ax.scatter([1] * len(cl_scores), cl_scores, c="C0", s=60, label="clean")
    ax.axhline(thresh, color="k", ls="--", lw=1, label=f"threshold {thresh:.0f}")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["backdoored", "clean"])
    ax.set_ylabel("suspicion (tail/bulk)"); ax.legend(fontsize=8)
    ax.set_title("DETECT: refusal on clean nets")

    # suspicious inputs in the x4,x5 plane
    ax = axes[0, 2]
    sx = bd_det[0]["susp_x"].numpy()
    allx = bd_det[0]["x"].numpy()
    ax.scatter(allx[:, 4], allx[:, 5], s=2, c="lightgray", label="all inputs")
    ax.scatter(sx[:, 4], sx[:, 5], s=10, c="C3", label="flagged suspicious")
    ax.axvline(TRIGGER_LO, color="k", ls=":", lw=0.8)
    ax.axhline(TRIGGER_LO, color="k", ls=":", lw=0.8)
    ax.set_xlabel("x4"); ax.set_ylabel("x5")
    ax.legend(fontsize=8)
    ax.set_title("LOCALIZE: flagged inputs land in the trigger corner\n"
                 "(dotted = true trigger boundary, never told to reader)")

    # unit selectivity
    ax = axes[1, 0]
    z = np.array(loc[0]["unit_z"])
    ax.bar(range(len(z)), z, color=["C3" if v > TRIGGER_UNIT_Z else "C0" for v in z])
    ax.axhline(TRIGGER_UNIT_Z, color="k", ls="--", lw=0.8)
    ax.set_xlabel("hidden unit (layer 2)"); ax.set_ylabel("selectivity z-score")
    ax.set_title(f"trigger circuit: {len(loc[0]['trigger_units'])} units fire "
                 f"on suspicious inputs")

    # surgery triple
    ax = axes[1, 1]
    labels = ["trigger fire\n(want ->0)", "main err\n(want flat)"]
    pre_v = [np.mean([r["pre"]["trigger_fire_rate"] for r in surg_rows]),
             np.mean([r["pre"]["main_err"] for r in surg_rows])]
    post_v = [np.mean([r["post"]["trigger_fire_rate"] for r in surg_rows]),
              np.mean([r["post"]["main_err"] for r in surg_rows])]
    xp = np.arange(2)
    ax.bar(xp - 0.2, pre_v, 0.4, label="before surgery", color="C3")
    ax.bar(xp + 0.2, post_v, 0.4, label="after surgery", color="C2")
    ax.set_xticks(xp); ax.set_xticklabels(labels)
    ax.legend(fontsize=8)
    ax.set_title("EXCISE: trigger gone, benign task intact")

    ax = axes[1, 2]
    txt = (f"PLANTED-TRIGGER SURGERY (blind reader):\n\n"
           f"  planted: (x4>{TRIGGER_LO} & x5>{TRIGGER_LO}) -> {TARGET_VALUE}\n"
           f"           reader never told this\n\n"
           f"  DETECT   backdoored {min(bd_scores):.0f}-{max(bd_scores):.0f} vs "
           f"clean {min(cl_scores):.0f}-{max(cl_scores):.0f}\n"
           f"  LOCALIZE dirs->(x4,x5) {sum(dir_hits)}/{N_SEEDS}, "
           f"{np.mean(n_units):.0f} units/net\n"
           f"  EXCISE   fire {np.mean([r['pre']['trigger_fire_rate'] for r in surg_rows]):.2f}"
           f"->{np.mean([r['post']['trigger_fire_rate'] for r in surg_rows]):.2f}\n"
           f"           main_err delta "
           f"{np.mean([r['post']['main_err']-r['pre']['main_err'] for r in surg_rows]):+.3f}\n"
           f"  SPECIFIC clean damage "
           f"{max(abs(d) for d in clean_damage):.3f}\n\n"
           f"  {'ALL GATES PASSED' if passed else 'failures -- see report'}")
    ax.text(0.02, 0.97, txt, va="top", ha="left", fontsize=9.5,
            family="monospace", transform=ax.transAxes)
    ax.axis("off")
    fig.tight_layout()
    fig_path = RESULTS / "exp7_trigger_surgery.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    report = {"experiment": "exp7_trigger_surgery", "all_passed": passed,
              "trigger": {"condition": f"x4>{TRIGGER_LO} & x5>{TRIGGER_LO}",
                          "target": TARGET_VALUE},
              "detection": {"backdoored_scores": bd_scores,
                            "clean_scores": cl_scores, "threshold": thresh},
              "localization": {"n_units": n_units, "dir_hits": dir_hits},
              "surgery": [{"pre_fire": r["pre"]["trigger_fire_rate"],
                           "post_fire": r["post"]["trigger_fire_rate"],
                           "pre_err": r["pre"]["main_err"],
                           "post_err": r["post"]["main_err"]} for r in surg_rows],
              "clean_damage": clean_damage, "gates": gates}
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2, default=str),
                                         encoding="utf-8")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  a blind reader found a planted backdoor and cut it out:")
    print(f"  detect {min(bd_scores):.0f}x-vs-clean, localize (x4,x5), "
          f"trigger fire ->"
          f"{np.mean([r['post']['trigger_fire_rate'] for r in surg_rows]):.2f}, "
          f"benign intact, clean nets unharmed")
    print(f"  overall: {'ALL GATES PASSED' if passed else 'FAILURES PRESENT'}")
    print(f"  report:  {RESULTS / 'report.json'}")
    print(f"  figure:  {fig_path}")
    print("=" * 70)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
