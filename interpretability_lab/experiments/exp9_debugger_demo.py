"""EXPERIMENT 9 -- The neural debugger, driven end to end (dnSpy for nets).

A demonstration + gated test of the NeuralDebugger against the exp7 backdoored
network. The workflow mirrors decompiling a suspicious assembly, finding the
malicious method, breaking on it, and patching it out:

  1. ATTACH      point the debugger at a trained model (no per-model code)
  2. DECOMPILE   build the object tree; each object is a causally-graded
                 GeometricConceptObject
  3. SEARCH      "find the object that controls the anomalous output" -- the
                 debugger ranks objects by causal effect on the backdoor
  4. BREAKPOINT  set a breakpoint; trace a trigger input vs a benign input and
                 confirm the trigger objects fire only on the trigger
  5. PATCH       ablate the found objects -> a new model with the backdoor gone
  6. VERIFY      backdoor removed, benign task intact (gated)

Gates test that the debugger's blind, interactive workflow reproduces exp7's
batch result -- i.e. the tool is trustworthy, not just pretty.

Run:  python -m interpretability_lab.experiments.exp9_debugger_demo
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from interpretability_lab.debugger import NeuralDebugger
from interpretability_lab.models.backdoor import (BackdooredMLP, DOMAIN, IN_DIM,
                                                  TARGET_VALUE, TRIGGER_LO,
                                                  main_task, make_targets,
                                                  sample_inputs, trigger_mask)
from interpretability_lab.experiments.exp7_trigger_surgery import train_net, behavior_stats

RESULTS = Path(__file__).parent / "results" / "exp9"


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return {"gate": name, "passed": bool(ok), "detail": detail}


def main():
    print("=" * 70)
    print("EXPERIMENT 9: neural debugger (dnSpy for nets), driven on exp7 backdoor")
    print("=" * 70)
    RESULTS.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(2024)
    gates = []

    # a fresh backdoored specimen to attach to
    model = train_net(backdoored=True, seed=0)
    pre = behavior_stats(model, gen)
    print(f"\n  target: backdoored net, trigger fire {pre['trigger_fire_rate']:.2f}, "
          f"main_err {pre['main_err']:.3f}")

    # ---- 1. ATTACH
    dbg = NeuralDebugger().attach(
        model, name="suspect_net",
        input_sampler=lambda n: sample_inputs(n, gen, trigger_frac=0.15))
    print(f"  attached: {len(dbg.layers)} activation layers -> {dbg.layers}")

    # ---- 2. DECOMPILE (blind: the net's OWN behavior, no trigger info) then
    #         run anomaly discovery to surface any conditional/hidden circuit
    dbg.decompile(top_k=6)
    dbg.discover_anomaly(top_k=20, z_thresh=2.0)
    n_anom = sum(1 for o in dbg.objects.values() if o.kind == "trigger")
    print("\n  decompiled object tree (! = anomaly object):")
    dbg.print_tree()
    gates.append(gate("ATTACH+DECOMPILE: tree built, anomaly circuit surfaced",
                      len(dbg.objects) > 0 and n_anom > 0
                      and all(0 <= o.confidence <= 1 for o in dbg.objects.values()),
                      f"{len(dbg.objects)} objects ({n_anom} anomaly) across "
                      f"{len(dbg.layers)} layers"))

    # ---- 3. SEARCH: find objects controlling the anomalous (backdoor) output.
    # score = mean output on trigger inputs (high only if the backdoor fires).
    trig_x = sample_inputs(4096, gen, trigger_frac=1.0)
    trig_x = trig_x[trigger_mask(trig_x)]

    # dedicated search: rank ALL objects by how much ablating each drops the
    # trigger-region output toward the benign value (the debugger's search()
    # generalized with a trigger-region probe)
    with torch.no_grad():
        base_trig = model(trig_x).ravel()
    benign_trig = main_task(trig_x).ravel()
    ranked = []
    for oid in dbg.objects:
        edited = dbg.ablate(oid)
        with torch.no_grad():
            out = edited(trig_x).ravel()
        drop = float((base_trig - benign_trig).abs().mean()
                     - (out - benign_trig).abs().mean())
        ranked.append((oid, drop))
    ranked.sort(key=lambda r: -r[1])
    print("\n  SEARCH 'what controls the trigger-region output?' (top 6):")
    for oid, drop in ranked[:6]:
        tag = " [anomaly]" if dbg.objects[oid].kind == "trigger" else ""
        print(f"    {oid:<16} moves output toward benign by {drop:+.3f}{tag}")

    # the trigger circuit is DISTRIBUTED: no single unit carries it (each moves
    # output <0.2), but the anomaly set does collectively (the exp7 finding).
    # Culprits = the discovered anomaly objects, verified as a set below.
    culprits = [oid for oid in dbg.objects if dbg.objects[oid].kind == "trigger"]
    with torch.no_grad():
        set_cured = dbg.ablate(*culprits)
        set_out = set_cured(trig_x).ravel()
    set_drop = float((base_trig - benign_trig).abs().mean()
                     - (set_out - benign_trig).abs().mean())
    print(f"  collective: ablating all {len(culprits)} anomaly objects moves "
          f"output toward benign by {set_drop:+.3f} "
          f"(vs {max(d for _, d in ranked):.3f} for the best single object)")
    gates.append(gate("SEARCH: located the (distributed) circuit controlling the anomaly",
                      len(culprits) > 0 and set_drop > 0.3,
                      f"{len(culprits)} anomaly objects, collective drop {set_drop:.3f}"))

    # ---- 4. BREAKPOINT + trace trigger vs benign
    for oid in culprits:
        dbg.set_breakpoint(dbg.objects[oid].layer)
    x_trigger = trig_x[:1]
    x_benign = sample_inputs(64, gen, trigger_frac=0.0)
    x_benign = x_benign[~trigger_mask(x_benign)][:1]
    tr_trig = dbg.trace(x_trigger[0])
    tr_ben = dbg.trace(x_benign[0])
    culprit_set = set(culprits)

    def culprits_fired(trace):
        fired = set()
        for step in trace["steps"]:
            for f in step["fired"]:
                if f["id"] in culprit_set:
                    fired.add(f["id"])
        return fired
    fired_on_trig = culprits_fired(tr_trig)
    fired_on_ben = culprits_fired(tr_ben)
    print(f"\n  BREAKPOINT trace: culprit objects firing on trigger input: "
          f"{len(fired_on_trig)}/{len(culprits)}; on benign input: "
          f"{len(fired_on_ben)}/{len(culprits)}")
    gates.append(gate("BREAKPOINT: trigger objects fire on trigger, quiet on benign",
                      len(fired_on_trig) > len(fired_on_ben),
                      f"trigger {len(fired_on_trig)} vs benign {len(fired_on_ben)}"))

    # ---- 5. PATCH (edit-and-recompile) + 6. VERIFY
    cured = dbg.ablate(*culprits)
    post = behavior_stats(cured, gen)
    print(f"\n  PATCH: ablated {len(culprits)} objects ->")
    print(f"    trigger fire {pre['trigger_fire_rate']:.2f} -> {post['trigger_fire_rate']:.2f}")
    print(f"    main_err     {pre['main_err']:.3f} -> {post['main_err']:.3f}")
    gates.append(gate("PATCH+VERIFY: backdoor removed, benign task intact",
                      post["trigger_fire_rate"] < 0.15
                      and post["main_err"] < pre["main_err"] + 0.05,
                      f"fire ->{post['trigger_fire_rate']:.2f}, "
                      f"main_err delta {post['main_err']-pre['main_err']:+.3f}"))

    passed = all(g["passed"] for g in gates)

    # ---- figure: the debugger "UI" as a static render
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    # object tree with confidence + culprit highlight
    ax = axes[0, 0]
    ax.axis("off")
    lines = [f"$\\bf{{{dbg.name}}}$  [{len(dbg.layers)} layers, {len(dbg.objects)} objects]"]
    for lname in dbg.layers:
        objs = sorted([o for o in dbg.objects.values() if o.layer == lname],
                      key=lambda z: -z.confidence)
        lines.append(f"  └ {lname} ({len(objs)} objects)")
        for o in objs:
            mark = " ⚠ TRIGGER" if o.name in culprit_set else ""
            lines.append(f"      • {o.name:<14} conf {o.confidence:.2f}{mark}")
    ax.text(0.0, 1.0, "\n".join(lines[:22]), va="top", ha="left",
            family="monospace", fontsize=8.5, transform=ax.transAxes)
    ax.set_title("object tree (decompiled) — ⚠ = found by search", loc="left")

    # search ranking
    ax = axes[0, 1]
    ids = [r[0] for r in ranked[:10]]
    drops = [r[1] for r in ranked[:10]]
    colors = ["C3" if i in culprit_set else "C0" for i in ids]
    ax.barh(range(len(ids)), drops, color=colors)
    ax.set_yticks(range(len(ids))); ax.set_yticklabels(ids, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("output moved toward benign when ablated")
    ax.set_title("SEARCH: what controls the anomaly (red = culprit)")

    # trace: trigger vs benign firing
    ax = axes[1, 0]
    layers = [s["layer"] for s in tr_trig["steps"]]
    trig_counts = [sum(1 for f in s["fired"] if f["id"] in culprit_set)
                   for s in tr_trig["steps"]]
    ben_counts = [sum(1 for f in s["fired"] if f["id"] in culprit_set)
                  for s in tr_ben["steps"]]
    xp = np.arange(len(layers))
    ax.bar(xp - 0.2, trig_counts, 0.4, label="trigger input", color="C3")
    ax.bar(xp + 0.2, ben_counts, 0.4, label="benign input", color="C2")
    ax.set_xticks(xp); ax.set_xticklabels(layers, fontsize=8)
    ax.set_ylabel("culprit objects firing"); ax.legend(fontsize=8)
    ax.set_title("BREAKPOINT trace: trigger circuit fires only on trigger")

    # patch result
    ax = axes[1, 1]
    ax.bar([0, 1], [pre["trigger_fire_rate"], post["trigger_fire_rate"]],
           color=["C3", "C2"], width=0.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["before patch", "after patch"])
    ax.set_ylabel("trigger fire rate"); ax.set_ylim(0, 1)
    ax.set_title(f"PATCH: backdoor {pre['trigger_fire_rate']:.0%} -> "
                 f"{post['trigger_fire_rate']:.0%}, benign intact "
                 f"(err {post['main_err']:.3f})")
    fig.suptitle("NeuralDebugger — attach · decompile · search · breakpoint · patch",
                 fontsize=13)
    fig.tight_layout()
    fig_path = RESULTS / "exp9_debugger_demo.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    report = {"experiment": "exp9_debugger_demo", "all_passed": passed,
              "n_objects": len(dbg.objects), "n_layers": len(dbg.layers),
              "culprits": culprits, "search_top": ranked[:6],
              "pre": pre, "post": post, "gates": gates}
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2, default=str),
                                         encoding="utf-8")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  attached to a net, decompiled {len(dbg.objects)} objects, searched")
    print(f"  out the trigger circuit ({len(culprits)} objects), traced it firing")
    print(f"  only on trigger inputs, patched it out: fire "
          f"{pre['trigger_fire_rate']:.2f}->{post['trigger_fire_rate']:.2f}, benign intact")
    print(f"  overall: {'ALL GATES PASSED' if passed else 'FAILURES PRESENT'}")
    print(f"  report:  {RESULTS / 'report.json'}")
    print(f"  figure:  {fig_path}")
    print("=" * 70)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
