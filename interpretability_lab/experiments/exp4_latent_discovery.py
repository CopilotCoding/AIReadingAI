"""EXPERIMENT 4 -- Blind latent-variable discovery.

Exp3 recovered the equation but failed its causal steering gate (R^2 = 0.02):
probe directions decode s = a+b perfectly yet are not what the circuit
listens to. Decodability is not causal reality.

Exp4 v1 then looked for a flat 2D linear subspace of layer-1 activations
carrying the state -- and correctly REFUSED: no such subspace exists
(rank-2 bottleneck preserves only 0.95; ~8 linear dims needed for 0.999).
Diagnosis showed why: the network keeps a 2D ABSTRACT state embedded on a
CURVED surface in activation space. Latent variables need not be linear
subspaces -- visible already at 10K params.

Final pipeline (reader is blind; ground truth enters only at grading):

  1. DISCOVER the function-level funnel where it lives: SVD of per-input
     output gradients. Spectral-gap rule picks k. Verify causally by
     projecting inputs onto the k-dim subspace: function must survive.
  2. MEASURE the state's embedding at layer 1: minimal LINEAR activation
     bottleneck preserving the function. k_lin >> k_func = curved embedding.
  3. STORY: sparse symbolic fit in the discovered coordinates; assemble
     back to input space.
  4. CAUSAL STEERING along the curved representation's tangent: per-point
     pushforward vectors v(x) = m1(x) o (W1 u_i). The story's own
     predictions must match. Control: pushforward of the discovered
     UNUSED direction must do ~nothing.
  5. NEGATIVE CONTROL: untrained net -> reader must refuse.

Run:  python -m interpretability_lab.experiments.exp4_latent_discovery
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import subspace_angles

from interpretability_lab.extraction import latent_discovery as ld
from interpretability_lab.extraction import symbolic_reader as sym
from interpretability_lab.experiments.exp3_composition import monomial_basis
from interpretability_lab.models.tiny import TinyMLP2

RESULTS = Path(__file__).parent / "results" / "exp4"
SPECIMEN = (Path(__file__).parent.parent / "corpus" / "data" / "exp3"
            / "exp3_3a_compose_seed0")
DOMAIN = (-2.0, 2.0)
H1 = H2 = 96


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return {"gate": name, "passed": bool(ok), "detail": detail}


def sample(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, generator=g) * (DOMAIN[1] - DOMAIN[0]) + DOMAIN[0]


def pieces(model, X):
    """Analytic forward pieces: h1, layer-1 mask, dy/dh1, dy/dx."""
    Xn = X.numpy().astype(np.float64)
    W1 = model.hidden1.weight.detach().numpy().astype(np.float64)
    b1 = model.hidden1.bias.detach().numpy().astype(np.float64)
    W2 = model.hidden2.weight.detach().numpy().astype(np.float64)
    b2 = model.hidden2.bias.detach().numpy().astype(np.float64)
    w3 = model.out.weight.detach().numpy().ravel().astype(np.float64)
    pre1 = Xn @ W1.T + b1
    m1 = pre1 > 0
    H = np.maximum(pre1, 0.0)
    m2 = (H @ W2.T + b2) > 0
    J = (m2 * w3) @ W2                                      # dy/dh1
    Gx = np.einsum("nh,nh,hd->nd", J, m1.astype(np.float64), W1)  # dy/dx
    return {"H": H, "m1": m1, "J": J, "Gx": Gx, "W1": W1}


def jac_fn(model, X):
    p = pieces(model, X)
    return p["H"], p["J"]


def z_monomials(k, max_deg=2):
    import itertools
    basis = {}
    for exps in itertools.product(range(max_deg + 1), repeat=k):
        if sum(exps) > max_deg:
            continue
        parts = [f"z{i+1}" + (f"^{p}" if p > 1 else "")
                 for i, p in enumerate(exps) if p > 0]
        name = "*".join(parts) if parts else "1"
        basis[name] = (lambda e: lambda Z: np.prod(
            [Z[:, i] ** p for i, p in enumerate(e)], axis=0))(exps)
    return basis


def eval_story(fit, basis, Z):
    y = np.zeros(len(Z))
    for n, c in fit["coefficients"].items():
        y += c * basis[n](Z)
    return y


def discover(model, seed=11):
    """Blind funnel discovery + story. Returns (claims, refusal_reason)."""
    X = sample(8192, seed)
    p = pieces(model, X)
    active = ld.input_active_subspace(p["Gx"])
    in_dim = X.shape[1]
    if active["k"] >= in_dim:
        # no dimensionality reduction; a compact story may still exist,
        # but for a random PWL function it will not -- try and refuse.
        active["basis"] = np.eye(in_dim)
        active["k"] = in_dim
    preservation = ld.input_projection_check(model, X, active["basis"])
    if preservation < 0.999:
        return None, (f"projection onto candidate {active['k']}D funnel does "
                      f"not preserve the function (R^2 = {preservation:.4f})")
    Xn = X.numpy().astype(np.float64)
    mux = Xn.mean(0)
    Z = (Xn - mux) @ active["basis"].T
    with torch.no_grad():
        net_y = model(X).numpy().ravel().astype(np.float64)
    basis = z_monomials(active["k"])
    fit = sym.fit_sparse(Z, net_y, basis=basis)
    if fit["rel_rmse"] > 0.05:
        return None, (f"no compact story in {active['k']} discovered coords "
                      f"(best rel RMSE {fit['rel_rmse']:.3f})")
    return {"X": X, "p": p, "active": active, "mux": mux, "Z": Z,
            "net_y": net_y, "fit": fit, "basis": basis,
            "preservation": preservation}, None


def main():
    print("=" * 70)
    print("EXPERIMENT 4: blind latent discovery on exp3's network")
    print("=" * 70)
    RESULTS.mkdir(parents=True, exist_ok=True)

    model = TinyMLP2(3, H1, H2)
    model.load_state_dict(torch.load(SPECIMEN / "weights.pt", weights_only=True))
    model.eval()
    print(f"\n  loaded specimen: {SPECIMEN.name} (reader did not train it)")

    claims, refusal = discover(model)
    gates = []
    if refusal:
        print(f"  reader refused: {refusal}")
        gates.append(gate("discovery produced a validated claim", False, refusal))
        (RESULTS / "report.json").write_text(json.dumps(
            {"experiment": "exp4_latent_discovery", "all_passed": False,
             "refusal": refusal}, indent=2), encoding="utf-8")
        return 1

    active, fit = claims["active"], claims["fit"]
    sv = np.array(active["singular_values"])
    print(f"  input-gradient spectrum (normalized): "
          f"{np.round(sv / sv[0], 4).tolist()}")
    print(f"  DISCOVERED: function funnels through a {active['k']}D subspace "
          f"of input space; projection preservation R^2 = "
          f"{claims['preservation']:.6f}")
    gates.append(gate("funnel discovered and causally validated by projection",
                      active["k"] < 3 and claims["preservation"] >= 0.999,
                      f"k = {active['k']}, preservation {claims['preservation']:.6f}"))

    for i in range(active["k"]):
        print(f"  z{i+1} direction: "
              f"{ld.pretty_direction(active['basis'][i], ['a', 'b', 'c'])}")

    # ---- grading vs ground truth (first consultation): span{a+b, c}
    truth = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).T
    ang = np.degrees(subspace_angles(active["basis"].T, truth))
    ok = float(ang.max()) < 5.0
    gates.append(gate("discovered subspace matches span{a+b, c}", ok,
                      f"principal angles {np.round(ang, 3).tolist()} deg"))

    # the discarded direction, in input terms
    full_basis = np.linalg.svd(claims["p"]["Gx"], full_matrices=False)[2]
    unused = full_basis[active["k"]]
    print(f"  discarded direction: "
          f"{ld.pretty_direction(unused, ['a', 'b', 'c'])}   "
          f"(reader claims the network ignores this)")

    # ---- embedding measurement at layer 1
    Xe = claims["X"]
    emb = ld.used_subspace(model, "act1", Xe, jac_fn, k_max=16, r2_floor=0.999)
    curve = {c["k"]: c["r2"] for c in emb["curve"]}
    k_lin = emb["k"]
    r2_at_kfunc = curve.get(active["k"], 0.0)
    print(f"  layer-1 embedding: linear bottleneck needs k = {k_lin} for 0.999 "
          f"(rank-{active['k']} gives only {r2_at_kfunc:.4f})")
    print(f"  finding: {active['k']}D abstract state stored on a CURVED "
          f"~{k_lin}D-linear surface in 96D activation space")
    gates.append(gate("state's activation embedding measured (curved, not flat)",
                      k_lin is not None and r2_at_kfunc < 0.999,
                      f"k_lin = {k_lin} vs k_func = {active['k']}"))

    # ---- story and assembly
    print(f"  story in discovered coords: {fit['expression']}  "
          f"(rel RMSE {fit['rel_rmse']:.2e})")
    story_y = eval_story(fit, claims["basis"], claims["Z"])
    Xn = claims["X"].numpy().astype(np.float64)
    assembled = sym.fit_sparse(Xn, story_y, basis=monomial_basis(3))
    print(f"  assembled to input space (verbatim): {assembled['expression']}")

    # significance filter, stated openly: the assembly transcribes the
    # network's own small imperfections (e.g. a 0.012*c residue, ~0.4% of
    # signal). Terms contributing < 1% of the signal's std are reported,
    # then pruned and the rest refit.
    b3 = monomial_basis(3)
    y_std = float(story_y.std())
    keep, dropped = [], []
    for n in assembled["support"]:
        contrib = abs(assembled["coefficients"][n]) * float(b3[n](Xn).std())
        (keep if contrib >= 0.01 * y_std else dropped).append(n)
    if dropped:
        sub = {n: b3[n] for n in keep}
        assembled = sym.fit_sparse(Xn, story_y, basis=sub, max_terms=len(keep))
        print(f"  pruned insignificant terms (<1% of signal): {dropped}")
        print(f"  assembled to input space (significant):  {assembled['expression']}")
    want = {"a*c": 1.0, "b*c": 1.0}
    if set(assembled["support"]) == set(want):
        worst = max(abs(assembled["coefficients"][n] - v) for n, v in want.items())
        ok, detail = worst < 0.02, f"(a+b)*c recovered, worst abs coef err {worst:.4f}"
    else:
        ok, detail = False, f"support {assembled['support']} != ['a*c', 'b*c']"
    gates.append(gate("story assembles to the generating rule", ok, detail))

    # ---- causal steering along the curved representation's tangent
    Xs = sample(1024, 44)
    ps = pieces(model, Xs)
    Zs = (Xs.numpy().astype(np.float64) - claims["mux"]) @ active["basis"].T
    preds, obs, per_coord = [], [], []
    for i in range(active["k"]):
        delta = 0.25 * float(claims["Z"][:, i].std())
        field = ps["m1"] * (ps["W1"] @ active["basis"][i])   # pushforward (N, 96)
        dy = ld.steer_activation(model, "act1", Xs, field, delta)
        Zpush = Zs.copy()
        Zpush[:, i] += delta
        pred = eval_story(fit, claims["basis"], Zpush) - eval_story(
            fit, claims["basis"], Zs)
        r2_i = 1 - float(((dy - pred) ** 2).sum()) / \
            (float(((pred - pred.mean()) ** 2).sum()) or 1e-12)
        per_coord.append({"coord": i + 1, "delta": delta, "r2": r2_i})
        print(f"  steering z{i+1} tangent by {delta:.3f}: story-predicted vs "
              f"observed dy R^2 = {r2_i:.4f}")
        preds.append(pred)
        obs.append(dy)
    pooled_pred = np.concatenate(preds)
    pooled_obs = np.concatenate(obs)
    r2_steer = 1 - float(((pooled_obs - pooled_pred) ** 2).sum()) / \
        float(((pooled_pred - pooled_pred.mean()) ** 2).sum())
    gates.append(gate("steering discovered state matches the story's predictions",
                      r2_steer > 0.9, f"pooled R^2 = {r2_steer:.4f}"))

    # ---- control: pushforward of the discovered-unused direction
    delta0 = 0.25 * float(claims["Z"][:, 0].std())
    field_null = ps["m1"] * (ps["W1"] @ unused)
    dy_null = ld.steer_activation(model, "act1", Xs, field_null, delta0)
    ratio = float(np.median(np.abs(dy_null)) /
                  (np.median(np.abs(pooled_obs)) + 1e-12))
    print(f"  control: steering the discarded direction's tangent moves output "
          f"{ratio:.1%} as much")
    gates.append(gate("output indifferent to the discovered-unused direction",
                      ratio < 0.2, f"ratio = {ratio:.1%}"))

    # ---- negative control
    torch.manual_seed(999)
    rand_model = TinyMLP2(3, H1, H2)
    _, rand_refusal = discover(rand_model, seed=12)
    print(f"  negative control (untrained net): "
          f"{'reader refused -- ' + rand_refusal if rand_refusal else 'READER CLAIMED A LATENT'}")
    gates.append(gate("reader refuses on an untrained network",
                      rand_refusal is not None, rand_refusal or "claimed"))

    passed = all(g["passed"] for g in gates)

    # ---- figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax = axes[0, 0]
    ax.bar(range(1, len(sv) + 1), sv / sv[0], color=["C0"] * active["k"]
           + ["C3"] * (len(sv) - active["k"]))
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(sv) + 1))
    ax.set_xlabel("input-gradient component"); ax.set_ylabel("singular value (norm.)")
    ax.set_title(f"funnel discovery: {active['k']} used directions (blue),\n"
                 f"1 discarded (red, {sv[active['k']] / sv[0]:.1%})")

    ax = axes[0, 1]
    ks = [c["k"] for c in emb["curve"]]
    rs = [c["r2"] for c in emb["curve"]]
    ax.plot(ks, rs, "s-", label="linear bottleneck at h1")
    ax.axhline(0.999, color="k", ls=":", lw=0.8)
    ax.axvline(active["k"], color="C2", ls="--", lw=1,
               label=f"abstract state dim = {active['k']}")
    ax.set_ylim(min(rs) - 0.02, 1.004)
    ax.set_xlabel("bottleneck dimension k"); ax.set_ylabel("preservation R^2")
    ax.legend(fontsize=8)
    ax.set_title(f"curved embedding: 2D state needs k = {k_lin} linear dims")

    ax = axes[0, 2]
    B = active["basis"]
    Bn = B / np.abs(B).max(axis=1, keepdims=True)
    im = ax.imshow(Bn, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["a", "b", "c"])
    ax.set_yticks(range(active["k"]))
    ax.set_yticklabels([f"z{i+1}" for i in range(active["k"])])
    for i in range(active["k"]):
        for j in range(3):
            ax.text(j, i, f"{Bn[i, j]:+.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("discovered state directions (input terms)\n"
                 f"span matches {{a+b, c}} to {ang.max():.2f} deg")

    ax = axes[1, 0]
    ax.scatter(pooled_pred, pooled_obs, s=5, alpha=0.4)
    lim = max(np.abs(pooled_pred).max(), np.abs(pooled_obs).max())
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax.set_xlabel("story-predicted dy"); ax.set_ylabel("observed dy")
    ax.set_title(f"tangent steering of discovered state (R^2 = {r2_steer:.3f})\n"
                 f"exp3's fixed-direction attempt scored 0.02")

    ax = axes[1, 1]
    ax.hist(np.abs(pooled_obs), bins=40, alpha=0.6, label="steer used state")
    ax.hist(np.abs(dy_null), bins=40, alpha=0.6, label="steer discarded dir")
    ax.set_xlabel("|dy|"); ax.legend(fontsize=8)
    ax.set_title(f"used vs unused (control ratio {ratio:.1%})")

    ax = axes[1, 2]
    txt = (f"BLIND CLAIMS (truth never consulted):\n\n"
           f"1. function funnels through {active['k']}D:\n"
           f"   z1 = {ld.pretty_direction(B[0], ['a','b','c'])}\n"
           f"   z2 = {ld.pretty_direction(B[1], ['a','b','c'])}\n"
           f"   ignores {ld.pretty_direction(unused, ['a','b','c'])}\n"
           f"2. state embedded CURVED at h1\n"
           f"   (k_lin = {k_lin} vs abstract {active['k']})\n"
           f"3. story: {fit['expression'][:38]}...\n"
           f"4. assembled: {assembled['expression']}\n\n"
           f"GRADE vs y = (a+b)*c:\n"
           f"   {'ALL GATES PASSED' if passed else 'failures -- see report'}")
    ax.text(0.02, 0.97, txt, va="top", ha="left", fontsize=9, family="monospace",
            transform=ax.transAxes)
    ax.axis("off")
    fig.tight_layout()
    fig_path = RESULTS / "exp4_latent_discovery.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    report = {"experiment": "exp4_latent_discovery", "all_passed": passed,
              "funnel": {"k": active["k"],
                         "singular_values": active["singular_values"],
                         "basis": active["basis"].tolist(),
                         "preservation": claims["preservation"],
                         "principal_angles_deg": ang.tolist()},
              "embedding": {"k_lin": k_lin, "curve": emb["curve"],
                            "r2_at_k_func": r2_at_kfunc},
              "story": fit, "assembled": assembled,
              "steering": {"per_coord": per_coord, "pooled_r2": r2_steer},
              "control_ratio": ratio, "gates": gates}
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2, default=str),
                                         encoding="utf-8")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  discovered: {active['k']}D state {{~(a+b), ~c}}, curved embedding "
          f"(k_lin={k_lin}), story assembles to {assembled['expression']}")
    print(f"  overall: {'ALL GATES PASSED' if passed else 'FAILURES PRESENT'}")
    print(f"  report:  {RESULTS / 'report.json'}")
    print(f"  figure:  {fig_path}")
    print("=" * 70)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
