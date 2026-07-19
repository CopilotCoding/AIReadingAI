# interpretability_lab

An experimental instrument for reading learned computation back out of neural
networks. Bottom-up: prove extraction works on networks where ground truth is
known, climb the parameter ladder until it breaks, and treat the breaking
point as data.

**Core axiom: we find what we find, not what we want to see.** Whether it's
human-readable, transformable into something human-readable, or irreducibly
alien — we will find out, and the answer is the result either way. (The
ladder has enforced this four times already: distributed multiplication,
probes-that-lie, curved latents, windowed induction — every one a refuted
human expectation kept on the record.)

Two success criteria, never conflated:

- **Behavioral equivalence** — the extracted description matches the network's
  outputs.
- **Algorithmic recovery** — the extracted description refers to the actual
  internal mechanism, verified **causally**: interventions predicted by the
  description must change the network's output the way the description says.

## Layout

```
interpretability_lab/
  models/          tiny architectures + training loops
  hooks/           activation capture for arbitrary PyTorch models
  extraction/      readers that turn parameters/activations into descriptions
  interventions/   ablation & causal verification
  corpus/          every trained model saved with its ground-truth rule
                   (training data for the Phase 3 interpreter network)
  experiments/     runnable experiments, each with hard PASS/FAIL gates
  activations/ features/ geometry/ visualization/   (later rungs)
```

## Experiment 0 — read-back at the bottom rung  ✅ ALL PASSED

`python -m interpretability_lab.experiments.exp0_readback`

| Rung | Model | Params | Ground truth | Recovered | Result |
|------|-------|--------|--------------|-----------|--------|
| 0a | 1 neuron | 2 | y = 3x + 5 | `y = 3*x + 5` (max coef err 1.4e-6) | PASS |
| 0b | 1 neuron | 3 | y = 2a − 7b + 1 | `y = 2*a - 7*b + 1` (max coef err 5.7e-6) | PASS |
| 0c | 2-2-1 ReLU MLP | 9 | XOR(a,b) | `XOR via [h0=OR, h1=AND]`, causally verified | PASS |

Gates for 0c (all required):
1. network learned XOR (behavioral)
2. extracted circuit composes to XOR
3. boolean abstraction reproduces the network's output when substituted in
4. **ablating each hidden unit flips exactly the inputs the extracted story
   predicts** (mechanistic)

### Finding worth recording

The first run of 0c **failed the mechanistic gates**, and the failure was
informative: the network did not learn clean on/off logic gates. Its second
hidden unit fired *weakly* (~0.07) for inputs (0,1)/(1,0) and *strongly*
(~1.25) for (1,1) — a graded unit. A naive reader that binarizes at
activation > 0 named it OR; magnitude-wise it functions as AND. The network's
actual mechanism is the counting solution:

```
out ≈ 1.12·(a+b) − 1.85·ReLU(a+b−1)      "exactly one input active"
```

Fix (in `extraction/logic_reader.py`): the reader searches over per-unit
activation thresholds and accepts only a gate naming whose reconstruction —
gate indicator × on-value, pushed through the real output weights —
reproduces the network's thresholded output on every input. With the searched
thresholds the story `OR(a,b) AND NOT AND(a,b)` validates and its ablation
predictions match observation exactly.

Lesson, stated once and kept: **ReLU units are not inherently boolean; any
gate-level description is a claim that must be validated against magnitudes
and causally, or it will silently misdescribe the mechanism.** This is the
distributed/graded-representation problem already visible at 9 parameters.

Artifacts: `experiments/results/exp0/report.json`, `exp0_readback.png`.
Convergence note: 2-2-1 XOR converged on seed 8 of 30 tried — init fragility
at minimal width, logged in the report.

## Experiment 1 — function recovery at ~100 params  ✅ ALL PASSED

`python -m interpretability_lab.experiments.exp1_function_recovery`

At this rung the network no longer *contains* the rule — it contains a
piecewise-linear approximation. The reader therefore makes two separated,
individually gated claims:

- **Mechanistic**: the exact PWL structure (knots, per-segment slopes) read
  analytically off the weights, rebuilt as a segment table via an independent
  code path, required to match the forward pass (max diff ~1e-6 achieved).
- **Symbolic**: sparse basis expression fitted to the *extracted mechanism*
  (never the training data), required to recover the generating rule. The
  residual between the two is reported as the **mechanism gap**.

| Task | Params | Ground truth | Recovered | Pieces | Gap (rel RMSE) | Result |
|------|--------|--------------|-----------|--------|----------------|--------|
| 1a | 97 | y = 3x² + 5x + 7 | `y = 7.001 + 5*x + 3*x^2` (worst coef err 0.02%) | 25 | 3.3e-3 | PASS |
| 1b | 97 | y = sin(x) | `y = 1*sin(x)` | 26 | 5.6e-3 | PASS |
| 1c | 97 | y = \|x\| | `y = 1*\|x\|` | 28 | 5.7e-4 | PASS |

Causal gate: every knot-bearing unit's ablation affected only its predicted
active region (23/23, 25/25, 27/27 units consistent).

### Findings worth recording

1. **Greedy symbolic selection failed first.** On [-3,3], cos(x) ≈ 1 − x²/2 +
   x⁴/24 is strongly correlated with any quadratic; greedy OMP grabbed
   `cos(x)` first and patched with `x^4`, producing a behaviorally-decent
   (gap 8.4e-3) but *wrong* rule — caught by the support gate. Fix in
   `extraction/symbolic_reader.py`: exhaustive search over all supports
   ≤ 4 terms (385 lstsq fits) with an explicit parsimony rule — smallest
   support within 1.5× of the best achievable error. Lesson: **on a bounded
   domain, basis families mimic each other; greedy selection can lock in the
   wrong family while matching behavior well.** Behavioral fit alone would
   never have caught this.

2. **The |x| network localized the kink.** Its knots cluster tightly at x=0
   (visible in the figure) — the network's internal structure mirrors the
   discontinuity in the target's derivative. Mechanism gap there is 6× smaller
   than for genuinely-curved targets, consistent with |x| being exactly
   representable by ReLUs while quadratics are not.

3. **Unit census** (32 hidden each): ~24–27 knot-bearing, 2–3 always-active
   (fold into the base line), 3–5 dead. Parameter count overstates mechanism
   size; the effective computation is the knot-bearing subset.

Artifacts: `experiments/results/exp1/report.json`, `exp1_function_recovery.png`.

## Experiment 2 — neural calculator stage 1 (~1K params)  ✅ ALL PASSED

`python -m interpretability_lab.experiments.exp2_calculator_stage1`

Two-input arithmetic on [-2,2]², 2→256→1 ReLU nets (1025 params). In 2D the
mechanism is a **polyhedral complex**: hidden-unit boundary lines tile the
plane into cells, each carrying an affine map. The reader extracts per-unit
tables (boundary line, contribution vector, census) and reconstructs the
forward pass by affine assembly; symbolic recovery runs on the reconstruction.

| Task | Ground truth | Recovered | Regions | Gap (rel RMSE) | Result |
|------|--------------|-----------|---------|----------------|--------|
| 2a | y = a + b | `y = 1*a + 1*b` (coef err 0.0000) | 9569 | 8.4e-4 | PASS |
| 2b | y = a − b | `y = 1*a − 1*b` (coef err 0.0000) | 9280 | 8.4e-4 | PASS |
| 2c | y = a · b | `y = 1*a*b` (coef err 0.0000) | 8068 | 1.4e-3 | PASS |

Causal gate: 225/225, 216/216, 228/228 significant units affected only their
predicted active half-plane under ablation.

### Findings worth recording

1. **Nominal mechanism size is a mirage for linear targets.** The a+b network
   has ~9.5K nominal linear regions, but its gradient is within 5% of (1,1)
   over 98.5% of the domain: the cells exist, their differences cancel. The
   honest mechanism statement is "effectively affine with gradient (1,1)" —
   gated as such. Region *count* measures architecture, not computation.

2. **The textbook multiplication hypothesis was REFUTED (partially).** Theory:
   ab = ((a+b)² − (a−b)²)/4 predicts hidden units aligned with the ±diagonals.
   First run gated on "majority of contribution on diagonals" — FAILED at
   42.2%. But the uniform-orientation null puts only 22.2% in those windows:
   the diagonals are enriched **1.9× over chance**, yet the network solves
   multiplication with a *distributed mixture* of ridge orientations, not the
   clean algebraic decomposition. Gate re-founded on enrichment vs null; the
   strong hypothesis stays refuted on the record. This is the first rung
   where the network's solution is *genuinely different* from the human one.

3. **Division deferred by design.** a/b has a pole at b=0; on a domain
   excluding it, division = multiplication by a reciprocal-shaped surface.
   Worth its own rung with a domain-restriction story, not a footnote here.

Artifacts: `experiments/results/exp2/report.json`, `exp2_calculator_stage1.png`.

## Experiment 3 — composition: y = (a+b)·c at ~10K params, depth 2  ⚠️ 1 GATE FAILED (kept on record)

`python -m interpretability_lab.experiments.exp3_composition`

3→96→96→1 (9793 params). Equation recovery worked: depth-2 affine assembly
matched the forward pass (1.1e-6); symbolic recovery from the mechanism gave
`y = 1*a*c + 1*b*c` (worst coef err 0.0002); negative control (untrained
net) correctly refused (rel RMSE 0.90).

### The failure that matters: PROBES LIE

Probes decoded s = a+b from layer 1 at R² = 1.0000. But steering the probe's
min-norm direction — predicted Δy = c·δ — scored **R² = 0.017**. The probe
direction reads s from correlations; it is *not* the direction the circuit
listens to. Decodability ≠ causal reality. Kept failing on the record as the
motivation for exp4.

## Experiment 4 — blind latent-variable discovery  ✅ ALL PASSED

`python -m interpretability_lab.experiments.exp4_latent_discovery`

The question changes from "what function?" to "**what internal state
exists?**" The reader is handed exp3's network *from the corpus* (it did not
train it) and no ground truth. Its blind claims, each causally validated:

1. **Funnel discovery**: input-gradient spectrum [1.0, 0.97, **0.019**] — the
   function uses a 2D subspace of input space. Verified by projection
   intervention: R² = 0.999975. Grading: discovered span matches {a+b, c} to
   **0.03°**; discarded direction = a−b.
2. **Curved embedding (headline finding)**: the 2D abstract state is NOT a 2D
   linear subspace of activation space — a rank-2 linear bottleneck at h1
   preserves only 0.95; **k = 7 linear dims** needed. Latent variables live
   on curved manifolds, visible already at 10K params (exp4 v1 looked for a
   flat subspace and correctly *refused* — that refusal was the discovery).
3. **Story + assembly**: quadratic story in discovered coords assembles to
   `y = 1.0*a*c + 1.0*b*c` (a 0.012·c residue — the network's own
   imperfection — reported, then pruned under a stated 1%-of-signal
   significance threshold).
4. **Causal steering, done right**: steering along per-point *pushforward
   tangent* vectors of the curved representation, with predictions from the
   story itself: pooled **R² = 0.989** (exp3's fixed-direction attempt:
   0.02). Control: steering the discovered-unused direction moves the output
   1.8% as much. Negative control: refuses on an untrained net.

## Experiment 5 — attention circuit in a tiny transformer (~40K params)  ✅ ALL PASSED

`python -m interpretability_lab.experiments.exp5_transformer_induction`

2-layer × 4-head attention-only transformer (37,780 params) trained on
variable-offset repeated sequences (positional shortcuts impossible). Data
rule: match-&-copy. Task accuracy 0.999.

**Blind circuit discovery** (causal, before looking at any attention
pattern): per-head ablations found a 6-head circuit with a clean double
dissociation — keep-only-circuit 0.993, ablate-circuit 0.045.

### The textbook induction circuit does NOT exist in this model

Prev-token scores ~0.10 (uniform), induction-target (j0+1) mass ~0.02 — yet
behavior is perfect match-&-copy. Diagnostics revealed the actual mechanism,
**windowed match-&-copy**: all four L0 heads (distributed) write a fuzzy
summary of tokens ~6–8 back into each position; L1H0/H3 content-match
against that *shifted* copy, attending j0+[4,9] with **0.90 mass** (baseline
0.25) — the retrieved window contains the successor token. Verified:

- composition: ablating the L0 heads collapses the L1 window match
  (0.90 → 0.57) and accuracy (0.999 → 0.056)
- discriminating OOD contrast: separated block repeats work (**0.78**) while
  single-token accidental repeats fail (**0.19**) — textbook induction
  predicts both high; a window matcher predicts exactly this split. The
  "failed" OOD generalization is the mechanism's own signature.
- negative control: untrained transformer → reader refuses.

Fourth instance of the ladder's central pattern: **SGD found a non-human
solution** (exp2 distributed multiplication, exp3 probe/causal split, exp4
curved latent, exp5 windowed induction). Methodological notes: attention
"pattern scores" are hypothesis tests, not measurements — score the
*measured* window, not the assumed offset; and analyses must run against a
**fixed corpus specimen** (CUDA retraining drifts instance-level details
like the within-block ramp shape, which is why the OOD gate tests the
block-vs-token contrast, not the ramp).

## Compression ledger (`experiments/results/ledger.md`)

params → effective units (minimal set preserving the model's own function to
R² ≥ 0.999) → symbolic terms:

| specimen | params | effective units | sym terms | units/term |
|---|---|---|---|---|
| exp0/xor | 9 | 2/2 | 3 | 0.7 |
| exp1/quadratic | 97 | 24/32 | 3 | 8 |
| exp1/sine | 97 | 25/32 | 1 | 25 |
| exp2/add | 1025 | 127/256 | 2 | 63.5 |
| exp2/multiply | 1025 | 179/256 | 1 | **179** |
| exp3/compose | 9793 | 143/192 | 2 | 71.5 |

The ratio grows ~250× across the ladder while extraction keeps validating:
many geometric objects collapse into few conceptual operations, and the
readers still recover them. (Nominal region count is deliberately NOT used —
exp2 showed it measures architecture, not computation.)

## Corpus

`corpus/data/exp0..exp5/…` — 11 specimens so far, each `weights.pt` + `meta.json`
(architecture, ground-truth rule, recovered description, pass/fail). Grows
with every experiment; this becomes the supervised training set for the
Phase 3 weight-space interpreter.

## Ladder status

```
   2 params   equation recovered exactly        ✅ (0a)
   3 params   equation recovered exactly        ✅ (0b)
   9 params   circuit recovered + causally verified ✅ (0c)
  97 params   mechanism (PWL segment table) + rule (3x²+5x+7, sin, |x|)
              both recovered, causally checked            ✅ (1a-1c)
  1025 params  polyhedral mechanism read + rule (a+b, a−b, a·b) recovered;
               multiplication solved by the net in a NON-human way
               (distributed orientations, 1.9× diagonal bias)  ✅ (2a-2c)
  9793 params  depth-2 composition (a+b)·c: rule recovered; probe-steering
               REFUTED (probes lie, R²=0.02)                   ⚠️ (3)
  9793 params  blind discovery: 2D internal state {a+b, c} found + causally
               verified (tangent steering R²=0.99); state lives on a CURVED
               7D-linear manifold, not a flat subspace         ✅ (4)
  37780 params attention-only transformer: 6-head circuit discovered blind
               (double dissociation); textbook induction REFUTED; actual
               mechanism = windowed match-&-copy, causally verified  ✅ (5)
  ~100K-1M     next candidates: multi-digit arithmetic with carry (algorithmic
               state over sequences), grokked modular addition (Fourier
               circuits), or Phase-2 interpretability-regularized training
  ...
```
