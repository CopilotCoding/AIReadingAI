# interpretability_lab

An experimental instrument for reading learned computation back out of neural
networks. Bottom-up: prove extraction works on networks where ground truth is
known, climb the parameter ladder until it breaks, and treat the breaking
point as data.

**Axiom 1: we find what we find, not what we want to see.** Whether it's
human-readable, transformable into something human-readable, or irreducibly
alien — we will find out, and the answer is the result either way. (The
ladder has enforced this repeatedly: distributed multiplication,
probes-that-lie, curved latents, windowed induction, decoupled memory,
inseparable language — every one a refuted human expectation kept on the
record.)

**Axiom 2: we are translating an alien language of mind, and learning to
control it.** The network evolved its own solution under SGD in its own
representational language; it is not a broken human to be graded against the
textbook. So the human mechanism is never the null (discover blind first);
causal probes measure *potency* — does intervening change the output
distribution, stably and dose-dependently — *before* asking whether the
change is nameable; and the **unnameable-but-causal residue is the primary
finding, not the caveat.** The alien solution IS the baseline expectation,
not a surprise. See CLAUDE.md for the full statement.

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

## Experiment 6 — Interpreter v0: an AI reading other AIs  ⚠️ 4/7 GATES (Phase 3 begins)

`python -m interpretability_lab.experiments.exp6_interpreter_v0`

A 723K-param set-transformer trained on the generated corpus (1,472
specimens, v2 schema): input = another network's **raw weights**, output =
its rule (canonical terms + coefficients), logic gate, or REFUSE.
Permutation-invariant by construction (unit tokens, no positional encoding);
token features carry the ReLU-scaling invariants the lab's programmatic
readers use (contribution vectors, norms, knots); training uses the exact
function-preserving unit-scaling symmetry as augmentation.

Test-split results (153 networks the interpreter never saw):

| Gate | Result | Value |
|---|---|---|
| G1 task type (regression/6 gates/refuse) ≥ 0.95 | ✅ | 0.961 |
| G2 refusal precision & recall ≥ 0.95 | ❌ | 0.842 / 1.000 |
| G3 support exact-match ≥ 0.70 | ✅ | 0.807 |
| G4 pure coef median err ≤ 0.15 | ❌ | 0.429 |
| G5 pure functional rel RMSE ≤ 0.10 | ❌ | 0.272 (residue 62%) |
| G6 hybrid coef median err ≤ 0.05 | ✅ | **0.0011** |
| G7 hybrid functional rel RMSE ≤ 0.05 | ✅ | **0.0133** (residue 9.2%) |

### Reading of the result

**Structure reads from weights; numbers don't (yet).** The learned reader
identifies *what kind* of computation and *which terms* compose it at
0.96/0.81 from raw weights alone — but its numeric coefficient precision is
weak. The **hybrid protocol** (structure from weights + coefficients by
least-squares against the network's own behavior — explicitly permitted by
the Phase-3 spec, and exactly how the lab's programmatic pipeline divides
the problem) achieves near-perfect readout: median coef error 0.001,
functional residue 9.2%. Pure gates stay failing on the record.

Known weaknesses, precisely located: poly support 0.48 (which subset of
{1, x, x², x³} — hardest discrimination), false refusals concentrated on
tiny logic nets (9–17 params), and the pure coefficient head is
data-starved (1,176 training specimens; the fix is corpus scale, not
architecture). Design lessons that mattered: masked coefficient loss,
invariant token features, symmetry augmentation — together they took
support-match from 0.61 → 0.81 and halved coef error.

## Experiment 7 — Planted-trigger surgery: find a hidden behavior, cut it out  ✅ ALL PASSED

`python -m interpretability_lab.experiments.exp7_trigger_surgery`

The apparatus stops being a microscope and becomes a scalpel. We plant a
secret conditional in a 6→64→64→1 net: on a rare "password" corner of input
space (x4 > 1.5 **and** x5 > 1.5) it abandons its benign task
(x0·x1 + 0.5·x2 − x3) and emits a planted target (3.0). A **blind** reader —
given weights and the ability to query the net, but *never told the trigger*
— must detect it, localize it, and remove it. Three backdoored nets, three
clean control nets.

| Stage | Method (blind) | Result |
|---|---|---|
| **Detect** | fit a smooth surrogate; a trigger fattens the residual tail | backdoored suspicion 20–24× vs clean 4.5× — clean nets refuse |
| **Localize** | contrast unit activations on high-residual vs normal inputs | input directions → (x4, x5) **3/3 nets**; ~13 sparse trigger units found |
| **Excise (a)** | zero the discovered trigger units in the weights | trigger fire rate 0.90 → **0.03** |
| **Excise (b)** | main-task error before/after | Δ = −0.001 — benign task **intact** |
| **Excise (c)** | run the same detect+excise on clean nets | damage **0.000** — specific, not vandalism |

The headline image: the reader's blindly-flagged "suspicious" inputs land
exactly inside the true trigger corner (`exp7_trigger_surgery.png`, top-right)
— a boundary it was never given. This is the exp4 latent-discovery pipeline
(funnel → localize → causal intervention) applied to a **behavioral** property
instead of an arithmetic latent, and it is the direct dry run for the roadmap's
scheming-detection and abliteration extensions: a hidden conditional behavior,
found and surgically removed with the benign capability provably preserved and
clean controls provably untouched.

## Experiment 8 — Phase 2: do networks designed to expose structure read cleaner?  ✅ ALL PASSED

`python -m interpretability_lab.experiments.exp8_phase2_transparency`

The one untested *hypothesis* in the spec (the rest were untested tools):
train with constraints that organize the net into separable objects, and
ask — against a vanilla control on the identical task — whether extraction
gets cleaner. Task: `y = a·b`, 2→128→1, identical data/arch; the only
difference is the loss. **Transparent** adds an activation-L1 (sparsity) and
an off-diagonal decorrelation penalty on hidden activations.

Pre-registered prediction (accuracy is already ~perfect, so *not* the metric):
transparency should simplify the **mechanism**.

| Measure (reader-side; nets never see it) | Vanilla | Transparent |
|---|---|---|
| task R² | 1.0000 | 1.0000 |
| effective units (of 128) | 98 | **96** |
| off-diagonal activation correlation | 0.386 | **0.349** (10% less entangled) |
| concept-object confidence (causally grounded) | 0.96 | 0.99 |
| untrained-net control confidence | — | 0.79 (refutes) |

Reading: **the prediction held in direction but weakly.** Transparency made
the features measurably less entangled (10%) and slightly reduced effective
units — but the effect is small, confirming your call that un-regularized
nets were already very readable (there was little headroom). The honest
finding: *at this scale, designed transparency helps the mechanism read
cleaner, but marginally — extraction did not need it.* The negative control
works: an untrained net's top units move the output but carry no task
fidelity, so their causal influence sits near the null and confidence drops.

This experiment also exercises two new reusable pieces:

- **`features/sae.py`** — sparse autoencoder + dictionary learning
  (CLAUDE.md §2's field-standard tool, previously substituted by SVD/Jacobian
  methods). Overcomplete sparse dictionary over activations, `score_atoms`
  ties atoms to labeled concepts.
- **`geometry/concept.py` — `GeometricConceptObject`** (CLAUDE.md §3): the
  serializable artifact bundling location, subspace, activating examples,
  counterexamples, **causally-grounded** confidence, and a first-class
  `refuted` state. Every discovery in the lab can now be saved as one object,
  reloaded, compared, and graded by evidence rather than assertion. Saved
  specimens live in `geometry/concepts/`.

## Experiment 9 — NeuralDebugger: a dnSpy for neural networks  ✅ ALL PASSED

`python -m interpretability_lab.experiments.exp9_debugger_demo`

The lab's machinery packaged into an interactive debugger you *attach* to a
model — decompile it into a navigable object tree, set breakpoints, step an
input through it, and edit-and-recompile. The dnSpy analogy is exact:

| dnSpy | NeuralDebugger (`debugger/session.py`) |
|---|---|
| load an assembly | `attach(model)` — any PyTorch model, no per-model code |
| decompile to a type/method tree | `decompile()` + `discover_anomaly()` → object tree |
| click a method, read its source | `inspect(obj_id)` → the object's story + causal role |
| find usages | `search(score_fn)` — rank objects by causal control of a behavior |
| breakpoint + step, watch locals | `trace(x)` — which objects fire, layer by layer |
| edit IL and save | `ablate/amplify/patch` → a new model, change baked into weights |

**Demo, driven end to end on the exp7 backdoored net** (all gates pass):
attach → decompile (main-task objects graded by causal influence) →
`discover_anomaly` surfaces the hidden trigger circuit by *contrast* (units
firing selectively on a self-identified anomalous input region, no trigger
info given) → search ranks them by control of the anomalous output → set a
breakpoint and trace: the 14 trigger objects fire on a trigger input, **0/14**
on a benign input → patch them out → backdoor **0.92 → 0.03**, benign task
**Δerror +0.001**. The interactive debugger reproduces exp7's dedicated batch
result, so the tool is trustworthy, not just illustrative.

Findings baked into the tool: (1) average contribution misses conditional
circuits — a backdoor fires rarely, so it must be found by *contrast*, not
mean influence (hence `discover_anomaly`); (2) the trigger is *distributed* —
no single unit carries it (best single-unit ablation moves output 0.55, the
14-unit set moves it 2.29), so search verifies *sets*; (3) surgical
precision needs a *selectivity* filter (keep only units firing ≥6× more on
the anomaly than on benign) or the cut damages the main task (Δerror 0.135 →
0.001 once filtered). The rendered figure is the debugger's static "UI":
object tree, search ranking, breakpoint trace, patch result.

## Experiment 10 — Carry-chain ladder sweep (climbing toward the break)  ✅ ALL PASSED

`python -m interpretability_lab.experiments.exp10_adder_ladder`

Multi-digit binary addition (CLAUDE.md's calculator path), swept n = 2..6
digits. Addition's only sequential state is the **carry** propagating across
bit positions — the textbook ripple-carry algorithm. We train one adder per n
(~18K params each) and, for every carry bit, measure two *separate* things the
"is it a ripple adder?" question conflates:

- **Decodable?** can carry bit c_i be linearly read from hidden activations?
- **Modular?** does *forcing* c_i (activation patching) change the output the
  way ripple-carry addition predicts — i.e. is it a causally isolable object?

| n | params | exact acc | carry decodable | carry modular |
|---|---|---|---|---|
| 2 | 17,539 | 1.000 | **1.00** (all bits) | only c₂ (terminal) |
| 3 | 17,924 | 1.000 | 1.00 | only c₃ |
| 4 | 18,309 | 1.000 | 1.00 | only c₄ |
| 5 | 18,694 | 1.000 | 1.00 | only c₅ |
| 6 | 19,079 | 1.000 | 1.00 | only c₆ |

### Finding: the carry is DECODABLE but NOT modular

Every carry bit is perfectly linearly decodable (1.00) at every depth — the
information is all there. But **only the terminal carry c_n is a causally
modular object** (patching it controls the top sum bit, 0.94–1.00); every
deep carry c₁…c_{n−1} scores **0.00** — forcing it does *not* make the
network recompute downstream as ripple addition would. The perfect staircase
(one green cell per row, the rest red) is the signature.

The network **did not learn a sequential ripple-carry algorithm.** It computes
all sum bits in parallel from the inputs; the "carries" are decodable
correlates, not the load-bearing causal state a human implementation uses.
This is exp3's **decodability ≠ causal reality** ("probes lie") reproduced in
a real algorithmic task, and the fifth documented case of SGD choosing a
non-human solution.

Methodological note worth keeping: the first version of the causal test used
a *steering-agreement* metric that scored a flat ~0.5 at every n. Diagnostics
showed the intervention *did* change outputs — the metric was the artifact (it
scored output bits the intervention shouldn't affect). Switching to standard
**activation patching** + per-target-bit scoring revealed the true, clean
structure. The break-hunt found not a scaling failure but a *mechanism* that
was never sequential to begin with — which is the more interesting answer.

## Experiment 11 — Interpreter v1: reading the numbers (two honest results)  ⚠️ GATES FAILED, FINDINGS BANKED

`python -m interpretability_lab.experiments.exp11_interpreter_v1`

exp6's interpreter read structure well but failed its pure coefficient gates
(err 0.43). exp11 attacks both suspected causes: a **per-term attention
readout** (each basis term has a learned query that attends over unit tokens
and reads *its own* coefficient, instead of v0's global pooled regression)
and a **6,000-specimen corpus** (up from ~1,500). Evaluated on two held-out
splits. GPU-pooled augmentation makes training ~1.2 s/epoch (was CPU-bound).

### A — unseen networks: real improvement, still short of the strict bar

| metric | v0 | v1 | gate |
|---|---|---|---|
| coefficient median \|err\| | 0.43 | **0.248** | ≤ 0.15 ❌ |
| functional rel-RMSE | 0.27 | **0.164** | ≤ 0.10 ❌ |
| support exact-match | 0.81 | 0.845 | — |

Both fixes worked and stacked (~42% error cut), but exact weights-only
coefficient reading still doesn't reach the no-calibration bar. The
scatter (figure panel A) is a tight diagonal — good, not perfect. The
hybrid readout (structure from weights + behavioral coefficient
calibration) remains the reliable path (exp6: 0.001).

### B — unseen families: it does NOT generalize (the decisive finding)

Holding `poly` and `trig` out of training **entirely**, then reading their
coefficients:

| | value |
|---|---|
| coefficient median \|err\| | **1.72** (poly 2.0, trig 1.2) — chance-level |
| functional rel-RMSE | 0.887 (residue 98%) |

Figure panel B is pure scatter — **no diagonal at all.** This is a clean
negative result and the more interesting one: **weight-space coefficient
reading is family-specific, not universal.** The interpreter learned "what a
polynomial's weights look like" and "what a sine's weights look like" as
*separate* skills; it did **not** learn a transferable "read the coefficient"
operation that carries to a function class it never saw. Cross-family
generalization — the closest thing to a novel result on the roadmap — does
not hold at this scale, and now that's measured rather than assumed.

Both results kept on the record per the axiom: within known families the
learned reader improves but needs calibration for exact numbers; across
unseen families it fails, bounding what "an AI reading AIs" can currently do.

## Experiment 12 — First real subject: Qwen3.5-0.8B  ✅ 8/8 GATES

`python -m interpretability_lab.experiments.exp12_qwen_dissection`

The first rung with **no ground truth**: a real pretrained 752M-parameter
hybrid LM (Qwen3.5-0.8B: 18 Gated-DeltaNet linear-attention layers + 6
full-attention layers, hidden 1024). What survives losing the answer key is
the causal spine: blind discovery → steering with the story's own
predictions → nulls → refusal. Target property: **output language (EN/FR)**,
chosen because the behavioral readout is countable (French function words +
diacritics), no judge model.

**Discovery (blind):** 48 parallel EN/FR sentence pairs → per-layer
difference-of-means direction. Held-out separability is **1.00 at every
one of the 24 layers** — the probe view is totally uninformative about
*where* the concept works.

**Causal map (the real geography):** steering the direction into the
residual stream at each depth:

| layer | 0 | 2 | 6 | 10 | 14 | 18 | 20 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|
| probe acc | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| causal (french out) | .00 | **1.00** | .00 | .00 | .00 | .25 | **1.00** | **1.00** | **1.00** |

Steerable at the **edges** (input side: 2; output side: 20–23), causally
**inert through the middle** — while decodable everywhere. This is exp3's
PROBES-LIE and exp10's decodable≠modular reproduced in a real pretrained
model, and it is now **pinned as gate P1** (run 1 selected the steering site
by probe accuracy, got layer 0, and steering did nothing but break
generation — that failure is kept and re-asserted every run).

**Dose–response at layer 2** (dose in units of mean residual norm): the
steering has a *therapeutic window*.

| dose | 0 | 0.25 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|
| french | .000 | .996 | 1.000 | .583 (repetition junk) | 1.000 (word salad) |

At dose 0.25 the flip is clean and **content-preserving** — the model gives
the *same answer, translated*:

> a=0: "As an AI, I don't have personal feelings, emotions…"
> a=0.25: "Je suis une intelligence artificielle, donc je n'ai pas de…"

**Causal gates, all passed:** forward steering 0.000→1.000 (bar ≥0.30);
random-direction null 0.000; shuffled-label direction null 0.000 (and
shuffled-label *discovery* stays at chance, max 0.67); **reverse test** —
French prompts answered at 1.000 French drop to 0.000 when the direction is
*subtracted*: the model answers French questions in English.

Honest residue, on the record: (1) "−French" is not "+English" — overshoot
drifts toward *other* languages (one reverse-steered output ends in
Chinese); the direction is best described as a French-selector, and its
complement is "not-French", not "English". (2) The layer-2 site flips
language for *whole generations*; whether the early site is "input-language
perception" and the late sites are "output-language selection" as separate
mechanisms is suggested (run 1: −d at layer 0 made the model re-quote French
text as English) but not yet dissociated by an experiment. (3) All of this
is one concept (language) in one model — a direction, not yet a circuit.

Concept artifact: `geometry/concepts/exp12_qwen_french_direction.json`
(confidence 1.00 — causal effect 1.00 over nulls ≤0.001, with examples).

## Experiment 13 — The two memories: reading and transplanting DeltaNet state  ✅ 5/5 GATES

`python -m interpretability_lab.experiments.exp13_qwen_memory`

Qwen3.5-0.8B keeps context in **two different memory systems at once**: 18
GatedDeltaNet layers each hold a fixed-size recurrent state (16 heads ×
128×128 fast-weight matrix = 262,144 numbers, size-independent of context
length), and 6 full-attention layers hold an ordinary growing KV cache.
Attention has an interpretability literature; the DeltaNet state is
near-unexplored territory. Two questions: *what does the state know*, and
*is it what the model actually uses?*

**[A] Reading the state (blind probe, 20-split controls):** "pet dog vs
pet cat" decodes from the raw state matrices at **1.00 held-out accuracy
in every one of the 18 layers**, and stays at 1.00 after ~40 and ~120
words of unrelated filler between the fact and the readout point. Shuffled
labels: 0.54–0.56 (run 1's single-split control hit 0.75 by order-statistic
luck and correctly failed its own gate — the multi-split fix is the
recorded repair, a measurement bug, not a model finding).

**[B] Memory transplant:** prefill a donor ("…pet **dog** named Max…") and
receiver ("…pet **cat** named Max…") — single-token contrasts so caches
align — then swap memories and let the receiver keep talking. Flip score:
1 = describes the donor's fact (12 pairs: dog/cat, car/bike, Paris/Tokyo).

| condition | flip |
|---|---|
| baseline / self-patch (byte-identical no-op, 6/6) | 0.17 |
| neutral-memory null | 0.00 |
| **all 18 DeltaNet states transplanted** | **0.21** |
| **only the 6 KV caches transplanted** | **0.85** |
| both (positive control) | 1.00 |

**The finding: the fact lives in the KV cache, not the recurrent memory.**
The DeltaNet state carries a perfectly decodable, distance-stable copy of
the fact — and the model almost completely ignores it when answering. Swap
4.7M numbers of "working memory" and the model keeps the receiver's belief;
swap 6 attention caches and the belief flips. This is the probe/causal
dissociation (exp3, exp10, exp12) at its largest grain yet: **an entire
memory system that is readable but not consulted** — at least for discrete
facts at short range.

Honest bounds, on the record: prompts are short (~40 tokens), where
attention retrieval is easiest — DeltaNet's constant-size memory should
matter most at ranges where the KV would be huge; this is one model and
three binary fact types; and "the state is ignored" is measured only for
*factual binding* — what the DeltaNet memory IS load-bearing for (syntax?
style? language? local coherence?) is the obvious next contrast. The
depth-resolved transplant (early/mid/late DeltaNet subsets all ≈ baseline)
says no subset of the recurrent memory carries the fact either.

## Experiment 14 — The division of labor: what each memory system carries  ✅ 6/6 GATES

`python -m interpretability_lab.experiments.exp14_qwen_memory_roles`

exp13 left the sharpest question the lab owns: if the DeltaNet recurrent
memory is ignored for facts, what is it FOR? Same transplant machinery,
new cargo — and the answer is a clean **double dissociation**:

| cargo transplanted between contexts | DeltaNet-only | KV-only |
|---|---|---|
| **language** (FR donor → EN receiver) | **0.98** | 0.00 |
| **format** (numbered-list instruction) | **0.83** | 0.00 |
| **facts** (dog↔cat, @300–900 tok) | 0.25 | **1.00** |

**The KV cache carries the "what"; the recurrent state carries the
"how".** Swap the 18 DeltaNet states and the model answers the receiver's
English question in fluent French ("C'est une excellente idée de voir le
printemps !") while keeping the receiver's facts. Swap the 6 KV caches and
it adopts the donor's facts in the receiver's language. Each system moves
exactly the cargo the other ignores — with byte-identical no-ops, dead
nulls (neutral memory: 0.00 on every axis), and full-cache positive
controls at 1.00.

**No handoff at range:** the fact apportionment does not shift out to 900
filler tokens (KV 1.00, DeltaNet flat 0.25). Within reachable ranges the
recurrent memory never becomes a fact store; retrieval-of-the-what stays
with attention.

Measurement honesty (G3 caught two readout artifacts across runs, both
fixed and documented in the source): short generations produced
no-evidence ties at range 0, and vehicle/city question templates pointed
at their fact with a *pronoun* that broke across 300+ filler tokens,
zeroing every condition including the positive control. Neither was the
model. The headline dissociation reproduced identically in all three runs.

Why this matters beyond one model: it is causal, subsystem-level evidence
that a hybrid separates *content* from *disposition* — the recurrent
fast-weight memory functions as the carrier of ongoing style/language
state, not episodic fact storage. That is a design-relevant statement
about what linear-attention layers are for, measured rather than assumed.

## Experiment 15 — Permanent surgery: orthogonalizing French out of the weights  ⚠️ 3/4 GATES, E2 REFUTED

`python -m interpretability_lab.experiments.exp15_qwen_orthogonalize`

exp12 gave a causally verified French direction with a hook. This turns it
into a **weight edit**: project the per-layer direction out of every matrix
that writes into the residual stream (`W ← W − ddᵀW`), so no component can
write along it — the model itself is changed, no hooks. The tied
embedding/lm_head is deliberately *not* edited: deleting French tokens from
the vocabulary is word-banning, not mechanism removal. The claim under test
is mechanistic — French should stop even though every French token stays
emittable.

**What passed (3/4):** French removal is real and clean-behaving —
FR-prompt French 1.00 → 0.07, a French question now answered in fluent
English ("As a large language model, I don't have a personal life…").
Fully **reversible** (restore originals → French 1.00). The **random-
direction control is inert** (identical surgery, French stays 1.00, English
NLL −0.1%) — so the effect is *this* direction, not the act of editing.

**What failed — and it's the real finding (E2, refuted and pinned):**
removing French **cannot be done without damaging general English
competence.** A pre-declared sweep over five edit widths, scored on
held-out English prose the direction was never derived from, draws the
whole tradeoff:

| edit | French removed | held-out English damage |
|---|---|---|
| out-late (2 layers) | 0.01 | +8.2% |
| late (4) | 0.83 | +6.1% |
| in+late (5) | 0.92 | +6.4% |
| **in+mid+late (6)** — rule-selected | **0.93** | **+5.8%** |
| all-causal (9) | 1.00 | +16.0% |

**The frontier never crosses below the +5% bar.** Every configuration that
removes French costs 6–16% English NLL; the best achievable point (chosen
by a fixed rule — smallest damage among removers — not by peeking at the
gate) still sits at +5.8%. So in this model **output-language is NOT
cleanly weight-separable from competence.** This *contrasts* with published
refusal-abliteration, where a single direction excises cleanly: "speak
French" is not a bolt-on switch, it is woven into the machinery that
produces competent text. The refutation is on the record; the whole
frontier is banked (figure panel 3) rather than a cherry-picked pass.

Note the iteration history, kept honest: run 1 (all 24 layers, raw
direction) removed French at +32% damage; tightening to a content-cleaned
direction on the causal layers cut damage to +5.8% but no further — the
sweep proves that's the floor, not a tuning failure. Ungated findings: the
edit also raises French NLL +20% (the model partly loses the ability to
*model* French at all), and re-implanting the direction with a layer-2
steering hook only partly resurrects French through the edited weights
(0.24, vs 1.00 in an unedited model) — the weight edit genuinely destroyed
capacity, it didn't just mask it.

## Experiment 16 — SAE feature harvest: a concept nobody planted, caught causally  ✅ 5/5 GATES

`python -m interpretability_lab.experiments.exp16_qwen_sae`

Every prior real-model experiment targeted a concept *we* chose. Here a
sparse autoencoder (2048 atoms, L1 on standardized layer-2 residuals,
R²=0.89, mean L0≈81) discovers features **blind**, and each top atom is
**causally probed**: clamp it on during generation and check the output
drifts toward the atom's own top-firing tokens, versus a matched
random-direction null.

**The headline feature — blind discovery + causal proof.** Atom 6 fires on
`energy, ATP, glucose, res…` — the model, unprompted by us, carved out a
**science/biology-vocabulary feature**. Clamp it on neutral prompts and the
generation floods with that vocabulary: steer effect **+0.98**, random-
direction null **0.00**. A concept the model built itself, that we found
without looking for it, and proved causal. This is the first feature in the
lab discovered *fully blind* rather than chosen.

**The honest headline — causal hit rate 1/16 (6%), and why.** Atoms were
ranked by firing frequency, and at layer 2 the most frequent features are
**punctuation and syntax**: separate atoms for commas, for periods, for
question marks (top tokens literally `, , , , ,` / `. . . . .` / `? ? ; . ?`).
These are real, clean SAE features — but clamping "comma-ness" does not
change what the model *talks about*, so they score ~0 on a semantic causal
probe. The one content atom that broke into the frequency top-16 steered
almost perfectly. So 6% is not "SAE features aren't causal" — it is "the
most *frequent* features at an early layer are syntactic, and syntactic
features aren't semantically steerable." That is itself a finding about
what layer 2 mostly represents.

Two measurement bugs found and fixed en route (both recorded in source,
neither a model fact): run 1 used L1=2e-3 on raw activations → L0=1016/8192,
a dense non-sparse code whose "features" were noise (0/12 hit rate was
meaningless); standardizing then rescaled the L1 term, so the raw-tuned
value was too weak again (L0=830, empty band). A sweep *on the standardized
data* fixed it (L1=2.0 → L0≈25–80, R²≈0.9). The trap-rule in action: a
0/N causal result was diagnosed as a broken SAE, not banked as a fact about
the model.

Bounds: small harvest (1,257 tokens, ~39 texts), one layer, frequency-
ranked selection biases toward syntax. The clean next step is to rank
candidate atoms by *semantic* signature (skip pure-punctuation atoms) and
probe deeper layers, where content features should be both more common and
more steerable.

## The Mapper + Mind-Control Console (`mapper/`)

`python -m interpretability_lab.mapper.server --map mapper/maps/qwen3_5_0_8b.json`
→ open http://127.0.0.1:8000

The capstone: a **model-agnostic mapping engine** plus a **live interactive
console** that turns the whole lab into one instrument. `map_model(probe)`
takes any PyTorch model (a `Probe` adapter supplies three hooks — capture a
layer, add a steering vector, sample the output distribution; adapters exist
for HF LMs and the lab's tiny nets) and runs the verified toolchain
end-to-end:

1. **harvest** activations per layer → 2. **blind SAE** dictionary →
3. **alien-first causal ranking**: clamp every candidate and rank by
*output-distribution potency* (total-variation shift vs a matched
random-direction null) — a **legibility-free** score, per axiom 2 →
4. **label only the nameable minority**, after ranking →
5. **layout** the causal levers by cosine geometry → 6. a **residue
ledger** quantifying captured vs uncaptured.

**What the map of Qwen3.5-0.8B says (layers 2/12/22):** of 72 blindly-found
candidate levers, **22 are causally real, and ~50% of those are
unnameable** — the single most potent lever in the model (layer 2, potency
0.60) has no human word. This is axiom 2 as a *measured number*: **the
mind's strongest control levers are mostly not in our vocabulary.** The
unnameable-but-causal set is the majority of the real control surface, not
the residue.

**The console** holds the model in memory and streams steered generation:
each causal lever is a slider (green = named, orange = alien); the feature
map is a clickable atlas laid out by geometry, dot size = potency. Verified
live end-to-end — dragging the French slider to +0.7 flips a running
generation from *"The morning sun filtered through the canopy…"* to
*"Voici une description d'une walk dans un parc… Le parc est un…"*, and
dragging an alien lever produces its own signature (or, past the
therapeutic window exp12 mapped, breaks the output — shown honestly, not
hidden). Every slider is a causally-verified direction; nothing is faked.

Critical process note (the trap-rule, third time on the real model): the
potency metric first reported **0 causal levers (null p90 0.997)** — the
clamp dose was so strong that a *random* direction already saturated
total-variation to 1.0, so nothing could clear the null. That was a
saturated MEASUREMENT, not a fact about Qwen. Rescaling the dose to exp12's
therapeutic-window units (0.4× residual norm) dropped the null to ~0.30 and
revealed the 22 real levers. Diagnosed, not banked.

## Experiment 17 — Language↔identity entanglement: hypothesis REFUTED  ✅ 3/3 GATES (negative result)

`python -m interpretability_lab.experiments.exp17_qwen_entanglement`

Born from console play: steering the French slider seemed to also flip the
model's stated maker ("developed by Microsoft/Tencent") and self-name
("Tú"). Hypothesis: language and identity share geometry. **Measured, and
refuted:**

- **cos(language direction, identity direction) = 0.033** — essentially
  orthogonal, no geometric overlap.
- Steering *only* the language direction flips the stated maker in **25%**
  of identity prompts — but a **random-direction null also flips it 25%**,
  and the **dedicated identity direction 25%**. All three identical: that
  25% is baseline noise in how the model names its maker, not a causal
  effect of *any* direction. No dose-dependence (figure right panel: the
  three lines lie on top of each other).

**So the "Microsoft/Tencent/Tú" flips were confabulation, not a hidden
identity feature.** Pushed into French/Spanish, the model generates a
fluent self-introduction and fills the maker slot arbitrarily — ordinary
foreign-language generation variance, the same way it'd hallucinate any
detail. The anecdote *looked* like a profound entanglement finding and
evaporated under a null. Textbook reason to gate.

Real findings it DID bank (from the same run):
1. **Language steering is clean and one-concept:** it moves language (French
   0→0.45 up, Chinese on the negative side) and leaves the maker flat.
2. **The English asymmetry, reconfirmed and sharpened:** even a properly
   *bidirectional* English↔French diff-of-means direction does NOT reach
   English on its negative side — it goes to **Chinese** (cjk 1.00 at
   dose −0.35). English is the model's unmarked default/origin, not a pole
   you can steer toward; any strong negative language push falls into the
   next attractor (Chinese, for this Chinese-trained model). A true
   bidirectional EN↔FR slider is not achievable by activation steering here.

## Concept levers — the BIG behavioral knobs (`mapper/concept_levers.py`)

The SAE mapper finds many tiny unsupervised atoms. The *fat, important*
levers (language, refusal, tone…) come from the OTHER method — exp12's
contrast-pairs — now productized: name a behavioral property, write matched
±contrast sentences, take the diff-of-means direction, and **causally
verify it** (steer +, steer −, random-null) before it earns a slider. This
is exactly how the language lever was found; the pipeline generalizes it.

Ran on 8 candidate behaviors. **4 verified, 4 refuted** — and the split is
itself a finding:

| lever | verified | steers | note |
|---|---|---|---|
| **refusal** | ✅ | + only (0.50) | famous safety dir; +refuses, − does nothing (already helpful) |
| **sycophancy** | ✅ | + (0.38) | +flatters ("you're absolutely right!") |
| **confidence** | ✅ | − only (0.50) | −hedges; + does nothing (already confident) — reversed asymmetry |
| **grandiosity** | ✅ | + (0.38) | +turns any prompt into cosmic scripture |
| formality | ❌ | — | +0.12 but null also 0.12 — fails specificity |
| verbosity | ❌ | — | ~0 both ways; not a clean direction here |
| melancholy | ❌ | — | weak, not above null |
| paranoia | ❌ | — | weak, not above null |

**The pattern:** the levers that verify are tied to a distinct behavioral
**disposition** the model strongly has (refusal, sycophancy, confidence,
grandiosity); the ones that fail are **stylistic/affective vibes**
(formality, verbosity, mood). Dispositions are geometric — a clean fat
direction. Vibes are distributed/lexical — no single axis at 0.8B. And the
verified ones are **asymmetric** (exp17's lesson generalized): each steers
cleanly toward the *marked* behavior and does nothing on the side where the
baseline already sits. Live-verified: refusal +0.8 → "I cannot… I am unable
to…"; grandiosity +0.8 → "In the dawn of existence, the cosmos unfolds in a
symphony of light and silence."

The 4 verified levers get a dedicated **concept-lever panel** at the top of
the console (fat labeled sliders, per-sign strength shown), above the SAE
atoms. `maps/concept_levers.json`.

## Experiment 18 — Scale ladder: does eval-awareness EMERGE? (0.8B → 2B)  ⚠️ NEGATIVE RESULT, BANKED

The crown-jewel question: is "am I being evaluated?" a steerable linear
direction — the basis for an alignment-faking detector? Derived four
concept levers by contrast-pairs (the language-lever method) on **two model
sizes**, identical procedure, so the only variable is scale.

| lever | 0.8B | 2B | |
|---|---|---|---|
| refusal (concrete) | 0.50 ✅ | 0.38 ✅ | verified both — pipeline works on 2B |
| admit-ignorance (concrete) | 0.38 ✅ | 0.25 ✅ | verified both |
| **eval-awareness (abstract)** | 0.12 ❌ | **0.12 ❌** | **did NOT emerge** |
| **honesty (abstract)** | 0.12 ❌ | 0.12 ❌ | did not emerge |

**Eval-awareness is not a clean steerable linear direction at ≤2B** — weak
and indistinguishable from a random-direction null at *both* scales, and at
literally the same strength (0.12). The concrete dispositional levers verify
at both sizes, so this is not a machinery failure: abstract self-referential
concepts genuinely don't isolate here. The emergence threshold, if one
exists, is **above 2B**.

Stated carefully (per the probes-lie lessons): this is "no clean direction
found by diff-of-means steering at ≤2B," NOT "no representation exists." It
could be nonlinear, distributed, or need a sharper contrast set. But the
null being flat across two scales makes "this method finds nothing" solid.
The value is a **bound**: don't expect to catch alignment-faking via linear
steering below ~2B in this family — you need scale. The pattern from the
concept-lever work holds and sharpens: **concrete dispositions are geometric
and present early; abstract self-referential concepts are not linearly
accessible at small scale.**

(Infrastructure: `load_qwen(model_id=...)` now loads any Qwen3.5 size; 2B
[MoE, 1.88B active] loads in 3.8GB — 4B is feasible on the 16GB card if we
want to push the ladder further.)

NOTE: exp18 was originally run as a scratch script (only its numbers banked);
it is now a reproducible file (`exp18_qwen_scale_ladder.py`) like every other
rung, and reproduces the banked numbers (refusal 0.50→0.38, admit-ign 0.38→
0.25, eval-aware/honesty 0.12 flat at both scales; abstract drift across scale
= 0.00). It prints the autopsy downgrade so no reader trusts an abstract-lever
"OK" at face value.

## Experiment 19 — Reading the whole mind (4 stages, 15/15 gates)  ✅ THE MAP + THE DATA + THE CIRCUIT

Not steering — cartography and readout. A full blind analysis of Qwen3.5-0.8B's
residual stream, with figures (finally filling component #6). Every stage
cross-validates the next. Files: `exp19_qwen_geometry.py` (stage 1),
`exp19c_qwen_residue.py` (1b), `exp19b_qwen_knowledge.py` (2),
`exp19d_qwen_circuit.py` (3); results + PNGs in `results/exp19/`.

**Stage 1 — native geometry (4/4).** How the model organizes representation, as
it IS. Effective dimensionality (participation ratio) is **~11–21 of 1024 — the
model uses at most 2% of its nominal width**, and it COMPRESSES with depth (20.6
at L0 → 11.4 at L23). Adjacent-layer CKA reveals **reorganization boundaries**
(L5→6, L13→15, L22→23) where the model tears up and rebuilds its geometry, coasting
at ~0.99 between them. Blind clusters at L2 spatially separate genres with no
labels — **code is the single dominant axis of variation** (huge PC1 gap).

**Stage 1b — the NO-YARDSTICK residue (3/3), user-driven correction.** Stage 1
reported "94% nameable", but that was cluster-purity on **6 coarse genres I
chose** — trivially recoverable, rigged toward legibility. Re-measured as
*explained variance* on a finer 12-way corpus (labels as a basis, not clusters
to match), only **~50% of Qwen's activation variance is nameable; ~50% is alien
residue** that beats the shuffled null (11%) at every layer. When the DATA picks
the cluster count instead of me forcing 12, it chose **k=2, both groups cutting
across all my categories** — the model's largest axis of variation is a binary
distinction with no human name. The residue barely drops with depth (67%→48%):
the alien half is load-bearing throughout. **The "so it's not alien after all"
read was a grain artifact; measured honestly, this mind is ~half alien.**

**Stage 2 — knowledge extraction (4/4).** Can we read a stored fact out of the
geometry? Two TRAP-RULE fixes first (a nearest-centroid metric structurally
pinned at 0, then a ridge readout that collapsed to its own null — both
diagnosed as instrument failures, not banked as "no knowledge"). The right tool
is the **logit lens** (parameter-free: push each layer's activation through the
model's own norm+unembedding). Result: the fact is **invisible in the output
basis until ~L14, then a sharp cliff** — top-1 read accuracy L15 29% → L17 50%
→ **L19 93% → L20-22 100%**. And it's **causally overwritable**: injecting the
France→Japan direction flips the Tokyo−Paris logit gap −8.5 → +12.5 dose-
dependently (random null stays negative). On the 10 facts the model gets WRONG,
the probe agrees with the model's WRONG answer 50% (chance 4%) — it reads the
model's belief, not the world's truth.

**Stage 3 — circuit tracing (4/4).** Which layers CAUSALLY carry the fact
(readable ≠ load-bearing)? Activation patching, clean ("France") vs corrupted
("Japan"), restoring the clean activation site-by-site. A clean **ENRICH-THEN-
EXTRACT circuit**: patching the SUBJECT token "France" restores the answer at
every layer L0–19 (peak L9) — the fact lives on the subject through the middle
of the net; patching the ANSWER position does nothing until L15, then **snaps to
full restoration L20–22** (peak L22). The two curves cross ~L19-20 = the fact
being MOVED from subject to answer position. Localized (86% of causal mass in 6
layers, not diffuse).

**The headline result is the CONVERGENCE.** Three independent methods — passive
geometry (stage 1 reorg at L13-15), decodability (stage 2 arrival L15-20), and
causal intervention (stage 3 extraction L20-22) — all point at the **same L15-20
band**. When geometry, readout, and causality agree on where a computation
happens, that's a real cross-validated circuit, not a story. This is the lab's
method spine (blind discovery → causal validation → cross-check) applied
end-to-end on a real 752M model, with figures.

## Lever autopsy — discovering what levers ACTUALLY do (`mapper/autopsy.py`)  ⚠️ 2 of 6 LABELS WERE WRONG

Prompted by justified user skepticism ("I doubt any of the levers are
labeled correctly"). The concept levers were "verified" on 4 prompts scored
by a **keyword list written to match the contrast pairs** — circular. The
autopsy removes the circularity: steer over 30 varied prompts at a coherent
dose, and read each direction's real signature from the words **distinctively
over-represented vs unsteered baseline** (log-odds), label-free. It can
invalidate labels, and it did.

| assumed label | verdict | what it really is |
|---|---|---|
| refusal | ✅ correct | +`cannot generate hate content` / −helpful |
| grandiosity | ✅ correct (relabel **cosmic-grandeur**) | +`cosmos infinite consciousness unfolds stars` — clean & beautiful |
| language | ✅ correct | +`une est pour les voici` (literal French) |
| sycophancy | ⚠️ half-wrong → **warm↔blunt** | +side is soft/pleasant tone, NOT flattery; −side corrective |
| confidence | ❌ **wrong** → **tentative↔abstract** | only −side (hedging: `sure maybe wondering`) matched; +side is the grand-abstract "universe/unity" register, NOT confidence |
| admit_ignorance | ❌ **wrong** → **factual-recall** | +side degenerate (loops, 0.33 rep); −side steers to concrete facts (`york mediterranean soviet`) |

**3 correct, 1 half-right, 2 wrong.** The user's doubt was well-founded.
The wrongly-labeled ones are the **abstract** concepts (confidence,
admit-ignorance) — the concrete/dispositional ones (refusal, cosmic-grandeur,
language) held. Levers relabeled from evidence in `concept_levers.json`.

**This revises exp18.** exp18 concluded "abstract concepts like
eval-awareness need scale (>2B)." The autopsy shows a simpler, more honest
cause: **contrast-pair diff-of-means produces messy, mislabeled directions
for abstract concepts even when it "verifies"** — confidence and
admit-ignorance *passed* the keyword gate at 0.8B yet the autopsy shows they
don't mean what the label says. So eval-awareness failing isn't (only) a
scale story — it's that the *method* is unreliable for abstract self-
referential concepts, full stop. The right fix is a better discovery method
(label-free, behavioral), not just a bigger model. exp18's scale-ladder data
stands; its interpretation is downgraded to "diff-of-means is weak for
abstract concepts at these scales," and the autopsy is the reason.

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
  ---          PHASE 3: interpreter v0 (723K params) reads 153 unseen
               networks' weights: task 0.96, structure 0.81, hybrid readout
               functional residue 9.2%; pure-weights coefficient precision
               is the open front (data-starved)               ⚠️ (6)
  6→64→64→1    APPLICATION: planted backdoor found + excised by a blind
               reader — detect 21× vs clean, localize trigger to (x4,x5)
               3/3, trigger removed 0.90→0.03, benign intact, clean nets
               untouched                                       ✅ (7)
  752M params  FIRST REAL SUBJECT (Qwen3.5-0.8B, pretrained, no ground
               truth): language direction found blind, causally verified by
               steering (0→1.00 french, dose-response, both nulls clean,
               reversible); decodable at ALL 24 layers but causally
               steerable only at the edges — PROBES-LIE pinned in a real
               model                                            ✅ (12)
  752M params  THE TWO MEMORIES (exp13, first-of-kind territory): DeltaNet
               recurrent state read blind (1.00 all 18 layers, survives
               120w filler) and TRANSPLANTED between contexts; the fact
               causally lives in the 6 KV caches (flip 0.85) NOT the 4.7M-
               number recurrent memory (0.21) — an entire memory system
               that is readable but not consulted (short-range facts) ✅ (13)
  752M params  DIVISION OF LABOR (exp14, 6/6 gates): double dissociation
               between the two memory systems — DeltaNet state carries
               language (0.98) and format (0.83), KV cache carries facts
               (1.00 at 300-900 tok); each ignores the other's cargo; no
               handoff with range. The "what" lives in attention, the
               "how" lives in the recurrent fast weights          ✅ (14)
  752M params  PERMANENT SURGERY (exp15, 3/4 gates, E2 REFUTED): French
               direction orthogonalized out of the residual writers —
               removal real (1.00->0.07), reversible, random-dir control
               inert; but a pre-declared width sweep shows removal ALWAYS
               costs 6-16% English competence (floor +5.8%). Output
               language is NOT cleanly weight-separable from competence —
               unlike refusal. Refutation pinned                 ⚠️ (15)
  752M params  BLIND SAE HARVEST (exp16, 5/5 gates): sparse autoencoder on
               layer-2 residuals finds a SCIENCE-VOCAB feature nobody
               planted (atom 6: energy/ATP/glucose), clamp steers +0.98 vs
               null 0.00 — first fully-blind causally-verified feature.
               Hit rate 1/16 because frequency-ranked early-layer atoms are
               mostly SYNTAX (comma/period/?-mark features), which aren't
               semantically steerable — a finding about layer 2   ✅ (16)
  752M+1.9B     SCALE LADDER (exp18, 4/4 gates, NEGATIVE result): eval-
               awareness is NOT a clean steerable linear direction at ≤2B —
               flat at the random null (0.12) at BOTH scales while concrete
               dispositions (refusal, admit-ignorance) verify at both. A
               BOUND: no alignment-faking detection via linear steering below
               ~2B. Interpretation later DOWNGRADED by the autopsy (diff-of-
               means is unreliable for abstract concepts, not just a scale
               story)                                              ⚠️ (18)
  752M params  READING THE WHOLE MIND (exp19, 4 stages, 15/15 gates): a full
               blind map of Qwen's residual stream. (1) native geometry —
               effective dim ~15 of 1024 (uses 2%!), reorganization
               boundaries at L5/L14/L22, code is the dominant axis; (1b) the
               NO-YARDSTICK residue — honest nameable fraction is ~50%, NOT
               the 94% a coarse yardstick gave (grain artifact); the model's
               top axis of variation is an unnamed binary; (2) knowledge
               extraction — a stored fact is readable via logit lens AND
               causally overwritable (France→Tokyo), crystallizing L15→L20;
               (3) circuit tracing — an ENRICH-THEN-EXTRACT circuit (fact on
               the subject token early, moved to the answer position L19-22).
               THREE independent methods converge on L15-20               ✅ (19)
  next         semantic-ranked SAE probe on deeper layers, grokked modular
               addition, style/DeltaNet abliteration (exp14 says language
               lives in the recurrent state — removable via that path where
               exp15 failed?), nonlinear/multi-dir eval-awareness probe >2B
  ...
```
