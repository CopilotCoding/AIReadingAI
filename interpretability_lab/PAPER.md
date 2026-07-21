# Reading the Machine: Causally-Gated Extraction of Learned Computation from Neural Networks, from 2 to 38,000 Parameters

*Research report, interpretability_lab — July 2026*

## Abstract

We ask whether the computation a neural network learns can be extracted back
into human-readable form — and whether the extraction can be *proven* to
describe the network's actual mechanism rather than merely imitating its
behavior. We built an experimental instrument that climbs a parameter ladder
(2 → 97 → 1,025 → 9,793 → 37,780 parameters) across architectures (linear
units, ReLU MLPs, depth-2 MLPs, attention-only transformers), holding every
claim to pre-registered pass/fail gates with causal verification, negative
controls, and pinned refutations. Extraction succeeded at every rung:
equations, boolean circuits, piecewise-linear mechanisms, latent variables,
and attention algorithms were all recovered and causally validated. The
recurring counter-finding is that **gradient descent routinely solves tasks
by non-human algorithms**: we document four cases where the textbook
mechanism was absent and the network's actual solution was different,
distributed, or geometrically curved. We trained a 723K-parameter
interpreter network *from scratch* to read other networks' raw weights: on
153 unseen networks it identifies the computation class at 0.96 accuracy and
the exact symbolic structure at 0.81, while precise coefficients require
behavioral calibration (hybrid readout: median coefficient error 0.001,
functional residue 9.2%). Structure, we find, is readable from weights;
numbers are not — yet. Finally, to show the instrument can *edit* and not
only describe, we planted a backdoor (a hidden conditional that hijacks the
output on a rare input corner) and had a blind reader — never told the
trigger — detect it (20–24× residual signal vs trigger-free controls),
localize it to the correct input directions and a sparse circuit, and excise
it from the weights: trigger behavior removed, benign task preserved
(Δerror −0.001), and the identical procedure on clean networks causing zero
damage. A controlled test of the roadmap's central *hypothesis* — that
networks trained under geometric constraints expose their structure more
readily — found the effect real but small (10% less feature entanglement,
same accuracy): at this scale un-regularized networks are already highly
readable, so designed transparency has little headroom, and is expected to
matter only where baseline readability degrades. Every discovery is emitted
as a serializable object carrying a confidence *grounded in causal
evidence*, not asserted.

**Core axiom.** *We find what we find, not what we want to see.* Gates test
the measured mechanism, never the hoped-for one; refuted hypotheses stay on
the record and are re-asserted on every run; readers must be able to refuse.

## 1. Method: the standard of evidence

Two claims are never conflated:

- **Behavioral equivalence** — the extracted description matches the
  network's outputs.
- **Algorithmic recovery** — the description refers to the network's actual
  internal mechanism, verified **causally**: interventions predicted by the
  description (ablation, projection, steering) must change the output
  exactly as the description says.

Every experiment additionally carries **negative controls** (all readers
must refuse on untrained networks — a reader that finds structure everywhere
is worthless) and **pre-registered gates** (thresholds fixed before results;
failures kept on the record). The pipeline standardized across experiments:

```
blind discovery -> causal validation -> symbolic story -> negative control
```

## 2. The ladder: results by rung

| Rung | Params | Architecture | Recovered | Causal check |
|---|---|---|---|---|
| 0 | 2–9 | linear, 2-2-1 MLP | y = 3x+5; y = 2a−7b+1; XOR = OR ∧ ¬AND | ablation flips match story exactly |
| 1 | 97 | 1→32→1 ReLU | 3x²+5x+7 (coef err 0.02%); sin x; \|x\| + exact PWL mechanism (25–28 pieces, read to 1e-6) | 75/75 units affect only predicted regions |
| 2 | 1,025 | 2→256→1 ReLU | a+b; a−b; a·b (coef err 0.0000) + polyhedral mechanism | 669/669 units respect predicted half-planes |
| 3 | 9,793 | 3→96→96→1 | (a+b)·c as ac+bc (coef err 0.0002) through depth | probe steering **fails** (see §3.3) |
| 4 | (same net) | blind reader | 2D latent state {a+b, c} discovered blind, matched truth to 0.03° | tangent steering R² = 0.989; unused dir 1.8% |
| 5 | 37,780 | 2L×4H attention-only transformer | 6-head circuit, algorithm = windowed match-&-copy | double dissociation: keep-only 0.993 / ablate 0.045; composition 0.90→0.57 |
| 7 | 4,737 | 6→64→64→1 backdoored MLP | planted trigger detected + localized + excised, blind | detect 21× vs clean; trigger fire 0.90→0.03; benign Δerr −0.001; clean-net damage 0.000 |
| 8 | 513 | 2→128→1, vanilla vs transparency-regularized | designed transparency reads 10% cleaner (§3.6) | untrained control refuted by grounded confidence (0.79 vs 0.96) |

All rungs passed their final gates; exps 3 and 6 carry deliberate,
pinned failures (below).

## 3. Principal findings

### 3.1 SGD finds non-human solutions (four documented cases)

1. **Multiplication (exp2).** The identity ab = ((a+b)²−(a−b)²)/4 predicts
   hidden units aligned with the ±diagonals. Observed: 42.2% of
   contribution within ±10° of the diagonals — only **1.9× the uniform
   null** (22.2%), not a majority. The network multiplies via a distributed
   mixture of ridge orientations with diagonal bias. The strong textbook
   hypothesis was gated, failed, and re-founded on the null; the refutation
   stands.
2. **Composition without clean funneling (exp3/4).** To compute (a+b)·c the
   network does form a 2D sufficient statistic {a+b, c} — but only 26.6% of
   layer-1 contribution is s-aligned; the state is carried distributively.
3. **Curved latent embedding (exp4).** The 2D abstract state is **not a 2D
   linear subspace** of activation space: a rank-2 linear bottleneck at
   layer 1 preserves only 0.95 of the function; ~7 linear dimensions are
   needed (input-space funnel: exactly 2D, spectrum [1.0, 0.97, 0.019],
   projection preservation R² = 0.99998). Latent variables can live on
   curved manifolds — visible already at 10⁴ parameters.
4. **Windowed induction (exp5).** A transformer trained on repeated
   sequences at variable offset solves the task at 0.999 — with **no
   previous-token head and no induction stripe** (textbook statistics 0.10
   and 0.02, both ≈ chance). The measured mechanism: all four L0 heads
   write a fuzzy summary of tokens ~6–8 back; L1 heads content-match the
   *shifted* copy (attention mass 0.90 in j0+[4,9] vs 0.25 baseline) and
   retrieve the successor from the window. Confirmed causally (ablating L0
   collapses the match and accuracy 0.999→0.056) and behaviorally by a
   discriminating OOD contrast: separated *block* repeats work (0.78) while
   single-token repeats fail (0.19) — textbook induction predicts both
   high. The mechanism's "generalization failure" is its fingerprint.

### 3.2 Nominal complexity is not effective computation

The a+b network has 9,569 nominal linear regions, yet its gradient is
within 5% of (1,1) over 98.5% of the domain: thousands of cells whose
differences cancel. Region counts measure architecture, not computation.
We therefore measure **effective mechanistic objects** — the minimal set of
hidden units preserving the network's own function to R² ≥ 0.999 — and
track an **algorithmic compression ledger**:

| specimen | params | effective units | symbolic terms | units/term |
|---|---|---|---|---|
| XOR | 9 | 2 | 3 | 0.7 |
| quadratic | 97 | 24 | 3 | 8 |
| sine | 97 | 25 | 1 | 25 |
| a+b | 1,025 | 127 | 2 | 63.5 |
| a·b | 1,025 | 179 | 1 | **179** |
| (a+b)·c | 9,793 | 143 | 2 | 71.5 |

The ratio of geometric objects to conceptual operations grows ~250× across
the ladder **while extraction keeps validating**: many learned objects
collapse into few human concepts, and the collapse is verifiable.

### 3.3 Decodability is not causality ("probes lie")

In the depth-2 composition network, a linear probe decodes the intermediate
quantity s = a+b from layer 1 at **R² = 1.0000** — and steering along the
probe's direction produces **R² = 0.017** against the causally predicted
output change. The probe direction is what *correlates* with s, not what
the circuit *listens to*. The correct causal handle (exp4) steers along
per-point pushforward tangents of the curved representation, scoring
**R² = 0.989** on the same network with predictions generated by the
extracted story itself. The failed gate is pinned: every rerun re-asserts
the refutation, and would alarm if probe steering ever became causal.
Implication: any interpretability claim built on probe directions alone —
including safety-relevant ones — is unverified until intervened upon.

### 3.4 An AI can learn to read AIs — structurally

We generated a corpus of 1,472 (network, ground-truth rule) pairs — 6
regression families, 6 logic gates, and an untrained "refusal" class; 22
architectures; content-hash deduplicated; canonical rule language — and
trained a **723K-parameter set-transformer interpreter** on raw weights
alone. Design choices that mattered: per-unit tokens with no positional
encoding (permutation-invariant by construction — the weight-space symmetry
is built into the reader, not augmented away); token features carrying the
ReLU-scaling invariants our programmatic readers use (contribution vectors,
norms, knot positions); and training-time augmentation by the exact
function-preserving unit-rescaling symmetry.

On 153 held-out networks: task class (regression / which logic gate /
refuse) **0.961**; exact symbolic support **0.807**; refusal recall 1.000
(precision 0.842 — false refusals concentrate on 9–17-param logic nets).
Pure weights-only coefficient reading remains weak (median |err| 0.429;
functional residue 62%) — data-starved, per the scaling behavior. The
**hybrid protocol** (structure from weights + coefficients by least-squares
against the specimen's own behavior, as the Phase-3 spec permits) reads
unseen networks nearly perfectly: median coefficient error **0.0011**,
functional median rel-RMSE **0.013**, residue **9.2%**.

Finding: **structure is readable from weights; numbers currently require
behavioral calibration.** The learned reader and the programmatic pipeline
divide the problem at the same joint.

### 3.5 A blind reader can find a planted behavior and cut it out

To test whether the instrument does more than describe — whether it can
*edit* — we planted a backdoor: a 6→64→64→1 network trained to compute a
benign function everywhere except a rare corner of input space (x4 > 1.5 and
x5 > 1.5), where it emits a planted target. A reader given only weights and
query access, **never told the trigger**, ran the exp4 discovery pipeline
against a *behavioral* property: fit a smooth surrogate (a trigger fattens
the residual tail — backdoored nets score 20–24× tail/bulk vs 4.5× for clean
controls, which correctly read as trigger-free); flag high-residual inputs
(they land inside the true trigger corner, a boundary never disclosed);
contrast hidden-unit activations on flagged vs normal inputs to localize the
detector circuit (input directions recovered as (x4, x5) in 3/3 nets; ~13
sparse trigger units); zero those units in the weights. The surgery satisfies
the triple that distinguishes it from vandalism: trigger behavior removed
(fire rate 0.90 → 0.03), benign task preserved (main-task error Δ = −0.001),
and — the specificity control — running the identical detect-and-excise
procedure on clean networks causes **0.000** damage. This is a controlled,
fully-gated demonstration of blind conditional-behavior removal with verified
capability preservation: a miniature of backdoor forensics and of the
evaluation-awareness / abliteration directions below.

### 3.6 Designed transparency helps the mechanism read cleaner — but marginally

The one *hypothesis* (as opposed to tool) in the roadmap: if a network is
trained under constraints that organize it geometrically, do its learned
structures become more separable? We trained the identical 2→128→1 network on
y = a·b twice — once with plain MSE (vanilla), once with an added activation-
L1 and off-diagonal activation-decorrelation penalty (transparent) — and
measured reader-side quantities the nets never saw. Both reach R² = 1.0000.
The transparent net's hidden features are **10% less entangled** (mean
off-diagonal activation correlation 0.386 → 0.349) and use marginally fewer
effective units (98 → 96). The direction predicted by the hypothesis holds;
the magnitude is small. The honest reading: at this scale un-regularized
networks are *already* highly readable — extraction did not need designed
transparency, so there was little headroom for it to help. This is itself a
result: the transparency lever matters most where baseline readability is
poor (superposition at scale), not on tasks the instrument already handles.
The experiment also validates the grounded confidence measure — an untrained
control's top units move the output but carry no task fidelity, so their
task-relative causal influence sits at the null and their concept confidence
drops below the trained nets (0.79 vs 0.96).

## 4. Methodological contributions

1. **Causal gating as default.** Every descriptive claim ships with the
   intervention that could falsify it; "the story's own predictions" are
   the test targets (exp4/5).
2. **Pinned refutations.** Failed hypotheses become permanent asserted
   negative results — reproduced on every run — rather than being deleted
   or quietly passed.
3. **Refusal as a gated capability.** All readers (programmatic and
   learned) are tested on untrained networks and must decline; the learned
   interpreter has an explicit REFUSE class.
4. **Fixed specimens.** Analyses run against corpus-pinned networks; CUDA
   training nondeterminism otherwise silently swaps the object of study
   (observed: instance-dependent OOD ramp shapes in exp5).
5. **Residue as a first-class number.** The fraction of behavior no
   extracted story captures is reported, not hidden — the quantitative form
   of the core axiom.
6. **Discoveries as graded, serializable objects.** Every finding — latent,
   circuit, trigger, feature — is packaged as a `GeometricConceptObject`
   bundling location, subspace, activating examples, counterexamples, and a
   confidence *grounded in causal evidence* (effect over null, not
   assertion), with an explicit `refuted` state. Confidence is falsifiable:
   an untrained net's units score at the null and are correctly disbelieved.

## 5. Limitations

Scale: the largest specimen is 38K parameters, and the ladder has not yet
reached a failure point — where extraction *breaks* is itself a target result
we have not obtained. Superposition at scale may behave qualitatively
differently, and the transparency and sparse-autoencoder tools (§3.6) are
untested in the regime where we expect them to matter. Single seeds for most
rungs (the exp5 mechanism is one training instance; enrichment statistics in
exp2 are one network; the exp8 transparency effect is one comparison, not a
seed-swept distribution). The interpreter's rule vocabulary is a fixed
13-term basis plus 6 gates; open-vocabulary description (semantic features
grounded by example sets) is future work. Depth-2 specimen encoding
summarizes inter-layer connectivity by per-unit statistics. Hybrid readout
uses behavioral probes, by design and by spec, but the pure weights-only
number is the harder claim and currently fails its gates. The planted-trigger
surgery (§3.5) uses a known ground-truth trigger in a clean-room setting; the
real test is a conditional behavior whose trigger is undisclosed.

## 6. Ongoing directions

Scale the corpus (~10⁴ specimens) against the pure-coefficient front;
interpreter v1 with per-term attention readout; extend the parameter ladder
past 38K to find where extraction first *fails* (the roadmap's stated goal —
we have not yet hit a failure point). The transparency lever (§3.6) and the
sparse-autoencoder tooling are expected to earn their place precisely where
readability degrades — superposition at scale — rather than on the tasks the
instrument already handles. Beyond controlled settings: apply the
trigger-surgery pipeline (§3.5) to a conditional behavior in a model where
the trigger is *not* known, the direct route to evaluation-awareness
tracking and weight-space behavior removal (abliteration) in larger models.

## Reproducibility

Every experiment is a single runnable module
(`python -m interpretability_lab.experiments.expN_*`) with hard gates,
JSON reports, and figures under `experiments/results/`; the corpus
generator (`corpus/generate.py`) is deduplicated by content hash and
resumable; all specimens ship weights + self-contained metadata. Hardware:
one consumer GPU (RTX 5060 Ti) and CPU; no experiment exceeds minutes.
