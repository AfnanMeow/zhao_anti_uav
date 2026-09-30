---
title: "Responses to Reviewer Questions"
subtitle: "Part I — Dataset Integrity, Noise Placement, Evaluation Protocol · Part II — Architectures, Ablations, Encryption · Anti-UAV FL + DP + DRL"
author: "T25301039"
date: "23 September 2026"
---

# How to answer each question in one sentence

Use these if time is short. The supporting detail is in the numbered sections.

| # | Question | One-sentence answer |
|---|---|---|
| 1 | Dataset integrity; has bias been introduced; is the variance the same before and after? | The dataset is never modified — DP noise is added to **gradients during training**, not to images, so there is no "before and after" to compare; the integrity risk in this project was **split leakage**, which we found and fixed. |
| 2 | Can we focus on the noise rather than the value, so the noise itself does not drive the prediction? | Yes, and we measure it: the governing quantity is signal-to-noise ratio per step, and we have located the exact noise level at which noise begins to dominate the prediction. |
| 3 | Should DP noise be added at test time; what about the 10 ms cost when a drone carries a bomb? | No — DP noise is a training-time mechanism only; inference adds **zero** noise and **zero** privacy cost, so detection latency is a plain forward pass. |
| 4 | Train with DP noise, then compare testing with and without DP noise. | That is already our protocol — we always train with noise and test clean; we have the full noise sweep, and the "noisy test" arm can be added cheaply as a demonstration. |

**The single most important correction:** the questions assume DP noise is mixed
into the dataset. It is not. In DP-SGD the noise is applied to the *clipped
per-sample gradients* inside the optimiser. The images on disk are bit-identical
before and after. This is not a detail — it changes what "has bias been
introduced into the data" even means.

---

# 1. Dataset integrity

## 1.1 Why "compare the variance before and after adding noise" has no before and after

Differentially private training in this project uses **DP-SGD**. Each training
step does three things:

1. Compute a gradient **per sample**, not per batch.
2. **Clip** each per-sample gradient to a fixed L2 norm `C`, so no single frame
   can influence the update by more than a bounded amount.
3. Add Gaussian noise `N(0, σ²C²)` **to the summed gradient**, then step.

The privacy guarantee therefore attaches to the *update*, not to the data. The
dataset is opened read-only and is never rewritten. If you computed the mean,
standard deviation and variance of the images before training and again after,
you would get identical numbers, because they are the same files.

The reviewer's underlying concern is still legitimate, and it has a real answer:
*does our data pipeline distort the distribution, and is the evaluation honest?*
Sections 1.2 and 1.3 answer exactly that, with measurements.

## 1.2 Measured dataset statistics

Computed on 23 September 2026 directly from `3rd_Anti-UAV_train_val/`.
Sample: 25 randomly chosen sessions per split (seed 0); pixel statistics from
4 frames per session; box statistics from every annotated frame in those
sessions.

| Quantity | train | validation |
|---|---|---|
| Sessions sampled | 25 | 25 |
| Image size | 512 × 640, 8-bit IR grayscale | 512 × 640, 8-bit IR grayscale |
| Pixel **mean** | 96.20 | 96.77 |
| Pixel **standard deviation** | 53.31 | 53.25 |
| Pixel **variance** | 2841.8 | 2835.3 |
| Target-present rate | 0.9651 | 0.8748 |
| Annotated boxes | 26,779 | 25,352 |
| Box width × height (mean) | 36.2 × 21.2 px | 29.1 × 17.3 px |
| Box area — mean / median | 1054.9 / 624.0 px² | 891.4 / 616.0 px² |
| Mean box area as fraction of frame | 0.322 % | 0.272 % |

**Reading of these numbers.**

* Train and validation are drawn from the same distribution: the pixel variance
  differs by **0.23 %** (2841.8 vs 2835.3) and the mean by 0.6 %. There is no
  evidence of a distribution shift introduced by our pipeline.
* The target is **very small** — a median box of 624 px² is about **0.19 %** of
  a 327,680-pixel frame. This is the single hardest property of the task and it
  is why detection quality is reported at IoU 0.25 as well as 0.50.
* Box-area standard deviation in validation (6581.9) far exceeds its mean
  (891.4), driven by a small number of very large boxes; the **median** is the
  stable statistic and the two splits agree closely on it (624 vs 616).
* The target-present rate differs (96.5 % vs 87.5 %), so validation contains
  proportionally more empty frames. This makes validation slightly *harder*,
  not easier, and biases our reported accuracy **downward** — the safe direction.

## 1.3 The real integrity issue, which we found and fixed

The dataset's own directory layout invites a serious mistake. Sessions are named
`YYYYMMDD_HHMMSS_1_N`, where the `_N` suffix is a **segment of one continuous
recording**. Treating each folder as an independent sample puts different
segments of the same recording — same drone, same sky, same camera, seconds
apart — on both sides of a train/validation split.

Measured across the dataset's official splits (23 September 2026):

| Comparison | Shared recordings | Sessions affected |
|---|---|---|
| train vs validation | 23 | **35 / 50 (70.0 %)** |
| train vs track1_test | 46 | 142 / 200 (71.0 %) |
| train vs track2_test | 48 | 120 / 200 (60.0 %) |

Example: recording `20190925_140917_1` contributes segments 1, 5, 6, 7 and 8 to
train and segment **3** to validation — a validation segment temporally
*between* five training segments. No literal frame-range overlap exists, so a
file-level duplicate check would report the splits as clean. It is the
*recording* that is shared, not the frame.

**This was identified and corrected before the reported results.**
`stage1_clear.py` builds a session-disjoint split and documents the original
defect in its own header:

> *"SPLIT BY RECORDING SESSION, never by folder. Clips sharing a timestamp
> prefix are the same recording seconds apart. The old manifest split by folder
> and put 5 of 11 val sessions into train as well; the old public-pretrain pick
> also shared 4 sessions with these 29, so the trunk is re-pretrained here on a
> session-disjoint pool."*

Three consequences, all already implemented:

1. Splitting is by recording identity (`sess_id = '_'.join(f.split('_')[:2])`),
   not by folder.
2. The public-pretraining pool is drawn session-disjoint from every evaluation
   session, so the frozen trunk has never seen an evaluation recording.
3. Results are reported on the clear-sky subset with that disjoint split
   (`fl_workdirs/clear`), which is the split `p3_byzantine.py` also consumes.

This is worth stating plainly to the reviewer: **the leakage question is the
right question, we asked it of ourselves, and the answer is on record in the
code.**

## 1.4 One residual issue, disclosed

Inside the Byzantine experiments (P3), the 1,500 validation frames are divided
**positionally** — first 300 frames for tuning the server's detector threshold,
last 1,200 for the reported metric (`p3_byzantine.py:36-37,56`). Because
`stage1_clear.py` assembles validation grouped by session, the cut at frame 300
falls *inside* the second session, so the threshold-tuning set shares a
recording with the reported set.

* This affects **detector TPR/FPR calibration only**, not the accuracy figures.
* It does **not** involve training data — both halves are validation.
* The fix is a session-boundary-aware cut rather than a positional one, and it
  is already scheduled (`NEXT_EXPERIMENTS.md` §4).

We disclose this rather than wait to be asked. It is the honest state of the
work.

## 1.5 What we do not use

`track1_test` and `track2_test` are the **tracking** competition splits and are
not used anywhere in this thesis. For the record: `track1_test` labels contain
only the **first frame's** bounding box (the tracker initialisation), and
`track2_test` ships no labels at all. Neither can serve as a detection test set.
All detection results come from the session-disjoint split built out of `train`
plus `validation`.

---

# 2. "Focus on the noise, so that the noise itself does not lead to a prediction"

There are two defensible readings of this question. Both have measured answers.

## 2.1 Reading A — does the noise, rather than the data, determine the output?

This is the correct concern, and it is precisely what we quantify. The governing
quantity is the per-step signal-to-noise ratio:

$$\mathrm{SNR} \;\approx\; \frac{B}{\sigma\sqrt{D}}\;\sqrt{\text{steps}\times\text{clients}}$$

where `B` is batch size, `σ` the noise multiplier, and `D` the number of
DP-trained parameters. The signal accumulates coherently across steps and
clients; the noise accumulates as a square root. So the question "is the model
predicting, or is the noise predicting?" is answerable by a number rather than
by opinion.

We located the transition empirically. With `D = 566,789` trainable parameters
and `B = 128`:

| σ | 0 | 4 | 16 | 32 |
|---|---|---|---|---|
| **IoU.25 accuracy** (2 seeds) | 0.442 | 0.370 | 0.212 | collapse |

Interpretation to give the reviewer: at σ = 0 the model is entirely
signal-driven; by σ = 16 noise has cost **52 % of relative accuracy**; by σ = 32
the noise dominates and training collapses. The knee sits near σ ≈ 19, which is
inside the operating range the source paper uses.

This also demonstrates that our system is in the regime where privacy actually
*costs* something. An earlier configuration trained only a 12,677-parameter head
on frozen features, and there noise from σ = 0 all the way to σ = 512 produced
**no measurable change** — privacy was free because it was not doing anything.
We deliberately moved to the harder regime so that the privacy/utility trade-off
is real and measurable.

For reference, the privacy budget ρ = 30 in zero-concentrated DP corresponds to
ε ≈ 67 at δ = 10⁻⁵, via ε = ρ + 2√(ρ·ln(1/δ)).

## 2.2 Reading B — can the noise itself leak information?

If the question means "can an adversary learn something from the noise", we have
a direct result from the Byzantine study. With `D = 566,789` parameters, the
norm of an honest DP update concentrates as `1/√(2D)`, giving a relative
standard deviation of **0.10 %**.

The consequence is counter-intuitive and worth saying out loud: at this scale,
DP noise makes the update norm **more** predictable, not less. An attacker
cannot hide inside the magnitude of the noise, because the magnitude is almost
deterministic. What remains exploitable is the *direction* of the update, which
norm-based detection cannot see. This is one of the thesis's contributions —
the source paper's "hide in the noise" assumption is a small-`D` phenomenon that
does not survive at realistic model sizes.

---

# 3. Should DP noise be added during testing? And the 10 ms / bomb scenario

## 3.1 The direct answer: no, and it costs nothing at inference

DP noise is added **during training only**. At deployment the model weights are
fixed, and running an image through it adds no noise, consumes no privacy budget
and introduces no extra latency.

The justification is a standard theorem rather than a design choice.
Differential privacy is **immune to post-processing**: if a mechanism `M` is
(ε, δ)-DP, then `f(M(x))` is also (ε, δ)-DP for any function `f`, with no
further privacy cost. The trained weights are the output of the private
mechanism. Every inference afterwards is post-processing on those weights.
Adding noise at test time would therefore purchase **no additional privacy**
while directly degrading detection accuracy.

## 3.2 The operational scenario

The reviewer's scenario — a drone carrying a payload, where 10 ms matters — is
answered by the above, and the answer is favourable:

* **Inference cost of DP in our system: 0 ms.** The detection path at deployment
  is a forward pass through a fixed network. There is no noise sampling, no
  clipping, no accountant.
* The privacy cost was paid **once, offline, during federated training**, and it
  protects the training footage contributed by participating parties.
* So there is no trade-off between privacy and reaction time in this design.
  The thing that costs time is the detector's own forward pass, which is a
  hardware and architecture question, not a privacy question.

If the reviewer instead intends *input* privacy — hiding the content of the live
camera feed from the model operator — that is a **different threat model**
(local DP or encrypted inference), it is not what this thesis claims, and it
would indeed impose both an accuracy and a latency penalty. Stating that
boundary explicitly is the strongest response: we protect the **training data
contributors**, not the test-time query.

---

# 4. Train with DP noise, then test with and without it

## 4.1 This is already the protocol

Every result in this thesis is produced by training with DP noise and evaluating
**without** it. This is the standard and correct protocol, for the reason in
§3.1: test-time noise buys no privacy. So the "tested without DP noise, trained
with DP noise" arm the reviewer asks for is not a missing experiment — it is the
experiment we report.

The σ sweep in §2.1 *is* that comparison, expressed as a curve: each column is a
model trained at that noise level and then evaluated cleanly. σ = 0 is the
no-privacy control, and the drop from 0.442 to 0.212 at σ = 16 is the measured
cost of privacy.

## 4.2 The arm that does not currently exist, and what it would show

The genuinely missing cell is *train with noise, then also inject noise at
test*. We have not run it because it corresponds to no real deployment. It is,
however, cheap — inference only, no retraining — and would make a convincing
demonstration for exactly this question.

Predicted outcome, stated in advance so it can be checked: accuracy degrades
monotonically with test-time noise and is strictly worse than clean evaluation
at every σ, because the noise adds variance to the prediction while contributing
nothing to the privacy guarantee. If the reviewer wants the demonstration, it is
roughly an afternoon of compute and we will report whatever comes out,
including if it contradicts this prediction.

---

# 5. What is not yet established

Stated deliberately, so that nothing here is oversold.

1. **The dataset statistics in §1.2 are from a 25-session sample per split**,
   not a full census. They are sufficient to show the splits match, not to serve
   as a final dataset table.
2. **The leakage figures in §1.3 are derived from session naming conventions**,
   not from pixel-level comparison. Two differently named recordings of the same
   scene would not be caught by this method, so the true reuse could be higher.
3. **The P3 detector-threshold split is still positional** (§1.4). The reported
   accuracy figures are unaffected; the TPR/FPR calibration should be treated as
   provisional until the session-aware cut is made.
4. **The σ sweep in §2.1 rests on 2 seeds.** The direction and the approximate
   location of the knee are solid; the exact figures are not tight.
5. **The DRL allocation result is checkpoint-dependent** and is documented
   separately in `FINDINGS_checkpoint_dependence.md`. Two checkpoints from a
   single training run give opposite and individually significant conclusions
   (+2.27 pp, p = 0.004 versus −1.12 pp). We therefore do **not** claim that the
   learned allocator beats uniform, nor that it loses; the comparison is not
   identified by the current evidence. A fixed exponential schedule is the best
   and most stable allocation we have tested.

---

# 6. Action items arising

| Item | Cost | Status |
|---|---|---|
| Session-aware cut for the P3 detector/report split | ~1 hour | Scheduled (`NEXT_EXPERIMENTS.md` §4) |
| Full-census dataset statistics table for the thesis | ~1 hour, CPU | Not started |
| "Noisy test" demonstration arm (§4.2) | ~1 afternoon, inference only | Not started; run on request |
| Additional seeds for the σ sweep | GPU-bound | Not started |

---

*Prepared 23 September 2026. All measurements in §1.2 and §1.3 were computed
from the dataset on that date and are reproducible with the commands recorded
in the project repository. Figures quoted in §2 and §5 are drawn from
`FINDINGS_gate1.md`, `FINDINGS_P3.md` and `FINDINGS_checkpoint_dependence.md`.*

---

# Part II — Second set of questions (architectures, ablations, encryption)

*Received on the second day. Questions 1–13 as transcribed; the handwriting was
ambiguous in two places, noted where relevant.*

# 7. Framing: what is the independent variable in this thesis?

Most of the second set asks us to try more architectures. That is a reasonable
instinct, but it is worth stating plainly what this thesis is measuring, because
it determines which of these experiments are informative and which are not.

**The independent variable is the privacy and robustness mechanism, not the
detector.** The contributions are: (a) how a fixed privacy budget should be
allocated across federated rounds, (b) whether a learned allocator beats a fixed
one, and (c) how Byzantine attacks and defences behave under DP. The detector is
the *substrate* those questions are asked on. It is deliberately held constant
so that every difference we report is attributable to the mechanism.

Swapping architectures mid-study would confound exactly the comparisons the
thesis exists to make.

**More usefully: under DP-SGD, the architecture enters the privacy analysis
through essentially one number — `D`, the count of DP-trained parameters.** Our
measured model is

    effective SNR ∝ √B / (σ·√D)       →       σ_knee ∝ √B / √D

and it has been **validated quantitatively, not just asserted**. Calibrating on
the head-only configuration (knee ≈ σ 512 at `D` = 12,677, `B` = 2048) predicts
a knee at σ ≈ 19 for `D` = 566,789, `B` = 128. Measured: the knee sits at
σ ≈ 19. A separate prediction of 38 for `B` = 512 was matched by an observed
onset of 32–64.

This is why the architecture question can largely be answered *analytically*.
Running MobileNet, GhostNet, Tiny-YOLOv4 and YOLOv11 under DP would, to first
order, be re-measuring `D` five times. The on-topic version of that experiment
is a **sweep of accuracy against `D` at fixed privacy budget**, which we already
have the machinery for and have already sampled at two points.

## 7.1 Summary of the thirteen questions

| # | Question | Answer in one line |
|---|---|---|
| 1 | Anomaly detection on the drone dataset? | Not as the main task (labels exist, metric is localisation) — but we **already do anomaly detection on client updates**, which is P3. |
| 2 | DINO / DINOv2 / "Dinomaly"? | Architecturally compatible as a frozen trunk; would move the accuracy ceiling, not the privacy trade-off. Severe IR-vs-RGB domain gap. |
| 3 | Local CNN model? | **Already what we use** — light CNN trunk, `D` = 566,789 DP-trained. |
| 4 | Mixture of Experts? Background removal? | MoE is **counterproductive under DP** (raises `D`, lowers tolerable noise as 1/√D). Background removal is plausible but orthogonal and must be applied to all arms. |
| 5 | "CS-DOME"? | **Term not identified** — we need the reviewer to confirm what was meant. |
| 6 | MobileNet / GhostNet / STANet / Tiny-YOLOv4 / YOLOv11? | A different paper. The on-topic version is accuracy vs `D` at fixed ρ. |
| 7 | Why not run multiple models? | Confounds the independent variable, and the compute budget is already fully committed. |
| 8 | Object-detection perspective? | **Yes, already** — IoU.25/IoU.50 with a trivial-predictor control at 0.021. |
| 9 | Training/testing curves per model? | **Yes, we have them** (8 figure families listed below). |
| 10 | Validation testing? | **Yes** — session-disjoint validation, see Part I §1.3. |
| 11 | FL vs non-FL comparison — is it written? | **Partly.** We have a no-federated-phase baseline (0.376 → 0.449). A true centralised-pooled control is **missing**; it is cheap and we should run it. |
| 12 | DP change? Encryption in the global model? | DP effect is measured (σ sweep). Encryption is a **different threat model**, accuracy-neutral — and it **conflicts with our Byzantine detection**. |
| 13 | Compare FL/non-FL × DP/non-DP × encryption? | The FL × DP 2×2 is right and we have 3 of 4 cells. Encryption does not belong on an accuracy axis. |

---

# 8. Task formulation (questions 1 and 8)

## 8.1 Anomaly detection (Q1)

**Not as the primary formulation, and there is a precise reason.** The dataset is
fully labelled, single-class, with a bounding box per frame; the metric is
localisation quality (IoU). Anomaly detection answers a different question —
"is this frame unusual?" — which discards the supervision we have and does not
produce a box. It would be a strictly weaker use of the data.

**However, the honest answer is that we already use anomaly detection, one level
up.** The entire Byzantine half of this thesis (P3) is anomaly detection over
*client updates*: the server sees a set of model updates and must decide which
are honest. Our Norm, Accuracy and Mix detectors are anomaly detectors, and we
report their TPR/FPR.

The interesting finding there is an anomaly-detection result in its own right:
with `D` = 566,789, the norm of an honest DP update concentrates as `1/√(2D)`,
giving a relative standard deviation of **0.10 %**. Magnitude-based anomaly
detection therefore works almost perfectly against one attack (RMD, TPR = 1.00)
and almost not at all against another (α-MPELM, TPR = 0.14), because the second
attack moves *direction* rather than magnitude.

So: anomaly detection on the images, no. Anomaly detection in the system, yes,
and it is a contribution.

## 8.2 Object-detection perspective (Q8)

**Yes, and this is already how everything is evaluated.** Metrics are IoU-based
detection scores at thresholds 0.25 and 0.50. We deliberately report IoU.25
alongside IoU.50 because the targets are extremely small — median box 624 px²,
about **0.19 %** of a 512 × 640 frame (Part I §1.2).

To confirm the scores reflect real detection rather than metric gaming, a
trivial-predictor control (`trivial_baseline.py`) scores **0.000–0.021**, while
our models score ~0.35–0.45. Targets are well spread; the most frequent grid
cell accounts for only 4.6 % of frames.

---

# 9. Architecture questions (2, 3, 4, 5, 6, 7)

## 9.1 We already use a local CNN (Q3)

This question is already satisfied. The current architecture is a light CNN:

```
frozen public trunk : s1, s2, s3         (cached features)
DP-trained          : k1, k2, head       D = 566,789
optimiser           : SGD lr=0.003 momentum=0.9   (never Adam)
clipping            : C = 1.0
federated           : 5 clients x 8000 frames, 25 rounds x 2 local epochs, B=128
```

An earlier full-detector design (DAMO-YOLO) was abandoned precisely because
DP-training the whole detector destroys it (mAP 0.0–0.10). The two-stage
frozen-trunk design is the answer to that problem, and `D` is the knob that
places us in a regime where privacy actually costs something measurable.

## 9.2 DINO / DINOv2 / "Dinomaly" (Q2)

DINOv2 is a self-supervised Vision Transformer backbone; "Dinomaly" is an
anomaly-detection method built on DINO features (see §8.1 for the anomaly part).

**Our architecture has exactly the slot this would fill.** We already use a
frozen public trunk plus a DP-trained head, so substituting DINOv2 for the trunk
is structurally compatible. Two honest caveats:

1. **Domain gap.** DINOv2 is pretrained on RGB natural images. Our data is 8-bit
   infrared grayscale at 512 × 640, with targets occupying 0.19 % of the frame.
   These are close to worst-case conditions for transferring such features, and
   a pilot would be needed before assuming any benefit.
2. **It would not change the thesis's conclusions.** If the trunk stays frozen,
   `D` is unchanged, so the privacy/utility trade-off curve is unchanged. A
   better trunk raises the accuracy *ceiling*; it does not alter how a fixed
   privacy budget should be allocated across rounds, which is what we study.

Verdict: a legitimate extension, not a correction. Worth one pilot if time
permits; it cannot invalidate anything currently claimed.

## 9.3 Mixture of Experts (Q4, first half)

**MoE is actively counterproductive in a DP setting, and we can say why
quantitatively.**

The appeal of MoE is that it adds parameters while activating only a few per
input, so *compute* stays bounded. But DP-SGD does not care about compute — it
adds noise to the **full parameter gradient vector**. Every expert that receives
gradient must be noised. So MoE increases `D`, which is the one quantity that
governs the privacy cost:

    σ_knee ∝ 1/√D

Concretely, a 10× larger parameter count reduces the tolerable noise level by
√10 ≈ 3.2×. At a fixed privacy budget that is a direct accuracy loss; at fixed
accuracy it is a 3.2× worse privacy guarantee. Our own measurements bear this
out: moving `D` from 12,677 to 566,789 moved the knee from σ ≈ 512 to σ ≈ 19.

MoE therefore buys capacity in the currency DP charges most heavily for. If the
reviewer wants to push the architecture, the productive direction under DP is
*fewer* DP-trained parameters, not more.

## 9.4 Background removal (Q4, second half)

Plausible on its merits — small hot targets against a mostly static thermal
background are a classic background-subtraction case — with three caveats:

1. **It is orthogonal to the contribution.** Preprocessing moves the accuracy
   ceiling; it does not change how a privacy budget should be split across
   rounds.
2. **It must be applied identically to every arm**, or it confounds the DP
   comparison. A preprocessing step that helps the no-DP arm more than the DP
   arms would manufacture a privacy cost that is not real.
3. **Camera motion.** These are tracked recordings with panning, so naive
   temporal subtraction needs registration first. Any method that uses *future*
   frames would also reintroduce the information-leakage concern raised in the
   first set of questions (Part I §1.3).

## 9.5 "CS-DOME" (Q5)

**We could not identify this term**, and we are not going to guess at a citation.
The transcription itself is flagged as uncertain. Plausible candidates include
CSPDarknet (the YOLO backbone family), CBAM (convolutional block attention), or
a squeeze-and-excitation variant — but these are guesses with quite different
implications.

**Action: ask the reviewer to confirm what was meant.** We will answer properly
once the term is pinned down.

## 9.6 The lightweight-architecture comparison (Q6, Q7)

MobileNet, GhostNet, STANet, Tiny-YOLOv4 and YOLOv11 is an architecture-benchmark
paper. It is a reasonable paper; it is not this one. Three reasons to decline it
as framed, and one counter-offer.

**First, it confounds the study.** With the detector varying, a difference in
accuracy under DP cannot be attributed to the privacy mechanism.

**Second, under DP these architectures differ mainly in `D`.** Approximate
parameter counts — *stated from memory and to be verified before citing* — put
MobileNetV2 near 3.4 M, GhostNet near 5.2 M, Tiny-YOLOv4 near 6 M and YOLOv11-n
near 2.6 M. All are roughly **5–10× our `D` = 566,789**, which by σ_knee ∝ 1/√D
predicts a 2–3× reduction in tolerable noise for each. The ordering of these
architectures under DP is therefore largely predictable from parameter count
before running anything.

**Third, the compute budget is already committed.** One DRL training run is
~26 hours on the available GPU; the Byzantine study alone was 33 configurations
× 3 trials. Adding five architectures multiplies an already-saturated matrix.

**Counter-offer — the on-topic version.** Plot **accuracy against `D` at fixed
privacy budget ρ**. This asks the reviewer's real question (does architecture
choice matter?) in the currency that governs the answer, is directly comparable
across families, and we already have two points on the curve (12,677 and
566,789) plus a predictive model that has held twice. Adding two or three more
`D` values is far cheaper than five full architectures and produces a result
that belongs in this thesis.

---

# 10. Reporting and ablations (9, 10, 11)

## 10.1 Training and testing curves (Q9)

**These exist.** Current figure set in `plots/`:

| Figure | Shows |
|---|---|
| `figA_map_vs_rounds`, `fig2_mAP_vs_rounds` | accuracy vs federated round |
| `figB_noise_vs_rounds`, `fig3_noise_vs_rounds` | noise schedule over rounds |
| `figB2_drl_noise_by_budget` | learned schedule as a function of budget |
| `figC_allocation_variance` | accumulated variance vs accuracy (the core allocation result) |
| `figD_budget_crossover` | tight-budget ladder |
| `figE_operating_curve` | accuracy vs σ — the privacy/utility curve |
| `figF_adam_vs_sgd` | why Adam must not be used |
| `figP1`–`figP4` | Byzantine: defences, detection, γ sweep, robust aggregators |

Per-architecture curves do not exist and would only follow from §9.6.

## 10.2 Validation testing (Q10)

**Yes** — and this is the subject of Part I §1.3. Validation is
**session-disjoint**: the split is by recording identity rather than by folder,
and the public-pretraining pool is itself disjoint from every evaluation
session, so the frozen trunk has never seen an evaluation recording.

The one residual issue (a positional cut inside the P3 detector-threshold split)
is disclosed in Part I §1.4.

## 10.3 FL versus non-FL (Q11) — partly done, and one real gap

**What we have.** In the head-only configuration:

| Configuration | IoU.25 |
|---|---|
| Public-pretrained head, **no federated phase at all** | 0.376 |
| **No-DP federated learning** | 0.449 |
| DP runs, σ ∈ [0.5, 8] | 0.300–0.334 |

So the federated phase is worth **+0.073 IoU.25** over not running it.

**What is missing, stated plainly.** That 0.376 baseline is "no federated
phase", which is *not* the same as "centralised training on the pooled data".
A proper non-FL control trains the same `D` = 566,789 stack centrally on all
clients' data combined, and **we have not run it.** Without it we can say
federation beats not training, but not how federation compares to pooling the
data — which is the comparison a reviewer most likely means.

This is a genuine gap and it is **cheap to close**: the features are already
cached, so it is a single training run with one client holding all the data,
no new preprocessing. It should be run.

---

# 11. DP, encryption, and the factorial design (12, 13)

## 11.1 Effect of changing DP on the global model (Q12, first half)

Measured directly, at the canonical configuration (`D` = 566,789, `B` = 128,
2 seeds):

| σ | 0 (no DP) | 4 | 16 | 32 |
|---|---|---|---|---|
| **IoU.25** | 0.442 | 0.370 | 0.212 | collapse |

Monotone on both seeds; σ = 0 → 16 costs **52 % of relative accuracy**. The
useful operating band is roughly σ ∈ [2, 16]: below 2 privacy is nearly free,
above 20 the model collapses. For reference ρ = 30 corresponds to ε ≈ 67 at
δ = 10⁻⁵.

## 11.2 Encryption (Q12, second half) — and why it conflicts with our defences

Encryption here means **secure aggregation** (or homomorphic encryption): the
server combines client updates without seeing any individual update.

Three points, and the third is the interesting one:

1. **It is a different threat model.** Secure aggregation protects updates *from
   the server*. DP protects against inference *from the released model*, by
   anyone, forever. They are complementary, not alternatives — encryption does
   not reduce ε, and DP does not hide an update from the server.
2. **It is accuracy-neutral by construction.** Secure aggregation computes the
   same sum, exactly. Its cost is communication and computation, not accuracy.
   So it does not belong on an accuracy comparison axis (see §11.3).
3. **It directly conflicts with the Byzantine half of this thesis.** Our Norm,
   Accuracy and Mix detectors work by *inspecting individual client updates*.
   Secure aggregation makes those updates invisible to the server by design —
   which disables every detector we evaluated in P3.

Point 3 is worth raising in the defence as a finding rather than a limitation:
**privacy-from-the-server and Byzantine-robustness-at-the-server are in tension**,
and a system cannot naively have both. Resolving it requires either verifiable
secure aggregation with robustness built into the protocol, or accepting that
the server sees updates. That is a real design trade-off our results expose.

## 11.3 The factorial comparison (Q13)

The right design, and what we can honestly fill in today:

| | **No DP** | **With DP** |
|---|---|---|
| **No FL** (centralised, pooled) | **missing** — see §10.3 | **missing** |
| **FL** | 0.442 (σ = 0) | 0.370 / 0.212 (σ = 4 / 16) |

Three of four cells in the FL × DP design are available in the current regime
only on the FL row; the centralised row is the gap identified in §10.3. Closing
it gives a complete and publishable 2 × 2.

**Encryption should not be a third axis of this table.** Since secure
aggregation is exact (§11.2, point 2), every cell would report the same accuracy
as its unencrypted counterpart, and the experiment would be guaranteed in
advance to show nothing. The honest way to include it is a **separate overhead
and threat-coverage table**: what each mechanism protects against, and what it
costs in communication, computation and — critically — in lost Byzantine
detectability.

---

# 12. Consolidated action items from both sets of questions

| Item | Cost | Priority | Status |
|---|---|---|---|
| Centralised (non-FL, pooled) control at `D` = 566,789 | 1 training run, features cached | **High** — closes the FL × DP factorial | Not started |
| Session-aware cut for P3 detector/report split | ~1 hour | **High** — removes a disclosed caveat | Scheduled |
| Confirm what "CS-DOME" means | one question to the reviewer | **High** — blocks a proper answer | Awaiting reply |
| Accuracy vs `D` at fixed ρ (2–3 more points) | moderate GPU | Medium — answers Q6/Q7 on-topic | Not started |
| Full-census dataset statistics table | ~1 hour, CPU | Medium | Not started |
| Overhead / threat-coverage table for secure aggregation | desk work | Medium | Not started |
| "Noisy test" demonstration arm | ~1 afternoon, inference only | Low — expected negative | Not started |
| DINOv2 trunk pilot | moderate | Low — cannot change conclusions | Optional |
| Additional seeds for the σ sweep | GPU-bound | Medium | Not started |

---

# 13. What to say if pressed on scope

A short, honest position that covers most of the second set:

> This thesis measures how a fixed privacy budget should be spent across
> federated rounds, and how that interacts with Byzantine attacks. The detector
> is held fixed on purpose, because it is the control variable — if it changes,
> none of the privacy comparisons mean anything. Under DP-SGD the architecture
> affects the privacy cost almost entirely through the number of privately
> trained parameters, and we sweep that quantity directly, with a predictive
> model that has now matched measurement twice. So the architecture question is
> answered in the currency that governs it, rather than by benchmarking five
> detectors and confounding the result.

