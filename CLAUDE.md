# Anti-UAV FL + DP + DRL Thesis — Session Handoff

Federated Learning + Differential Privacy + DRL noise allocation for Anti-UAV
drone detection, adapted from a MARL paper's method (author reference:
`train_CIFAR10_author.py`), using DAMO-YOLO (via ModelScope) instead of the
paper's toy CNN. Canonical script: **`fl_damoyolo_final.py`**.

## GPU POWER CAP — re-apply after every reboot

This machine **hard-powers-off** under sustained full GPU load (observed 10 and
11 Sep: ~98% utilisation for 10-15 min, twice). The card is an RTX PRO 6000
Blackwell with a 600 W default limit; training bursts were measured at 538 W,
which most likely trips the PSU's overcurrent protection.

Fix, in an **Administrator** PowerShell:

    nvidia-smi -pl 300

Verified: peak draw 538 W -> 317 W, peak temp 68 -> 60 C, and runs got *faster*.
`nvidia-smi -pm 1` (persistence mode) is **not supported on Windows**, so the cap
is **lost on every reboot / driver reload** and must be re-applied.

`gpu_guard.py` checks the cap at import and adapts: light duty-cycling when the
cap is present, aggressive throttling plus a loud warning when it is not. It is
wired into `gate_DB_sweep.client_round`, the choke point all experiment scripts
share, so every run inherits it. Tunable via GPU_TARGET_UTIL / GPU_MAX_POWER /
GPU_MAX_TEMP env vars. Throttling is numerically neutral (sleeps only) —
verified by identical results with and without it.

## ⚠ 2026-10-01 — P3-H ROUND 4 (36 runs): direction attacks at convergence are OPEN

`p3_only/p3h_heterogeneous_sigma/FINDINGS_P3H_round4.md`. Against a flip attacker
that never backs off, cosine catches all 3 in rounds 1-5 then goes blind as the
gradient signal dies; damage only -0.041 -> -0.029 (P14/P15/P17 failed). Only the
backing-off α-MPELM is largely stopped (P16 held, marginal). Do NOT claim α-MPELM
is solved: the cosine detector is a partial, early-training defence.

## ⚠ 2026-10-01 — P3-H ROUND 3 COMPLETE (240 runs): cosine detector, aggregators, split leak

Read **`p3_only/p3h_heterogeneous_sigma/FINDINGS_P3H_round3.md`** (prereg
`PREREG_round3.md`). Cosine (direction) detector: P9 FAILED (TPR 0.14 vs α-MPELM)
but P10 HELD (damage -0.043 -> ~+0.01) — it works by DETERRENCE: catches all
attackers in round 1, α-MPELM's back-off keeps γ<0.5 after. The direction signal
dies by round ~5, so a non-backing-off or late-onset attacker is untested and likely
evades it. Median costs -0.11 even without attack (P13). Arm G needs only 2 clean
rounds. Positional split confirmed as the cause of the old Mix/accuracy-detector
results, and of the 0.47 vs 0.65 IoU gap (report set, not the trunk).

## ⚠ 2026-10-01 — P3-H ROUND 2 COMPLETE (150 runs): DETECTION RESULT, NO DAMAGE RESULT

Read **`p3_only/p3h_heterogeneous_sigma/FINDINGS_P3H_round2.md`** (supersedes the
decisive section below for everything except P1-P3). Pre-registered in
`PREREG_round2.md`. P4, P5a, P6, P7 held; **P5b and P8 failed** — heterogeneous σ
does not increase damage (twice now), so the damage claim is dropped for good.
Story: per-client history detection works under constant σ (1.00/0.00) but adaptive
allocation breaks it (0.66/0.71 FPR), forcing trust in declared σ, which a matched
liar defeats (1.00 -> 0.24). Accuracy detector is silent on the clean split, so the
original P3 "Mix neutralises α-MPELM" does NOT reproduce. Figures: `figD1`, `figD2`,
`figH1`-`figH3` + `defence_summary.csv` in `p3_only/p3h_heterogeneous_sigma/figures/`.

## ⚠ 2026-09-30 — P3-H DECISIVE SUBSET RUN: P1 HELD, P2 FAILED, P3 HELD

Read **`p3_only/p3h_heterogeneous_sigma/FINDINGS_P3H_decisive.md`**. 21 runs on the
5090. Heterogeneous budgets widen the honest-norm band 0.11% -> 10.1% and drop
Norm-raw TPR 1.00 -> 0.11; sigma-normalisation restores it to 1.00. But undefended
RMD damage does NOT grow (-0.0081 vs -0.0063, paired diff ~0). **Per the
pre-registered rule the reframe fails; the full grid was not run.** §9's fallback
applies: a detection result only. Code: `p3_only/p3h_heterogeneous_sigma/code/`
(`p3h.py` resumes per round; `launch.ps1` runs detached). The Stage-1 cache was
rebuilt (seeded) on the 5090 — absolute IoU.25 ~0.47, not ~0.65.

## PROPOSED REFOCUS — `p3_only/` (2026-09-25, PLAN ONLY)

All P3-track material lives in **`p3_only/`**. Start with `p3_only/README.md`.
Plan: `p3_only/fresh_beginning.md`. GPU costs: `p3_only/COSTS.md` (**revised 2026-09-28**: total is
**~6-15 h**, not 3-5 h — per-run time is uncertain 3x, since `p3_byzantine.py`'s
own "3x7x3 would be ~8 hr" comment implies ~2.5 min/run vs the 0.5-1 min
previously assumed, and no P3 wall-clock was ever logged. **Run the 15-run
decisive subset first (~30-60 min): it settles predictions 1-3 and can save 14 h
and an abstract rewrite.** Code, not compute, is the bottleneck: 1-2 days). Original
P3 is frozen as a byte-checked snapshot in `p3_only/baseline_uniform_sigma/`
(originals remain at their old paths). New P3-H outputs go to
`p3_only/p3h_heterogeneous_sigma/results/`, never `exp_data/`. P3 cannot run
on the 1650 Super (~13 GB feature cache vs 4 GB VRAM); use the 5090 or the lab
card. Note: every `.png` in `plots/` is a git-LFS pointer, not an image; the
PDFs are real.

**2026-09-27 — DRL track closed, and the figure gap found.**
Final MNIST run completed 800 episodes (resumable script held; snapshots 50..750
in `DRL_Nets/snapshots/`). **Best actor 668 emits a FLAT schedule, Sigma^2
penalty 1.0000x — it converged to uniform.** Accuracies (authors' test_MNIST.py,
n=1, sigma~0.54): DRL 0.9580 / Uniform 0.9586 / Exponential 0.9578 /
Arithmetic 0.9599, all inside 0.21pp. The schedule is deterministic so n=1 is
adequate for the penalty claim; the accuracy gap (-0.0006) measures nothing.
Read: the converged agent independently rediscovers uniform optimality in the
low-sigma regime. The paper's gain is a property of undertrained checkpoints.
**Do NOT write that their results are unreliable or fabricated — their number
reproduces (actor 114, +2.27pp vs published +3.02pp). The defensible claim is
about their PROTOCOL** (one checkpoint, episodes=1) and the converged policy.

**Figure gap found 2026-09-27 (see `p3_only/fresh_beginning.md` §6a, new):**
P3's headline finding — honest norm relative sd 0.10% from 1/sqrt(2D) — has
**no figure anywhere**. And `exp_data/p3_byzantine.json` logs only
`{acc,gamma,tp,fp,detected}` with sigma as a scalar, so there are **no
per-client update norms in any existing output**. Consequence: the uniform-sigma
CONTROL panel of the key new figure also needs a re-run, not just the
heterogeneous arm. P3-H must log per-client-per-round norm, sigma, declared
sigma, malicious flag and detector decision — then all six planned figures are
post-processing over one JSON.

**⚠ CORRECTIONS 2026-09-27 — `p3_only/fresh_beginning.md` §2b supersedes parts of
§2a/§4. Read §2b before writing any claim.** Verified from code + stored data:

* The drone DRL test **did** use heterogeneous per-client budgets
  (`lightcnn_publication_plots.py:180`, ρ~U(20,40)). That is NOT why it lost — its
  schedule was uniform's (1.669-2.320 vs 1.681-2.318) and the test lacked CRN.
* **Cross-client σ relative sd = 8.58%, IDENTICAL for DRL / Uniform / Exponential /
  Arithmetic** (`exp_data/lightcnn_pub_fig23_data.json`). Spread is set by the
  BUDGETS, not the policy. Chapter-3 sentence: *round-to-round allocation is a
  utility question; client-to-client dispersion is a security question.*
* So "adaptive allocation opens the channel" is **too strong**. Per-client budget
  **heterogeneity** opens it — present in the literature's BASELINE (their Uniform
  arm is uniform *over rounds per client*, not across clients). Adaptive only makes
  the spread non-stationary and client-reported.
* Predicted band: 0.094% (homogeneous, measured) → **~8.6% (~90x wider)**.
* **Two existing P3 findings are conditional on homogeneous σ** and P3-H may revise
  them: Norm TPR 1.00, and the -0.017 resilience ceiling.
* σ=1.83 = sqrt(2·2/(30/25)); ρ=30 is the **midpoint of the drone range [20,40]**,
  i.e. the average client. Frame as progression, not error: P3 did the homogeneous
  case, P3-H relaxes it. Keep σ=1.83 permanently as the control arm.
* **Missing arm A′** = heterogeneous budgets + uniform over rounds (the literature's
  real baseline). Centre the σ ladder on 1.83 so only spread varies. **Use A′, not
  the adaptive arm, in the decisive 15-run subset** — the claim is that BUDGETS open
  the channel, so A′ tests it without the adaptive confound.
* **Arm G added 2026-09-28** (`fresh_beginning.md` §5a): per-client HISTORICAL
  baseline detector — compare each client to its own past norms, not to other
  clients. Under uniform allocation a client's norm is constant to 0.094%, so this
  is as sharp as the homogeneous detector **while tolerating budget heterogeneity**;
  under adaptive allocation no stable baseline exists so it fails. It is the control
  that justifies the σ-normalised detector (otherwise a reviewer asks why not just
  use per-client history). **Zero GPU — post-processing of the §2b logs.**
  Predictions 6 and 7 pre-registered.
* Rejected framings (with reasons, §2b C8): "uniform σ solves poisoning" (α-MPELM is
  a direction attack, TPR 0.14 even at uniform σ); "just impose uniform σ" (costs
  throttling to the strictest budget); "adaptive closes the gap by converging"
  (convergence is over rounds, the gap is across clients).
* **NO CIFAR evidence exists** — `test_CIFAR10_paired.py` was never run. The
  convergence-to-uniform result is MNIST + drone only.

**FRAMING (how the two halves are one thesis) — `p3_only/fresh_beginning.md`
§2a, added 2026-09-27.** Both halves are projections of one equation,
`Δ_i,t = clip_C(g_i,t) + N(0, σ_i,t² C² I_D)`: **sum over rounds** gives
accumulated variance `Σσ_t²` → utility (the allocation question); **spread across
clients in one round** gives `‖Δ_i‖ ∝ σ_i` concentrating at `1/sqrt(2D)` →
detectability (the security question). One-line brand: **"privacy budget
allocation is a security parameter, not just a utility knob."**

The DRL null is **load-bearing, not an apology** — it is half of a cost/benefit
argument. You need "no utility benefit" for "unjustified security cost" to land.
Structural fix so it doesn't read as two stapled papers: **move the 1/sqrt(2D)
norm-concentration result out of the P3 results and into the theory chapter,
beside the Σσ_t² result**, both derived from the same Δ. Then Ch5 is §5.1
allocation (null with a mechanism) and §5.2 security (channel, defence, limit).

Teammate's proposal: drop all DRL-vs-uniform claims, focus the thesis on
Byzantine resilience, with the bridge claim "adaptive allocation opens a
model-poisoning channel". **That claim is untested**: P3 ran every client at
one σ=1.83 (`p3_byzantine.py:52`). Mechanism is sound (noise dominates, norm ∝
σ, so heterogeneous per-client σ widens the honest norm band), but only
**cross-client** σ heterogeneity within a round opens it, not round-to-round
variation. Next: supervisor sign-off → fix P3 positional split → run P3-H
(heterogeneous σ, σ-normalised detector, σ-lying attacker arm; predictions
pre-registered in §6). Current title asserts uniform optimality and must change.

## E5 — SCHEDULE HUNT (2026-09-25, NOT YET RUN) — `NEXT_EXPERIMENTS.md` §3d

**The drone DRL "loss" is RNG noise:** its σ schedule is Uniform's (≤0.013
apart, penalty 1.0000×), and `lightcnn_publication_plots.py:179` never seeds
torch and runs arms back-to-back without CRN. E5a adds CRN (prediction: an
exact tie, not a win). E5b tests whether ANY fixed schedule beats Uniform at
T=50, ρ ∈ {1, 2} (ρ=0.5 at T=50 breaks the σ≤20 cap). Pilot seeds 9000+i, then
confirmation on disjoint seeds 9500+i. **Only if a schedule is confirmed does
the DRL chase resume**, and a DRL win must then beat that schedule, not just
Uniform. 5090/lab card only.

## REVIEWER Q&A — `DEFENSE_QA.md` / `.docx` (2026-09-23)

Answers to four reviewer questions (dataset integrity/bias, "focus on the noise
not the value", test-time DP noise + latency, train-DP/test-clean comparison),
written to be delivered verbally or handed over. Key points:

* **DP noise is applied to gradients, not to the dataset** — the images on disk
  are bit-identical, so "compare the variance before and after" has no before
  and after. This misconception underlies three of the four questions.
* **Measured dataset stats** (25 sessions/split, 23 Sep): train vs validation
  pixel variance 2841.8 vs 2835.3 (0.23% apart), median box 624 vs 616 px²
  (~0.19% of frame). No pipeline-induced distribution shift.
* **Recording-level leakage in the dataset's OWN splits**: 70% of validation
  sessions share a recording with train (71% track1, 60% track2); `_N` suffixes
  are segments of one continuous recording. No literal frame overlap, so a
  file-level duplicate check misses it. **Already fixed** — `stage1_clear.py`
  splits by recording and re-pretrains the trunk on a session-disjoint pool.
* **Disclosed residual**: `p3_byzantine.py:36-37,56` cuts val positionally
  (300 detector / 1200 report), landing inside a session. Affects TPR/FPR
  calibration only, not accuracy. Fix scheduled.
* **No test-time noise, ever** — DP is immune to post-processing, so inference
  adds 0 ms and 0 privacy cost. Answers the "10 ms / drone with a bomb" question
  favourably. Input privacy at inference is a different threat model, out of scope.
* `track1_test`/`track2_test` are the **tracking** splits and are unused
  (track1 ships first-frame-only boxes; track2 ships none).

**Part II (2026-09-23, second set of 13 questions — architectures/ablations):**

* **The independent variable is the privacy mechanism, not the detector.** Under
  DP-SGD the architecture enters almost entirely through **D**; σ_knee ∝ √B/√D
  has now matched measurement twice (predicted 38 → observed 32-64; predicted
  19 → observed 19). So "try 5 architectures" is re-measuring D five times; the
  on-topic version is **accuracy vs D at fixed ρ**.
* **MoE is counterproductive under DP** — it raises D, and noise is added to the
  full gradient vector regardless of sparse activation. 10x params → √10 ≈ 3.2x
  less tolerable noise.
* **Anomaly detection: we already do it** — P3's Norm/Accuracy/Mix detectors are
  anomaly detectors over client *updates*. Not appropriate at image level
  (labelled, single-class, localisation metric).
* **Encryption conflicts with the Byzantine half**: secure aggregation hides
  individual updates, which is exactly what our detectors inspect. Raise as a
  finding (privacy-from-server vs robustness-at-server tension), not a gap.
  It is also accuracy-neutral, so it must NOT be a third axis of an accuracy
  factorial.
* **REAL GAP FOUND — the non-FL control is missing.** The 0.376 baseline is
  "no federated phase", NOT "centralised on pooled data". A true centralised
  control at D=566,789 has never been run. Features are cached so it is one
  run. **Highest-priority action item** — it completes the FL x DP 2x2.
* **"CS-DOME" could not be identified** — ask the reviewer before answering.

## ACTIVE WORK — read `NEXT_EXPERIMENTS.md` first

**2026-09-27: `NEXT_EXPERIMENTS.md` was rewritten around P3-H.** The DRL /
allocation experiments E1-E5 are archived verbatim in
**`NEXT_EXPERIMENTS_superseded_DRL.md`** — nothing deleted, but do not run them
without a deliberate decision to reopen the DRL track. E1 (measure `a_t`) is the
one to reopen if a reviewer challenges the scoping of Theorem 3.1.

The active plan is: fix the P3 positional split -> re-run the uniform-σ control
**with per-client norm logging** -> P3-H arms B-F -> six figures (figH1-figH6).
The uniform-σ arm is **kept permanently as the control**, not replaced: the
uniform-vs-heterogeneous contrast is the result, and it is the reference panel or
leftmost point in figH1/H3/H4/H5.

The section below is the older DRL-era framing, retained for context.

Two reviews (2026-09-20) found the uniform-optimality theorem is
under-justified: the load-bearing step is that every round's noise reaches the
final model with **equal weight** (`V ∝ Σσ_t²`), not clipping. If weights `a_t`
are not flat the optimum is `ρ_t ∝ √a_t`, which is non-uniform. **E1** measures
`a_t` by single-round noise pulses; **E2** tests the same question via a
matched-variance permutation (`front-clean` vs `back-clean`, identical `Σσ²`).
Full protocol, decision rules and the resulting thesis edits are in
`NEXT_EXPERIMENTS.md`. **E3** removes a confound in the DRL experiment: our
agent's flat schedule is an artefact of the actor sigmoid being saturated at the
bottom of the [1,20] action range (uniform sits at output 0.044; the authors'
MNIST agent sat at 0.28-0.42). That finding is written up in full in
**`FINDINGS_actor_saturation.md`** — it also notes that the source paper's
reported gain is monotone in actor resolution (0.42 -> 3.0pp, 0.11 -> 0.15pp,
0.044 -> none), which E3 tests. Run order: **E4 -> E3 -> E1 -> E2**. **E4** is new and highest-value: re-run the
authors' OWN CIFAR-10 test harness with n=10 paired trials instead of the n=1 it
ships with (`test_CIFAR10_paired.py`, already written). Needs only DRL_Nets/ from
the Windows box; no drone data, no Stage-1 cache. ~6 h.

## ⚠⚠ 2026-09-22 — CHECKPOINT SELECTION FLIPS THE CONCLUSION (headline result)

Read **`FINDINGS_checkpoint_dependence.md`** first. Two checkpoints from the SAME
training run, same code, same data, same n=10 paired protocol:

| DRL vs | actor 114 | actor 612 |
|---|---|---|
| Uniform | +0.0227, 9/10, p=0.004 | -0.0112, 2/10, p=0.051 |
| Exponential | +0.0029, 7/10, p=0.557 | **-0.0265, 0/10, p=0.00003** |
| Arithmetic | +0.0085, 6/10, p=0.123 | **-0.0220, 0/10, p=0.002** |

DRL goes from FIRST place to LAST. The three fixed schedules move by <=0.64pp;
all the instability is in the learned policy (2.75pp, sign reversal). Cause:
actor 612 learned a FLAT schedule (7.200->7.106, penalty 1.0000x) because it was
trained in the unrepaired regime (sigma~0.53) where uniform IS optimal; tested at
sigma~7 where decaying schedules win, that policy is wrong. **Better training ->
worse transfer.** Predicted before the run from the schedule shape alone.

Consequence: the source paper's `agent.load(episode=350)` + `episodes=1` protocol
can produce EITHER conclusion. Their +3.02pp reproduces (actor 114 gave +2.27pp)
but is not evidence the method works. **Do NOT claim DRL beats uniform, or that
it loses — the comparison is not identified.** What survives: a one-line fixed
exponential schedule is best and most stable in both runs.

## 2026-09-21 — THESIS'S UNIFORM-OPTIMALITY CLAIM REFUTED IN THE HIGH-NOISE REGIME

Read **`FINDINGS_regime_dependence.md`** before touching Chapter 3 or 5.

Authors' unmodified MNIST code with their unit bug fixed, **n=10 paired**, actor
114: DRL 0.9061 / Exponential 0.9032 / Arithmetic 0.8976 / **Uniform 0.8834**.
DRL beats uniform by **+2.27pp, 95% CI [+0.94,+3.59], 9/10 wins, p=0.004** —
their published +3.02pp REPRODUCES. Uniform is the WORST arm, and
Spearman(accumulated variance, accuracy) = **+1.000**, i.e. exactly inverted
versus the -0.767 the thesis reports for the drone task.

Reconciliation: the weighted objective `V = sum_t a_t sigma_t^2`. At the drone
operating point (sigma~1.83) a_t is near-flat so uniform is optimal; at the
paper's (sigma~6-9) a_t is steep, so decreasing schedules win and the variance
penalty is irrelevant. Theorem 3.1 is not wrong — assumption (iii) fails at
high sigma, so its scope must be narrowed to the regime it was measured in.

What survives: **DRL ~ Exponential (+0.29pp, p=0.557)** — a one-line fixed
decaying schedule matches the learned policy, so the gain comes from being
DECREASING, not from being LEARNED. That is the defensible negative result and
it is about MARL, not about allocation. **E1 (measure a_t) is now the central
experiment.** Caveat: actor 114 is undertrained (544/800), so this is a lower
bound on DRL.

**2026-09-21 — first real run of the AUTHORS' unmodified `test_MNIST.py` (actor
114, n=1): DRL 0.9563 / Uniform 0.9550 / Exponential 0.9552 / Arithmetic 0.9565.**
Their claimed MNIST gain is +3.02pp; measured here it is +0.13pp, and Arithmetic
beats DRL. All four arms sit inside 0.15pp. Training is still running (544/800,
best actor still 114), so this is provisional and undertrained relative to their
tested episode 195.

Two conversion discrepancies found the same day, written up in
**`FINDINGS_authors_code_bugs.md`**: (a) the authors' action-mapping QP bounds
omit the `/bs` that `sigma2rho` uses, so the sigma_min=1 clamp never binds and
applied sigma is ~0.5, not in [1,20] as the paper states; (b) our port dropped
the `/bs` entirely, a sqrt(200)~14x sigma rescale versus theirs. (b) invalidates
the sigmoid-position table in `FINDINGS_actor_saturation.md` — that doc now
carries a correction header and its 'resolution -> gain' pattern must not be
cited. Re-run our agent with the authors' conversion restored BEFORE E3.

Known-wrong claims currently in the LaTeX, pending those results: the §5.1.6 DRL
causal claim (the learned schedule is flat, penalty 1.0000×), the per-layer
2.37× figure, the §3.2.3 "no cross-client structure" sentence (aggregation
weights are cross-client structure — see **`FINDINGS_cross_client.md`**), the
"don't use Adam" recommendation, and the missing ε values (ρ=30 → ε≈67). P3
detector results are unsafe until the val/report split is made session-disjoint.
All detailed in `NEXT_EXPERIMENTS.md` §6.

## Status as of last session

Root-caused why the DRL agent could never beat uniform DP-noise allocation
and why reported accuracy looked like either ~99.9% (overfitting) or stuck
at 0 mAP. Three real bugs were found and fixed in `fl_damoyolo_final.py`
(each has a detailed docstring at its definition — read those for full
rationale, this file just indexes them):

1. **`trust_remote_code=True` missing** from both `pipeline(...)` calls
   (`Net.__init__`, `evaluate_map()`). Without it, modelscope 1.31.0 raises,
   and `Net.__init__`'s except-fallback silently replaced the whole model
   with a dummy `nn.Linear(1,1)` — i.e. the entire FL loop was training a
   no-op stub with no visible error. Fixed.

2. **`Server.aggregate()`** was doing plain FedAvg (including the detection
   head) across all clients. Averaging the head destroys score calibration
   (verified live: global mAP top-score ~0.012 vs local ~0.89 pre-fix).
   Fixed: backbone/neck still get real FedAvg; head is taken from one client
   instead of blended. See `aggregate()`'s docstring/comments.

3. **Detection-head seeding** (`Net.__init__`'s `init_mode`). Full head reset
   — even DAMO-YOLO's own architecturally-correct `ZeroHead.init_weights()`
   calibrated init — destabilizes the regression branch (DFL bucket decode is
   very sensitive to weight perturbation) and collapses training to
   predicting background everywhere within ~5-25 gradient steps
   (`loss_bbox`/`loss_dfl` -> exactly 0.0, permanently). Root-caused via an
   isolated diagnostic harness that instruments the OTA assigner directly,
   bypassing the full FL loop. Fixed by adding `'seed_perturb'` mode
   (see `_perturb_head_cls()` docstring): perturbs ONLY the classification
   layers (`cls_convs`, `gfl_cls`) with scaled Gaussian noise
   (`HEAD_PERTURB_SCALE = 0.3`, picked via an 8-epoch sweep — 0.3 stable,
   1.0 persistently unstable, 2.0 reproduced the full-reset collapse), and
   leaves box regression + backbone/neck untouched. This is now
   `INIT_MODE`'s default and is **verified working**: round 1 of every
   episode trains cleanly, internal COCOeval AP50 climbing 0.30 -> 0.90+
   within a single 8-epoch round, reproduced identically across 3 separate
   episodes on real hardware (GTX 1650 Super, 4GB).

Also fixed two redundant-download bugs that were silently re-fetching the
125MB weight file from modelscope.cn on every `Net()`/`evaluate_map()` call
and on every trainer subprocess round (~8x wall-clock slowdown, real network
-stall risk on flaky connections): `pipeline()` calls now point at the local
cache dir directly instead of the hub model id; the trainer's `build_trainer`
kwargs no longer pass a redundant `model=` argument alongside `cfg_file`+
`cache_path` (see the `NOTE:` comment right above that kwargs dict).

## THE OPEN ISSUE — read this before doing anything else

The DP-noise-scale problem described in earlier handoffs (per-element noise
swamping the clipped signal in `Training.train_one_round()`) was real, and the
project responded to it with a full redesign rather than a patch: a **light CNN
port** (`fl_lightcnn_final.py`), then **real DP-SGD via opacus**
(`fl_dpsgd_final.py`), then the current **two-stage frozen-trunk design**
(`stage1_public_pretrain.py` + `stage2_gate.py`). DAMO-YOLO is no longer in the
experimental loop.

**Current status: Gate 1 has been run, re-analysed, and it does NOT pass.**
Full write-up with numbers, controls and repro commands: **`FINDINGS_gate1.md`**
— read that before touching the gate. In brief:

1. The original flat Gate 1 curve was an **artifact of seeding the DP head from
   public pretraining** — it left only 0.376 → 0.449 IoU.25 of headroom, so every
   DP run just sat at its init. (`stage2_gate.py`'s "smooth degradation" verdict
   line is an unconditional `print` and means nothing.)
2. **Adam must not be used for DP-SGD here**: it normalises the noise scale away
   entirely (16× σ → no change), while SGD transmits it. Clipping is not the
   culprit.
3. Re-run properly — random head, SGD, real dynamic range (floor 0.002, ceiling
   0.399), trivial-predictor control at 0.021, 3 seeds — **noise has no
   measurable effect from σ=0 to σ=512, i.e. from ε=∞ down to ε≈0.104.**
   Seed variance (±0.04) exceeds the whole effect.

The structural finding: DP-training the **whole** detector destroys it
(mAP 0.0–0.10), DP-training only a **12.7K head** on frozen public features makes
noise irrelevant. The thesis needs the regime in between. The governing quantity
is per-step SNR = B/(σ√D) × √(steps·clients), so the knobs are **D** (how much of
the trunk is DP-trained) and **B** — not C, and not σ alone.

**Fork resolved (2026-09-10): option B — tune D and B to reach the live regime.
DONE, and the live regime is found.** Cutting the trunk after `s3`
(`stage1b_recache.py`) raises the DP-trained parameter count to **D=566,789**,
and B=128 places the degradation knee at σ≈19 — inside the paper's σ∈[1,20]:

| σ | 0 | 4 | 16 | 32 |
|---|---|---|---|---|
| IoU.25 (2 seeds) | 0.442 | 0.370 | 0.212 | collapse |

σ=0→16 now costs 52% relative accuracy (head-only: σ=0→512 cost nothing).
Two things were required to get here, both non-obvious:
* **SGD, never Adam** — Adam normalises the noise scale away entirely.
* **`k1/k2` seeded from the public trunk, head random** (`make_init`) — the
  deeper stack collapses to all-background if DP-trained from scratch, while the
  random head is what preserves dynamic range (floor 0.000, ceiling 0.451).

Canonical config, repro command and the paired-design caveat are in
`FINDINGS_gate1.md`.

**FINAL RESULT (2026-09-11): uniform is PROVABLY optimal under clipping.**

Accumulated noise variance ~ sum_t sigma_t^2 = sum_t 2*el/rho_t, while clipping
pins each round's signal contribution at C regardless of sigma_t. So allocation
moves only the noise term, and minimising sum(1/rho_t) s.t. sum(rho_t)=rho gives
rho_t = const by Jensen. **Uniform is the analytic optimum; every deviation is
strictly worse.**

Confirmed across 13 allocations in 3 families (power / bang-bang / Dirichlet),
all paired with common random numbers:
Spearman(sum sigma^2, diff) = **-0.767, p=0.0022**; Pearson(log) -0.851,
p=0.00023. Figure: `plots/figC_allocation_variance.png`.

Tested and rejected: early stopping (neutral, T=20 +0.0011) and a mild
front-load (k=-0.25 bounds any gain at **+0.0003**, n=16). The earlier k=-1
"+0.0023" did NOT replicate (-0.0013 on disjoint seeds) — as did the earlier
n=30 k=+1 "+0.0107". Both were noise. **Do not cite either.**

Why the premise fails: the paper assumes round-varying signal magnitude
(large gradients early => early rounds tolerate more noise). Under per-sample
clipping the contributed signal is pinned at C every round, so that variation
does not exist. The assumption is incompatible with DP-SGD itself.

**DOOR 1 (tight budgets) CLOSED, 2026-09-12.** Pre-specified ladder,
rho = 0.5 / 2 / 8 x T = 3/6/12/25, n=10 paired with CRN. No allocation beats
uniform at any budget; every *significant* entry is a loss (T=6 at rho=8:
-0.039, p=0.018; T=3 at rho=8: -0.128, p=0.003). At rho=0.5 point estimates turn
positive (T=6 +0.040) but win counts are 4/10 and 5/10 — coin flips driven by
outliers. Paired sd grows 0.060 -> 0.103 -> 0.173 toward the cliff, so the noise
grows faster than the effect: n=73 (~28 hr) would be needed to resolve +0.040,
and the agent's reward sd (0.173) is 4x the best candidate gain. **Unlearnable
even if real.** Figure: `plots/figD_budget_crossover.png`.

Trial 1 of that run showed +0.230 (+46%) and did not survive replication — the
fourth single-run result to evaporate in this project. Cliff-adjacent single
runs carry no information.

All experiment data is catalogued in **`DATA_INDEX.md`** (16 result files, which
are superseded, and six suggested figures).

Full derivation, tables and open routes: `FINDINGS_gate1.md`.
**P3 (Byzantine, 12-14 Sep) is scoped in `P3_BYZANTINE_NOTES.md`** — attacks are
RMD and alpha-MPELM, detectors Norm/Accuracy/Mix, from `exploiting_dp.pdf`.

## P3 COMPLETE (2026-09-13) — see `FINDINGS_P3.md`

33 configs x 3 trials, 10 clients, m = 10/20/30%. Headline results:

* **Paper's claim reproduced exactly**: Norm detector catches RMD at TPR=1.00,
  a-MPELM at TPR=0.14.
* **But the premise inverts at scale.** Honest update-norm relative sd is 0.10%
  (1/sqrt(2D) concentration, D=566,789) — DP noise makes the norm MORE
  predictable. The paper's "hide in the noise magnitude" room is a small-D
  phenomenon. What remains is direction, which norm detection cannot see.
* **The architecture is inherently resilient**: worst-case damage is -0.017
  IoU.25 (-2.7% relative), undefended, at 30% malicious. The tight norm band
  caps attack amplitude.
* **Mix detection neutralises both attacks** (a-MPELM 0.6649 vs 0.6485 baseline)
  at a 28% honest-client false-positive cost.
* **Classical robust aggregators are counterproductive under DP**: median costs
  -0.208 with NO attacker, Krum -0.073 under RMD, because averaging is what
  suppresses DP noise (sqrt(N)) and they discard it.

Figures: `plots/figP1..figP4`.

## Files

- `fl_damoyolo_final.py` — the canonical script, all fixes above are here.
- Everything under `/run/media/afnanmeow/Shared/tmp_smoketest/` was a Linux
  home-machine (GTX 1650 Super, 4GB) smoke-test scratch environment — a
  reduced-config copy (`fl_smoketest.py`) plus a Python 3.10 venv with
  pinned deps, used purely to validate fixes on real hardware before the
  full lab-machine run. Not needed on a new machine; not portable
  (Linux-specific venv, hardcoded Linux paths). Safe to ignore/delete there.
- No git repo here (`git init` was never run). Diagnostic scratch scripts
  used to isolate the head-init and noise-scale bugs were one-off and not
  preserved — the *findings* are captured in this file and in the fixed
  code's docstrings; the throwaway diagnostic harnesses themselves don't
  need to survive.
