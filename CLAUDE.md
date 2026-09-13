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
