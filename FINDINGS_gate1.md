# Gate 1 re-analysis — result, root cause, and what it means

Session of 2026-09-10. Supersedes the "GATE 1 verdict: smooth degradation"
line printed by `stage2_gate.py` (that line is an unconditional `print`, not a
computed verdict — it fires regardless of the data).

## 1. The original Gate 1 curve was an artifact of the head init

`stage2_gate.py` seeds the DP-trained head from `public_pretrained.pt`. Measured
directly (`diag_gate1.py`):

| | cell | IoU.25 | IoU.50 |
|---|---|---|---|
| public-pretrained head, **no FL at all** | 0.269 | **0.376** | 0.204 |
| every DP run, σ ∈ [0.5, 8] | 0.257–0.271 | 0.300–0.334 | 0.168–0.186 |
| no-DP FL | 0.349 | 0.449 | 0.261 |

Every DP run sat **at or slightly below its own starting point**. The "flat
curve" was the init value, not graceful degradation — nothing had degraded
because nothing had ever improved. The federated phase was only ever worth
0.376 → 0.449 IoU.25, far too narrow a band to resolve a noise curve in.

## 2. Two real mechanisms, both confirmed by measurement

**(a) Adam destroys the noise *scale*.** `diag_gate1_causes.json`: under Adam,
16× more noise changes nothing (σ=0.5 → 0.336, σ=8 → 0.320) and nothing moves
between round 5 and 25. Under SGD, identical noise gives clean monotone
degradation (0.311 / 0.311 / 0.247 at σ=0.5 / 2 / 8) that deepens over rounds.
Adam's per-coordinate normalisation rescales noise-dominated gradients to
fixed-size steps, so σ stops reaching the weights. **DP-SGD here must not use
Adam.** Clipping is *not* the culprit: clip-only at both C=1 and C=10 still
learns.

**(b) Noise averages out.** `probe_noise.py` confirms the mechanism is correct —
endpoint displacement is exactly linear in σ (0.0063 / 0.100 / 0.400 at
σ=1/16/64). But median per-sample gradient norm is 26.8 vs C=1.0, so clipping
bites 27×, and SNR = B/(σ√D) is independent of C once it binds. Signal then
accumulates coherently across steps and clients while noise accumulates as √:
gain ≈ √(steps)·√(clients) ≈ √200·√5 ≈ 31. Even σ=16 ends up signal-dominated.

## 3. Corrected gate: random head, SGD, real dynamic range

Head initialised randomly (floor 0.002 IoU.25) and learned from scratch under DP
on the frozen public trunk; no-DP ceiling 0.399. Trivial-predictor control
(`trivial_baseline.py`) scores 0.000–0.021, so ~0.35 is real detection, not
metric gaming (targets are well spread; most common cell = 4.6% of frames).

3 seeds, score = mean IoU.25 over rounds 20–25 (`knee_seeds.json`):

| σ | ε (δ=1e-5) | mean | sd |
|---|---|---|---|
| 0 (no DP) | ∞ | 0.340 | ±0.043 |
| 4 | 4.40 | 0.313 | ±0.041 |
| 16 | 0.916 | 0.311 | ±0.041 |
| 64 | 0.204 | 0.305 | ±0.042 |
| 256 | 0.109 | 0.283 | ±0.076 |
| 512 | 0.104 | 0.300 | ±0.080 |
| 1024 | 0.103 | 0.243 | ±0.072 |

**Seed variance exceeds the entire noise effect from σ=0 to σ=512.** The init
seed matters more than three orders of magnitude of DP noise. (An apparent
"knee at 256–512" in the single-run sweep `knee_sigma_sweep.json` was seed
noise and does not survive replication. The σ=1024 "peak 0.414" was likewise a
single excursion in an oscillating trajectory: 0.358 → 0.414 → 0.265 → 0.135.)

## 4. What this means

The tension is structural, and Stage 1 jumped clean over the middle of it:

* DP-train the **whole** detector (648K params, B=200) → noise destroys it
  (`dpsgd_bigdata_test.json`: mAP 0.0–0.10 at every schedule)
* DP-train only a **12.7K head** on frozen public features (B=2048) → noise is
  irrelevant from ε=∞ all the way down to ε≈0.1

The thesis needs the regime in between, where noise both matters *and* permits
learning. The governing quantity is per-step SNR = B/(σ√D) times the averaging
gain √(steps·clients), so the knobs are **D** (how much of the trunk is
DP-trained) and **B**, not C and not σ alone.

Note the flip side: at σ=16 this configuration achieves **ε<1 at no accuracy
cost whatsoever** (0.311 vs 0.340 no-DP, well inside seed noise). That is a
genuinely strong privacy result — it is just not a result about *allocating*
noise, because there is no accuracy gradient for any allocator to exploit.

## 5. Reproduce

`diag_gate1.py` · `diag_gate2.py` · `gate1_corrected.py` · `probe_noise.py` ·
`trivial_baseline.py` · `find_knee.py` · `knee_seeds.py`
→ `exp_data/diag_gate1_causes.json`, `diag_gate2_randomhead.json`,
`gate1_corrected.json`, `knee_sigma_sweep.json`, `knee_seeds.json`
All run on cached Stage-1 features; each config is ~25 s.

---

# Recalibration (option B): placing the knee inside the paper's σ range

Decision taken 2026-09-10: tune **D** and **B** to reach the regime where noise
both matters and permits learning, then run the schedule/DRL comparison there.

## The SNR model, and that it actually predicts

Effective SNR ≈ (B / (σ·√D)) · √(steps · clients), with
steps = ROUNDS·EPOCHS·(n/B). Since steps ∝ 1/B this collapses to

    effective SNR  ∝  √B / (σ·√D)      →      σ_knee ∝ √B / √D

Calibrating the constant off the head-only run (knee ≈ σ512 at D=12,677,
B=2048) gives effective SNR ≈ 1.11 at the knee, hence **σ_knee ≈ 1.69·√B** at
D=566,789. Predicted: B=512 → 38, B=128 → 19, B=32 → 10.

## Setup change that made the deeper stack trainable

Cutting the trunk after `s3` (`stage1b_recache.py`) raises the DP-trained
parameter count 12,677 → **566,789** (√D 113 → 753) while still caching, so runs
stay affordable. But DP-training `k1+k2+head` **from scratch collapses to
all-background at every lr tried** (no-DP control = 0.000 at lr 0.01).

Fix (`make_init`, mode `public_tail`): seed `k1/k2` from the public trunk and
leave only the head random. Both are still DP-trained — D is what is
*privatised*, not what is *random* — but optimisation is now stable and the
random head preserves dynamic range:

    floor 0.000  →  no-DP ceiling 0.451   (8 rounds, lr=0.003; beats the
                                           head-only ceiling of 0.399)

lr calibration at B=512, 8 rounds, no DP: 0.003 → 0.451, 0.01 → 0.435,
0.03 → 0.414. **lr = 0.003.**

## Result: a real accuracy-vs-noise gradient exists at D=566,789

B=512, 25 rounds, SGD lr=0.003, score = mean IoU.25 over r20–25
(`exp_data/gateDB_B512_s3.json`):

| σ | 0 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| IoU.25 | 0.402 | 0.360 | 0.395 | 0.367 | 0.333 | 0.251 |

Degradation onset ~32–64, substantial by 128 — against the head-only result
where σ=0→512 produced nothing at all. **Predicted knee 38, observed onset
32–64: the SNR model is usable for choosing B.** Run-to-run scatter is still
±0.02–0.035 (see the σ=8 vs σ=16 inversion), so any final claim needs paired
seeds, not a single sweep.

Remaining gap: the paper operates at σ∈[1,20], where this configuration is still
nearly flat (0.402 / 0.360 / 0.395 at σ=0/8/16). B=128 is predicted to move the
knee to ≈19, inside the range — under test.

## The live regime (confirmed, 2 seeds)

B=128 puts the knee at the predicted σ≈19, **inside** the paper's operating
range (`gateDB_B128_s3.json`, `gateDB_B128_s3_seed5678.json`):

| σ | seed 1234 | seed 5678 | mean |
|---|---|---|---|
| 0 (no DP) | 0.431 | 0.452 | **0.442** |
| 4 | 0.397 | 0.342 | **0.370** |
| 16 | 0.325 | 0.098 | **0.212** |
| 32 | 0.015 | — | collapse |

Monotone on both seeds. σ=0→16 costs 52% relative accuracy, against the
head-only configuration where σ=0→512 cost nothing measurable. **There is now a
real accuracy gradient for a noise allocator to exploit.**

### Canonical config for the schedule/DRL phase

```
frozen public trunk : s1, s2, s3          (cached, fl_workdirs/stage1/cut_s3/)
DP-trained          : k1, k2, head        D = 566,789
init                : k1,k2 from public_pretrained.pt ; head random
optimiser           : SGD lr=0.003 momentum=0.9   (NOT Adam -- see section 2a)
clipping            : C = 1.0
federated           : 5 clients x 8000 frames, 25 rounds x 2 local epochs, B=128
score               : mean IoU.25 over rounds 20-25
useful sigma band   : ~[2, 16]   (<2 nearly free, >20 collapses)
```

Repro: `python gate_DB_sweep.py --B 128 --lr 0.003 --sigmas 0 4 16 32`

### Caveat carried into the next phase

Between-seed spread at σ=16 is wide (0.325 vs 0.098) — expected near a cliff,
but it means the schedule comparison **must** use the project's existing paired
design (identical init and per-client budgets within each trial, scored over the
last 3–6 rounds), not unpaired sweeps. The σ=8 vs σ=16 inversion at B=512 and
the retracted "knee at 256–512" earlier in this document are both reminders of
how easily single runs mislead here.

### Next step

Run k=0 (uniform) vs k=+1 (less noise late, the paper's predicted optimum) in
this regime, paired, at ρ chosen so mean σ lands in [2,16] — the same
pre-registered single-hypothesis design as `confirm_k1_advantage.py` /
`dpsgd_final_test.py`, which is now testing something that can actually move.

---

# The thesis experiment, first run (clear-sky subset)

## Setup changes

Restricted to the 29 hand-picked clear-sky sequences, **split by recording
session** (`stage1_clear.py`). Two leakage problems in the old split were found
and fixed on the way:

* the old manifest split by folder, putting **5 of 11 val sessions into train**
  as well (e.g. `20190925_222534` clips 3,7 in train and 1,2 in val);
* **4 of the 20 public-pretrain folders shared a session with the clear-29**, so
  the frozen trunk had already seen sibling clips of the new val set.

The trunk is therefore re-pretrained on a session-disjoint 97-folder pool.
Final split: 5 × 3,808 train, 1,500 val, 17 sessions, zero session overlap.

Rationale for the subset: on the cluttered sequences the detector sits at
0.0–0.10 mAP, and degradation cannot be measured against a floor — every
schedule scores alike. Removing that floor effect is what makes the allocation
question measurable. **These results describe the clear-sky subset, not
anti-UAV detection in general.** The comparison itself is unaffected: both arms
see identical data.

## σ curve on the new split (`gateDB_B128_clear.json`)

| σ | 0 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| IoU.25 | 0.769 | 0.557 | 0.499 | 0.492 | 0.371 | 0.047 |

Floor 0.001, ceiling 0.769 — a far wider dynamic range than the old split
(0.002 → 0.451), with real slope across the whole of σ∈[1,20].

## Result: k=+1 vs uniform, n=30 paired (`paired_schedule_clear.json`)

ρ=30 fixed (the authors' own CIFAR-10 midpoint), el=2 → σ_t = 2/√ρ_t:
uniform σ=1.83 every round; k=+1 sweeps σ 6.58 → 1.32. No round hits the σ≥1 clip.

    k=+1 minus k=0 = +0.0107 ± 0.0466 (sd)     17/30 wins
    paired t=+1.260   p=0.218 two-sided, 0.109 one-sided
    95% CI [-0.0067, +0.0281]
    arm means: uniform 0.6481, k=+1 0.6588  (+1.65% relative)

**Direction matches the paper's prediction, magnitude is consistent with the
gains the authors themselves report (~1% MNIST, ~0.2% CIFAR-10), but n=30
cannot resolve it.** Observed effect size dz=0.230.

This is the first configuration in the project where the premise is not
provably inert — every earlier setup had no accuracy-vs-σ gradient at all, so
no allocation could differ. The mechanism is visible in the schedule: k=+1
spends its high noise (σ≈6.6) early, where the σ curve is flat (0.499 at σ=4 vs
0.492 at σ=8 — almost free), and buys low noise (σ≈1.3) late, where the curve is
steepest.

## Power

| test | power | n needed | extra trials | extra time |
|---|---|---|---|---|
| one-sided α=.05 | 80% | 117 | +87 | ~7.8 hr |
| one-sided α=.05 | 90% | 162 | +132 | ~11.8 hr |
| two-sided α=.05 | 80% | 149 | +119 | ~10.7 hr |

One-sided is defensible: the hypothesis is directional and pre-registered.

**Treat this n=30 as exploratory.** Extending it after seeing the result would
be optional stopping; the clean route is a fresh confirmatory sample on disjoint
seeds, analysed once — the same exploratory→confirmatory pattern already used in
`verify_paper_premise.py` → `confirm_k1_advantage.py`.

### Variance reduction worth adding first

sd=0.0466 is what drives the required n. The two arms currently draw
**independent** DP-noise and shuffling streams — `make_init` seeds once per
trial, then the arms run sequentially off whatever RNG state follows. Seeding
both arms identically (common random numbers) correlates their noise draws and
should cut the sd of the *difference* materially. If sd falls 0.0466 → ~0.035,
required n drops from 117 to ≈66 (~5.9 hr). Cheap to implement, worth a 10-trial
pilot to confirm before committing a night.

---

# Headroom: what is the CEILING on any allocation? (n=20, CRN)

`headroom_sweep.py` -> `exp_data/headroom_clear.json`. Same live regime, rho=30,
all six schedules share one init AND one RNG stream per trial (common random
numbers). CRN cut the sd of paired differences from 0.0466 to 0.0171 — a 3.2x
tightening, so these 20 trials carry the precision of ~150 without it.

Budget fairness was checked explicitly: effective rho spent is 30.0 for every
arm except k=-1 (26.1, underspends = conservative) and k=+3 (29.6). The sigma<=20
clip touches only rounds with negligible allocations, so no arm buys an advantage.

| k | sigma over rounds | mean diff | 95% CI | wins | p |
|---|---|---|---|---|---|
| -1.0 | 1.00 -> 3.57 (rises) | **+0.0023** | [-0.0012, +0.0058] | 11/20 | 0.19 |
| 0.0 | 1.83 flat (uniform) | — | — | — | — |
| +0.5 | 3.38 -> 1.51 (falls) | -0.0018 | [-0.0041, +0.0006] | 9/20 | 0.13 |
| +1.0 | 6.58 -> 1.32 | -0.0045 | [-0.0125, +0.0035] | 10/20 | 0.25 |
| +2.0 | 20.0 -> 1.09 | -0.0214 | [-0.0447, +0.0018] | 8/20 | 0.069 |
| +3.0 | 20.0 -> 1.00 | **-0.0539** | [-0.0878, -0.0201] | 5/20 | **0.0034** |

## The premise is inverted, not merely absent

The ordering is monotone in k: +0.0023, -0.0018, -0.0045, -0.0214, -0.0539.
Every step toward the source paper's prescription makes things worse, and the
extreme is significant (p=0.003). The only arm above uniform is k=-1, which
puts LOW noise early and lets it RISE later — the reverse of the paper.

Mechanism: the DP-trained head starts random and must build a detector in the
first few rounds. Corrupting those destroys the foundation and the run never
recovers (visible as the collapse trajectories in the sigma sweep). Late rounds
are fine-tuning, which tolerates noise. The authors' "large gradients early, so
early rounds tolerate more noise" is a pretrained-regime intuition that does not
hold when the privatised part is learned from scratch.

## Bound on the DRL

Best achievable gain +0.0023, **upper 95% bound +0.0058 (<=0.84% relative)**,
against per-run reward noise of 0.0171. The prize is a third of the agent's own
measurement noise.

This bounds the published method specifically, not just power schedules: the
paper's agent observes **(round t, remaining budget rho_k)** and nothing else, so
for a fixed budget its policy is a deterministic function of t — i.e. exactly a
fixed schedule, the class tested here.

## Supersedes the n=30 paired run

The earlier `paired_schedule_clear.json` result (k=+1 at +0.0107, "direction
matches the paper") was measured without CRN at sd=0.0466 and was not
significant (p=0.22). At 3.2x the precision, k=+1 is **-0.0045**. That apparent
gain was noise and should not be cited.

## Still open

* **Tight budgets.** Everything here is at rho=30 (uniform sigma=1.83), a smooth
  part of the curve. The sigma curve has a cliff between 16 and 32; rho<=1 puts
  uniform sigma at 10-20, on that cliff, where concentrating budget into a few
  usable rounds could plausibly beat spreading it thin. Untested. If it wins
  there, the winning shape is likely concentration (k<0 / bang-bang), not the
  paper's smooth decay. Test as a PRE-SPECIFIED rho ladder (0.5, 2, 8, 30) with
  all results reported — sweeping rho until one is significant would manufacture
  the result.
* **Non-power allocations.** Random draws from the budget simplex would bound all
  fixed schedules, not just this family.
* **An agent with model feedback** (loss / gradient norm in the state) escapes the
  fixed-schedule class entirely and is not bounded by this table. Genuinely
  beyond the published method, and the one route by which a learned policy could
  still win.

---

# FINAL RESULT: uniform is provably optimal under clipping

## The argument

Let sigma_t be the noise multiplier in round t, with per-round zCDP cost
rho_t = 2*el/sigma_t^2 and budget sum_t rho_t = rho.

Noise accumulates into the global model as a random walk, so the accumulated
noise variance is proportional to

    sum_t sigma_t^2  =  sum_t 2*el/rho_t .

The signal does NOT depend on the allocation. Per-sample gradients are clipped
to C every round, so each round contributes at most C of signal regardless of
sigma_t. (Measured directly in `probe_noise.py`: median per-sample gradient norm
26.8 against C=1.0, so clipping binds by 27x, and the update norm at sigma=0 is
0.1288 — the noise adds in quadrature on top of a fixed signal.)

Allocation therefore moves only the noise term, and

    minimise sum_t 1/rho_t   s.t.   sum_t rho_t = rho

is solved by rho_t = const, by Jensen (1/x is convex) or AM-HM, with minimum
T^2/rho. **Uniform allocation is the analytic optimum, and every non-uniform
schedule is strictly worse.**

This subsumes the earlier 630-run DP-FedAvg finding ("clipping normalises
gradient magnitude away") and explains the original complaint that started the
project: the RL could not beat uniform because uniform is optimal.

## Empirical confirmation across 3 schedule families

`plots/figC_allocation_variance.png` — 13 allocations, all measured paired with
common random numbers:

    Spearman(sum sigma_t^2, accuracy diff) = -0.767   p = 0.0022
    Pearson(log sum sigma_t^2, diff)       = -0.851   p = 0.00023

A single scalar orders power schedules, bang-bang schedules and random Dirichlet
draws alike. Uniform sits at the minimum, 1.00x, and nothing beats it.

| allocation | variance penalty | measured diff |
|---|---|---|
| k=-0.25 | 1.04x | -0.0007 |
| k=+0.5 / k=-0.5 | 1.18x | -0.0018 / -0.0014 |
| k=-0.75 | 1.47x | -0.0015 |
| k=+1 / k=-1 | ~1.99x | -0.0045 / -0.0013 |
| front-clean, back-clean | 7.86x | -0.0744 / -0.0470 |
| k=+2 | 10.15x | -0.0214 |
| rand-B, rand-A | 11.0x / 18.7x | -0.0945 / -0.1115 |
| k=+3 | 21.03x | -0.0539 |

## Two hypotheses tested and rejected in the process

**Early stopping** (`truncation_test.py`). Spending rho over T<25 rounds buys a
lower sigma in each. Neutral at best: T=20 +0.0011, T=8 -0.058. The averaging
gain sqrt(T) exactly offsets the sigma ~ sqrt(T) budget cost.

**A mild front-load** (`fine_negk_test.py`, n=16). k=-1 and k=+1 have nearly
identical variance penalties (1.99x vs 1.98x) but measured +0.0023 and -0.0045
in the headroom run, suggesting a timing benefit for clean early rounds that
could be bought cheaply at k=-0.25 (1.04x penalty). Tested on disjoint seeds:

| k | mean diff | 95% CI | p |
|---|---|---|---|
| -0.25 | -0.0007 | [-0.0018, **+0.0003**] | 0.15 |
| -0.50 | -0.0014 | [-0.0036, +0.0008] | 0.21 |
| -0.75 | -0.0015 | [-0.0049, +0.0019] | 0.37 |
| -1.00 | -0.0013 | [-0.0062, +0.0036] | 0.59 |

**k=-1 did not replicate** (+0.0023 -> -0.0013), so the timing asymmetry the
hypothesis rested on was itself noise. The k=-0.25 interval bounds any gain from
mild front-loading at **+0.0003 (+0.04% relative)** — the tightest null in the
project.

## Bottom line for the DRL

Uniform is the optimum, not merely a strong baseline. A learned policy can at
best tie it, and every deviation it explores costs accuracy in proportion to
sum_t sigma_t^2. This holds for the published method specifically: its state is
(t, rho_k) with no model feedback, so its policy is a fixed schedule, and fixed
schedules are exactly what the theorem covers.

The premise fails because the paper assumes round-varying signal magnitude
("gradients are large early, so early rounds tolerate more noise"). Under
per-sample clipping the contributed signal is pinned at C every round, so that
variation does not exist. The assumption is incompatible with the DP-SGD
mechanism the method is built on.

## Per-layer allocation is closed analytically too (no GPU needed)

Earlier in the session I recommended per-layer allocation as the axis with the
best odds, on the grounds that this project had already observed a large
head-vs-backbone sensitivity asymmetry. **That recommendation was wrong, and the
same inequality that makes uniform optimal across rounds rules it out.**

Split the model into groups g with D_g parameters, noise multiplier sigma_g:

    privacy (one Gaussian release, per-group scales):  rho = (1/2) sum_g 1/sigma_g^2
    accumulated noise variance:                        V   = sum_g sigma_g^2 * D_g

Minimising V under the budget gives rho_g proportional to sqrt(D_g), with

    V_min = (sum_g sqrt(D_g))^2 / (2 rho)      vs      V_global = D_total / (2 rho)

and by Cauchy-Schwarz (sum_g sqrt(D_g))^2 >= sum_g D_g, with equality only in the
degenerate single-group case. **Splitting is always worse**: you pay composition
for every group you release separately. For our k1/k2/head split the best
possible per-layer scheme is 2.37x worse than one global clip
(1,345,614 vs 566,789).

Across rounds the same inequality with equal D per round reduces to AM-HM, which
is the uniform result. Both axes are therefore closed by one argument, and
neither needs an experiment.


---

# DOOR 1 CLOSED: tight budgets (n=10 paired, CRN)

The last open route with a real argument behind it. The variance theorem assumes
accuracy is a smooth monotone function of sum sigma_t^2; near the collapse cliff
that assumption is weakest, so allocation might matter there. Tested with a
PRE-SPECIFIED ladder — three budgets, four arms, all reported
(`tight_budget_ladder.py` -> `exp_data/tight_budget_ladder.json`, which also
stores full per-round trajectories for every arm).

Spend the whole budget over the first T rounds, then stop: sigma = sqrt(4T/rho).
Every arm spends rho exactly; no sigma touches the [1,20] clip.

| rho | uniform (T=25) | T=12 | T=6 | T=3 |
|---|---|---|---|---|
| **0.5** (sigma 14.1) | 0.5609 | +0.036 (5/10, p=0.42) | +0.040 (4/10, p=0.48) | -0.050 (4/10, p=0.48) |
| **2.0** (sigma 7.07) | 0.6361 | -0.025 (3/10, p=0.37) | -0.034 (3/10, p=0.30) | -0.123 (2/10, **p=0.014**) |
| **8.0** (sigma 3.54) | 0.6402 | -0.022 (2/10, p=0.12) | -0.039 (1/10, **p=0.018**) | -0.128 (1/10, **p=0.003**) |
| 30 (sigma 1.83) | 0.7027 | — | -0.058 (T=8, n=1) | — |

**No allocation significantly beats uniform at any budget.** Every significant
result in the table is a LOSS. At rho=0.5 the point estimates turn positive, but
the win counts are 4/10 and 5/10 — coin flips. The positive mean comes from a few
large asymmetric outliers (concentrating the budget occasionally rescues a run
that uniform would have collapsed) rather than consistent superiority.

## Why it cannot be rescued by more trials

Paired-difference sd grows sharply toward the cliff, because outcomes there are
bimodal — the model either survives the noise or collapses:

| rho | 8.0 | 2.0 | 0.5 |
|---|---|---|---|
| sd of paired diff | 0.060 | 0.103 | **0.173** |

Resolving the +0.0404 point estimate at rho=0.5 would need **n=73 (~28 hr)**, and
with 4/10 wins it is very likely chasing noise. More importantly, **the noise
grows faster than the effect**: the best candidate gain (+0.040) sits well below
the sd (0.173) of the very signal an agent would have to learn from. Even if a
small gain exists at the cliff, it is unlearnable.

Figure: `plots/figD_budget_crossover.png`.

## What genuinely does change with budget

The penalty for deviating from uniform shrinks monotonically as the budget
tightens — slope of mean-diff against log(rho) is -0.021 / -0.029 / -0.028 for
T=12 / 6 / 3. So the theorem's assumption really does weaken toward the cliff,
just never enough to flip the ordering. That is a reportable gradient inside a
null result, and it is the honest characterisation: uniform's margin erodes at
tight budgets but is never overturned.

## Cautionary note for the write-up

Trial 1 at rho=0.5 showed T=6 beating uniform by **+0.230 (+46%)** — the largest
effect measured in the whole project. It did not survive: n=2 gave +0.071, n=6
gave -0.0003, n=10 gave +0.040 with 4/10 wins. This is the fourth time in this
project that a single-run result evaporated under replication (after k=+1
+0.0107, k=-1 +0.0023, and the original Gate 1 curve). Cliff-adjacent regimes
are where this happens most; single runs there carry essentially no information.
