# SUPERSEDED — DRL / allocation experiments E1-E5 (archived 2026-09-27)

**These are no longer the active plan. Do not run them without a deliberate
decision to reopen the DRL track.** Archived from `NEXT_EXPERIMENTS.md` when the
thesis refocused on Byzantine resilience.

## Why they were retired

The DRL track closed on 2026-09-27. The final MNIST run completed all 800
episodes and its best actor (668) emits a **flat** schedule with Sigma^2 penalty
**1.0000x** — the converged agent *is* uniform allocation. Six full training runs
produced no win. A flat schedule cannot beat or lose to uniform by allocation,
because it is the same allocation, so no additional trials can change that.

E3 (actor resolution), E4 (their win rate) and E5 (schedule hunt) all exist to
find or explain a DRL/schedule win. That question is answered.

E1 (measure `a_t`) and E2 (matched-variance permutation) are **not** win-chasing
— they test the scope of the uniform-optimality theorem, which the high-sigma
MNIST result (`FINDINGS_regime_dependence.md`) already shows is narrower than
the thesis claims. They are archived with the rest because the refocused thesis
demotes allocation optimality from a headline claim to background. **If a
reviewer challenges the scoping of Theorem 3.1, E1 is the experiment that
answers them** — reopen this file in that case.

Nothing here is deleted; it is preserved verbatim below.

---

## 2. E1 — measure the weight profile `a_t`

### Idea

Hold every round at the uniform operating point, then add a **single extra noise
pulse at exactly one round** `t*`. The drop in final accuracy, per unit of added
variance, estimates `a_{t*}`. Sweep `t*` across training to get the shape.

### Protocol

Baseline arm: `σ_t = 1.826` for all `t = 1..25`.

Pulse arm `t*`: identical, except `σ_{t*} = σ_pulse`.

Added variance at that round is `Δ_{t*} = σ_pulse² − 1.826²`. Then

```
a_hat(t*)  =  [ IoU_baseline − IoU_pulse(t*) ]  /  Δ_{t*}
```

Report the **normalised** profile `a_hat(t) / mean_t a_hat(t)` — only the shape
matters, the constant is absorbed into the optimisation.

* Probe rounds: `t* ∈ {1, 4, 7, 10, 13, 16, 19, 22, 25}` (9 probes).
* `σ_pulse`: **pilot first** (see below). Start at `σ_pulse = 6`, giving
  `Δ ≈ 36 − 3.33 = 32.7`.
* `n = 20` paired trials per probe, CRN, baseline and pulse sharing the stream.
* Total: 9 probes × 20 trials × 2 arms = 360 runs. At ~25–40 s per 25-round run
  on cached features this is roughly 3–4 h. Baselines can be shared across
  probes within a trial (one baseline per trial, not one per probe), which cuts
  it to ~200 runs.

### Pilot before committing the full grid

Run `t* ∈ {1, 13, 25}` at `n = 4` and check:

1. **Measurable**: the pulse drop exceeds the paired noise floor (~0.017).
   If drops are < 0.02, raise `σ_pulse` to 8 or 10.
2. **Linear**: run `t* = 13` at `σ_pulse ∈ {4, 6, 8}` and confirm the drop grows
   roughly with `Δ` (i.e. `a_hat` is stable across pulse sizes). If `a_hat`
   falls as `σ_pulse` rises, you are out of the linear regime — reduce the pulse.
   **This check is not optional**: `a_t` is only meaningful as a local
   derivative, and the σ curve has a cliff between 16 and 32.
3. **No clipping of the pulse**: confirm `σ_pulse` does not hit the `σ ≤ 20`
   clamp, and that the pulse round's `ρ_t` is simply not budget-normalised here
   — E1 deliberately does **not** hold `Σρ_t` fixed, because we are measuring a
   derivative, not comparing budgets. Say so explicitly in the write-up.

### Decision rule (fix before looking at results)

Fit `a_hat(t)` against `t` by weighted least squares, weights `1/se²`.

* **Flat** if the 95% CI of the slope contains 0 **and** the fitted range across
  `t` is under 1.3×. → uniform is optimal; rewrite §3.2 to rest on measured
  equal weighting.
* **Not flat** otherwise. → compute `ρ_t ∝ √a_hat(t)`, normalise to `ρ = 30`,
  and **run that schedule against uniform, paired, n = 20**. That single arm is
  the money result: a derived optimal schedule beating uniform.

Report the profile with error bars regardless of outcome.

### Expected, so you are not surprised

`a_t` is probably **mildly increasing** (late noise hurts more). If so the
derived optimum is a mild `k > 0`. But note the variance penalty grows fast:
`k=+0.5` already costs 1.18× and measured −0.0018. So the predicted optimum is
likely `k ≈ +0.1…+0.3` with a gain of order 0.001–0.003 — **at or below the
0.017 noise floor.** If that is what you find, the honest result is:

> uniform is optimal to within measurement precision, and the weighted theory
> explains why the residuals in Fig 5.1 are structured rather than random.

That is a better paper than the current one and it is still a negative result
for the DRL. Plan for it.

---

## 3. E2 — matched-variance permutation contrast

### Idea

`front-clean` and `back-clean` use the **same σ multiset in reverse order**, so
`Σσ² = 655.0` for both, exactly 7.86× the uniform minimum. The variance-only
theory says they must score the same. If they do not, ordering matters and the
single-scalar claim is incomplete.

### Current state of this data — read before trusting it

`exp_data/allocation_bound.json`, `rho=30`, CRN, **n = 2**:

```
front-clean   scores [0.5286, 0.5882]   diff vs uniform  mean -0.0744
back-clean    scores [0.5239, 0.6476]   diff vs uniform  mean -0.0470
back-clean − front-clean  =  [-0.0047, +0.0593]         mean +0.0273
```

**The two trials disagree in sign.** The +0.027 mean is carried entirely by one
trial. This is exactly the profile of the four results in this project that
evaporated under replication. Treat it as a hypothesis, not evidence. It is
currently cited in no thesis claim, and it must not be until this experiment
returns.

### Protocol

* Arms: `uniform`, `front-clean`, `back-clean`, all sharing one CRN stream per
  trial. Keep `uniform` in the design as the anchor even though the contrast of
  interest is `back − front`.
* `n = 20` paired trials. Extend to `n = 30` only if pre-declared; do not extend
  after seeing the result (that is optional stopping — the project already
  flagged this in `FINDINGS_gate1.md`).
* Primary statistic: paired mean of `back-clean − front-clean`, with 95% CI and
  win count. **One pre-registered test, two-sided.**
* Secondary, report but do not treat as primary: `k=+2` (10.15×) vs `rand-B`
  (11.02×), and `k=+3` (21.03×) vs `rand-A` (18.73×). These are only
  *approximately* variance-matched, so they cannot carry the argument — only the
  exact permutation pair can.

### Decision rule

* CI excludes 0 → ordering matters at matched variance. Report as a measured
  limitation of the single-scalar model, and it corroborates a non-flat `a_t`
  from E1. The two experiments should **agree on direction**: `back-clean`
  winning (quiet late) implies `a_t` increasing.
* CI contains 0 at n=20 → the residual structure in Fig 5.1 is not established.
  Say so, drop the claim, and keep the Spearman result as-is.

**Cross-check**: E1 and E2 are measuring the same underlying quantity two
different ways. If they disagree in direction, something is wrong with the
setup — do not write either up until that is resolved.

---

## 3b. E3 — the actor was saturated; give it usable resolution

**This is the experiment that speaks directly to the supervisor's demand, and it
is confound-removal, not result-manufacturing.**

> Full write-up of this finding, with the source-paper parameters, the
> arithmetic, and the reproduction snippets: **`FINDINGS_actor_saturation.md`**.
> The summary below is enough to run the experiment.

### The observation

The authors' Fig. 2 shows the learned noise level **decreasing over rounds** — a
curve — and they justify it with exactly the premise this thesis attacks
("the magnitude of the gradient at the early stage of training is larger than
that at the latter stage"). Our agent instead learned a flat line: σ 1.878 →
1.896, `Σσ²` penalty **1.0000×**, indistinguishable from uniform.

The reason is almost certainly mechanical. The actor ends in a sigmoid scaled to
`[σmin, σmax] = [1, 20]` in every setting, theirs and ours. Where the operating
point sits inside that range differs enormously:

| setting | ρ | T | E | uniform σ | actor sigmoid output | reported gain |
|---|---|---|---|---|---|---|
| MNIST (theirs) | 2–4 | 20 | 4 | 8.94 – 6.32 | **0.42 – 0.28** | 3.0 pp |
| CIFAR-10 (theirs) | 20–40 | 25 | 4 | 3.16 – 2.24 | **0.11 – 0.065** | 0.15 pp |
| **ours** | 30 | 25 | 2 | **1.826** | **0.0435** | none |

Our operating point sits at 4.4% of the action range, deep in the saturated tail
of the sigmoid, where the actor cannot resolve variation. A near-constant raw
action, pushed through the action-mapping affine projection that enforces
`Σρ_t = ρ`, produces *exactly* uniform — which is why the penalty is 1.0000×
rather than merely close to it.

Note the monotone pattern: resolution 0.42 → 0.11 → 0.044, reported gain
3 pp → 0.15 pp → none. That is a testable hypothesis **about the source paper's
own result** — that the reported gain tracks how much actionable range the actor
had, rather than allocation being intrinsically beneficial.

### Why this must be run

As it stands a panelist or reviewer can say: *"your agent output a flat
schedule, so you never actually tested adaptive allocation — you tested a
saturated network."* That objection is correct and currently unanswerable. It
has to be closed whichever way the result falls.

### Engineering gap — read before scheduling E3

**The DDPG agent is not wired to the calibrated pipeline, and that bridge does
not exist yet.** `class Actor` lives in `fl_damoyolo_final.py` and is imported
by `fl_lightcnn_final.py` / `lightcnn_publication_plots.py` — the LightCNN
pipeline, 8 rounds. Every calibrated-pipeline script (`gate_DB_sweep`,
`allocation_bound`, `headroom_sweep`, `tight_budget_ladder`, ...) runs
**fixed schedules only**.

The bridge is small, because the interface is already a per-round σ vector:

1. Agent emits a σ schedule for the episode (its existing action-mapping
   affine transform already enforces `Σρ_t = ρ`).
2. Feed that vector to `e1_weight_profile.run_sched(...)`, which accepts an
   arbitrary per-round schedule and returns mean IoU.25 over rounds 20–25.
3. Reward = per-round ΔIoU.25 on **held-out** val, never a client's own shard.
4. Update the agent; repeat across episodes.

Budget roughly half a day of coding plus the training episodes. Do this before
committing GPU time to E1, or E3 will stall waiting on it.

```
sigma range [1, 4]   ->  uniform (1.826) sits at sigmoid output 0.275
sigma range [1, 3]   ->  uniform sits at 0.413
```

Both bracket the authors' MNIST band. Use `[1, 4]` as primary.

* Arms: `DRL(range=[1,20])` (reproduces the flat result), `DRL(range=[1,4])`,
  `uniform`, `exponential`, `arithmetic`.
* Paired, CRN, `n = 20`, held-out scoring — **never** the client's own shard.
* In the calibrated pipeline (D=566,789, B=128), not the LightCNN pipeline the
  current §5.1.6 uses.
* Report the learned schedule for each arm alongside its `Σσ²` penalty. The
  schedule plot is as much the result as the accuracy number.

### Decision rule (fix before looking)

1. **Still flat at [1,4]** → saturation was not the cause; the agent genuinely
   finds nothing. This is the *strongest possible* version of the negative
   result, because the confound is now excluded. Report it as such.
2. **Curved but does not beat uniform** → the agent explores real non-uniform
   schedules and still loses, which is Theorem 3.1 confirmed against a competent
   adversary rather than a saturated one. Also a strong result, and it explains
   the authors' curve without needing their gain to be real.
3. **Curved and beats uniform** → Route B is live. Cross-check immediately
   against E1: the learned curve should have the same direction as `ρ_t ∝ √a_t`.
   If the agent wins while E1 says `a_t` is flat, suspect a leak before
   believing it.

Outcome 3 is the supervisor's win. Outcomes 1 and 2 are better science and
should be written as the main result if that is what the data says.

### Cross-check with E1

E1, E2 and E3 measure the same underlying question three ways. They must agree
on direction:

* E1 `a_t` increasing ⇔ E2 `back-clean` > `front-clean` ⇔ E3 learned σ decreasing.
* Their Fig. 2 shows decreasing σ, which corresponds to increasing `a_t`.
* If two of the three disagree, stop and debug rather than write up.

---

## 3c. E4 — the source method's actual win rate (HIGHEST VALUE, ~6 h)

**This is the experiment that refutes the paper on its own benchmark, and it
does not depend on our DRL winning anything.** Nobody has ever run it, including
the authors.

### What is missing from the literature

The authors report CIFAR-10 `47.75%` vs `47.6%` (a `0.15pp` gain) and MNIST
`91.05%` vs `88.03%` (`3.0pp`), each from **a single run**. Their stated
*"we test 100 times on the test sets"* re-evaluates **one trained model** 100
times, which measures test-set sampling noise, not run-to-run training
variance. No win rate has ever been published for this method.

### The numbers that make this decisive

Run-to-run noise measured in our own replication is ~3pp at n=1, i.e. **20× the
CIFAR-10 effect being claimed**. Power analysis for a paired t-test
(80% power, α=0.05 two-sided):

| claim | trials needed to resolve |
|---|---|
| CIFAR-10 `0.15pp`, sd = 0.03 | **~3,140 paired trials** |
| CIFAR-10 `0.15pp`, sd = 0.015 (optimistic) | ~785 |
| MNIST `3.0pp`, sd = 0.03 | ~8 |
| MNIST `3.0pp`, sd = 0.015 | ~2 |

So the **CIFAR-10 claim is unresolvable at any sample size the authors could
plausibly have run**, while the **MNIST claim is genuinely checkable at n≈8**.
Be fair about that distinction in the write-up: MNIST may well be a real effect;
CIFAR-10 cannot be.

### Protocol (run the AUTHORS' OWN code, not our port)

Two harnesses are generated directly from `Author/Code/`, **unmodified except
for the trial count**:

```
author_test_CIFAR10_paired.py    <- Author/Code/test_CIFAR10.py
author_test_MNIST_paired.py      <- Author/Code/test_MNIST.py
```

The only substantive diff is `episodes = 1` → `N_TRIALS` (default 10), plus an
appended paired-statistics block and renamed output files. The DP mechanism,
the agent, the baselines and the federated loop are byte-for-byte theirs.

Verified facts about their originals — quote these, they matter:

* **`episodes = 1` is in the authors' own source** (`test_CIFAR10.py:34`,
  `test_MNIST.py:33`). It is not something we introduced.
* **Their loop is already correctly paired**: `net_init` is drawn per episode
  and every arm loads from it; `rhos_init` is drawn once and reused. They built
  a paired design and then ran it a single time, so the variance it was
  designed to control was never measured.
* **They test a selected checkpoint**: `agent.load(episode=350)` for CIFAR-10,
  `agent.load(episode=195)` for MNIST, both hardcoded, out of ~800 training
  episodes.
* Settings match the paper: CIFAR-10 `N=10, T=25, ρ∈[20,40], E=4`;
  MNIST `N=10, T=20, ρ∈[2,4], E=4`.

```
# needs the authors' trained actors in ./DRL_Nets/ :
#   actor_CIFAR10_350.pth, actor_target_CIFAR10_350.pth,
#   critic_CIFAR10_350.pth, critic_target_CIFAR10_350.pth      (MNIST: _195)

N_TRIALS=2  python author_test_MNIST_paired.py     # timing probe
N_TRIALS=10 python author_test_MNIST_paired.py     # MNIST first -- see below
N_TRIALS=10 python author_test_CIFAR10_paired.py
```

Each writes `exp_data/<TAG>_paired_stats.json` with per-trial values, win
counts, paired mean differences, 95% CIs and p-values against all three
baselines. Original outputs are never overwritten.

**Run MNIST first.** It is the claim that is actually resolvable: 3.0pp against
~3pp noise needs only n≈8, so n=10 genuinely settles it. CIFAR-10 at n=10
resolves ~2.7pp and therefore cannot confirm or refute a 0.15pp claim — for
CIFAR the reportable result is the *bound*, not a verdict. Prefer n=20 on both
if the timing allows.

**If the authors' checkpoints are unavailable**, retrain with
`Author/Code/train_MNIST.py` / `train_CIFAR10.py` and keep every saved episode,
then test the same episode number they used. Note in the write-up whether the
tested actor is theirs or a retrain.

### Decision rule (fix before looking)

* **CI contains 0** (expected) → no reproducible advantage on the authors' own
  benchmark, with their code, their released agent, at 2.5× their training
  budget. Report the win count and the ~3,140-trial figure.
* **CI excludes 0, DRL ahead** → the method does work on that benchmark and our
  drone negative result is task-specific. That is also a publishable finding,
  and it would redirect the whole thesis — so do not hope for either outcome.
  This is most plausible on **MNIST**, where the claimed effect is large enough
  to be real.
* **CI excludes 0, DRL behind** → strongest possible refutation.

### Two claims, do not conflate them

| | what it measures | cost |
|---|---|---|
| **(a)** freeze one trained agent, test it n times | *is this particular agent better?* | cheap — this harness |
| **(b)** retrain the agent from scratch n times, test each | *does the method reliably produce better agents?* | n × 2000 episodes |

A paper claims **(b)**. The authors measured one instance of **(a)**, once.
(a) is sufficient to refute them; (b) is the gold standard and is worth doing
only if (a) comes out ambiguous.

### Selection bias — and why ours is conservative

`LOAD_EPISODE = 181`, described in their code as *"load best model by episode
number"*, so the tested agent is a **selected** checkpoint. Our drone actor was
selected the same way: `lightcnn_publication_plots.py` states it uses the
checkpoint from *"the run that beat Uniform 5/5 in the fair shared-init
comparison"* — a 5/5 that was later found to be scored on each client's own
training shard and to reverse on held-out data (0.823 vs 0.906).

**Note the direction.** Both actors were picked for having looked good, which
biases results **in favour of DRL**. Ours still lost 3/8 on held-out data. A
negative result obtained with the selection bias pointing the other way is
conservative, and should be stated as such rather than hidden.

---

## 3d. E5 — Schedule hunt: does ANY schedule beat uniform? (added 2026-09-25)

**One question: is there any gain for a learned allocator to find?** A DRL
agent can only beat Uniform if *some* better schedule exists. Fixed schedules
are cheap to test, so test them first. If none beats Uniform, DRL can't either,
and the chase stops without days of agent training. If one does, chase it.

### Why this, and why now

The drone DRL "loss" is not what it looks like (`exp_data/lightcnn_pub_fig23_data.json`,
n=8, held-out):

* **The drone DRL emits the Uniform schedule.** σ differs from Uniform's by
  ≤ 0.013 in all 8 trials; Σσ² penalty 1.0000× (Exp/Arith: 1.052×).
* **Its −4.3pp "loss" (p=0.116, 3/8 wins) is random-number noise, not policy.**
  `lightcnn_publication_plots.py:179` seeds `random`/`numpy` once per trial,
  **never seeds `torch`**, and runs the four arms back to back. Each arm gets
  different DP noise and different shuffling. Two near-identical schedules end
  up 6.8pp apart (sd of the paired difference).
* The frozen-trunk experiments already use CRN (`tight_budget_ladder.py:57`).
  The drone DRL test does not.

And the tight-budget ladder (`exp_data/tight_budget_ladder.json`, uniform, CRN,
n=10) hints at the one place a non-uniform schedule might win:

| ρ | T=3 | T=6 | T=12 | T=25 |
|---|---|---|---|---|
| 8 | 0.512 | 0.602 | 0.618 | **0.640** |
| 2 | 0.513 | 0.602 | 0.611 | **0.636** |
| 0.5 | 0.511 | **0.601** | 0.597 | 0.561 |

At loose budgets more rounds is better. At ρ=0.5 (σ=14.1 at T=25) it reverses:
fewer, cleaner rounds did better (+0.040, **not significant**: 4/10 wins,
sd(diff) 0.13–0.21). That's a non-uniform strategy an agent could learn, if
it's real.

### Key fact: more rounds spreads the same budget thinner

At fixed total ρ, uniform σ grows as √T (σ = √(2·el·T/ρ), el=2). Going
25 → 50 rounds multiplies every round's σ by 1.41. This does **not** add
budget.

### E5a — CRN fix for the drone DRL test (a correction, not a win)

In `lightcnn_publication_plots.py`'s Fig 2/3 loop, re-seed **before each arm**
with the trial's seed:

```python
seed = 5000 + t
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
```

`rhos` and `init_sd` are already drawn once per trial and shared. Keep that.

* **Prediction:** |DRL − Uniform| < 0.005 in every trial, i.e. an exact tie.
  The −4.3pp disappears.
* **If it doesn't tie:** there's another nondeterminism source (data-loader
  workers, opacus RNG, CUDA atomics). Find it before trusting any drone
  comparison.
* **What this does NOT do:** create a win. DRL and Uniform are the same schedule.
* Runs where the lightcnn drone setup and the DRL checkpoint live (Windows box
  / 5090). Same cost as the original 8-trial test.

### E5b — fixed-schedule hunt at T=50 (the real test)

Frozen-trunk pipeline, canonical config (§1: D=566,789, B=128, 5 clients, SGD
lr=0.003, C=1.0). **All clients use the same schedule.** Only rounds vary, not
clients. (This also keeps the norm band tight, see "Link to the P3 plan" below.)

**Budgets:** ρ ∈ {1, 2} at T=50.

**ρ=0.5 at T=50 is OUT OF SPEC and excluded from the main grid.** Uniform
already sits at σ=20.0, and every decaying/increasing schedule needs σ up to
~24. The [1, 20] clamp would then silently overspend the privacy budget. Run
it only as a labelled side arm with the cap raised and the realised Σρ_t
logged.

**Arms (σ per round, computed with the same formulas as the authors' baselines):**

| Arm | Idea | ρ=1: σ first → last | ρ=2: σ first → last |
|---|---|---|---|
| Uniform-50 | reference | 14.14 → 14.14 | 10.00 → 10.00 |
| Exp-decreasing | less noise late (the MNIST winner pattern) | 16.99 → 12.02 | 12.02 → 8.50 |
| Arith-decreasing | same, linear | 17.32 → 12.25 | 12.25 → 8.66 |
| Exp-increasing | **direction control**: should LOSE if decreasing wins | 12.02 → 16.99 | 8.50 → 12.02 |
| Truncate-25 | spend it all in 25 rounds, then stop | 10.00 (25 rds) | 7.07 (25 rds) |
| Truncate-12 | spend it all in 12 rounds, then stop (the ladder hint) | 6.93 (12 rds) | 4.90 (12 rds) |

Exp/Arith use `sigma_exponential` / `sigma_arithmetic` from
`lightcnn_publication_plots.py:67-85` (ρ grows ×2 across rounds for Exp).
Truncated arms train only their active rounds. Rounds with zero budget get no
DP-trained update.

**Score:** mean IoU.25 over the last 3 *active* rounds (`SCORE_LAST=3`, as in
the ladder). Also store final-round IoU.25 and the full trajectory.

**CRN:** every arm in a trial uses the same seed (re-seed torch/cuda/numpy/
random before each arm, cudnn deterministic). Unlike the ladder, most arms here
share the same T, so the random streams stay aligned and sd(diff) should be much
smaller than the ladder's 0.13–0.21. That's why this hunt may succeed where
Door 1 couldn't. **Measure it in the pilot. Don't assume it.**

### Pre-registered seeds and stages

* **Stage 1: pilot.** n=10 per (ρ, arm), seeds `9000 + i`, i = 0..9.
  2 budgets × 6 arms × 10 = 120 runs.
* **Candidate:** any arm with mean diff vs Uniform-50 > 0 **and** ≥ 7/10 wins.
  Record sd(diff) for every arm.
* **Stage 2: confirmation.** Candidates only, on **disjoint** seeds
  `9500 + i`, n from a power calculation on the Stage-1 sd(diff) (80% power,
  α=0.05), **minimum 20**.
* **Confirmed** = paired p < 0.05 and positive mean diff on the Stage-2 seeds.
  Only confirmed results count. Four single-run "wins" have already evaporated
  in this project (see `CLAUDE.md`).

### Decision rule (fixed before looking)

* **No arm confirmed →** no gain exists at (ρ, T=50) either. The DRL chase stops
  there. Report it as a negative result. It extends "uniform is optimal at this
  operating point" to 50 rounds.
* **An arm confirmed → a schedule exists. Chase it.** Port the DDPG agent into
  the frozen-trunk pipeline (~1 day code + est. 7–13 h on the 5090), train at
  that (ρ, T), and test with CRN against **both** Uniform and the confirmed
  fixed schedule.
* **A DRL "win" must beat the confirmed fixed schedule, not just Uniform.**
  Beating only Uniform gets the critique already made of the source paper: "a
  one-line fixed schedule does the same" (MNIST actor 114 vs Exponential,
  p=0.557).
* **Direction check:** if Exp-decreasing wins but Exp-increasing *also* beats
  Uniform, the effect is not "less noise late". Investigate before chasing.

### Cost and where it runs

* ~1 min per 50-round run on the 5090 (estimate; timing-pilot first).
  Stage 1 ≈ **1.5–2 h**. Truncated arms are cheaper. Stage 2 depends on how many
  candidates survive.
* **Not runnable on the 1650 Super:** the frozen-trunk cache is ~13 GB vs 4 GB
  VRAM (`p3_only/COSTS.md`). Use the 5090 or the lab card (`nvidia-smi -pl 300`
  after reboot).
* Check the 5090's PyTorch has `sm_120` first (`p3_only/COSTS.md`).

### Link to the P3 plan (`p3_only/fresh_beginning.md`)

A winning schedule here varies σ **across rounds, identically for every
client**. Per `fresh_beginning.md` §4.2 that does **not** open the Byzantine
channel: the server compares clients within a round, and they all share σ. So
the two halves agree. *Round-varying allocation can help accuracy; client-
heterogeneous allocation opens a poisoning channel.* If E5b confirms a
schedule, that's the sentence that ties the thesis together.

### Outputs

`exp_data/E5a_drone_crn.json`, `exp_data/E5b_schedule_hunt.json`. Per §5:
per-trial arrays, σ per round per arm, `seed_base`, `crn: true`, realised Σρ_t
per arm, full trajectories, and the machine/GPU each run used.

---
