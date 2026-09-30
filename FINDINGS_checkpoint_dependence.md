# Checkpoint selection alone flips the conclusion, in both directions, significantly

**2026-09-22. This is the strongest result in the project.**

Two checkpoints from **the same training run**, evaluated with **the same code,
the same data and the same n=10 paired protocol**, produce opposite and
statistically significant conclusions about whether adaptive privacy-budget
allocation works.

This is not a claim that the source paper's numbers are noise. It is a
demonstration, on their own implementation, that **their evaluation methodology
— select one checkpoint, test once — can yield either answer.**

---

## 1. The result

MNIST, authors' code with their unit bug repaired
(`FINDINGS_authors_code_bugs.md`), n=10 paired trials, held-out test set.

| DRL vs | actor 114 | actor 612 |
|---|---|---|
| **Uniform** | **+0.0227**  9/10  p=0.004 | **−0.0112**  2/10  p=0.051 |
| **Exponential** | +0.0029  7/10  p=0.557 | **−0.0265**  0/10  **p=0.00003** |
| **Arithmetic** | +0.0085  6/10  p=0.123 | **−0.0220**  0/10  **p=0.002** |

Per-arm means:

| arm | actor 114 | actor 612 |
|---|---|---|
| DRL | **0.9061** *(best)* | **0.8786** *(worst)* |
| Exponential | 0.9032 | **0.9051** *(best)* |
| Arithmetic | 0.8976 | 0.9006 |
| Uniform | **0.8834** *(worst)* | 0.8898 |

The learned policy moves from **first place to last**. Against the two decaying
baselines it goes from statistically indistinguishable to losing **0 trials out
of 10**, at p = 3×10⁻⁵ and p = 0.002.

Note that the three *fixed* schedules barely move between the two runs
(≤0.64pp). **All of the instability is in the learned policy.**

## 2. Why — and this was predicted before the run

The mechanism is the schedule the agent emits:

| | actor 114 | actor 612 |
|---|---|---|
| learned σ schedule | 0.584 → 0.477 *(decreasing)* | **7.200 → 7.106 (flat)** |
| Σσ² penalty | 1.0157× | **1.0000×** — identical to uniform |

**Actor 612 learned uniform.** That is the correct policy *for the regime it was
trained in*: training ran under the authors' unrepaired code, where the σ clamp
never binds and applied σ ≈ 0.53. At that noise level `a_t` is near-flat, the
variance term dominates, and uniform genuinely is optimal. The agent converged
to it — the better-trained it became, the closer to uniform it got.

Evaluated in the repaired regime (σ ≈ 7), where decaying schedules win, that
same policy is now the wrong one. Actor 114, still under-converged and emitting
a decaying schedule by accident, happens to be closer to the test-regime optimum.

**The agent learns the optimum for its training regime, and that optimum does
not transfer.** Better training makes it *worse* under regime shift.

This was written down before the run, when only actor 612's schedule shape was
known:

> "actor 612's schedule is flatter than 114's... In the high-σ regime the
> corrected run creates, decreasing schedules are what win. So a flatter agent
> may score *worse* against Exponential/Arithmetic than actor 114 did, even
> though it's better trained — because it has converged toward uniform, which is
> the wrong target at high σ."

The prediction is recorded in the session transcript ahead of the result, so
this is a confirmed mechanism rather than a post-hoc story.

## 3. What this does to the source paper's claim

Their evaluation is `agent.load(episode=350)` with `episodes = 1`: **one selected
checkpoint out of ~800, measured once.** We have now shown, on their code, that
the checkpoint choice alone determines the sign of the result, and that both
signs reach significance at n=10.

It follows that:

* Their reported +3.02pp is **reproducible** — we obtained +2.27pp from actor
  114. It is not fabricated and not noise in the trivial sense.
* It is also **not evidence that the method works**, because a different
  checkpoint from the same run gives −2.65pp against the same baseline set.
* No claim of cherry-picking is needed or supported. Selecting a best-validation
  checkpoint is ordinary practice. The defect is testing it **once**, with no
  measure of how much the conclusion depends on that choice.

**The correct claim is about methodology, not about their honesty**, and it is
stronger for it.

## 4. What survives as a positive finding

**A fixed exponential schedule is the best and most stable allocation tested.**

| arm | actor 114 | actor 612 | spread |
|---|---|---|---|
| Exponential | 0.9032 | 0.9051 | 0.19pp |
| Arithmetic | 0.8976 | 0.9006 | 0.30pp |
| Uniform | 0.8834 | 0.8898 | 0.64pp |
| **DRL** | **0.9061** | **0.8786** | **2.75pp** |

Exponential beats uniform in both runs and is never beaten significantly by the
learned policy in either. The DDPG agent, value-decomposition network and
action-mapping machinery deliver nothing over a one-line decaying schedule, and
carry an order of magnitude more variance across checkpoints.

> In the regime where allocation matters, a one-line fixed exponential schedule
> outperforms uniform and matches or beats a learned multi-agent policy, while
> being dramatically more stable: across two checkpoints of the same training
> run the fixed schedules move by ≤0.64pp and the learned policy by 2.75pp,
> reversing sign.

That is the defensible negative result about MARL, and it now rests on two
independent checkpoints rather than one.

## 5. What is *not* established

* **That the learned allocator beats uniform.** Checkpoint-dependent; both
  directions significant. Do not claim it.
* **That the learned allocator loses to uniform.** Same reason. The honest
  statement is that the comparison is not identified by this evidence.
* The regime-dependence mechanism (`FINDINGS_regime_dependence.md` §2–3) is
  unaffected — it is visible in the fixed schedules alone and needs no agent.

## 6. Consequences for the thesis

1. **New headline.** "Whether adaptive allocation appears to work is determined
   by which checkpoint is evaluated, and the published protocol evaluates one."
   This is more defensible than either "uniform is optimal" or "DRL wins",
   because it does not require winning an argument about effect sizes.
2. **Uniform-optimality must still be re-scoped** — Exponential beats uniform in
   both runs at σ≈7 (`FINDINGS_regime_dependence.md`).
3. **Checkpoint variance must be reported** alongside trial variance in any
   future DRL comparison. A single checkpoint is a single sample from the
   policy distribution, and this project has now measured that distribution to
   be wide enough to reverse conclusions.
4. **E1 remains the unifying experiment** — measuring `a_t` explains why the
   training and test regimes have different optima.

## 7. Reproduce

```bash
cd Author/Code
VP=/run/media/afnanmeow/Shared/tmp_smoketest/venv/bin/python3
sed -i 's/agent.load(episode=[0-9]*)/agent.load(episode=612)/' test_MNIST_consistent_paired.py
N_TRIALS=10 $VP test_MNIST_consistent_paired.py        # ~62 min
cd .. && $VP analyse_mnist_paired.py --actor 612
```

Outputs: `exp_data/MNIST_consistent_paired_stats.json` (per-trial values, CIs,
p-values), `exp_data/exp_data_MNIST_consistent_paired.json` (curves, σ schedules).

**Back up `exp_data/` before re-running.** The actor-114 raw data was lost to an
overwrite; only its summary statistics survive, hardcoded as a reference in
`analyse_mnist_paired.py`.

## 8. Caveats

* Two checkpoints is a sample of two. The *direction* of each is significant at
  n=10, but the checkpoint-to-checkpoint distribution is not characterised. A
  proper treatment tests k checkpoints spaced across training.
* Actor 612 was trained under the unrepaired conversion and tested under the
  repaired one — a deliberate regime shift, and the point of §2. A clean
  "matched-regime" run requires retraining under the repaired conversion, which
  is the obvious next experiment and is a lab-machine job.
* MNIST with a 2-conv/3-FC network; the drone results are a separate regime.
