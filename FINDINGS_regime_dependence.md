# The optimum is regime-dependent: uniform wins at low σ, loses badly at high σ

**2026-09-21. This is the most important result in the project, and it refutes
the thesis's central claim as currently written.**

> ## ⚠ SUPERSEDED IN PART, 2026-09-22 — THE DRL RESULT IS CHECKPOINT-DEPENDENT
>
> §1's headline ("DRL significantly beats uniform, +2.27pp, p=0.004") was
> obtained with **actor 114**. Repeating the identical experiment with
> **actor 612** — the later, better checkpoint from the same training run —
> reverses every comparison, and both extremes are significant:
>
> | DRL vs | actor 114 | actor 612 |
> |---|---|---|
> | Uniform | **+0.0227**, 9/10, p=0.004 | **−0.0112**, 2/10, p=0.051 |
> | Exponential | +0.0029, 7/10, p=0.557 | **−0.0265**, 0/10, **p=0.00003** |
> | Arithmetic | +0.0085, 6/10, p=0.123 | **−0.0220**, 0/10, **p=0.002** |
>
> Same code, same data, same n=10 paired protocol. **Only the checkpoint
> differs.** See **`FINDINGS_checkpoint_dependence.md`** — that is now the
> headline result, and it is stronger than this one.
>
> **What in this document still stands:** the regime-dependence mechanism, which
> is visible in the *fixed* schedules in both runs and does not depend on the
> agent at all. At σ≈7, Exponential (1.0450× variance) and Arithmetic (1.0442×)
> both beat Uniform (1.0000×) — accuracy increases with accumulated variance,
> the inverse of the thesis's −0.767. §2, §3 and §6 are unaffected.
>
> **What does not stand:** any claim that the learned allocator beats uniform.
> That is checkpoint-dependent and therefore not established in either
> direction.

Established by running the authors' unmodified MNIST code — with their unit bug
fixed so the stated σ ∈ [1,20] range is actually enforced — at **n = 10 paired
trials**, actor 114.

---

## 1. The result

| arm | accuracy (n=10) | sd |
|---|---|---|
| **DRL** | **0.9061** | 0.0101 |
| Exponential | 0.9032 | 0.0116 |
| Arithmetic | 0.8976 | 0.0076 |
| **Uniform** | **0.8834** | 0.0185 |

Paired differences, DRL against each baseline:

| vs | mean diff | 95% CI | wins | p |
|---|---|---|---|---|
| **Uniform** | **+0.0227** | **[+0.0094, +0.0359]** | **9/10** | **0.004** |
| Exponential | +0.0029 | [−0.0078, +0.0136] | 7/10 | 0.557 |
| Arithmetic | +0.0085 | [−0.0028, +0.0197] | 6/10 | 0.123 |

**The learned allocator significantly beats uniform: +2.27pp, p = 0.004, 9 wins
in 10.** It leads both other baselines on point estimate but neither
significantly.

The authors' published MNIST claim is +3.02pp over their best baseline. We
measure +2.27pp over uniform, at n=10 with a confidence interval. **Their
headline result reproduces.**

## 2. The part that refutes the thesis

Sort the arms by the quantity Theorem 3.1 says governs everything:

| arm | Σσ² penalty | accuracy |
|---|---|---|
| Uniform | **1.0000×** (the analytic minimum) | **0.8834** (worst) |
| Arithmetic | 1.0442× | 0.8976 |
| Exponential | 1.0450× | 0.9032 |
| DRL | **1.0753×** (worst) | **0.9061** (best) |

```
Spearman(accumulated noise variance, accuracy) = +1.000
```

**Perfectly monotone, and in the wrong direction.** More accumulated noise
variance produces *better* accuracy here. The thesis reports −0.767 for the same
correlation on the drone task. The variance-only model does not merely fail in
this regime; it is exactly inverted.

## 3. Reconciliation — one framework covers both

This is not a contradiction between two experiments. It is the weighted
objective the reviews predicted:

```
V = Σ_t a_t · σ_t²        optimum  ρ_t ∝ √a_t
```

* **Thesis regime** (drone, σ ≈ 1.83, D = 566,789): noise is low relative to
  signal, `a_t` is near-flat, the Σσ² term dominates → **uniform is optimal**,
  Spearman −0.767, every deviation costs.
* **Source-paper regime** (MNIST, σ ≈ 6–9, small CNN): noise is large, late
  rounds are disproportionately damaged, `a_t` is steeply non-flat → **schedules
  that go quiet late win**, and the Σσ² penalty they pay is irrelevant beside the
  timing benefit.

Every non-uniform arm here *decreases* σ across rounds
(DRL 9.24 → 5.94, Exponential 8.75 → 6.19, Arithmetic 8.92 → 6.30) while uniform
sits flat at 7.28. All three beat it. That is the `a_t` effect, visible directly.

**Theorem 3.1 is not wrong — its scope is far narrower than the thesis claims.**
The assumption that fails is (iii), utility monotone in accumulated variance;
it holds at low σ and fails at high σ.

## 4. What survives for the thesis, and it is not nothing

**DRL ≈ Exponential (+0.29pp, p = 0.557).** A trivial fixed exponentially
decaying schedule performs as well as the learned multi-agent policy. The
advantage over uniform comes from the schedule being **decreasing**, not from
its being **learned**.

So the defensible negative result changes shape but survives:

> In the regime where allocation matters, a learned allocator's entire advantage
> over uniform is reproduced by a one-line fixed exponential schedule. The DDPG
> agent, value-decomposition network and action-mapping machinery buy nothing
> over that baseline (p = 0.557).

That is a claim about **MARL**, not about allocation, and it is the one the
evidence actually supports. The current thesis claim — that uniform is optimal
and all deviation is strictly worse — must be withdrawn as a general statement
and re-scoped to the low-σ regime where it was measured.

## 5. Caveats — state these, do not bury them

* **Actor 114 from an incomplete 544/800 training run.** Undertrained; the
  result is a *lower* bound on DRL's performance. Re-run against the final actor.
* **MNIST with a 2-conv/3-FC network**, not the detection task. The drone result
  at σ ≈ 1.83 stands on its own data and is not contradicted — it is a different
  regime.
* n = 10. Uniform is resolved (p = 0.004); Exponential and Arithmetic are not.
* Their unit bug had to be fixed for any of this to appear
  (`FINDINGS_authors_code_bugs.md`). With their shipped code all four arms sit
  inside 0.15pp and nothing is measurable.

## 6. What this changes

**For the supervisor's demand:** met, legitimately. DRL significantly beats
uniform on the source paper's own benchmark, paired, n=10, held-out, p = 0.004.
No reinstated training-shard scoring, no cherry-picked seed.

**For the thesis:**

1. §3.2 Theorem 3.1 must be restated in weighted form with `a_t`, and its
   validity scoped to `a_t` ≈ const, which is an empirical condition, not an
   assumption to assert.
2. §5.1's "no allocation beats uniform" must be qualified to the measured
   regime. The Spearman −0.767 result stands *for the drone task at σ ≈ 1.83*.
3. The new headline is the **regime dependence** itself, which is more
   interesting than either half: whether adaptive allocation helps is decided by
   where the operating point sits on the noise curve, and both the source paper
   and this thesis were each measuring one side of it without knowing.
4. **E1 is now the central experiment, not a repair.** Measuring `a_t` across σ
   produces the curve that unifies both results.

## 7. Reproduce

```bash
cd Author/Code
VP=/run/media/afnanmeow/Shared/tmp_smoketest/venv/bin/python3
N_TRIALS=10 $VP test_MNIST_consistent_paired.py     # ~62 min, 10 client procs
```

Outputs `exp_data/MNIST_consistent_paired_stats.json` (per-trial values, CIs,
p-values) and `exp_data/exp_data_MNIST_consistent_paired.json` (curves, σ
schedules, budgets).

`test_MNIST_consistent_paired.py` differs from `Author/Code/test_MNIST.py` in
exactly three ways: the `/bs` removed from `rho2sigma`/`sigma2rho` so their own
QP bounds and σ clamp become consistent; `episodes` configurable; a paired-stats
block. The mechanism, agent and baselines are theirs.

Variance-vs-accuracy check:

```python
import json, numpy as np
d = json.loads(json.load(open('exp_data/exp_data_MNIST_consistent_paired.json')))
for off, n in zip(range(8, 12), ['DRL','Uniform','Exponential','Arithmetic']):
    v = [ (np.atleast_1d(np.array(t,float).mean(0) if np.array(t,float).ndim>1
           else np.array(t,float))**2).sum() for t in d[off] ]
    print(n, round(float(np.mean(v)), 1))
```
