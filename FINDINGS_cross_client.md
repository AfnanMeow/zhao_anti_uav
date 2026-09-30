# Cross-client structure: budget pooling is illegal, but inverse-variance weighting is free

Found 2026-09-20, prompted by the question *"if clients could share budgets among
themselves as a genuinely collaborative MARL agent, would that beat uniform?"*

Short answer: **the maths is real, the mechanism is not a valid DP mechanism,
and the same gain is available legally through aggregation weights instead.**
This corrects an over-strong sentence in Chapter 3 (see §6).

---

## 1. The claim, formalised

Currently each client carries its own budget, so the constraint is per-client:

```
V_per-client  =  (2E T² / N²) · Σ_i 1/ρ_i        s.t.  Σ_t ρ_{i,t} = ρ_i  for each i
```

The objective separates across `i`, so each client independently goes uniform in
time — that is Theorem 3.1 applied N times.

If budgets could instead be **pooled** into a single constraint
`Σ_{i,t} ρ_{i,t} = R = Σ_i ρ_i`, AM–HM runs over all `N·T` variables at once and
every `ρ_{i,t}` becomes equal to `R/(NT)`:

```
V_pooled  =  2E T² / R
```

The ratio is

```
V_per-client / V_pooled  =  (1/N²) · (Σ_i ρ_i) · (Σ_i 1/ρ_i)   ≥  1
```

by Cauchy–Schwarz (take `a_i = √ρ_i`, `b_i = 1/√ρ_i`, so `(Σ 1)² = N² ≤
(Σρ_i)(Σ1/ρ_i)`), **with equality if and only if every `ρ_i` is equal.**

So the claim is mathematically correct: pooling strictly beats per-client uniform
whenever budgets are heterogeneous. It is not a misunderstanding.

## 2. How large is it

The ratio is **scale-invariant** — it depends only on the *shape* of the budget
distribution, not its magnitude. `U[a, 2a]` gives the same answer for any `a`,
which is why MNIST and CIFAR-10 come out identical below.

| setting | mean variance ratio | variance cut |
|---|---|---|
| our measured ρ (5 clients, 20.85–39.63) | 1.0305 | **3.0%** |
| their CIFAR-10, ρ ~ U[20,40], N=10 | 1.0357 | 3.4% |
| their MNIST, ρ ~ U[2,4], N=10 | 1.0359 | 3.5% |
| hypothetical ρ ~ U[1,100], N=10 | 2.2122 | 54.8% |
| hypothetical ρ ~ U[0.5,100], N=10 | 2.5229 | 60.4% |

Calibration against our own measurements: a 1.18× variance penalty cost
−0.0018 IoU.25 and a 1.04× penalty cost −0.0007 (`FINDINGS_gate1.md`, headroom
and fine-negk tables). So a 1.03× *advantage* is worth roughly **+0.001**, which
is an order of magnitude below the paired noise floor of 0.017.

**In every budget distribution this literature actually uses, the gain is real
but unmeasurable.** It only becomes substantial under wildly unequal budgets
(the U[1,100] row), which nobody uses.

## 3. Why pooling is not a valid DP mechanism

Privacy budget is not transferable currency. If client A "gives" budget to
client B, then B adds less noise — but B's noise is what protects **B's** data.
A's consent has no bearing on B's guarantee. Concretely:

* Clients with `ρ_i < R/N` end up spending more than they were allocated, so
  their data receives a **weaker** guarantee than promised. That is a violation,
  not a trade.
* Clients with `ρ_i > R/N` spend less than allowed. Harmless, but it is their
  allowance being discarded.

The entire gain is therefore obtained by degrading the privacy of exactly the
clients who asked for the most protection.

**And it is self-defeating.** The only clean way to pool is for all parties to
agree up front to one common budget — but a common budget means all `ρ_i` are
equal, which is precisely the equality case where the ratio is **1.000 and the
gain is zero**. The benefit comes from the heterogeneity that legitimate pooling
would eliminate. It cannot be rescued by consent.

(The one setting where a pooled budget *is* legitimate: when the "clients" are
not distinct data owners but devices belonging to a single curator holding one
global budget. Then the per-client constraint was simply the wrong model to
begin with — and one would set equal budgets, returning the gain to zero.)

## 4. The legitimate version: inverse-variance aggregation weighting

The same gain is available with no privacy cost at all, by changing the
**aggregation weights** rather than the budgets.

FedAvg uses `a_i = 1/N`. Let `a_i` be free subject to `Σ_i a_i = 1`
(unbiasedness). The aggregate noise variance is `Σ_i a_i² σ_i²`; minimising it
gives

```
a_i  ∝  1 / σ_i²            V_min = 1 / (Σ_i 1/σ_i²)
```

against `V_FedAvg = (1/N²) Σ_i σ_i²`, so

```
V_FedAvg / V_min  =  (1/N²)(Σ_i σ_i²)(Σ_i 1/σ_i²)
```

Since a client uniform in time has `σ_i² = 2ET/ρ_i`, this is **the identical
expression** to §1 — the same `(1/N²)(Σρ_i)(Σ1/ρ_i)`. Same ceiling, but:

* it is **pure server-side post-processing** of already-privatised updates, so
  the DP guarantee and the accountant are untouched;
* it requires no transfer, no renegotiation, and no consent;
* clients with tight budgets (noisier updates) are simply trusted less, which is
  the statistically correct thing to do.

### Two caveats before implementing it

1. **Data volume.** If clients hold different amounts of data, the variance-optimal
   weight balances information content against noise, roughly `a_i ∝ n_i/σ_i²`,
   not `1/σ_i²` alone. Standard FedAvg already weights by `|D_i|`; the correct
   combination weights by both.
2. **Non-IID drift.** The derivation assumes every `Δ_i` is an unbiased estimate
   of the same quantity. Under non-IID shards they are not — each points toward
   its own local optimum — so down-weighting a noisy client also down-weights its
   distinct data. On heterogeneous partitions this trades noise reduction against
   client drift and should be measured, not assumed.

## 5. Is this worth claiming?

Modest but defensible. The source paper uses heterogeneous budgets
(`ρ_k ~ U[20,40]`, `U[2,4]`) together with uniform FedAvg weighting, leaving
~3.5% of noise variance on the table for free. Unlike the DRL result, this does
not have to beat anything to be correct — it is an exact statement about the
variance-optimal linear combination.

It is also the honest home for the "collaboration" intuition. There *is* genuine
cross-client structure in this problem; it simply lives in the aggregation
weights, not in the allocation of a fixed budget over rounds. A
credit-assignment mechanism such as a value-decomposition network would have
something to discover here, which is not true of the allocation problem.

Expected effect in our regime: ~+0.001, below the noise floor. Claim it as an
analytic result with a measured bound, **not** as a demonstrated accuracy gain.

## 6. Chapter 3 correction required

`chapters/chapter_3.tex`, `\paragraph{Across clients.}` (lines 114–125) ends:

> "There is no cross-client structure for a credit-assignment mechanism such as
> a value-decomposition network to discover."

**This is wrong as written.** It is true only of the per-round allocation of a
fixed per-client budget. It is false in general, because the aggregation weights
are cross-client structure and uniform FedAvg is suboptimal under heterogeneous
budgets by exactly `(1/N²)(Σρ_i)(Σ1/ρ_i)`.

Replace with something that says: the allocation problem separates across
clients, so each faces Theorem 3.1 independently; budgets cannot be pooled
because privacy budget is a property of a data owner, and pooling would weaken
the guarantee of the most privacy-sensitive clients while the equal-budget case
that consent would produce carries zero gain; but the aggregation weights do
*not* separate, and inverse-variance weighting captures a
`(1/N²)(Σρ_i)(Σ1/ρ_i)` variance reduction — approximately 3.5% under the budget
distributions used in this literature, and below our measurement floor.

Also note this is the same lever one review flagged: *"'no cross-client
structure' overlooks aggregation weights. With heterogeneous client budgets,
which Chen et al. use, inverse-variance weighting is a real cross-client lever."*

## 7. Reproduce

```python
import json, numpy as np
def ratio(r):
    r = np.asarray(r, float); N = len(r)
    return (1/N**2) * np.sum(r) * np.sum(1/r)

d  = json.load(open('exp_data/lightcnn_pub_fig23_data.json'))
rh = np.array(d['all_rhos'], float)              # trials x clients
print([round(ratio(rh[t]), 4) for t in range(rh.shape[0])])   # ~1.03

rng = np.random.default_rng(0)
print(np.mean([ratio(rng.uniform(20, 40, 10)) for _ in range(20000)]))   # 1.0357
print(np.mean([ratio(rng.uniform(1, 100, 10)) for _ in range(20000)]))   # 2.2122
```

## 8. Status

Analytic result, verified numerically against our own measured budgets. No new
GPU runs required for the claim itself. If you want the empirical version, the
cheap test is one paired arm — FedAvg uniform weighting vs `a_i ∝ n_i/σ_i²` at
`n = 20`, in the calibrated pipeline — but expect it to land inside the noise
floor and be prepared to report it as a bounded null.
