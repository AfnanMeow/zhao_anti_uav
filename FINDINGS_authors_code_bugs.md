# The source implementation does not run the mechanism its paper describes

Found 2026-09-21, from the first real run of the authors' unmodified
`Author/Code/test_MNIST.py` (actor 114). Two separate defects, both verifiable
from their source in under a minute. Together they mean **the published
experiments did not evaluate the method as specified.**

---

## 1. The σ range stated in the paper is not the range applied

The paper specifies `[σ_min, σ_max] = [1, 20]` for both MNIST and CIFAR-10.
Their code declares the same. But the box constraint that is supposed to
enforce it is written in different units from the conversion it constrains:

```python
rho_used_min = 2 / sigma_max**2 * epoch_local      # = 0.0200   <- no /bs
rho_used_max = 2 / sigma_min**2 * epoch_local      # = 8.0000   <- no /bs

def sigma2rho(sigma, el, bs):
    return 2 * el / (bs * (sigma**2))              # <- WITH /bs
def rho2sigma(rho, el, bs):
    return np.sqrt((2 * el) / (bs * rho))          # <- WITH /bs
```

With `el=4, bs=200`:

| | range |
|---|---|
| bounds used in the action-mapping QP | `0.0200 … 8.0000` |
| values `sigma2rho` can actually return | `0.000100 … 0.0400` |

The upper bound is **200× larger than anything reachable**, so the constraint
never binds on that side and the `σ_min = 1` floor is inoperative.

**Measured consequence.** In a real run of their unmodified code the applied σ
is **0.477 – 0.649**, i.e. roughly half of their own stated minimum, across
every arm:

```
DRL          0.584 -> 0.477        Uniform      0.530 flat
Exponential  0.637 -> 0.450        Arithmetic   0.649 -> 0.459
```

So every number in the paper was produced at a noise scale about **14× lower**
than the specification the paper reports. Whatever the results mean, they are
not results for σ ∈ [1, 20].

## 2. Which form was intended — and it is not the one they shipped

This is decisive, and it is arithmetic rather than opinion. Evaluate their own
bounds against both candidate conversions:

```
their QP bounds                        : [0.020000, 8.0000]

sigma2rho WITHOUT /bs, at [σ_max, σ_min] : [0.020000, 8.0000]   <- EXACT MATCH
sigma2rho WITH    /bs, at [σ_max, σ_min] : [0.000100, 0.0400]   <- no match
```

`rho_used_min/max` are *defined* as `2/σ²·el` at the range endpoints. They match
the **no-`/bs`** conversion exactly, at both ends. The `/bs` inside
`rho2sigma`/`sigma2rho` is therefore the inconsistent element: it contradicts
the bounds written three lines above it in the same file.

### Consequence for our port

`fl_damoyolo_final.py` and its descendants omit the `/bs`:

```python
def rho2sigma(rho, el, bs): return float(np.sqrt(2 * el / rho))
def sigma2rho(sigma, el, bs): return 2 * el / (sigma**2)
```

**This is the internally consistent form** — it is what their own bounds were
written for, and under it the `[1, 20]` clamp works as specified. Our port did
not introduce an error; it removed one. **Do not "restore" the `/bs`.** Doing so
would reintroduce their defect and break the σ clamp.

(The `bs` parameter is still accepted and ignored in our version. Cosmetic;
leave it, or drop the argument. It changes no result.)

## 3. Scope — what this does and does not affect

**Does not affect any thesis result.** None of the scripts producing the
thesis's experiments call these functions at all:

```
gate_DB_sweep.py   allocation_bound.py   headroom_sweep.py   tight_budget_ladder.py
truncation_test.py fine_negk_test.py     p3_byzantine.py     stage2_gate.py
```

All are clean; they build schedules inline as
`np.sqrt(2 * EPOCH_LOCAL / rho)` — the consistent form. The conversion appears
only in the deprecated DRL scripts (`fl_damoyolo_final.py`,
`fl_lightcnn_final.py`) and the CIFAR-10 ports.

**Does affect ε reporting.** Our ρ values are not on the same scale as the
paper's, so `ρ=30 → ε≈67` should not be presented as comparable to their
budgets. The correct fix is not to patch the formula but to **report ε from
Opacus's RDP accountant** using the actual `(σ, sample_rate, steps)` — which is
what applies the noise in our pipeline anyway — rather than from an ad-hoc
per-round ρ expression.

## 4. Combined with the sample size, the evaluation cannot support the claim

Independently of the conversion, their evaluation design:

* `episodes = 1` in both test scripts (`test_CIFAR10.py:34`, `test_MNIST.py:33`)
  — **a single measurement per arm.**
* `agent.load(episode=350)` / `agent.load(episode=195)` — a **selected**
  checkpoint out of ~800.
* *"We test 100 times on the test sets"* — 100 re-evaluations of **one trained
  model**, which measures test-set sampling noise, not run-to-run training
  variance.
* Their loop is **already correctly paired** (`net_init` redrawn per episode and
  shared by every arm; `rhos_init` drawn once and reused) — the design controls
  exactly the variance they then never measured by running it once.

Measured run-to-run spread in this project is ~3pp at n=1. Their CIFAR-10 claim
is 0.15pp. Resolving an effect that size against that noise needs **~3,140
paired trials** (80% power, α=0.05); ~785 even on an optimistic noise estimate.
The MNIST claim (3.0pp) is the only one resolvable at feasible n (~8).

## 5. First replication result (provisional)

Authors' unmodified code, their MNIST configuration, actor 114, n=1:

| arm | final accuracy |
|---|---|
| DRL | 0.9563 |
| Uniform | 0.9550 |
| Exponential | 0.9552 |
| **Arithmetic** | **0.9565** |

Claimed MNIST gain **+3.02pp**; measured **+0.13pp**, and DRL is not the best
arm — Arithmetic is. All four lie within **0.15pp**. Absolute accuracy (95.6%)
*exceeds* their reported 91.05%, so this is a healthy reproduction, not a broken
one; the methods simply do not separate.

**Provisional**: n=1, and actor 114 against their tested 195, with training at
544/800. Actor 114 has nevertheless remained "best" for 430 episodes.

## 6. What is fair to write, and what is not

Supportable, and each independently checkable from their repository:

> The released implementation does not evaluate the mechanism the paper
> specifies. The action-mapping box constraint is written in different units
> from the conversion it bounds, so the stated noise range σ ∈ [1,20] is never
> enforced and the applied noise is roughly 14× smaller. The evaluation reports
> a single run per arm from a selected checkpoint, with error bars computed by
> re-evaluating one trained model rather than by repeating training, so no
> run-to-run variance was measured. The reported CIFAR-10 effect (0.15pp) is an
> order of magnitude below the run-to-run spread of the same experiment.

Not supportable, and should not be written: any claim of intent — cherry-picking,
concealment or dishonesty. There is no evidence of selective reporting, and
`episodes = 1` plus a fixed-checkpoint evaluation was common practice in this
literature. Unit inconsistency and an under-powered design are errors, not
misconduct, and the specific version above is both stronger and safe to defend.

## 7. Reproduce

```bash
VP=/run/media/afnanmeow/Shared/tmp_smoketest/venv/bin/python3
$VP -c "
el,bs,smin,smax=4,200,1.0,20.0
print('QP bounds        : [%.6f, %.4f]'%(2/smax**2*el, 2/smin**2*el))
print('no /bs  reachable: [%.6f, %.4f]'%(2*el/smax**2, 2*el/smin**2))
print('with/bs reachable: [%.8f, %.6f]'%(2*el/(bs*smax**2), 2*el/(bs*smin**2)))"
```

Applied σ and accuracies from their run:

```python
import json, numpy as np
d = json.loads(json.load(open('Author/Code/exp_data/exp_data_MNIST.json')))
for i, n in enumerate(['DRL','Uniform','Exponential','Arithmetic']):
    print(f'{n:12s} final={d[i][0][-1]:.4f}')
for off, n in zip(range(8,12), ['DRL','Uniform','Exponential','Arithmetic']):
    s = np.array(d[off][0], float); m = np.atleast_1d(s.mean(0) if s.ndim>1 else s)
    print(f'{n:12s} sigma {m[:3].round(3)} ... {m[-3:].round(3)}')
```

## 8. In progress

`Author/Code/test_MNIST_consistent.py` — their code with the `/bs` removed so
the `[1,20]` clamp works as specified, i.e. **the experiment they intended**.
Running now against actor 114; output to
`exp_data/exp_data_MNIST_consistent.json`.

Caveat to state when reporting it: actor 114 was *trained* under the
inconsistent conversion, so testing it under the corrected one is a
policy/environment mismatch. It isolates the conversion's effect on the applied
schedule, but a clean version requires retraining under the corrected
conversion. That is a lab-machine job.

The reciprocal test ("theirs in ours" — our drone pipeline with `/bs` restored)
is **not currently runnable**: `DRL_Nets/lightcnn_checkpoint.pth`,
`fl_workdirs/lightcnn_shards` and `single_test_coco/` are all absent after the
rebuild.
