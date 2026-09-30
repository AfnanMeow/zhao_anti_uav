# The flat DRL schedule is an actor-saturation artefact, not a null result

Found 2026-09-20, from an observation about the source paper's Fig. 2. This is a
**confound in our own DRL experiment**, and it also yields a testable hypothesis
about the source paper's reported gains. Experiment to settle it: **E3** in
`NEXT_EXPERIMENTS.md`.

> ## ⚠ CORRECTION 2026-09-21 — THE MECHANISM BELOW IS WRONG AS STATED
>
> **The σ figures in §2 are defensible after all, but the mechanism is not.**
>
> The two codebases use different conversions (`FINDINGS_authors_code_bugs.md`):
> theirs divides by batch size, ours does not. Their own QP bounds prove the
> **no-/bs form is the intended one**, so the §2 table — computed with that form
> — describes their *intended* σ correctly. Their *applied* σ is ~14× lower
> (measured 0.48–0.65) because of the unit bug in their code.
>
> **What is wrong is the causal story.** In their pipeline the actor's `[1,20]`
> output only sets the schedule's **shape**: it is converted to `rho_used`,
> affine-projected onto the budget constraint, then converted back. The applied
> σ *level* is set by the budget, not by where the sigmoid sits. So "the
> operating point sits at 4% of the dial, therefore the actor cannot resolve
> variation" does not follow.
>
> **Direct evidence against it**: their unmodified `test_MNIST.py` (actor 114)
> produced a clearly non-flat schedule, σ 0.584 → 0.477, penalty 1.0157×, under
> the same `[1,20]` action range. Shape expression works.
>
> **What survives and still needs explaining**: our drone agent's schedule is
> flat (1.0000×) while theirs is not. Real difference, cause unknown.
> Candidates: the conversion mismatch changing where the projection lands;
> reward SNR (ΔIoU sd 0.017 vs Δaccuracy sd ~0.003); or our agent correctly
> finding uniform.
>
> The "resolution → reported gain" monotone pattern in §3 mixes intended and
> applied σ across settings and **should not be cited** until recomputed
> consistently. E3 remains worth running, but re-derive its motivation first.

---

## 1. The observation

The source paper's Fig. 2 plots the learned noise level per communication round,
one curve per client privacy budget. Those are **curves**, sloping downward.
Their own text (p. 949):

> "the noise level **decreases** as the training proceeds. This is because the
> magnitude of the gradient at the early stage of training is larger than that at
> the latter stage... Our method has learned to save privacy budget in the early
> stage and use it aggressively in the later stage."

Our agent produced a **flat line** instead:

| arm | σ over rounds | Σσ² penalty |
|---|---|---|
| Uniform | 1.890 flat | 1.0000× |
| **DRL (ours)** | **1.878 → 1.896** | **1.0000×** |
| Exponential | 2.277 → 1.610 | 1.0525× |
| Arithmetic | 2.315 → 1.637 | 1.0521× |

Our DRL is uniform to four decimal places. Note also that the only genuinely
non-uniform arms here are the *fixed baselines*, not the learned policy — the
opposite of the intended design.

## 2. The mechanism

The actor ends in a sigmoid rescaled to `[σmin, σmax]`, which is `[1, 20]` in
**every** setting — theirs and ours. To emit noise level σ the sigmoid must
output `(σ − 1) / 19`. Where the operating point falls inside that range differs
by an order of magnitude:

```
σ_uniform = sqrt(2 · E · T / ρ)          sigmoid output = (σ − 1) / 19
```

| setting | ρ | T | E | σ_uniform | **sigmoid output** | reported gain |
|---|---|---|---|---|---|---|
| MNIST (theirs) | 2 | 20 | 4 | 8.944 | **0.418** | 3.02 pp |
| MNIST (theirs) | 4 | 20 | 4 | 6.325 | **0.280** | 3.02 pp |
| CIFAR-10 (theirs) | 20 | 25 | 4 | 3.162 | **0.114** | 0.15 pp |
| CIFAR-10 (theirs) | 40 | 25 | 4 | 2.236 | **0.065** | 0.15 pp |
| ours — LightCNN | ~32 | 8 | — | 1.681 | **0.036** | none (−0.043) |
| ours — calibrated | 30 | 25 | 2 | 1.826 | **0.044** | not yet run |

Our operating point sits at **3.6–4.4% of the action range**, deep in the
saturated tail of the sigmoid, where the actor cannot resolve variation at all.

Worse, the failure is then *locked in* by the action-mapping module. That module
applies an affine transform to the raw actions to enforce `Σ_t ρ_t = ρ` exactly.
A near-constant raw action, projected onto the budget constraint, yields
precisely the uniform allocation. That is why the measured penalty is exactly
`1.0000×` and not merely close to it — the flatness is produced by the
projection, not learned.

Their source parameters, for the record (paper §IV):

```
MNIST    : rho_k ~ U[2, 4],    B=200, E=4, C=4, sigma in [1,20], T=20, eta=0.1
CIFAR-10 : rho_k ~ U[20, 40],  B=200, E=4, C=6, sigma in [1,20], T=25, eta=0.1
N = 10 clients; policy and critic are 3 fully-connected layers; sigmoid output.
```

Our LightCNN run used **their own CIFAR-10 budget range** — measured ρ per
client ∈ [20.85, 39.63], mean 32.18 — and still landed at 0.036, because our
T=8 rather than 25. So the gap is not a difference in privacy budget; it is a
difference in where `(ρ, T, E)` places σ inside a fixed `[1,20]` action range.

## 3. The pattern this exposes in the source paper

Ordering all three settings by actor resolution:

```
sigmoid resolution   0.42 / 0.28   →   0.11 / 0.065   →   0.044 / 0.036
reported gain        3.02 pp           0.15 pp            none (−0.043)
```

The reported gain is **monotone in how much actionable range the actor had.**
This is a hypothesis about the source result, not a proven claim, and it admits
two readings:

* **Artefact reading.** The gain tracks actor headroom rather than any property
  of allocation. Where the agent has resolution it can move accuracy around;
  where it does not, it emits uniform and ties. Their CIFAR-10 gain of 0.15 pp,
  published without error bars, sits at the low-resolution end.
* **Genuine reading.** Where the agent has room to act it finds a real gain, and
  our configuration simply denied it that room.

Both are consistent with the data available. **E3 distinguishes them**, by giving
our agent MNIST-equivalent resolution and seeing what it does.

## 4. Why this must be fixed regardless of outcome

As the thesis currently stands, §5.1.6 reports a flat learned schedule and a
−0.043 deficit against uniform, and attributes the deficit to the agent learning
a non-uniform schedule. That attribution is **false** — the schedule is flat and
its variance penalty is 1.0000×, so Theorem 3.1 predicts DRL ≈ uniform and
cannot explain any deficit. (Separately, at n=8 with every CI crossing zero, that
table is noise.)

More seriously, a panelist or reviewer can object:

> Your agent emitted a flat schedule. You therefore tested a saturated network,
> not adaptive allocation. The negative result does not follow.

That objection is correct and currently unanswerable. Closing it is
confound-removal, and it is required whichever way the result falls.

## 5. The fix

Rescale the action range so the operating point sits where theirs did:

| range | uniform σ = 1.826 sits at |
|---|---|
| `[1, 20]` (current) | 0.043 |
| `[1, 8]` | 0.118 |
| **`[1, 4]`** | **0.275** |
| `[1, 3]` | 0.413 |

`[1, 4]` lands inside the authors' MNIST band (0.28–0.42) and is the primary
choice for E3; `[1, 3]` is the alternative. This changes only the resolution the
actor has over its action space — it does not change the privacy accounting, the
budget, or the set of schedules that are reachable in principle.

## 6. Consequences for the three outcomes

| E3 outcome | meaning |
|---|---|
| still flat at `[1,4]` | saturation was not the cause; the agent genuinely finds nothing. **Strongest** form of the negative result, because the confound is excluded. |
| curved, still loses | Theorem 3.1 confirmed against a competent agent rather than a saturated one; also explains their curve without their gain being real. |
| curved, and wins | the optimum is non-uniform; cross-check the direction against E1's `a_t` profile before believing it. |

Direction consistency across experiments (all three measure the same thing):

```
E1: a_t increasing   ⇔   E2: back-clean > front-clean   ⇔   E3: learned σ decreasing
```

Their Fig. 2 shows decreasing σ, i.e. increasing `a_t`. If two of the three
disagree, debug rather than write up.

## 7. Reproduce

Learned schedules and variance penalties:

```python
import json, numpy as np
d = json.load(open('exp_data/lightcnn_pub_fig23_data.json'))
for k in ['Uniform','DRL','Exponential','Arithmetic']:
    a = np.array(d['all_sigmas'][k], dtype=float)   # trials x clients x rounds
    print(k, np.round(a.mean((0,1)), 3), (a**2).sum(-1).mean())
```

Source parameters: `mutool draw -F txt MARL.pdf | sed -n '905,925p'`
Their Fig. 2 description: same command, lines 949–962.

Sigmoid placement: `(sqrt(2*E*T/rho) - 1) / 19`.

## 8. Status

Not yet in the thesis. Do not write it up until E3 returns — the mechanism is
established (the arithmetic above is not in doubt), but the *consequence*, i.e.
whether the agent behaves differently once given resolution, is exactly what is
unmeasured.
