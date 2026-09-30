# Fresh beginning — refocusing the thesis on Byzantine resilience

**2026-09-25.** Proposal from a teammate, assessed against what the data
actually supports. This is a **plan, not a result**. Nothing here has been run
yet. Read §3 before writing a single sentence of the new abstract.

---

## 1. The proposal (translated from the original Banglish message)

> Let's change the thesis. We have two parts: MARL + Byzantine resilience.
>
> - In the MARL part we **claim nothing** — not that DRL is better, not that
>   anything else is better.
> - We put **full focus on Byzantine resilience**, because everything in that
>   part lines up.
> - MARL stays as it is.
> - The abstract changes slightly, something like: *"Adaptive budget allocation
>   opens the door to model poisoning, as malicious updates can hide in the
>   threshold limit. And we are solving that."*
> - That makes a less bullshit paper.
> - No more fighting over DRL vs uniform. Whatever results we have, stay.
> - The thesis title stays the same.

## 2. Verdict

**The direction is right, and it's a better thesis than the current one.** But
the sentence the whole reframe depends on is **untested**. As written, the
abstract would claim something our experiments never measured.

What's good about it:

- **It drops the weakest argument.** The DRL-vs-uniform result is
  checkpoint-dependent: actor 114 beats uniform (+2.27pp, p=0.004) and actor
  612 from the same run loses (−1.12pp). No claim either way is defensible
  (`FINDINGS_checkpoint_dependence.md`). Reporting that as an observation,
  without taking a side, is honest and safe.
- **P3 is the cleanest work in the project.** It reproduces the source paper
  exactly (Norm detector: RMD TPR=1.00, α-MPELM TPR=0.14), then shows the
  paper's premise flips at realistic model size (`FINDINGS_P3.md`).
- **It gives the two halves a reason to be in one thesis.** Right now they are
  two projects stapled together. "Adaptive allocation creates a poisoning
  channel" connects them.

## 2a. Framing: one thesis, one variable, two consequences

*Added 2026-09-27. This is the answer to "how do we stop this looking like two
projects stapled together?" — and it is structural, not cosmetic.*

### The unifying object

Both halves of the thesis are projections of the same quantity. Every DP
federated update is

```
Δ_i,t  =  clip_C(g_i,t)  +  N(0, σ_i,t² C² I_D)
```

Take two different views of that one noise term:

| View | Quantity | Gives you | Whose question |
|---|---|---|---|
| **Sum over rounds `t`** | accumulated variance `Σ_t σ_t²` | **utility** cost of allocation | the allocation literature's |
| **Spread across clients `i` in one round** | `‖Δ_i‖ ∝ σ_i`, concentrating at `1/√(2D)` | **detectability** of a malicious update | nobody's, until now |

Same noise. Sum it over time and you get the accuracy consequence. Spread it
across clients and you get the security consequence. **The literature has only
ever taken the first projection.**

### The claim, in one sentence

> **Privacy budget allocation is a security parameter, not just a utility knob.**

That sentence covers both halves, so neither is a subplot.

### The arc

1. The literature proposes adaptive per-client privacy budget allocation and
   claims accuracy gains (Chen et al., ICC 2024).
2. **We test that claim on their own code and their own benchmark: it does not
   hold.** The agent trained to convergence emits a flat schedule (actor 668,
   800 episodes, Σσ² penalty 1.0000×) — it *is* uniform allocation. The published
   gain is checkpoint-dependent (`FINDINGS_checkpoint_dependence.md`).
3. **Nobody asked what adaptive allocation costs.** We do: per-client σ
   heterogeneity widens the honest update-norm band, which is precisely an
   attacker's hiding room (§4).
4. We close it (σ-normalised detection, §5) and show honestly where the fix
   breaks (σ-lying attacker, Arm F).
5. **Uniform allocation dominates on both axes.** The recommendation is not
   "tune the allocator better" — it is "do not adaptively allocate: it gains
   nothing and costs security."

### Why the DRL null result is load-bearing, not an apology

This is the part to internalise before defending it. Step 2 is **half of a
cost–benefit argument**, not a failed chapter.

You need *no benefit* for *unjustified cost* to land. If adaptive allocation
genuinely bought +3pp accuracy, the security cost in step 3 would be a trade-off
worth debating, and the paper would be weaker for it. Because it buys nothing,
the security cost is pure loss and step 5 is decisive.

So the six training runs that produced no win are not wasted work being quietly
dropped — they are the evidence for the first half of the thesis's conclusion.
Present them that way: a null **with a mechanism** (the agent converges to the
analytic optimum in the low-σ regime), not an unexplained loss.

### The structural move that actually fixes the "stapled" problem

The `1/√(2D)` norm-concentration result currently lives in the P3 results
(`FINDINGS_P3.md`). **Move it into the theory chapter, beside the `Σσ_t²`
result.** Derive `Δ` once, then take both projections in the same chapter:

* **Ch 3 (theory)** — one derivation of the DP update; then §3.x accumulated
  variance (utility) and §3.y norm concentration (security), from that same
  equation.
* **Ch 5 (results)** — §5.1 allocation: a null with a mechanism. §5.2 security:
  the channel, the defence, its limit.
* **Ch 6 (discussion)** — uniform dominates both axes.

Two results derived from one equation in one chapter reads as a single thesis.
The same two results in two unconnected chapters reads as two papers. That is
the whole difference, and it costs nothing but section ordering.

### Title, revisited under this framing

See §7 for the candidate list. Under this framing two more become available, and
they describe the contribution rather than only the attack:

* *Privacy Budget Allocation as a Security Parameter in Byzantine-Resilient
  Federated Learning*
* *The Unpriced Cost of Adaptive Privacy Budget Allocation in Federated
  Anti-UAV Detection*

The first is closer to the contribution; the second is likelier to be read.

### Caveat

Steps 3 and 4 are **predictions** until P3-H runs. This framing is safe to build
the thesis *structure* around now — chapter order, which result sits where — but
the abstract cannot assert the channel until predictions 1 and 2 hold (§6).

## 2b. Corrections to the mechanism, verified from code and stored data (2026-09-27)

*Several statements in §2a and §4 were too strong. Verified against the source
and against stored results; these supersede them.*

### C1. The drone DRL test DID use heterogeneous per-client budgets

`lightcnn_publication_plots.py:180`:

```python
rhos = [float(np.random.uniform(RHO_MIN, RHO_MAX)) for _ in range(N)]   # 20.0, 40.0
```

Per-client budgets, drawn independently — the same design as the authors, only a
different range. **So heterogeneous budgets are not why the drone DRL lost.** The
reason is the one already recorded: its schedule is uniform's (σ range
1.669–2.320 vs uniform's 1.681–2.318, penalty 1.0000×) and the test ran arms
back-to-back without CRN and never seeded torch, so the measured difference was
RNG noise.

### C2. Cross-client σ spread is set by the BUDGETS, not by the allocation policy

Measured from `exp_data/lightcnn_pub_fig23_data.json` (8 trials, 5 clients):

| Arm | cross-client relative sd of σ |
|---|---|
| DRL | **8.58%** |
| Uniform | **8.58%** |
| Exponential | **8.58%** |
| Arithmetic | **8.58%** |

**Identical**, because every arm receives the same per-client `rhos` draw. This is
the temporal-vs-cross-client distinction appearing directly in data:

> **Round-to-round allocation is a utility question. Client-to-client dispersion
> is a security question.** The allocation policy controls the first and cannot
> touch the second.

Put that sentence in Chapter 3 verbatim. It has now decided three separate
framing questions.

### C3. Therefore the claim wording must change

**Too strong (as written in §2a step 3 and §4):** "adaptive allocation opens the
poisoning channel."

**Correct:** *per-client budget heterogeneity* opens it — and that is present in
the **baseline** of the whole personalised-DP literature, not only under adaptive
allocation. Even the uniform arm has 8.58% spread (`test_MNIST.py` builds Uniform
as `client.rho_total / CR_Total` — uniform **over rounds**, per client, *not*
uniform across clients).

What adaptive allocation adds on top:

1. the spread becomes **non-stationary** (σ_i,t moves every round), defeating
   per-client historical calibration — a converged agent gives static spread,
   which the server *can* calibrate against;
2. the allocation depends on **client-reported** state, which is the σ-lying
   channel (Arm F).

This makes the finding **bigger**, not smaller: the exposure is in the literature's
default setting.

### C4. Quantified prediction, from our own configuration

The honest norm is noise-dominated — the measured 0.10% relative sd matches
`1/sqrt(2D) = 0.094%` at D=566,789 exactly, which *is* the evidence that noise
dominates the norm. Hence norm ∝ σ, and:

| Setting | Honest-norm relative sd |
|---|---|
| P3 as run (σ identical) | **0.094%** — measured |
| Allocation experiments (σ spread 8.58%) | **~8.6%** — predicted |

**~90x wider band.** This is no longer a hand-wave: the input (8.58%) is measured
from our own stored runs.

### C5. Two existing P3 findings are CONDITIONAL on homogeneous σ

P3-H may revise, not merely extend, these:

* "Norm detector catches RMD at TPR 1.00" — that holds against a 0.094% band.
* "Architecture is inherently resilient, worst case -0.017 IoU.25" — that ceiling
  came from the tight band capping attack amplitude.

Better to discover this ourselves than in review.

### C6. Where σ = 1.83 comes from, and why the two pipelines differ

```
sigma = sqrt(2*el / rho_per_round),  rho_per_round = rho_total / T
rho_total = 30, T = 25, el = 2  ->  sqrt(4/1.2) = 1.8257  ->  1.83
```

`p3_byzantine.py:52` documents it: "uniform at rho=30, the canonical operating
point". **ρ=30 is the midpoint of the drone budget range [20,40]** — i.e. the
*average client's* budget. So P3 did not ignore heterogeneity carelessly; it took
the mean of the heterogeneous range and held it fixed, a legitimate simplification
for a first study. Frame it in the thesis as progression, not error correction:
**P3 characterised the homogeneous case; P3-H relaxes it.**

The two pipelines use different constants (allocation: T=8, el=7, N=5; gate/P3:
T=25, el=2) but both land σ in 1.7–2.3, so the operating points are comparable.

### C7. Missing arm A′, and the ladder must be centred on 1.83

| Arm | per-client ρ | over rounds | represents |
|---|---|---|---|
| A | identical (σ=1.83) | constant | P3 as run — the floor, **keep permanently as control** |
| **A′** | **heterogeneous** | **constant** | **the literature's real baseline — was MISSING** |
| B–F | heterogeneous | adaptive | what the paper proposes |

Without A′ you cannot separate "heterogeneous budgets are risky" from "adaptive
allocation is risky" — different claims, different targets.

**Centre the σ ladder on 1.83**, i.e. draw ρ around 30 rather than adopting a new
range, so mean σ is held constant and only the *spread* varies. Otherwise two
things change at once and the band widening cannot be attributed to spread.

### C8. Framings tested and rejected

* **"Uniform σ solves poisoning."** No — α-MPELM moves update *direction*, not
  magnitude, and is caught at TPR 0.14 even under uniform σ. Uniform σ closes the
  **magnitude-hiding channel** only; direction attacks still need Mix. Scope the
  claim or a reviewer will quote our own table back.
* **"Just impose uniform σ per client."** Not free: either clients breach their
  own budgets (not permitted) or everyone is throttled to the strictest budget
  (utility cost). Heterogeneous budgets are the premise of personalised DP, not an
  implementation choice.
* **"Adaptive allocation closes the gap by converging to uniform."** No —
  convergence is over *rounds*; the gap is *across clients*. C2 shows DRL and
  Uniform have identical 8.58% cross-client spread. Convergence removes the
  time-variation only, making the band stationary and calibratable, not narrower.
  It also hands the authors their win back and yields the absurd recommendation
  "run a DDPG agent for 800 episodes to obtain uniform allocation" — to which the
  answer is: then use uniform.

### C9. There is NO CIFAR evidence

`test_CIFAR10_paired.py` was written but **never run**. The convergence-to-uniform
result rests on **MNIST (actor 668) and the drone task** — two settings. Do not
write "MNIST, CIFAR and drone".

## 3. The gap: P3 never tested adaptive allocation

`p3_byzantine.py:52`:

```python
SIGMA = 1.83    # uniform at rho=30, the canonical operating point
```

**Every client, every round, same σ.** So what P3 established is the
*opposite condition* from the proposed abstract:

| | Tested? | Result |
|---|---|---|
| Uniform σ: can attackers hide in the norm band? | **Yes** (P3) | **No.** Honest norm rel. sd = 0.10%; the band is too tight. Worst undefended damage −0.017 IoU.25. |
| Adaptive σ: can attackers hide in the norm band? | **No** | **Unknown.** This is the claim the new abstract makes. |

Until the second row is measured, "adaptive allocation opens the door to model
poisoning" is a hypothesis.

## 4. Why the hypothesis is probably true

### 4.1 Mechanism

A client's DP update is its clipped gradient signal plus Gaussian noise. At our
scale the noise dominates the norm:

```
noise norm / signal norm  ≈  σ·√D / B  =  1.83 × 753 / 128  ≈  10.8
```

So an honest update's norm is **approximately proportional to σ**.

- **Uniform:** every client has the same σ → honest norms are nearly identical
  → tight band (0.10%) → a poisoned update sticks out.
- **Heterogeneous σ:** honest norms spread by roughly the same factor as σ.
  With σ spanning [1, 20], the band is up to **~20× wide**. A poisoned update
  can sit inside it.

Analogy: a guard weighs bags. If every honest bag weighs 10 kg, a 13 kg bag is
caught. If honest bags range from 1 to 20 kg, a 13 kg bag walks through.

### 4.2 The subtlety that makes or breaks the claim

**What opens the channel is σ that varies *across clients within a round*,
not σ that varies across rounds.**

The server compares clients **against each other in the same round**. If
adaptive allocation only changes σ from round to round, but every client
follows the same schedule, all honest norms in a given round still match. The
band stays tight, and a per-round median-based threshold adapts automatically.

So the abstract must say *per-client* or *heterogeneous* allocation, not just
"adaptive." Fortunately the source paper's own setup is heterogeneous: each
client gets its own random total budget (`Author/Code/train_MNIST.py:537`,
`rho_total = np.random.uniform(low=rho_min, high=rho_max)`) and its own agent
picking σ. That's the exact condition that opens the channel.

### 4.3 Why it should also increase damage

P3 found the architecture "inherently resilient" **because** the tight norm band
caps how large an attacker's update can be. Widen the band and that cap
loosens: the attacker can scale up its update and still pass. So the prediction
is not only lower detection but **more damage** when undefended. This matters.
Without it, we'd be "solving" a problem that only costs 2.7%.

## 4.4 Framing: the claim covers MARL but does not depend on any trained agent

**Thesis sentence:**

> Adaptive per-client privacy budget allocation, such as the MARL scheme of
> Chen et al., opens a model-poisoning channel: malicious updates can hide
> within the spread of honest noise levels.

Wording matters. It's the *attacker* that hides, inside the spread of honest
noise levels. Not "the noise hides."

**No existing DRL agent can drive P3.** The drone agent lives in
`fl_damoyolo_final.py` (DAMO-YOLO pipeline). P3 runs on the frozen-trunk
pipeline (`gate_DB_sweep.py`, D = 566,789), which has no agent. The MNIST actor
was trained on a different model and state space, so it can't be used either.
An agent-driven arm would mean porting DDPG into the P3 pipeline and training
it fresh: ~1 day of code plus an estimated 7–13 h on the 5090.

**It isn't needed.** In both the authors' code and ours, most of the per-client
σ difference comes from the **random per-client total budget**, not the agent
(`Author/Code/train_MNIST.py:537`, `fl_damoyolo_final.py:957`: `rho_total =
np.random.uniform(rho_min, rho_max)`). Actor 612 learned a near-flat schedule
and still produces different σ per client. P3-H's random-budget arm reproduces
that condition directly.

**Keeping the claim agent-free is safer.** DRL outcomes are checkpoint-dependent
(`FINDINGS_checkpoint_dependence.md`). A claim resting on one trained agent
would inherit that, and could be dismissed as checkpoint-picking, which is the
exact criticism made of the source paper.

**MARL makes the fix harder.** Each client's own agent picks its σ from local
state the server can't see, so the server can't verify a client's σ. It has to
trust what the client reports. That makes **Arm F (attacker lies about its σ)
the main threat model**, not a side test. σ-normalised detection is only sound
if declared σ is bounded or checkable (capped at the allowed range, or checked
against the client's remaining budget).

## 4.5 Parallel track: is there any accuracy gain for allocation? (E5)

**Archived 2026-09-27** — E5 now lives in
`NEXT_EXPERIMENTS_superseded_DRL.md`, not in the active runbook. A fixed-schedule
hunt at T=50, ρ ∈ {1, 2}, with CRN. It was to be run *before* any further DRL
agent work; if a schedule beat Uniform, the DRL chase would be back on, aimed at
a gain known to exist.

It is retired because step 2 of §2a is settled: the converged agent emits
uniform, so there is no known gain to aim at. Reopen it only if you decide to
reargue the utility half — and note that under §2a's framing you do **not** need
a utility gain for the thesis to work; you need its *absence*.

It fits this plan. Any schedule it finds varies σ across **rounds**,
identically for all clients, which per §4.2 does not open the poisoning
channel. The combined thesis sentence would be: *round-varying allocation can
help accuracy; client-heterogeneous allocation opens a poisoning channel.*

## 5. The fix: σ-normalised norm detection

Divide each client's update norm by its σ before thresholding. The noise-driven
spread cancels and the band shrinks back toward 0.10%.

**Honest limitations. State these, don't hide them:**

1. **The server has to know each client's σ.** That's fine if the server
   assigns budgets. If clients choose their own σ (the MARL setup), the server
   has to trust what they report.
2. **A malicious client can lie about its σ.** It could claim a large σ so its
   inflated update normalises to look honest. Test this as its own arm, and
   bound it. For example, the server can cap declared σ at the allowed range,
   or check that a declared σ matches the client's remaining budget.
3. **Normalisation doesn't help against α-MPELM.** That attack moves update
   *direction*, not magnitude. Norm detection already only catches it at
   TPR=0.14 under uniform σ. The Mix detector is still needed for it.

## 5a. Arm G — per-client historical baseline detection (added 2026-09-28)

**Why this arm exists: it is the control that justifies §5.** A reviewer will ask
why we built a σ-normalised detector when the server could simply compare each
client against its own past updates. Without a measured answer, §5 looks like it
solves a problem that has an easier solution sitting next to it.

### The idea

Do not compare clients to one another at all. Track each client's own update norm
across rounds and flag deviations from **its own** history.

### Why it should work under uniform allocation — and work very well

Under uniform allocation a client's σ is constant across rounds, so its update
norm is constant, and *within* one client the norm concentrates at the same
`1/sqrt(2D) = 0.094%` as the homogeneous case. A per-client baseline is therefore
**as sharp as the homogeneous-σ detector** while tolerating arbitrary budget
heterogeneity. Using the §2b C4 numbers: client A (ρ=20) sends 1782 ± 1.7 every
round, so a 1700 update sits ~48 sd out.

**Consequence: uniform allocation with heterogeneous budgets is not inherently
vulnerable.** The wide band only bites a detector that insists on comparing
clients to each other — which is exactly what `p3_byzantine.py` does today
(`BETA1 = 3.0`, MADs from the cross-client median).

### Why it should fail under adaptive allocation

The agent gives each client a different σ every round, so there is no stable
per-client norm to learn. The baseline never settles and the defence is
unavailable. This is the "non-stationary" half of §2b C3, made measurable.

### Predictions (pre-registered, alongside §6)

6. **Arm G under A′** (heterogeneous budgets, uniform over rounds): RMD detection
   approaches the homogeneous-σ value (~1.00) despite the wide cross-client band.
7. **Arm G under adaptive allocation:** detection degrades substantially relative
   to prediction 6, because no stable baseline exists.

If 6 and 7 both hold, the argument becomes: *uniform allocation admits a
trust-free defence; adaptive allocation destroys it, forcing the server onto
σ-normalisation — which requires trusting a client-reported value and is
therefore attackable (Arm F).* Considerably stronger than σ-normalisation alone.

### Honest limitations — state them, do not bury them

1. **It needs a clean window.** The baseline must form before the attacker acts;
   an attacker poisoning from round 1 has no history to violate. Report how many
   rounds the baseline needs to become useful.
2. **It does not help against α-MPELM**, which preserves magnitude and moves
   direction. No norm-based method does.
3. **It is not in the current code.** `p3_byzantine.py` thresholds across clients,
   not within one. Arm G is a new rule, not a new experiment.

### Cost: zero GPU

Arm G consumes the same per-client-per-round norms that §2b already mandates. It
is a post-processing pass over the logged JSON. **Another reason not to skip the
§2b logging** — without it Arm G is impossible and would need a full re-run.

## 6. The experiment (P3-H: heterogeneous σ)

This is a targeted subset of P3, not a new project. The existing uniform-σ P3
runs become the **control arm**.

**Setup:** same as `p3_byzantine.py` (10 clients, D=566,789, B=128, session-
disjoint `fl_workdirs/clear`), except per-client σ is heterogeneous.

- σ assignment: draw each client's ρ like the authors do (uniform over
  [ρ_min, ρ_max]) and convert to σ. Also run a deterministic spread
  (e.g. σ ∈ {1, 2, 4, 8, 16}) so the band width is controlled.
- Attacks: RMD, α-MPELM.
- Malicious fraction m: 10 / 20 / 30%.
- Trials: 3 (matching P3).

**Arms:**

| Arm | σ | Detector | Purpose |
|---|---|---|---|
| A — control (**numbers exist; must be re-run for figures**) | uniform 1.83 | Norm / Accuracy / Mix | the floor |
| **A′** | **heterogeneous, uniform over rounds** | Norm (raw) / none | **the literature's real baseline — use this in the decisive subset** |
| B | heterogeneous, adaptive | none | undefended damage |
| C | heterogeneous, adaptive | Norm (raw) | does the channel open? |
| D | heterogeneous, adaptive | **Norm, σ-normalised** | does the fix close it? |
| E | heterogeneous, adaptive | Mix | direction attack still covered? |
| F | heterogeneous, attacker **lies** about σ | Norm, σ-normalised | the defence's limit |
| **G** | A′ **and** adaptive | **per-client historical baseline** | **zero GPU (post-processing); the control that justifies D — see §5a** |

**Arm A needs a re-run, verified 2026-09-27.** Its *numbers* exist
(`exp_data/p3_byzantine.json`), but that file records only
`{acc, gamma, tp, fp, detected}` per round and stores σ as a single scalar
(`"sigma": 1.83`). There are **no per-client update norms in any existing
output**, so the uniform-σ control panel of figH1 and the honest baseline of
figH2/figH3 cannot be produced from what we have. Re-run the uniform-σ
configuration with the per-client logging specified in §6a — it is a small
subset of the original 33 configs, and it also finally gives the 0.10%
concentration result a figure.

**Predictions, written down before running:**

1. **Arm C vs A:** Norm detector TPR against RMD drops well below its uniform-σ
   value of 1.00.
2. **Arm B vs A:** undefended damage is larger than the uniform-σ worst case
   of −0.017 IoU.25.
3. **Arm D:** σ-normalisation restores RMD detection to near 1.00.
4. **α-MPELM:** normalisation does not rescue it (TPR stays low under C and D);
   Mix remains the defence.
5. **Arm F:** a σ-lying attacker defeats naive normalisation unless declared σ
   is bounded.

**If prediction 1 or 2 fails, the reframe fails.** That would mean the channel
doesn't open in practice (for example, local-training signal dominates the norm
more than §4.1 assumes). Run this *before* rewriting the abstract.

**Also fix first:** the P3 detector/report split is currently positional and
cuts inside a session (`p3_byzantine.py:36-37,56`, `NEXT_EXPERIMENTS.md` §4).
Make it session-aware before P3-H, so both arms share a clean split.

## 6a. Figures — what exists, what is missing, what P3-H must produce

### What the four existing P3 figures show

All from the **uniform-σ control** (σ = 1.83 every client, every round). PDFs in
`baseline_uniform_sigma/figures/`. Note the `.png` copies in `plots/` are
git-LFS pointers, not images — regenerate from the PDFs or re-run
`make_p3_figures.py`.

| Figure | Content | Corresponds to |
|---|---|---|
| **figP1** | Final IoU.25 per defence, per attack, at m = 30% | their Fig 9 |
| **figP2** | TPR vs malicious ratio (left), honest-client FPR vs ratio (right) | their Figs 5–7 |
| **figP3** | Degree of poisoning γ over rounds — α-MPELM adapting to the defence | their Fig 8 |
| **figP4** | Accuracy trajectories with/without a robust aggregator, attack vs none | their Fig 10 |

### The gap that matters: the headline result has no figure

**P3's strongest finding is unplotted.** `FINDINGS_P3.md` reports that the honest
update-norm relative standard deviation is **0.10%**, a consequence of
`1/√(2D)` concentration at D = 566,789 — and that this is *why* the source
paper's "hide in the noise magnitude" premise fails at scale. There is currently
no figure for it anywhere in the thesis.

That is also precisely the figure the new claim needs, because the new claim is
that heterogeneous σ *widens* that band. So one figure does double duty: it
visualises the existing contribution and carries the new one.

### Figures P3-H must produce

Output to `p3h_heterogeneous_sigma/figures/`, PDF **and** PNG (PNG written as a
real file, not LFS).

**figH1 — the norm band (THE figure).** Two panels side by side, honest update
norms as a distribution (histogram or strip plot), with the detector threshold
drawn as a line and malicious updates overlaid as distinct marks.

* Left: **uniform σ** — band is a spike (0.10% rel sd), malicious marks fall
  clearly outside the threshold. Detection is trivial.
* Right: **heterogeneous σ** — band is broad, malicious marks fall *inside* it.

If the claim is true, this single figure proves it to a reader in one glance,
and if it is false the figure shows that too. **Design the experiment so this
figure is producible**, i.e. log per-client update norms every round, not just
aggregate TPR.

**figH2 — norm vs σ scatter, and why normalisation works.** Per-client update
norm on y, that client's σ on x, one point per client per round. Honest clients
should lie on a line (norm ∝ σ once noise dominates); malicious clients sit off
it. Second panel: the same data after dividing by σ — the line collapses to a
point cloud and the malicious points become outliers again.

This is the mechanism figure. It justifies σ-normalised detection visually
rather than by assertion, and it is nearly free given figH1's logging.

**figH3 — band width vs σ spread.** x: σ spread (σ_max/σ_min, or sd(σ)/mean(σ)).
y: honest-norm relative sd. Overlay the predicted relation (relative sd of norms
≈ relative sd of σ in the noise-dominated regime) against the measured points.
Anchor the left end at the measured uniform-σ value of 0.10%.

Purpose: shows the effect is quantitatively understood, and lets a reader read
off how much heterogeneity is needed before detection degrades. This converts
"the band widens" from a claim into a calibrated curve.

**figH4 — detection vs σ spread, per detector, per attack.** TPR on y, σ spread
on x; series for Norm (raw), Norm (σ-normalised), Mix, and **per-client baseline
(Arm G)**; separate panels for RMD and α-MPELM. The heterogeneous analogue of
figP2, with spread on the x-axis instead of malicious fraction.

Expected shape if predictions 1, 3, 4, 6 and 7 hold: raw Norm falls with spread;
σ-normalised Norm stays flat near 1.00 for RMD; **per-client baseline stays flat
under uniform-over-rounds allocation but falls under adaptive**; all stay low for
α-MPELM, where only Mix works.

**figH5 — damage vs σ spread.** Final IoU.25 drop on y, σ spread on x,
undefended vs each defence. Establishes that the channel has an accuracy
consequence and not merely a detection-metric one. Anchor against the uniform-σ
worst case of −0.017 IoU.25 (prediction 2 says heterogeneous is worse).

**figH6 — the σ-lying attack and its bound.** TPR under σ-normalised detection
on y, against the attacker's overstatement ratio (declared σ / true σ) on x.
Two series: naive normalisation, and normalisation with declared σ capped to the
allowed range or cross-checked against remaining budget.

Purpose: shows the defence's limit *and* the fix. This is the figure that keeps
the paper honest — it is the one a reviewer will look for, and having it
pre-empts the obvious attack on the contribution.

### Priority

figH1 and figH2 are mandatory — without them there is no visual evidence for
the central claim. figH4 and figH6 are needed for the defence contribution.
figH3 and figH5 are strongly desirable and come almost free from the same run,
provided per-client per-round norms and σ are logged.

**Logging requirement, decided now:** P3-H must record, for every trial, round
and client: the update norm, that client's σ, its declared σ (where it differs),
whether it is malicious, and the detector's decision. Every figure above is then
a post-processing step over one JSON, and no figure requires re-running the
experiment. The original `p3_byzantine.py` logged aggregates only, which is why
figH1 cannot be produced from existing data.

## 7. What changes in the thesis

### Claims to drop

- "Uniform allocation is optimal" as a general statement. It's refuted at high σ
  (`FINDINGS_regime_dependence.md`).
- "DRL beats uniform" or "DRL loses to uniform". The comparison isn't
  identified (`FINDINGS_checkpoint_dependence.md`).

### Results that stay, as observations rather than claims

- Regime dependence: at low σ (drone, 1.83) uniform wins; at high σ (MNIST ~7)
  decaying schedules win.
- Checkpoint dependence: learned schedules are unstable across checkpoints of a
  single run. This now **supports** the security story. Learned per-client
  allocation is exactly the heterogeneous-σ condition, and it's unpredictable.
- The authors' `/bs` unit bug (`FINDINGS_authors_code_bugs.md`).
- A fixed exponential schedule was the best and most stable allocation tested.

### Title — cannot stay the same

The current title is *"The Optimality of Uniform Privacy Budget Allocation in
Byzantine-Resilient Federated Learning for Anti-UAV Detection"*. It makes
exactly the allocation claim the proposal drops. Candidates:

- *Heterogeneous Privacy Budget Allocation Opens a Model-Poisoning Channel in
  Federated Anti-UAV Detection*
- *Hiding in the Noise Band: Byzantine Risks of Adaptive Differential Privacy
  in Federated Anti-UAV Detection*

Choose after P3-H. If prediction 3 holds, add "…and How to Close It".

### New abstract skeleton (only after P3-H confirms)

> Adaptive, per-client differential-privacy budget allocation has been proposed
> to improve utility in federated learning. We show it has a security cost. At
> realistic model sizes, DP noise makes honest update norms highly predictable
> under uniform allocation (relative sd 0.10%), which caps attack amplitude and
> lets norm-based detection catch magnitude attacks reliably. Heterogeneous
> allocation widens that band by the ratio of client noise levels, so poisoned
> updates can hide within it. [P3-H numbers.] We propose σ-normalised
> detection, which restores [number], and characterise its limits against
> clients that misreport their noise level. We also find that the utility gains
> attributed to learned allocation are checkpoint-dependent and reproduced by a
> fixed decaying schedule.

## 8. Supervisor — do this before rewriting anything

He said: *"make DRL win as you claimed in the earlier version, or the panel
will defer."* This plan deliberately stops trying to make DRL win. Don't let
the defence be the first time he hears that.

Pitch it as a stronger story, not a retreat:

1. We found a security flaw in adaptive allocation (the P3-H mechanism).
2. We have half the evidence already (uniform-σ P3), and one targeted
   experiment completes it.
3. The DRL result is reported honestly as checkpoint-dependent, and that
   instability is itself part of the security argument.

If he insists on a DRL win regardless, that's a different conversation.
Resolve it before investing in the rewrite.

## 9. Risks

| Risk | Consequence | Mitigation |
|---|---|---|
| Norm band doesn't widen enough (signal dominates more than §4.1 assumes) | Core claim fails | Run P3-H first; the deterministic σ-spread arm tests the mechanism directly |
| Damage doesn't increase under heterogeneous σ | "Solving" a harmless problem | Prediction 2 is pre-registered; if it fails, reframe as a detection result only |
| σ-lying attacker beats normalisation | Fix looks weak | Report it; propose bounded declared σ; this is an honest open problem |
| Supervisor rejects the reframe | Wasted rewrite | §8 before any LaTeX changes |

## 10. Order of work

MNIST and P3 are **independent** and run on different machines, so do them
**in parallel**. MNIST is not a prerequisite for P3-H: under this plan it only
supports the "learned allocation is unstable" observation, which actors 114
and 612 already show.

**Critical path for P3 (~1 day):**

1. Talk to the supervisor (§8). This gates everything, and it isn't a compute job.
2. Fix the P3 positional split, and write P3-H (code only, ~half a day).
3. On the 5090 or lab card: timing pilot → P3 control re-run → P3-H (~3–5 h).
4. Only if predictions 1–2 hold: rewrite the title, abstract and chapters.

**In parallel, off the critical path:** MNIST keeps running on the 1650 Super
(~33 h left from episode 212). Don't move it to the 5090 while P3 needs that
GPU (see `COSTS.md`).
