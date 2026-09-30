# Next experiments: P3-H (heterogeneous sigma) — attack, defence, and its limit

Rewritten 2026-09-27. The DRL/allocation experiments E1-E5 that used to fill this
file are archived verbatim in **`NEXT_EXPERIMENTS_superseded_DRL.md`** — nothing
was deleted, but do not run them without a deliberate decision to reopen that
track.

Rationale and thesis consequences: **`p3_only/fresh_beginning.md`** (that is the
plan; this file is the runbook). Costs: `p3_only/COSTS.md`.

**Re-apply the power cap first: `nvidia-smi -pl 300` in an Administrator
PowerShell.** It is lost on every reboot and the machine hard-powers-off
without it (see `CLAUDE.md`).

**P3 cannot run on the 1650 Super** (~13 GB feature cache vs 4 GB VRAM). Use the
RTX 5090 or the lab RTX PRO 6000.

---

## 0. Why these, and what is at stake

The DRL track is closed. The final MNIST run completed 800 episodes and its best
actor (668) emits a **flat** schedule, Sigma^2 penalty **1.0000x** — the converged
agent *is* uniform allocation. Six training runs, no win. Because a flat schedule
is the same allocation as uniform, no number of extra trials can change that
verdict, so chasing it further spends GPU time for a known answer.

The thesis therefore refocuses on the Byzantine half, which has four findings
that stand on their own data (`FINDINGS_P3.md`): the source paper's detection
claim reproduced exactly (Norm catches RMD at TPR 1.00, alpha-MPELM at 0.14), the
premise inverting at scale (honest norm relative sd 0.10% from `1/sqrt(2D)`), the
architecture being inherently resilient (worst case -0.017 IoU.25), and classical
robust aggregators being counterproductive under DP (median costs -0.208 with no
attacker).

**The one thing that is claimed but not tested.** The bridge claim — that
adaptive budget allocation opens a model-poisoning channel because malicious
updates can hide inside the honest norm band — has never been measured. All of
P3 ran a single global sigma (`p3_byzantine.py:52`, `SIGMA = 1.83`; the saved
JSON stores `"sigma": 1.83` as a scalar). Under one global sigma the honest band
is a 0.10% spike, so there is *no* room to hide; that is why Norm detection hits
TPR 1.00.

The channel, if it exists, comes from **cross-client sigma spread within a
round**, not from a schedule that varies over rounds. The server compares clients
against each other inside a round, so if every client shares that round's sigma
they all look alike regardless of how sigma moves over time. Per-client budget
heterogeneity is what widens the band — and the authors' own design draws
`rho_total ~ U(2.0, 4.0)` independently per client, so the adaptive setting
manufactures exactly that spread.

P3-H tests it, defends it, and then breaks the defence honestly.

### Run order

**Fix the split -> re-run uniform-sigma control with logging -> P3-H arms B-F ->
figures.** The split fix (§4) comes first so both arms share a clean split. The
uniform-sigma control is re-run second, because its *numbers* exist but its
per-client norms do not (see §2b).

---

## 0b. STEP ZERO — build the Stage-1 cache (nothing runs without it)

`fl_workdirs/stage1/cut_s3/` is **not shipped with the dataset; it is
generated.** If it is missing, nothing is wrong — it has simply not been built
yet. One command does the whole thing:

```
python stage1_clear.py          # GPU, one-time, ~20-40 min
```

It reads only `./3rd_Anti-UAV_train_val/train`, and writes **everything** the
experiments need into `./fl_workdirs/clear`:

```
public_pretrained.pt            frozen trunk (s1,s2,s3) + public head
feat_client_0..4.pt             cached cut_s3 features, 5 clients
tgt_client_0..4.pt              targets
feat_val.pt / tgt_val.pt        held-out val
manifest.json                   val sessions, client sessions, public folders
```

Two things that cost time if you get them wrong:

* **`stage1b_recache.py` is NOT needed.** It belongs to the older `bigdata`
  path. `stage1_clear.py` does its own cut_s3 caching (see its comment
  "cache frozen s1->s3 features (the cut_s3 point)").
* **Paths differ between the two pipelines.** `gate_DB_sweep.make_init` defaults
  to `./fl_workdirs/stage1/public_pretrained.pt`, but the clear-sky run writes
  to `./fl_workdirs/clear/`. Always pass `--data ./fl_workdirs/clear`.
  `e1_weight_profile.py` already defaults to the clear path and resolves the
  trunk beside it.
* `single_test_coco/` is **not** required for P3-H — only
  `lightcnn_publication_plots.py` uses it.

---

## 1. Canonical configuration (do not vary between arms)

From `FINDINGS_gate1.md`, the live regime:

```
frozen public trunk : s1, s2, s3        (cached in ./fl_workdirs/clear by stage1_clear.py)
DP-trained          : k1, k2, head      D = 566,789
init                : k1,k2 from public_pretrained.pt ; head random  (make_init, mode public_tail)
optimiser           : SGD lr=0.003 momentum=0.9        (NEVER Adam)
clipping            : C = 1.0
federated           : 5 clients x 3,808 frames, 25 rounds x 2 local epochs, B=128
score               : mean IoU.25 over rounds 20-25
budget              : rho = 30  ->  uniform sigma = 1.826 every round
```

Reference points: floor (random head) 0.001, trivial predictor 0.021, no-DP
ceiling 0.769, uniform-at-rho-30 ≈ 0.70.

**Pairing is mandatory.** Every arm inside a trial shares one initialisation and
one RNG stream (common random numbers). This is what makes n=20 usable; without
it the paired-difference s.d. is 0.0466 instead of 0.0171 (a 2.7× reduction in
s.d., 7.4× in variance — so ~20 CRN trials carry the precision of ~150 unpaired).
`allocation_bound.py` already implements CRN correctly; copy its seeding.

---

---

## 2. P3-H — the experiment

Authoritative design, with rationale and the five pre-registered predictions:
`p3_only/fresh_beginning.md` §5-§6. Summarised here for running.

Outputs go to `p3_only/p3h_heterogeneous_sigma/results/`, **never `exp_data/`**.
Code goes to `p3_only/p3h_heterogeneous_sigma/code/`.

**Setup:** as `p3_byzantine.py` — 10 clients split from the 5 cached shards,
D = 566,789, B = 128, 15 rounds, 3 trials, session-disjoint `fl_workdirs/clear`,
attacks RMD and alpha-MPELM, malicious fraction m = 10/20/30% — except per-client
sigma is heterogeneous.

Two sigma-assignment modes, run both:

* **Realistic:** draw each client's rho as the authors do, uniform over
  `[rho_min, rho_max]`, and convert to sigma. This is the setting the claim is
  about.
* **Controlled:** a deterministic ladder (e.g. sigma in {1, 2, 4, 8, 16}) so the
  band width is a known independent variable. This is what makes figH3 and
  figH4 interpretable — without it, "spread" is not a controlled quantity.

### Arms

| Arm | sigma | Detector | Purpose |
|---|---|---|---|
| **A** control | uniform 1.83 | Norm / Accuracy / Mix | baseline band + detection (re-run, see §2b) |
| **B** | heterogeneous | none | undefended damage |
| **C** | heterogeneous | Norm (raw) | does the channel open? |
| **D** | heterogeneous | Norm, **sigma-normalised** | does the fix close it? |
| **E** | heterogeneous | Mix | direction-based attack still covered? |
| **F** | heterogeneous, attacker **lies** about sigma | Norm, sigma-normalised | the defence's limit |

**Arm A is kept permanently as the control, not replaced.** The frozen snapshot
in `p3_only/baseline_uniform_sigma/` stays untouched, and the uniform-sigma
configuration appears as the reference panel or leftmost point in figH1, figH3,
figH4 and figH5. The uniform-vs-heterogeneous contrast *is* the result.

### Decision rule (fixed before looking)

* **Prediction 1 — Arm C vs A:** Norm TPR against RMD drops well below 1.00.
* **Prediction 2 — Arm B vs A:** undefended damage exceeds the uniform-sigma
  worst case of -0.017 IoU.25.
* **Prediction 3 — Arm D:** sigma-normalisation restores RMD detection toward 1.00.
* **Prediction 4 — alpha-MPELM:** normalisation does not rescue it under C or D
  (it moves direction, not magnitude); Mix remains the defence.
* **Prediction 5 — Arm F:** a sigma-lying attacker defeats naive normalisation
  unless declared sigma is bounded.

**If prediction 1 or 2 fails, the reframe fails** — the channel does not open in
practice, and the abstract must not assert it. Run this *before* rewriting
anything.

## 2b. Logging requirement — decided now, do not skip

`p3_byzantine.py` records only `{acc, gamma, tp, fp, detected}` per round, with
sigma as a single scalar. **There are no per-client update norms in any existing
output**, verified 2026-09-27. Consequence: the uniform-sigma control panel of
figH1 and the honest baselines of figH2/figH3 cannot be built from what we have,
so Arm A needs a re-run even though its accuracy and TPR numbers are known.

P3-H must record, for every **trial x round x client**:

```
norm            L2 norm of that client's update
sigma_true      the sigma it actually used
sigma_declared  what it told the server (differs only in Arm F)
malicious       bool
rejected        the detector's decision
```

With that, all six figures are post-processing over one JSON and no figure ever
requires re-running the experiment. This is the single cheapest insurance in the
whole plan — the original P3 has to be re-run precisely because it was omitted.

## 3. Figures P3-H must produce

Full specification: `p3_only/fresh_beginning.md` §6a. Output PDF **and** a real
PNG (every `.png` in `plots/` is a git-LFS pointer, not an image — do not repeat
that) to `p3_only/p3h_heterogeneous_sigma/figures/`.

| Figure | Content | Priority |
|---|---|---|
| **figH1** | The norm band: uniform sigma (tight spike, malicious outside threshold) vs heterogeneous (broad band, malicious inside). **This is the paper.** | mandatory |
| **figH2** | Per-client norm vs sigma scatter — honest on a line, malicious off it; second panel after dividing by sigma. The mechanism figure. | mandatory |
| **figH3** | Honest-norm band width vs sigma spread, with the predicted relation overlaid, anchored at the measured 0.10%. | high |
| **figH4** | TPR vs sigma spread, per detector (Norm raw / Norm normalised / Mix), per attack. Heterogeneous analogue of figP2. | mandatory |
| **figH5** | Damage (IoU.25 drop) vs sigma spread, undefended vs each defence, anchored at -0.017. | high |
| **figH6** | The sigma-lying attack: TPR under normalised detection vs overstatement ratio, naive vs capped declaration. Shows the limit *and* the fix. | mandatory |

**Note on the existing figure set:** P3's headline result — the 0.10% norm
concentration that inverts the source paper's premise — currently has **no figure
anywhere in the thesis**. figH1 and figH3 fix that while carrying the new claim,
so one figure family does both jobs.

The four existing figures (figP1 defence comparison, figP2 TPR/FPR vs malicious
ratio, figP3 gamma over rounds, figP4 with/without robust aggregator) all remain
valid as the uniform-sigma control and are kept.

## 4. PREREQUISITE — fix the P3 detector/report split before P3-H

**P3 session-disjoint re-split.** `p3_byzantine.py:36-37,56` splits the 1,500
val frames as "first 300 detector / last 1200 reported", a **positional** slice.
`stage1_clear.py:102` assembles val as `for s in val_sess: for f in by_sess[s]`
— grouped by session. With ~1,500 frames over ~6 sessions the cut at frame 300
lands inside the second session, so the server's detector-validation set shares
recording sessions with the reported metric set.

This is the same leak fixed twice in §4.1, and it plausibly explains the
anomalies: Accuracy and Mix raise no-attack accuracy (0.6656 / 0.6655 vs 0.6485
undefended) while rejecting 19–28% of honest clients, which is what selecting on
near-duplicates of the test set looks like. Krum under α-MPELM (0.676) beating
both the clean baseline (0.649) and Krum-without-attack (0.639) is the same
smell.

Fix: choose detector sessions and report sessions disjointly by session id, then
re-run the P3 grid. Until then **no detector-based P3 number is safe to
publish.**

---

## 5. Conventions

* Write results to `p3_only/p3h_heterogeneous_sigma/results/p3h_<arm>.json`.
  **Never to `exp_data/`** — that directory holds the frozen uniform-σ baseline.
* Store **per-trial arrays, never just means.** This project has twice been
  unable to answer a question from stored results: `allocation_bound.json` kept
  only n=2 means, and `p3_byzantine.json` kept only aggregates — which is why the
  uniform-σ control must now be re-run (§2b). Include `seed_base`, the per-round
  trajectory, and the per-client records of §2b.
* Every run must pass through `gate_DB_sweep.client_round` so `gpu_guard.py`
  throttling applies.
* Log each client's realised σ and its spent budget, and confirm no client
  touches the `σ ∈ [1, 20]` clamp — the controlled ladder tops out at 16
  deliberately for that reason. If the clamp binds, the σ spread is no longer the
  independent variable and figH3/figH4 become uninterpretable.
* Record the detector threshold actually used each round, not just the decision.
  figH1 draws that threshold as a line; without it the figure cannot be built.

---

## 6. What to hand back

Enough for a session to resume with no context:

1. The per-arm JSON files, with the full per-trial x round x client records
   specified in §2b (norm, σ_true, σ_declared, malicious, rejected, threshold).
2. Honest-norm relative sd for the uniform-σ control and for each σ spread —
   i.e. the measured version of the 0.10% figure, which is what figH3 plots.
3. TPR/FPR per detector per attack, for arms A and C-F.
4. Which of the five decision-rule branches fired, **stated before any
   interpretation** — in particular whether predictions 1 and 2 held, since the
   reframe depends on them.
5. Undefended damage (Arm B) against the -0.017 IoU.25 uniform-σ reference.
6. For Arm F: the overstatement ratio at which σ-normalised detection fails, and
   whether capping declared σ restores it.
7. The six figures, PDF and real PNG.

Then the thesis edits follow mechanically:

* §3.2 — restate the theorem in weighted form `V = Σ a_t σ_t²`, with uniform as
  the `a_t` = const corollary. Stop attributing the result to clipping. Scope it
  to the regime measured: at σ≈7 on MNIST, decaying schedules beat uniform
  (`FINDINGS_regime_dependence.md`), so assumption (iii) fails at high σ. The
  experiment that would measure `a_t` directly is **E1, archived** in
  `NEXT_EXPERIMENTS_superseded_DRL.md` — reopen it only if a reviewer challenges
  the scoping.
* §3.2.3 — **delete the per-layer 2.37× claim.** Under a fair normalisation
  `Σ C_g² = C²`, Cauchy–Schwarz runs the other way:
  `V_split = (Σ C_g √D_g)² / 2ρ ≤ C² D / 2ρ`. Splitting is not worse; the 2.37×
  compares against a looser clip.
* §3.2.3 `\paragraph{Across clients.}` — **the closing sentence is wrong**:
  *"There is no cross-client structure for a credit-assignment mechanism such as
  a value-decomposition network to discover."* True only of per-round allocation
  of a fixed per-client budget; false in general, because the **aggregation
  weights** are cross-client structure. Uniform FedAvg is suboptimal under
  heterogeneous budgets by exactly `(1/N²)(Σρ_i)(Σ1/ρ_i)`, and inverse-variance
  weighting `a_i ∝ 1/σ_i²` captures it as pure post-processing at zero privacy
  cost. ~3.5% variance under the budget distributions this literature uses
  (below our 0.017 noise floor), up to 55% under genuinely unequal budgets.
  Keep the non-transferability argument — pooling budgets *is* illegal, and
  self-defeating besides, since the equal-budget case consent would produce has
  zero gain. Full derivation, magnitudes and replacement wording:
  **`FINDINGS_cross_client.md`**.
* §4.2 / §6.2 — **remove the Adam recommendation.** Adam post-processes already
  privatised gradients: the guarantee holds and the accountant is right. What we
  found is that σ becomes a weak lever on utility in this regime — an
  experimental-design pitfall, not a privacy failure. Adam scored 0.320 vs SGD
  0.247 at σ=8, i.e. *better* utility at equal privacy.
* §5.1.6 — the DRL section's causal claim is **false and must go**: the learned
  schedule is flat (1.878 → 1.896, `Σσ²` penalty **1.0000×**, identical to
  uniform). The theorem predicts DRL ≈ uniform and DRL ≥ exponential/arithmetic;
  Table 5.3 shows it losing to all three. At n=8 with every CI crossing zero
  that table is noise. **Resolved 2026-09-27:** the MNIST agent trained to
  convergence (800 episodes, actor 668, explore at its 0.01 floor) also emits a
  flat schedule, penalty **1.0000×** — so this is not an artefact of our
  pipeline. The converged agent independently arrives at uniform. Report it that
  way: a null with a *mechanism* (convergence to the analytic optimum in the
  low-σ regime), not an unexplained loss. Per the refocus
  (`p3_only/fresh_beginning.md` §7) all DRL-beats-uniform claims are dropped.
  Do **not** write that the source paper's results are unreliable or fabricated
  — their number reproduces (actor 114 gave +2.27pp vs their published +3.02pp).
  The defensible criticism is of their *protocol*: one checkpoint, `episodes=1`.
* Everywhere — report ε. At δ=1e-5, `ε = ρ + 2√(ρ ln(1/δ))` gives ρ=0.5→5.30,
  ρ=2→11.60, ρ=8→27.19, **ρ=30→67.17**. The headline experiments run at ε≈67,
  which is not a meaningful guarantee, and must be disclosed. State adjacency,
  δ, and whether the reported ε uses this zCDP conversion or the Opacus RDP
  accountant with Poisson subsampling — they differ.
* Fix `Σσ²` s.d. reduction 3.2× → **2.7×** (0.0466/0.0171 = 2.73; the ~150-trial
  equivalence is unaffected, it comes from the 7.4× variance ratio).
* §3.1 — noise is added to the **sum** of clipped gradients (σC, i.e. σC/B on
  the mean), not to the batch average as currently written.
* Terminology — "head" is used for both the 566,789-parameter DP block and the
  12,677-parameter layer. Pick one.
