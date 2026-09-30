# Technical Review Report
## Adaptive Privacy Budget Allocation in Differentially Private Federated Learning

**Project:** Anti-UAV FL + DP + DRL thesis (BRAC University, T25301039)
**Review period:** 2026-09-20 — 2026-09-21
**Subject:** Thesis manuscript, experimental record, and the source method's
released implementation (Chen et al., ICC 2024)
**Status:** Central thesis claim **requires re-scoping**; source paper's headline
result **reproduced**; a new and stronger result identified

---

## 1. Executive summary

This review was triggered by two independent external critiques of the thesis
manuscript. It expanded into a full audit of the experimental record and, at the
reviewers' prompting, into direct replication of the source method.

**Four outcomes, in order of consequence:**

1. **The thesis's central claim is refuted in the high-noise regime.** Uniform
   privacy-budget allocation is *not* universally optimal. At the source paper's
   operating point the learned allocator beats uniform by **+2.27pp
   (95% CI [+0.94, +3.59], 9/10 wins, p = 0.004)**, and the correlation between
   accumulated noise variance and accuracy is **+1.000** — exactly inverted from
   the −0.767 the thesis reports. Theorem 3.1 is not wrong; its scope was
   overstated.

2. **The source paper's headline result reproduces.** Their claimed MNIST gain
   of +3.02pp was measured here at +2.27pp with a confidence interval. Earlier
   characterisations of that result as "noise" were wrong and have been retracted.

3. **Their released code does not reproduce their own paper.** A unit
   inconsistency in the action-mapping constraint means the stated noise range
   σ ∈ [1,20] is never enforced; applied σ is ≈14× smaller. Run as shipped, all
   four allocation arms lie within 0.15pp and nothing is measurable.

4. **A defensible negative result survives, and it is about MARL rather than
   allocation.** The learned policy is statistically indistinguishable from a
   one-line fixed exponential schedule (**+0.29pp, p = 0.557**). The advantage
   over uniform comes from the schedule being *decreasing*, not from its being
   *learned*.

**Five errors were found in the thesis manuscript** (§4.2), of which two were
introduced during this review cycle and self-corrected. All are documented.

---

## 2. Scope and method

### In scope
- Thesis manuscript (LaTeX source, all chapters)
- Experimental record: 20 result files under `exp_data/`
- Source method implementation: `Author/Code/{train,test}_{MNIST,CIFAR10}.py`
- Project's own DRL implementations and the two-stage frozen-trunk pipeline

### Method
Every quantitative claim in the manuscript was traced to its underlying data
file and recomputed. Claims that could not be traced were flagged rather than
accepted. Where a disputed mechanism could be tested directly, it was tested
rather than argued.

### Verification standard applied
A claim was accepted only if it could be reproduced from stored data by an
independent computation. Three claims failed this standard and were withdrawn
(§4.2, items 1, 3, 5).

---

## 3. Workflow

| Phase | Activity | Outcome |
|---|---|---|
| 1 | Rewrite manuscript from `thesis_rewrite.tex` into the BRACU template | 6 chapters, 37pp, 0 LaTeX errors, 13 figures installed |
| 2 | Triage two external reviews | 7 of 8 substantive criticisms upheld |
| 3 | Verify every numeric claim against source data | 3 claims withdrawn, 2 corrected |
| 4 | Specify follow-up experiments E1–E4 with pre-registered decision rules | `NEXT_EXPERIMENTS.md` |
| 5 | Audit the source method's released code | 2 implementation defects identified |
| 6 | Replicate the source method (n=1, then n=10 paired) | Central claim refuted; source claim reproduced |

### Notable process decisions

- **Pre-registration.** Decision rules for every experiment were fixed in
  writing before results were seen, to prevent post-hoc interpretation.
- **Refusal to fabricate.** A request to "make the DRL win" was met by
  identifying three legitimate routes rather than by reinstating a known-invalid
  evaluation. The win was subsequently obtained legitimately (§5.3).
- **Self-correction.** Two findings produced during this review were later
  falsified by further evidence and were retracted in place rather than quietly
  removed (§4.2, items 3 and 5).

---

## 4. Findings

### 4.1 Severity classification

| Severity | Definition | Count |
|---|---|---|
| **Critical** | Invalidates a published or thesis claim | 3 |
| **Major** | Materially misleads, or blocks a result | 4 |
| **Minor** | Correctness or presentation defect | 6 |

### 4.2 Critical findings

---

**C-1 — The uniform-optimality claim does not hold at high noise**
*Status: confirmed, n=10 paired, p = 0.004*

The thesis asserts that uniform allocation minimises accumulated noise variance
and that "every non-uniform schedule is strictly worse." Measured at the source
paper's operating point (σ ≈ 6–9):

| arm | Σσ² penalty | accuracy |
|---|---|---|
| Uniform | 1.0000× *(analytic minimum)* | 0.8834 *(worst)* |
| Arithmetic | 1.0442× | 0.8976 |
| Exponential | 1.0450× | 0.9032 |
| DRL | 1.0753× *(worst)* | 0.9061 *(best)* |

`Spearman(accumulated variance, accuracy) = +1.000`

**Root cause.** The proof's load-bearing step is not clipping, as the manuscript
states, but the assumption that every round's noise reaches the final model with
equal weight. The correct objective is `V = Σ a_t σ_t²` with optimum
`ρ_t ∝ √a_t`. At low σ, `a_t` is near-flat and uniform is optimal. At high σ,
late-round noise is disproportionately damaging, `a_t` is steep, and decreasing
schedules win. Both experiments are correct; they sample different regimes.

**Required action.** Restate Theorem 3.1 in weighted form; scope its conclusion
to `a_t ≈ const`, an empirical condition requiring measurement (experiment E1).

---

**C-2 — Source implementation does not run the mechanism its paper specifies**
*Status: confirmed by arithmetic and by measurement*

`Author/Code/test_MNIST.py`, `test_CIFAR10.py`:

```python
rho_used_min = 2 / sigma_max**2 * epoch_local     # 0.0200   <- no /bs
rho_used_max = 2 / sigma_min**2 * epoch_local     # 8.0000   <- no /bs
def sigma2rho(sigma, el, bs): return 2*el / (bs * sigma**2)   # WITH /bs
```

The action-mapping box constraint is expressed in different units from the
conversion it bounds. Reachable values are `[0.0001, 0.04]`; the bounds are
`[0.02, 8.0]`. The upper bound is **200× larger than anything attainable**, so
the `σ_min = 1` floor never binds.

**Measured consequence.** Applied σ is 0.477–0.649 — roughly half their own
stated minimum, ≈14× below specification. Run as shipped, all four arms fall
within 0.15pp and no effect is measurable.

**Which form was intended** is determined by their own code: `rho_used_min/max`
match the **no-`/bs`** conversion exactly at both endpoints, and the `/bs`
version not at all. The project's port, which omits `/bs`, is therefore the
internally consistent one and must not be "corrected."

---

**C-3 — Evaluation design cannot support the published claim**
*Status: confirmed from source*

- `episodes = 1` in both test scripts — a single measurement per arm.
- `agent.load(episode=350)` / `(195)` — a selected checkpoint of ~800.
- *"We test 100 times on the test sets"* re-evaluates **one trained model**,
  measuring test-set sampling noise rather than run-to-run training variance.
- Their loop is **already correctly paired** (shared `net_init` and `rhos_init`
  across arms) — the design controls precisely the variance never measured.

Power analysis against measured run-to-run spread (~3pp at n=1):

| claim | paired trials needed (80% power, α=0.05) |
|---|---|
| CIFAR-10, +0.15pp | **~3,140** |
| MNIST, +3.0pp | ~8 |

The CIFAR-10 claim is unresolvable at any feasible sample size. The MNIST claim
is resolvable, and it **reproduces** (§5.3).

---

### 4.3 Major findings

**M-1 — Manuscript §5.1.6 stated a causal mechanism contradicted by its own data.**
The section attributed a −0.043 deficit to the agent learning a non-uniform
schedule. The learned schedule was flat, variance penalty **1.0000×**. The
theorem predicts parity, and cannot explain any deficit. *Introduced during this
review; retracted.*

**M-2 — Per-layer allocation claim is mathematically wrong.**
The "2.37× worse" figure compares a split scheme against a looser clip. Under a
fair normalisation `ΣC_g² = C²`, Cauchy–Schwarz runs the other way:
`V_split = (Σ C_g √D_g)²/2ρ ≤ C²D/2ρ`. Splitting is not worse. *Withdraw the
figure.*

**M-3 — "Do not use Adam with DP-SGD" is incorrect as a privacy statement.**
Adam post-processes already-privatised gradients; the guarantee and the
accountant are unaffected. Measured utility was *better* under Adam (0.320 vs
0.247 at σ=8). The real finding is that σ becomes a weak lever in that regime —
an experimental-design pitfall, not a privacy failure. *Remove the
recommendation.*

**M-4 — P3 detector results rest on a leaked split.**
`p3_byzantine.py` splits validation positionally ("first 300 / last 1200") while
`stage1_clear.py` assembles val grouped by recording session. With ~1,500 frames
over ~6 sessions the cut falls inside a session, so the server's detector-tuning
set shares sessions with the reported metric set. This is the third instance of
a leak already fixed twice. It plausibly explains the anomalies: detectors
*raising* no-attack accuracy while rejecting 19–28% of honest clients, and Krum
under α-MPELM (0.676) exceeding both the clean baseline (0.649) and Krum without
attack (0.639). **No detector-based P3 number is safe to publish until re-split.**

---

### 4.4 Minor findings

| ID | Finding | Action |
|---|---|---|
| m-1 | "no cross-client structure" is false — aggregation weights are cross-client structure; inverse-variance weighting captures `(1/N²)(Σρ)(Σ1/ρ)`, ~3.5% variance | Correct §3.2.3 |
| m-2 | CRN s.d. reduction stated as 3.2×; actual is **2.7×** (0.0466/0.0171). The ~150-trial equivalence is unaffected (variance ratio 7.4×) | Correct |
| m-3 | §3.1 places DP noise on the batch *average*; standard DP-SGD adds it to the *sum* (σC/B on the mean). As written it reads as a B-fold privacy error | Correct |
| m-4 | ε never reported. At δ=1e-5: ρ=0.5→5.30, ρ=2→11.60, ρ=8→27.19, **ρ=30→67.17**. Headline experiments run at ε≈67 | Report ε from the Opacus RDP accountant, not the ad-hoc ρ expression |
| m-5 | "head" denotes both the 566,789-parameter DP block and the 12,677-parameter layer | Disambiguate |
| m-6 | Thesis title asserted a MARL framework the work disproves; `exp_data/plot.py` fabricates baseline curves analytically to land 2.5%/1.8%/2.0% below DRL | Title replaced; fabricated figures excluded |

---

## 5. Experiments executed

### 5.1 Manuscript reconstruction

Rewrote all six chapters from `thesis_rewrite.tex` into the BRACU template.
Verified: **0 LaTeX errors, 0 undefined references, 0 overfull boxes, 37 pages,
13 figures embedded, 26 cross-references and 21 citation keys resolved.**
Replaced an unsupported "replication on MNIST and CIFAR-10" placeholder with an
honest provenance section, since no MNIST data existed and CIFAR-10 was n=1.

### 5.2 Source replication — as shipped (n=1)

| arm | accuracy |
|---|---|
| DRL | 0.9563 |
| Uniform | 0.9550 |
| Exponential | 0.9552 |
| Arithmetic | 0.9565 |

Spread **0.15pp**. Applied σ ≈ 0.5 (see C-2). Nothing separable.

### 5.3 Source replication — unit bug fixed (n=10 paired) ★

Actor 114, MNIST, their code with `/bs` removed so their own σ clamp functions.

| arm | accuracy | s.d. |
|---|---|---|
| **DRL** | **0.9061** | 0.0101 |
| Exponential | 0.9032 | 0.0116 |
| Arithmetic | 0.8976 | 0.0076 |
| **Uniform** | **0.8834** | 0.0185 |

| DRL vs | mean diff | 95% CI | wins | p |
|---|---|---|---|---|
| **Uniform** | **+0.0227** | **[+0.0094, +0.0359]** | **9/10** | **0.004** |
| Exponential | +0.0029 | [−0.0078, +0.0136] | 7/10 | 0.557 |
| Arithmetic | +0.0085 | [−0.0028, +0.0197] | 6/10 | 0.123 |

Spread **4.00pp** — a 27× increase over the shipped configuration. Applied σ
≈ 6–9, within specification. Every non-uniform arm decreases σ across rounds
(DRL 9.24→5.94); uniform is flat at 7.28 and finishes last.

**Caveat:** actor 114 is from an incomplete 544/800 training run, so this is a
*lower bound* on the learned policy's performance.

### 5.4 Re-analysis of existing records (no new compute)

- **Project's own DRL under the source's evaluation protocol.** Across 8 stored
  paired trials, a single-run evaluation would have reported a DRL win **3/8 =
  38%** of the time, the luckiest draw showing **+4.26pp** — larger than the
  authors' published MNIST gain. Truth at n=8: **−0.043**, CI [−0.0997, +0.0136].
- **Matched-variance permutation.** `front-clean` / `back-clean` share an
  identical σ multiset (7.86× penalty) but differ by 0.027 — however at **n=2
  with the two trials disagreeing in sign**. Recorded as hypothesis, not evidence.
- **Cross-client pooling.** Analytic: gain is `(1/N²)(Σρ)(Σ1/ρ)`, ≈3.0–3.5% for
  the budget distributions in use, ~55% only under extreme heterogeneity. Not a
  valid DP mechanism, and self-defeating (consent implies equal budgets, where
  the gain is exactly zero). The legal equivalent is inverse-variance
  aggregation weighting, which captures the identical factor as pure
  post-processing.

---

## 6. Status of thesis claims

| Claim | Status | Action |
|---|---|---|
| Uniform minimises accumulated noise variance | **Stands** (mathematically) | Restate in weighted form |
| Uniform is optimal; all deviation strictly worse | **REFUTED at high σ** | Re-scope to measured regime |
| Σσ² orders all outcomes (Spearman −0.767) | **Stands for the drone task at σ≈1.83** | Qualify explicitly |
| Clipping is what removes the signal variation | **Withdrawn** | Equal weighting is the real assumption |
| Published learned allocators are bounded by uniform | **REFUTED** | Their MNIST result reproduces |
| Learned allocator ≈ trivial fixed schedule | **NEW, supported** (p = 0.557) | Promote to headline |
| Per-layer splitting always worse (2.37×) | **Withdrawn** | Delete |
| No cross-client structure exists | **Withdrawn** | Aggregation weights are structure |
| Do not use Adam with DP-SGD | **Withdrawn** | Reframe as design pitfall |
| DP noise caps Byzantine attacker amplitude | **Unsafe** | Weak attacker; split leak (M-4) |
| Update norm concentrates as 1/√(2D) | **Stands** | Frame as observation, not contribution |

---

## 7. Deliverables

### Documentation
| File | Purpose |
|---|---|
| `FINDINGS_regime_dependence.md` | **C-1.** The refutation, reconciliation, and what survives |
| `FINDINGS_authors_code_bugs.md` | **C-2, C-3.** Implementation defects, with proofs |
| `FINDINGS_actor_saturation.md` | Superseded hypothesis, retained with correction header |
| `FINDINGS_cross_client.md` | m-1. Pooling analysis and the legal alternative |
| `NEXT_EXPERIMENTS.md` | E1–E4 protocols with pre-registered decision rules |
| `CLAUDE.md` | Session handoff and index |
| `REVIEW_REPORT.md` | This document |

### Code
| File | Purpose |
|---|---|
| `Author/Code/test_MNIST_consistent.py` | Their code, unit bug fixed |
| `Author/Code/test_MNIST_consistent_paired.py` | ...plus n-trial paired statistics |
| `author_test_MNIST_paired.py` / `author_test_CIFAR10_paired.py` | Their unmodified code, trial count configurable |
| `e1_weight_profile.py` | E1 harness: measures `a_t` by single-round noise pulses |

### Data
`exp_data_MNIST_consistent_paired.json` (10 trials, curves, σ schedules),
`MNIST_consistent_paired_stats.json` (per-trial values, CIs, p-values),
`exp_data_MNIST_consistent.json`, `exp_data_MNIST.json`.

---

## 8. Outstanding work

| Priority | Item | Cost | Blocker |
|---|---|---|---|
| **P0** | **E1** — measure `a_t` across σ. Now the central experiment: produces the curve unifying both regimes | ~4 h | Stage-1 cache (`python stage1_clear.py`) |
| **P0** | Re-run §5.3 against the **final** actor (114 is undertrained) | ~1 h | Training at 600/800 |
| P1 | Restate Theorem 3.1 in weighted form; re-scope §5.1 | — | E1 result |
| P1 | P3 session-disjoint re-split, then re-run the grid | ~2 h | — |
| P1 | Apply the 11 manuscript corrections (§4.2–4.4) | ~3 h | — |
| P2 | E4 on CIFAR-10 — bound only, not a verdict (needs ~3,140 trials) | ~6 h | Trained actor |
| P2 | Strong Byzantine attacks (ALIE, IPM, min-max, Fang) | — | P3 re-split |
| P3 | E2, E3 — reconsider; E3's premise weakened by C-2 | — | — |

### Operational note
The 5090 training host is running ~4× slower than the 4090 it replaced
(12 h to reach 600/800 vs 6 h for the full run). Probable cause: torch built
without `sm_120` (Blackwell) kernels, forcing PTX JIT fallback. Diagnostic:

```
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(), torch.cuda.get_arch_list())"
```

`sm_120` absent from the arch list confirms it; remedy is torch ≥2.7 / CUDA 12.8.

---

## 9. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Panel challenges the now-refuted uniform-optimality claim | **High** | Re-scope before submission; the regime-dependence framing is stronger |
| P3 results withdrawn under scrutiny of the split | **High** | Re-split and re-run before any security claim |
| §5.3 result attributed to an undertrained actor | Medium | Re-run against final actor |
| Over-claiming against the source authors | Medium | Report documented defects only; make no claim of intent |
| Single-run results resurfacing in the manuscript | Medium | Five such results have failed replication in this project; require paired n≥10 |

---

## 10. Conclusion

The review found the manuscript's central claim to be **correct in the regime it
was measured and incorrect as stated in general**. The source paper's result,
previously characterised as irreproducible noise, **reproduces** once its own
implementation defect is repaired.

The resulting position is stronger than either prior claim. Whether adaptive
privacy-budget allocation helps is decided by where the operating point sits on
the noise curve — and the source paper and this thesis were each measuring one
side of that curve without knowing the other existed. The remaining negative
result, that a learned multi-agent policy is statistically indistinguishable
from a one-line fixed schedule, is narrower, better supported, and survives the
regime problem intact.

**Recommended framing:** lead with the regime dependence, support it with the
`a_t` measurement from E1, and retain the MARL-versus-fixed-schedule comparison
as the negative result. Withdraw the universal optimality claim.

---

*Prepared 2026-09-21. All quantitative claims in this report are reproducible
from the data files listed in §7; reproduction commands are given in the
corresponding findings documents.*
