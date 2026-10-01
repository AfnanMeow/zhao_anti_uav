# P3-H decisive subset — results (2026-09-30)

Machine: CSE-12B21L-D05, RTX 5090, torch 2.7.1+cu128, driver limit 600 W with a
540 W software cap (`GPU_SOFT_CAP`). Data: `results/p3h_decisive.json` (7 configs x
3 trials, 15 rounds, full per-trial x round x client logs per NEXT_EXPERIMENTS §2b).
Analysis: `code/analyse_decisive.py`. Code: `code/p3h.py`.

## Verdict (pre-registered rule, fresh_beginning.md §6, stated before interpretation)

| Prediction | Result | Evidence |
|---|---|---|
| 1. heterogeneous sigma opens the channel | **HELD** | Norm-raw TPR vs RMD 1.00 -> **0.11** (FPR 0.01 -> 0.19) |
| 2. undefended damage exceeds -0.017 | **FAILED** | hetero -0.0081 (-1.8% rel.), uniform -0.0063 (-1.4% rel.) |
| 3. sigma-normalisation restores detection | **HELD** | Norm-sn TPR **1.00**, FPR 0.01 |

**Prediction 2 failed, so by the pre-registered rule the reframe fails: the full
grid was NOT run and the abstract must not assert the channel causes damage.**
Thresholds used (not fixed numerically in the plan, chosen before the run):
P1 TPR <= 0.5, P3 TPR >= 0.9.

## Numbers

| config | IoU.25 | TPR | FPR | honest-norm rsd | rsd after /sigma |
|---|---|---|---|---|---|
| uniform, no attack | 0.4756 | – | 0.00 | 0.112% | 0.112% |
| hetero, no attack | 0.4769 | – | 0.00 | **10.06%** | 0.158% |
| uniform, RMD, Norm | 0.4796 | 1.00 | 0.01 | 0.108% | |
| hetero, RMD, Norm | 0.4719 | **0.11** | **0.19** | 10.39% | |
| uniform, RMD, undefended | 0.4693 | – | – | 0.109% | |
| hetero, RMD, undefended | 0.4688 | – | – | 10.39% | |
| hetero, RMD, Norm-sn | 0.4816 | **1.00** | 0.01 | 10.39% | 0.154% |

Paired extra damage from heterogeneity (hetero minus uniform, per trial):
-0.0029, 0.0000, -0.0023. Effectively zero.

## What is established

* **The band widens ~90x, as predicted.** Honest-norm rsd 0.11% -> 10.1%
  (§2b C4 predicted ~8.6% from the budget spread; realised sigma spread here is
  ~9-10%). Measured, not extrapolated. The 1/sqrt(2D) result reproduces on this
  cache (0.088-0.112% vs 0.094% predicted).
* **Raw-norm detection breaks under the literature's baseline (Arm A').** Uniform
  over rounds, heterogeneous budgets, no adaptive policy: TPR 1.00 -> 0.11, and the
  detector now rejects honest high-sigma clients (FPR 0.19) more often than attackers.
* **Dividing by sigma fixes it completely**: rsd back to 0.15%, TPR 1.00, FPR 0.01.

## Why prediction 2 failed — read before redesigning anything

The pre-registered test compares damage with NO defence. Undefended FedAvg never
inspects norms, and RMD (fixed gamma=0.5, 0.90x honest norm) does not adapt to the
band, so heterogeneity has no causal path to extra damage in that arm. The result
is a real null for RMD, but it does not test whether an attacker that EXPLOITS the
wide band (e.g. scales its update up to the band edge while passing Norm-raw) can do
more damage. That would be a NEW hypothesis — it must be pre-registered before it is
run, not substituted for prediction 2 after seeing this result.

fresh_beginning.md §9 already lists the fallback for exactly this outcome:
"if [damage] fails, reframe as a detection result only".

## Caveats

* Absolute accuracy (~0.47) is lower than the original P3 (~0.65): the Stage-1
  trunk was rebuilt (seeded) on this machine, and the report set changed with the
  session-disjoint split. Comparisons are paired within this cache.
* The val set is 3 sessions, not 4 — `stage1_clear.py`'s [:1500] truncation drops
  `20190926_102042` entirely (true of the original P3 too). Detector: `20190925_140917`
  (351 frames); report: the other two (1,149).
* n=3 trials. Trial 3 carries most of the damage in both arms (-0.016 / -0.018).
* Wall-clock per 15-round run varied 1.8-30 min with the guard reporting 0 s slept
  in slow trials: the slowness is external (OS / remote-desktop load), not throttling.

## Not run (by the rule above)

Full grid: Arm A with Accuracy/Mix, adaptive arms B-E, sigma-lying Arm F, sigma
ladder, Arm G (needs `--onset` runs), alpha-MPELM, m=1/2. All are implemented in
`code/p3h.py` and can be launched with `code/launch.ps1` if the decision changes.
