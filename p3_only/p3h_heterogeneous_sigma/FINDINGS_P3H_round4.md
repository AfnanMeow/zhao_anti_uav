# P3-H round 4 — can the cosine detector stop a harder direction attacker? (2026-10-01)

Pre-registration: `PREREG_round4.md` (incl. the author's recorded expectations).
Verbatim analysis: `results/round4_analysis.txt` (`code/analyse_round4.py`).
Figure: `figures/figD5_direction_attack_limits`. 36 runs, hetero sigma, m=3, 3 trials.

## Verdicts

| # | Prediction | Result | Numbers | Recorded expectation |
|---|---|---|---|---|
| P14 | cos catches a never-backing-off flip (TPR>=0.70, FPR<=0.10) | **FAILED** | TPR 0.45, FPR 0.01 | fail (correct) |
| P15 | cos halves its damage (onset 0) | **FAILED** | -0.041 -> -0.029 (30% less) | uncertain |
| P16 | cos halves damage of adaptive alpha-MPELM starting round 6 | **HELD (marginal)** | -0.041 -> -0.0198 (bar -0.0205) | fail (wrong) |
| P17 | cos halves damage of never-backing-off flip starting round 6 | **FAILED** | -0.041 -> -0.033 | fail (correct) |

Pre-registered decision: at least one held -> report as a PARTIAL defence with its
conditions. P16's margin is 0.0007 with trial range -0.010 to -0.029: treat as marginal.

## Mechanism

* Cosine catches **all 3 attackers in each of rounds 1, 2, 3 and 5 (2.3 in round 4)**, then
  1.3 / 1.3 / 0.3 / 1.3 and ~0 from round 10 — the shared gradient signal dies as
  training converges (round-3: honest cosine 0.128 -> 0.008 -> 0.0003).
* The final model is still damaged: the attack keeps going in the late rounds,
  where no detector can see direction. Final accuracy is set by those rounds.
* P16 held for the same reason as round 3: adaptive alpha-MPELM is caught in rounds
  6-7 and its back-off rule disarms it. Its best defence was Cosine + Norm-sn:
  -0.041 -> -0.006.

## What the thesis can claim

The cosine detector is a **partial** defence: it detects direction attacks only while
the shared gradient signal is above the DP noise (the first ~5-8 rounds here), and it
neutralises an attacker only if that attacker backs off when caught. Against a
direction attacker that keeps flipping until convergence it cuts damage by about a
fifth to a third (30% from round 1, ~20% from round 6), and nothing tested removes it.
**Direction attacks at convergence under DP are an open problem** — the same 1/sqrt(D)
noise that makes norm detection so sharp (the P3 headline) makes direction
detection blind once the signal fades. That is the honest closing limit of P3-H.
