# P3-H round 4 — pre-registration (written 2026-10-01, BEFORE any round-4 run)

Question: does the cosine detector CATCH a direction attack, or only deter
alpha-MPELM's back-off rule (FINDINGS_P3H_round3.md)? Round 3 showed the honest
cosine signal decays from 0.128 (round 1) to ~0.008 (round 5) to ~0.0003 (round 13).

Setup as rounds 1-3: hetero sigma (per-client budgets, the literature baseline),
10 clients, m=3, 3 trials, 15 rounds, seeds 770000+t, session-disjoint split.
Damage = final IoU.25 (mean of last 3 rounds) minus the clean hetero run of the
same trial. TPR/FPR counted only over rounds in which the attackers are active.

## Attacks

* `mpelm_fixed`, onset 0 — alpha-MPELM direction attack with gamma fixed at 1.0
  (adv = -delta + 0.3 r, rescaled to the honest norm). Never backs off.
* `mpelm`, onset 5 — the paper's adaptive alpha-MPELM, silent for rounds 1-5,
  attacking from round 6 (its gamma has grown to 1.0 by then, since it is never
  caught while silent).
* `mpelm_fixed`, onset 5 — never backs off AND starts late.

Defences: none, norm_sn (reference), cos, cos_sn. 12 configs, 36 runs.

## Predictions (thresholds fixed now)

* **P14:** vs `mpelm_fixed` onset 0, `cos` TPR >= 0.70 and FPR <= 0.10.
* **P15:** vs `mpelm_fixed` onset 0, `cos` damage <= 0.5 x undefended damage (magnitudes).
* **P16:** vs `mpelm` onset 5, `cos` damage <= 0.5 x undefended damage.
* **P17:** vs `mpelm_fixed` onset 5, `cos` damage <= 0.5 x undefended damage.

Author's expectation, recorded so it cannot be adjusted later (not a criterion):
P14 fails, because the signal fades by round 5; P15 is uncertain; P16 and P17 fail.

Decision rule: if P15, P16 and P17 all fail, the thesis states that the cosine
detector only deters an attacker that both starts early and backs off when caught,
and that direction attacks under DP remain an open problem. If any of them hold, it
is reported as a partial defence with exactly the conditions under which it held.
