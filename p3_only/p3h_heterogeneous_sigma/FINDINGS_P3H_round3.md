# P3-H round 3 — results (2026-10-01)

Pre-registration: `PREREG_round3.md` (written before any round-3 run). Verbatim
analysis: `results/round3_analysis.txt` (`code/analyse_round3.py`). 240 runs,
RTX 5090, m=3 (30%), 3 trials, 15 rounds, same seeds and split as rounds 1-2.

## Verdicts (stated before interpretation)

| # | Prediction | Result | Numbers |
|---|---|---|---|
| P9 | cosine detector catches alpha-MPELM (TPR>=0.70, FPR<=0.10) | **FAILED** | TPR 0.13-0.14, FPR 0.04-0.06 |
| P10 | cosine detector halves alpha-MPELM damage | **HELD** | -0.043 -> **+0.009 / +0.009 / +0.012** |
| P11 | cosine detector costs <= 0.010 with no attack | HELD | -0.003 / -0.000 / +0.005 |
| P12 | cos+Norm-sn no worse than best norm defence vs band attacker | **FAILED** | see note |
| P13 | median costs >= 0.050 with no attack (P3 replication) | HELD | -0.112 / -0.106 / -0.106 |
| R3-5 | Mix discrepancy | **LEAK** | accuracy-detector honest FPR 0.18 on the positional split (0.00 on session-disjoint) |

**P12 note.** It fails only in the hetero setting, and only because Cosine+Norm-sn
scored ABOVE the clean baseline (+0.0089), while the rule as written compared
absolute magnitudes (|+0.0089| > 0.0018 + 0.005). That `abs()` was a specification
error in the pre-registration — it penalises improvement. The verdict stays FAILED
as registered; the substance is that cos_sn did not regress (uniform +0.0064,
adaptive -0.0034, both pass).

## How the cosine detector "stops" alpha-MPELM — deterrence, not detection (figD4)

* Round 1: honest cosine with the others' mean = 0.128, attackers ~0.000 -> all 3
  caught. alpha-MPELM's own episodic rule then cuts gamma by 0.6x (0.5 -> 0.3).
* With gamma < 0.5 the update is no longer flipped (adv = (1-2 gamma) delta + ...):
  the attacker sends a scaled-down HONEST direction. Occasional later catches keep
  gamma at 0.2-0.4 for the whole run; undefended it reaches 1.0 by round 5.
* Hence TPR is low (P9 failed) but the damage vanishes (P10 held).
* **The limit:** the honest cosine decays 0.128 (r1) -> 0.008 (r5) -> 0.0003 (r13)
  as training converges; after ~round 5 honest and malicious are indistinguishable.
  The defence works only because the attacker (a) attacks from round 1, while the
  direction signal is strong, and (b) backs off when caught. **A flip attacker that
  does not back off, or that starts after ~round 5, is untested and would likely
  evade it.** State this in the thesis; do not claim alpha-MPELM is "solved".

## Robust aggregators under DP (figD3)

| | no attack | RMD | alpha-MPELM | band attacker |
|---|---|---|---|---|
| Median | **-0.11** | -0.12 | **-0.18 to -0.19** | **-0.25 to -0.27** |
| Trimmed mean | -0.002 | -0.009 | -0.049 (worse than undefended -0.043) | -0.046 |
| Krum | -0.08 / -0.01 / +0.02 | -0.09 / -0.06 / -0.01 | -0.04 / -0.01 / +0.02 | same |

(Krum columns: uniform / hetero / adaptive.) Median is harmful in every case —
worse than no defence even under attack — replicating P3's finding (P13). Krum
rejects ~90% of clients by construction; it gets better as sigma spreads because it
then tends to pick the LEAST-NOISY (lowest-sigma) client's update. Descriptive only.

## Arm G clean-window length (R3-4)

Under hetero, the per-client history detector needs only **2 clean rounds**:
onset 2 / 3 / 5 / 8 all give TPR 1.00, FPR 0.00 vs RMD. vs the band-exploiting
attacker (onset 5): TPR 0.90, FPR 0.00.

## Damage vs sigma spread (R3-3, figH5)

Band attacker, undefended: -0.167 / -0.167 / -0.168 / -0.168 / -0.169 from 0% to
25% sigma spread — flat. Defended (Norm or Norm-sn): -0.003 to -0.011 throughout.
Confirms P2/P8: sigma heterogeneity changes detection, not damage.

## Mix discrepancy resolved (R3-5) — and the accuracy-level caveat with it

On the original positional split, with the current trunk: accuracy-detector honest
FPR 0.18 (Mix active FPR 0.30), and **report-set IoU.25 = 0.659** vs the original
P3's 0.6485. So:
1. the original 19-28% honest rejections and the "Mix neutralises alpha-MPELM"
   result came from the positional-split leak (pre-registered rule: FPR >= 0.10);
2. the ~0.47 vs ~0.65 accuracy gap is the REPORT SET (which frames are scored),
   not the rebuilt trunk. The rebuilt trunk matches the original.
Both earlier caveats about the rebuilt trunk can be withdrawn.

## Not pre-registered — do not claim

* Several defended configs score slightly above clean (+0.004 to +0.012). Within
  trial noise (ranges ~+-0.01); possibly selection of less-noisy updates. Not tested.
