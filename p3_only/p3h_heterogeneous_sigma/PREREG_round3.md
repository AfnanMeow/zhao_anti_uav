# P3-H round 3 — pre-registration (written 2026-10-01, BEFORE any round-3 run)

Same setup as rounds 1-2 unless stated: 10 clients, m=3 (30%), 3 trials, 15 rounds,
seeds 770000+t, session-disjoint split, sigma settings uniform / hetero / adaptive
as in PREREG_round2.md. Results in results/, never exp_data/. Thresholds are fixed
here and must not change after results are seen.

## R3-1 — cosine (direction) detector against alpha-MPELM

**Detector `cos`.** Each round, for every client i:
c_i = cos(Delta_i, mean_{j != i} Delta_j), computed on all 10 submitted updates
before any rejection. Reject i if c_i < median(c) - 3 * MAD(c) (one-sided: only
updates pointing AWAY from the others are rejected). Same beta=3 as the norm
detectors. `cos_sn` = cos OR norm_sn (the direction and the magnitude detector
together). Per-client c_i and the threshold are logged in EVERY round-3 run.

Rationale: an honest update = shared signal + independent DP noise, so c_i is
slightly positive; alpha-MPELM with gamma > 0.5 flips its own update, so its c_i
should be negative. Whether the shared signal clears the ~1/sqrt(D) = 0.0013 noise
floor of the cosine is exactly what is unknown.

Runs: {uniform, hetero, adaptive} x attack {rmd, mpelm, mpelm_scale} x defence
{cos, cos_sn}, plus clean runs (no attack) with defence cos for its cost.

* **P9:** vs alpha-MPELM, `cos` TPR >= 0.70 and FPR <= 0.10 in all three sigma settings.
* **P10:** vs alpha-MPELM, damage under `cos` is at most half the undefended damage
  (|d_cos| <= 0.5 |d_none|, means, round-2 undefended runs as reference) in all three.
* **P11:** on clean runs, `cos` costs <= 0.010 IoU.25 vs no defence, in all three.
* **P12:** vs the band-exploiting attacker, `cos_sn` damage <= the best single
  round-2 norm defence's damage in magnitude + 0.005, in all three (no regression).

If P9 fails, the thesis states that no tested detector stops alpha-MPELM.

## R3-2 — robust aggregators under heterogeneous sigma

Aggregators replacing the mean: krum (n_mal = m, as P3), coordinate-wise median,
trimmed mean (20%). Runs: {uniform, hetero, adaptive} x attack {none, rmd, mpelm,
mpelm_scale}. For Krum "rejected" = every client not selected.

* **P13 (replication of P3):** with NO attack, median costs >= 0.050 IoU.25 vs the
  mean aggregator, in all three settings.
* Krum and trimmed mean: descriptive.

## R3-3 — damage vs sigma spread (figH5)

ladder:S, S in {0.05, 0.1, 0.2, 0.4}, band-exploiting attacker x {none, norm, norm_sn}
plus a clean run per level. S=0 is the uniform arm from round 2. Descriptive, no
prediction (P2 and P8 already decided the damage question; this is the figure).

## R3-4 — how many clean rounds Arm G needs

hetero, RMD, no active defence, attack onset in {2, 3, 8} (5 exists from round 2);
plus hetero, band-exploiting attacker, onset 5. Same Arm G rule as round 2 (window =
the onset rounds; onset 1 is excluded because a 1-round window has no spread).
Reported: the smallest onset with TPR >= 0.90 and FPR <= 0.10. Descriptive.

## R3-5 — separating the Mix discrepancy

Re-run with the ORIGINAL positional split (first 300 val frames = detector set,
remaining 1,200 = report set) on the current trunk: uniform x {clean, alpha-MPELM}
x defence mix. Statistic: the accuracy detector's own honest-client FPR on clean runs.

* If FPR >= 0.10 (vs 0.00 on the session-disjoint split): the old 19-28% honest
  rejections are attributed to the positional-split leak.
* If FPR < 0.02: the leak is NOT the cause; the rebuilt trunk / lower accuracy level is.
* In between: inconclusive, reported as such.

## Order

R3-1 (cosine) first, then R3-5, R3-2, R3-4, R3-3.
