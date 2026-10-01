# P3-H round 2 — pre-registration (written 2026-09-30, BEFORE any round-2 run)

Context: the decisive subset (FINDINGS_P3H_decisive.md) gave P1 held, P2 failed,
P3 held. Per fresh_beginning.md §9 the thesis falls back to a DETECTION result.
Round 2 runs the detection arms that do not depend on P2, plus one new, separately
pre-registered damage test. Every threshold below is fixed now and must not be
changed after results are seen. All runs: m=3 (30%), 3 trials, 15 rounds, same
seeds (770000+t) and cache as the decisive subset, results in results/.

## Sigma settings

* uniform  — every client sigma=1.826 (Arm A)
* hetero   — rho_i ~ U(20,40), constant over rounds (Arm A', literature baseline)
* adaptive — same rho_i, spent on a per-client exponential schedule
  rho_it ~ exp(k_i t/(R-1)), k_i ~ U(-1,1): sigma_i,t changes every round, per client

## Group 1 — defence grid (predictions 1, 3, 4 across all sigma settings)

sigma in {uniform, hetero, adaptive} x attack in {rmd, mpelm} x defence in
{none, norm, norm_sn, mix, mix_sn} + clean baseline per setting.
Descriptive, plus prediction 4 (already in §6): **alpha-MPELM Norm-raw and Norm-sn
TPR both <= 0.5 in hetero and adaptive; Mix / Mix-sn remain its only defence.**

## Group 2 — Arm G, per-client historical baseline (predictions 6, 7 of §5a)

Runs: RMD, no active defence, attack onset at round 5 (rounds 0-4 clean), for
uniform, hetero, adaptive. Detector (post-processing, does not affect aggregation):
for client i, base_i = median of its norms in rounds 0-4; rsd = median over clients
of std/mean of their own round 0-4 norms; flag i at round r>=5 if
|norm_i,r / base_i - 1| > 3 * rsd.

* **P6:** hetero, Arm G TPR >= 0.90 and FPR <= 0.10.
* **P7:** adaptive, Arm G degrades vs P6: TPR drops by >= 0.30 OR FPR rises by >= 0.20.

## Group 3 — Arm F, sigma lying (prediction 5)

adaptive, defence norm_sn:
* F1: RMD, malicious declare lie*sigma_true, lie in {0.8, 0.9, 1.1}, cap off/on.
  RMD's norm is 0.90x honest, so lie=0.9 is the matched lie.
* F2: band-exploiting attacker (below) that declares sigma_true * s_t (its own
  inflation factor), cap off/on. Cap = clamp declared sigma to the honest range.

* **P5a:** F1 lie=0.9, no cap: Norm-sn TPR <= 0.50.
* **P5b:** F2 no cap: Norm-sn TPR <= 0.50; F2 with cap: TPR higher than without
  cap by >= 0.30.
  (Expected but NOT a pass criterion: the cap does not rescue lie=0.9, because 0.9x
  lies inside the honest range. Reported either way.)

## Group 4 — NEW damage test, band-exploiting attacker (prediction 8)

Attack `mpelm_scale`: alpha-MPELM direction attack whose norm is multiplied by a
per-attacker factor s_t: s <- min(3, 1.1 s) if not rejected last round, else
s <- max(1, 0.7 s); starts at 1. Under uniform sigma the 0.1% band pins s near 1;
under heterogeneous sigma it can inflate to the band edge undetected.

Runs: {uniform, hetero, adaptive} x defence {none, norm, norm_sn}.
Damage = final IoU.25 (mean of last 3 rounds) minus the clean baseline of the same
sigma setting and trial.

* **P8:** under Norm-raw defence, damage(hetero) - damage(uniform) <= -0.010 in the
  mean AND negative in >= 2 of 3 trials.
* **P8b:** under Norm-sn, |damage(hetero) - damage(uniform)| <= 0.005 in the mean
  (normalisation removes the extra damage).

If P8 fails, the damage claim is dropped for good; no further attacker variants
will be tried in search of one.

## Group 5 — controlled sigma ladder (figH3/figH4, descriptive)

ladder:S, S in {0.05, 0.1, 0.2, 0.4}, RMD, defence norm (Norm-sn is logged as a
shadow). S=0 is the uniform arm. No prediction; checks honest rsd tracks the
sigma spread.

## Not run

m = 1, 2 (m=3 only); robust aggregators (Krum/median/trimmed, P3 already showed
them counterproductive under DP).

## Addendum (2026-09-30, before any group-4 result was inspected)

Group 4 also runs defences mix and mix_sn, so every attack has the full
none / norm / norm_sn / mix / mix_sn comparison for the defence figures. These
extra configs are DESCRIPTIVE only; P8 and P8b are unchanged and use only
norm and norm_sn.
