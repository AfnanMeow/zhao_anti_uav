# P3-H round 2 — results (run 2026-09-30 -> 2026-10-01)

Pre-registration: `PREREG_round2.md` (written before any round-2 run; addendum
before any group-4 result). Analysis output verbatim: `results/round2_analysis.txt`
(`code/analyse_round2.py`). Figures + table: `figures/` (`code/make_defence_figures.py`).
Machine: RTX 5090, 540 W software cap. m = 3 of 10 clients (30%), 3 trials each,
15 rounds, seeds 770000+t, session-disjoint split. 150 runs + the 21 decisive ones.

## Verdicts (stated before interpretation)

| # | Prediction | Result | Numbers |
|---|---|---|---|
| P4 | alpha-MPELM invisible to Norm and Norm-sn (hetero, adaptive) | **HELD** | TPR 0.11-0.21 |
| P5a | matched lie (0.9x) defeats Norm-sn, adaptive, no cap | **HELD** | TPR 1.00 -> 0.24 |
| P5b | tracking liar defeats Norm-sn AND cap restores >= +0.30 | **FAILED** | 0.15 -> 0.24 with cap (+0.09) |
| P6 | per-client history (Arm G) works under hetero | **HELD** | TPR 1.00, FPR 0.00 |
| P7 | Arm G degrades under adaptive | **HELD** | TPR 0.66, FPR **0.71** |
| P8 | band-exploiting attacker does more damage under hetero (Norm-raw) | **FAILED** | diff +0.0084 (hetero LESS damaged) |
| P8b | Norm-sn removes the extra damage | HELD (vacuous: there was none) | diff +0.0006 |

**P8 failed, so per the pre-registration the damage claim is dropped for good.**
Together with decisive-subset P2, heterogeneous sigma has now twice failed to
increase damage. The result is a DETECTION result.

## The defence picture (figD1, table `figures/defence_summary.csv`)

IoU.25 change vs the clean run of the same sigma setting and trial (mean of 3):

| attack | defence | uniform | hetero | adaptive |
|---|---|---|---|---|
| RMD | none | -0.006 | -0.008 | -0.008 |
| RMD | Norm | +0.004 (TPR 1.00) | -0.005 (TPR **0.11**, FPR 0.19) | -0.007 (TPR **0.09**) |
| RMD | Norm-sn | +0.004 (1.00) | +0.005 (**1.00**, FPR 0.01) | +0.007 (**1.00**, FPR 0.02) |
| alpha-MPELM | none | -0.042 | -0.043 | -0.043 |
| alpha-MPELM | best defence | -0.006 (Mix) | -0.011 (Mix-sn) | -0.021 (Norm-sn) |
| band attacker | none | **-0.167** | **-0.169** | **-0.168** |
| band attacker | any norm defence | -0.002 to -0.013 | | |

* **Norm-sn is the only defence that works against RMD in every sigma setting**
  (TPR 1.00, FPR <= 0.02). Raw Norm collapses the moment sigma varies across clients.
* **Controlled ladder (figH3a): raw Norm fails already at 3.2% sigma spread**
  (TPR 0.15); Norm-sn holds to 25% (0.98). Honest-norm rsd tracks sigma spread
  at ~0.93x across the whole ladder: the band is set by the noise.
* **The band-exploiting attacker is the most damaging (-0.17, ~35% of accuracy)
  and every norm defence contains it** (to -0.002..-0.013), in all sigma settings.
  That is why P8 failed: the norm defence caps its inflation even when the band is wide.
* **alpha-MPELM is not stopped by anything** (TPR <= 0.24 for every detector); defences
  only reduce its damage partly. It carries damage in direction, which no norm sees.

## The accuracy detector is effectively silent — a change from the original P3

Alone, it fires at TPR <= 0.08 and FPR 0.00 in every Mix run, so **Mix ~= Norm**.
The original P3 had it rejecting 19-28% of honest clients and reported that "Mix
neutralises alpha-MPELM". Neither reproduces. NEXT_EXPERIMENTS §4 predicted the old
numbers came from the positional val split leaking the report set into the detector
set; this is consistent with that, but it is **confounded** with the rebuilt trunk
(IoU ~0.47 vs ~0.65 changes what a fixed beta2=0.02 drop means). Do not cite the
original Mix-vs-MPELM result without that caveat.

## The argument this supports (figH1, H2, H3)

1. Per-client budgets (even uniform over rounds, the literature's baseline) widen
   the honest norm band ~100x and break cross-client Norm detection.
2. Under constant per-client sigma, comparing each client to **its own history**
   restores detection perfectly (1.00 / 0.00) with no trust in the client (P6).
3. **Adaptive allocation destroys that option** — 0.66 caught, 71% of honest
   clients wrongly rejected (P7) — because sigma moves every round.
4. The remaining fix, Norm-sn, needs the client's **declared** sigma. An attacker
   that lies by the matched ratio drops it to 0.24 (P5a), and capping declared sigma
   to the honest range does nothing (0.9x lies inside that range; F1 capped = uncapped).
5. A lying band-attacker (F2) gets through Norm-sn (TPR 0.15), and the cap recovers
   only +0.09 (P5b failed: the adaptive honest range, sigma 1.26-2.80, is too wide to bind).

**One line:** adaptive budget allocation removes the trust-free defence and forces
the server to trust client-reported noise levels, which an attacker can exploit.
**Limit of the claim:** in this system that exploit costs detection, not accuracy —
the norm defences still bound the damage (F2 damage -0.014 vs -0.013 honest-declaring).

## Not pre-registered — do not claim

* alpha-MPELM under Norm-raw looks worse in hetero (-0.030) than uniform (-0.011).
  Post-hoc, n=3, trial ranges overlap.
* m = 1, 2 not run; robust aggregators not re-run.

## Caveats

Same as FINDINGS_P3H_decisive.md: rebuilt trunk (absolute IoU ~0.47), 3-session
val set, n=3. The "adaptive" arm is a per-client exponential schedule standing in
for a learned policy (§2b C3); a converged DRL agent's schedule is flat.
