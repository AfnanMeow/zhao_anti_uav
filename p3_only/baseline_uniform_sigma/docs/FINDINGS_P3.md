# P3 — Byzantine resilience of the DP-FL anti-UAV architecture

Grid: 33 configs x 3 trials, 10 clients x 1,904 frames (so m = 1/2/3 is exactly
the paper's 10/20/30%), sigma=1.83 (uniform at rho=30), 15 rounds. Server
detector val (300 frames) disjoint from the reported metric set (1,200).
Data: `exp_data/p3_byzantine.json`, `exp_data/p3_envelope.json`.

## 1. The attacker's hiding room does not exist here (and why)

`p3_envelope.py` measured the honest update-norm distribution:

| sigma | honest update norm | relative sd |
|---|---|---|
| 0 (no DP) | 0.6917 +- 0.6548 | 95% |
| 1.83 | 2.2487 +- 0.0187 | 0.83% |
| 4.0 | 4.8853 +- 0.0095 | 0.19% |
| 8.0 | 9.7586 +- 0.0100 | **0.10%** |

The paper's premise is that DP noise gives an attacker cover to hide inside. In
high dimensions the opposite holds: the norm of a Gaussian vector concentrates
as 1/sqrt(2D), which at D=566,789 predicts a relative sd of 0.094% — measured
0.102%. **DP noise makes the update norm MORE predictable, not less.**

Note the un-noised norm varies by 95% across clients; it is the DP noise that
imposes the regularity. So the paper's camouflage effect is a **small-D
phenomenon** (their MNIST/CIFAR models have percent-scale 1/sqrt(2D)), and it
disappears at detector scale.

Consequence: a norm detector here is extremely tight, which **caps attack
amplitude**. The only move left to an attacker is to keep the norm and change
the DIRECTION, which a norm detector cannot observe. a-MPELM is therefore
implemented as a norm-preserving direction attack — the strongest available.

## 2. The paper's central claim reproduces exactly

| attack | Norm-detector TPR |
|---|---|
| RMD (conventional) | **1.00** |
| a-MPELM (adaptive) | **0.14** |

The conventional attack is caught every single time; the adaptive one walks
through. This is the paper's headline result, reproduced on a real detector.

`figP3_gamma.png` shows the episodic-memorisation loop working: gamma escalates
to 1.0 when the attacker is never caught, and is driven down to ~0.14 once a
detector starts firing.

## 3. Full results (m = 30%, the hardest ratio tested)

| defence | no attack | RMD | a-MPELM | TPR (MPELM) | FPR on honest |
|---|---|---|---|---|---|
| none | 0.6485 | 0.6410 | **0.6312** | — | 0.00 |
| Norm | 0.6521 | 0.6446 | 0.6428 | 0.14 | 0.10 |
| Accuracy | 0.6656 | 0.6635 | 0.6616 | 0.24 | 0.19 |
| **Mix** | 0.6655 | 0.6583 | **0.6649** | **0.32** | 0.28 |
| Krum | 0.6386 | **0.5759** | 0.6760 | 0.00 | 0.00 |
| median | 0.4402 | 0.4333 | **0.3168** | 0.00 | 0.00 |
| trimmed mean | 0.6441 | 0.6373 | 0.6282 | 0.00 | 0.00 |

Figure: `plots/figP1_defence_comparison.png`.

## 4. Three findings that go beyond the paper

**(a) The architecture is inherently resilient.** The worst any attack achieves
is **-0.017 IoU.25 (-2.7% relative)** — and that is undefended, at 30% malicious
clients. The mechanism is the envelope result: with a ~2% norm band the attacker
cannot amplify, so the strongest legal move is reversing direction at matched
norm, and three reversed updates out of ten merely scale the average down rather
than corrupting it. Resilience here is a *consequence of DP*, not of a defence.

**(b) Mix detection fully neutralises both attacks.** Under Mix, a-MPELM scores
0.6649 against a 0.6485 no-attack baseline — the attack costs nothing. It is the
only defence that holds against both. The price is a **28% false-positive rate**:
it discards more than a quarter of honest clients' contributions. That trade
should be reported alongside detection rate; a defence that throws away a
quarter of legitimate participants is not free.

**(c) Classical robust aggregators are COUNTERPRODUCTIVE under DP.**
Coordinate-wise median costs **-0.208 with no attacker at all** — an order of
magnitude more than any attack in this study. Krum costs -0.073 under RMD. The
reason is structural: averaging is precisely what suppresses DP noise (by
sqrt(N)), and median/Krum discard that averaging, so they forfeit the noise
reduction while also breaking the update's directional structure. In DP-FL the
standard Byzantine toolkit can hurt more than the threat it addresses.

(Krum under a-MPELM scores 0.6760, the highest cell in the table — selecting the
single most central update happens to exclude direction-reversed outliers. But
it is not a reliable defence: the same mechanism loses badly to RMD.)

## 5. What can honestly be claimed

> A differentially-private federated anti-UAV detector that is Byzantine-resilient
> to adaptive, DP-camouflaged model poisoning at up to 30% malicious clients:
> undefended degradation is bounded at 2.7% relative, and Mix (norm+accuracy)
> detection removes it entirely at the cost of a 28% honest-client rejection rate.

Scope honestly: this covers the two attacks implemented (RMD, a-MPELM as a
norm-preserving direction attack) at m <= 30%, on the clear-sky subset, at
sigma=1.83. A stronger adaptive attacker — for instance one that optimises the
direction against the accuracy detector rather than simply reversing — is not
covered and would be the natural next test.

## Figures

* `figP1_defence_comparison` — final IoU.25 per defence per attack (m=30%)
* `figP2_detection` — TPR vs malicious ratio, plus honest-client FPR
* `figP3_gamma` — degree of poisoning over rounds (their Fig 8)
* `figP4_robust_agg` — trajectories with/without Krum and median (their Fig 10)
