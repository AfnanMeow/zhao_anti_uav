# P3 — Byzantine resilience (12–14 Sep)

Source: `exploiting_dp.pdf` — Hossain, Badsha, La, Islam, Khalil,
"Exploiting Gaussian Noise Variance for Dynamic Differential Poisoning in
Federated Learning", *IEEE Transactions on Artificial Intelligence* 6(11),
Nov 2025. Extracted text: `exploiting_dp.txt`.

## Core idea

DP's own Gaussian noise gives an attacker cover. A malicious client can hide
adversarial perturbations *inside* the legitimate noise envelope, so anomaly
detectors tuned to accept honest DP updates cannot separate them (their Fig. 2:
a tight detection range R flags honest LDP updates as anomalous; a widened R'
lets adversarial updates through).

## The two attacks to simulate

1. **RMD — Random Malicious Device** (the conventional baseline). A fraction
   m of clients return randomly falsified parameters.
2. **α-MPELM — adaptive model poisoning via episodic loss memorisation**
   (the paper's contribution). The attacker adapts injected noise per round
   using memorised loss history, tuning the perturbation to stay inside the
   detector's accept region while still degrading convergence.

Attack strength swept over m = 10%, 20%, 30% of n clients; paper uses ε=0.7.

## The three detectors to implement (defence side)

* **Norm** detection — flag updates whose norm falls outside an accepted range
  (parameter β1, their Fig. 5)
* **Accuracy** detection — flag by validation-accuracy influence (β2, Fig. 6)
* **Mix** — norm + accuracy combined (γ = degree of influence, Fig. 7)

Baseline detectors come from ref [26] in the paper; the paper's claim is that
α-MPELM deceives all three better than RMD does.

## Fit to our setup

Maps cleanly onto what already exists: 5 session-disjoint clients, FedAvg over
DP-SGD updates, known per-round σ, cached features so a full 25-round run is
~163 s. m=1 of 5 clients = 20%, squarely in the paper's swept range. The DP
machinery, the σ schedule and the honest-update norm statistics are all already
instrumented (`probe_noise.py` measures update norms directly), which is exactly
what a Norm detector needs to be calibrated against.
