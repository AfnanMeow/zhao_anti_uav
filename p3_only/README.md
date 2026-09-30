# p3_only — Byzantine-resilience track

Created 2026-09-25. Everything for the P3-focused direction of the thesis lives
here: the plan, what it costs, the existing P3 results frozen as a baseline,
and a place for new results.

Start with **`fresh_beginning.md`**.

## Layout

```
p3_only/
├── README.md                     this file
├── fresh_beginning.md            THE PLAN: refocus on Byzantine resilience (moved here from repo root)
├── COSTS.md                      GPU time estimates, 1650 Super vs RTX 5090
│
├── baseline_uniform_sigma/       SNAPSHOT of the original P3 (13 Sep 2026) — do not edit
│   ├── code/        p3_byzantine.py, p3_envelope.py, make_p3_figures.py
│   ├── data/        p3_byzantine.json, p3_envelope.json
│   ├── figures/     figP1..figP4 (.pdf)
│   ├── docs/        FINDINGS_P3.md, P3_BYZANTINE_NOTES.md
│   └── reference/   exploiting_dp.pdf / .txt  (the source Byzantine paper)
│
└── p3h_heterogeneous_sigma/      NEW EXPERIMENT (not yet written or run)
    ├── code/        P3-H script goes here
    ├── results/     P3-H JSON outputs go here
    └── figures/     P3-H plots go here
```

## baseline_uniform_sigma — frozen snapshot

These are **copies**, byte-checked against the originals on 2026-09-25. The
originals are still at their old paths (`p3_byzantine.py`, `exp_data/…`,
`plots/…`) because other scripts and docs reference them there. Treat this
folder as the record of what P3 was on 13 Sep. Don't edit it.

Configuration: 10 clients, **uniform σ = 1.83** for every client in every
round, 15 rounds, 3 trials, 33 configs. Attacks: RMD, α-MPELM. Detectors:
Norm / Accuracy / Mix. Malicious fraction 10/20/30%.

In the new plan this is the **control arm**: the uniform-σ condition, where
the honest norm band is 0.10% wide and attackers can't hide.

Data checksums (sha256):

```
661f43264e6a07f8ab4fd183b59ff084282e9895bc379d7f71d8d4f374bf7121  data/p3_byzantine.json
b1d82c891c915ba532aa71f0245764b5ad2aaef3df253360a1754947e3f892e9  data/p3_envelope.json
```

### Known issues carried by this snapshot

1. **Positional detector/report split.** `p3_byzantine.py:36-37,56` splits the
   1,500 val frames as "first 300 detector / last 1,200 report". The cut lands
   inside a session. This only affects TPR/FPR calibration, not accuracy. Fix
   it before P3-H, then re-run this arm so the two compare like-for-like.
2. **No `.png` figures.** Every `.png` in the repo's `plots/` folder is a git-LFS
   pointer file (text with a hash), not an image. The PDFs are real, and
   they're what the LaTeX uses. To get PNGs, re-run `make_p3_figures.py` on
   `data/p3_byzantine.json`, or pull the real files from wherever the LFS
   objects live.

## p3h_heterogeneous_sigma — the new experiment

Tests the claim the refocus depends on: **per-client heterogeneous σ widens
the honest norm band, so poisoned updates can hide in it**, plus the
σ-normalised detector that should close the gap. Arms, predictions and
pass/fail rules are in `fresh_beginning.md` §6.

**Status: empty. Nothing written or run yet.**

When it runs:

- write outputs to `p3h_heterogeneous_sigma/results/`, never to the repo-level
  `exp_data/` (so the uniform-σ baseline can't get overwritten)
- save every run's JSON, including the ones that contradict the predictions
- record which machine and GPU each run used (results have been seen to change
  across machines)

## Order of work

MNIST and P3 are **independent**, so run them in parallel on different
machines. MNIST does **not** block P3. See `fresh_beginning.md` §10.

- **P3 critical path (~1 day):** supervisor → split fix + P3-H code → timing
  pilot, control re-run and P3-H on the 5090 or lab card → rewrite only if
  predictions 1–2 hold.
- **In parallel:** MNIST on the 1650 Super (~33 h left). Keep it off the
  5090 while P3 is using it.
