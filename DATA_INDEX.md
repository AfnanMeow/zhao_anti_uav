# Experiment data index — everything available for figures

All raw results live in `exp_data/*.json`. Nothing has been deleted; superseded
results are kept and labelled as such so the write-up can show the progression.
Every file is written incrementally (after each trial), so even interrupted runs
hold usable data.

Regime for everything below unless stated: frozen public trunk `s1-s3`,
DP-trained `k1+k2+head` (D=566,789), SGD lr=0.003, B=128, 5 clients x 3,808
frames, 25 rounds x 2 local epochs, clear-sky session-disjoint split, metric
IoU.25 on 1,500 held-out frames.

## Headline results

| file | what | n | use for |
|---|---|---|---|
| `headroom_clear.json` | power schedules k = -1 … +3 vs uniform, CRN | 20 | main headroom table; dose-response in k |
| `fine_negk.json` | k = -0.25 … -1.0 vs uniform, CRN | 16 | tight null near uniform; refutes the "sweet spot" |
| `allocation_bound.json` | bang-bang + random Dirichlet vs uniform | 2 | shows non-power allocations also lose (screening only) |
| `truncation_test.json` | early stopping T = 8 … 25 at rho=30 | 1 | truncation is neutral at loose budget (screening only) |
| `tight_budget_ladder.json` | **rho = 0.5 / 2 / 8 x T = 3/6/12/25** | 10 | the budget-crossover result; **also stores full per-round trajectories** |
| `paired_schedule_clear.json` | k=+1 vs uniform, NO CRN | 30 | superseded — cite only as a precision cautionary tale |

## Calibration / mechanism (the story of how the regime was found)

| file | what | use for |
|---|---|---|
| `gate1_noise_curve.json` | original Gate 1, head seeded from public | the flat curve that was an init artifact |
| `diag_gate1_causes.json` | Adam vs SGD, clip-only controls | **the Adam-washes-out-sigma finding** |
| `diag_gate2_randomhead.json` | random head, optimiser/batch calibration | dynamic-range recovery |
| `gate1_corrected.json` | corrected gate, sigma 0 … 16, per-round | trajectories under noise |
| `knee_sigma_sweep.json` | sigma 0 … 1024 with per-round trajectories + eps | knee hunt; peak-then-decay curves |
| `knee_seeds.json` | same, 3 seeds, with epsilon | shows seed variance exceeded the whole effect |
| `gateDB_B512_s3.json` | sigma sweep at D=566,789, B=512 | SNR model validation (predicted knee 38) |
| `gateDB_B128_s3.json`, `..._seed5678.json` | same at B=128, two seeds | **the live regime**, predicted knee 19 |
| `gateDB_B128_clear.json` | sigma 0/2/4/8/16/32 on the clear split | **the headline noise curve, 0.769 -> 0.047** |
| `tight_budget_ladder_trial1_throttled.json` | trial 1 before the guard fix | identical to the rerun; evidence throttling is numerically neutral |

## Figures already generated (`plots/`)

* `figC_allocation_variance.png/pdf` — **the main theory figure**: accumulated
  noise variance orders all 13 allocations, Spearman -0.767, p=0.0022
* `figA_map_vs_rounds`, `figB_noise_vs_rounds`, `figB2_drl_noise_by_budget`,
  `fig2_mAP_vs_rounds`, `fig3_noise_vs_rounds` — earlier light-CNN era

## Figure suggestions not yet made

1. **Noise curve** — IoU.25 vs sigma on the clear split (`gateDB_B128_clear.json`)
   with the trivial-predictor floor (0.021) and no-DP ceiling (0.769) marked.
2. **Budget crossover** — best T vs rho from `tight_budget_ladder.json`; the
   regime boundary where uniform stops being optimal.
3. **Trajectories** — per-round IoU.25 for T=3/6/12/25 at each rho (the ladder
   stores these in full), showing collapse vs survival.
4. **Adam vs SGD** — `diag_gate1_causes.json`, the flat-vs-sloped pair; makes the
   optimiser point in one panel.
5. **Dose-response in k** — `headroom_clear.json` + `fine_negk.json` combined,
   mean diff vs k with CIs; monotone, uniform at the peak.
6. **Privacy-utility** — epsilon vs IoU.25 from `knee_seeds.json`
   (eps=0.916 at sigma=16 with no accuracy cost, on the OLD split).

## Reproduce

```
python gate_DB_sweep.py --B 128 --lr 0.003 --sigmas 0 2 4 8 16 32 \
    --data ./fl_workdirs/clear --public ./fl_workdirs/clear/public_pretrained.pt
python headroom_sweep.py
python fine_negk_test.py
python tight_budget_ladder.py
python make_allocation_figure.py
```

Cached features (`fl_workdirs/clear/`, ~16 GB) are rebuildable with
`python stage1_clear.py` (~3 min) if the directory is lost.

---

# P3 — Byzantine resilience

Source paper: `exploiting_dp.pdf` (Hossain et al., IEEE TAI 6(11), 2025);
extracted text `exploiting_dp.txt`; scoping notes `P3_BYZANTINE_NOTES.md`.

Setup: the 5 cached clear-sky shards split into **10 clients x 1,904 frames**, so
the malicious ratio m = 1/2/3 gives exactly the paper's 10 / 20 / 30%. sigma=1.83
(uniform at rho=30), 15 rounds, 3 trials per config, 33 configs. The server's
detector validation set (300 frames) is disjoint from the reported metric set
(1,200 frames).

| file | what |
|---|---|
| `p3_envelope.json` | honest update-norm envelope vs sigma — the foundation for every detector threshold |
| `p3_byzantine.json` | the full grid: per-round accuracy, gamma, TP/FP and the rejected-client set for every (attack, defence, m, trial) |

## The envelope result (already in hand)

| sigma | honest update norm | relative sd |
|---|---|---|
| 0 | 0.6917 +- 0.6548 | 95% |
| 1.83 | 2.2487 +- 0.0187 | 0.83% |
| 4.0 | 4.8853 +- 0.0095 | 0.19% |
| 8.0 | 9.7586 +- 0.0100 | **0.10%** |

High-dimensional concentration: a Gaussian norm in D dimensions has relative sd
1/sqrt(2D) = 0.094% at D=566,789, and 0.102% was measured. **DP noise makes the
update norm MORE predictable, not less**, so the paper's "hide in the noise
magnitude" room does not exist at this model size — it is a small-D phenomenon.
What survives is direction, which a norm detector cannot see; hence a-MPELM is
implemented here as a norm-preserving direction attack.

## Figures (`make_p3_figures.py`, tolerates partial data)

* `figP1_defence_comparison` — final IoU.25 per defence per attack at m=30%
* `figP2_detection` — TPR vs malicious ratio (their Figs 5-7) + honest-client FPR
* `figP3_gamma` — degree of poisoning over rounds (their Fig 8)
* `figP4_robust_agg` — accuracy trajectories with/without Krum and median (their Fig 10)
