# Compute cost of the P3-focused plan

**Revised 2026-09-28** (supersedes the 2026-09-25 version). Estimates for the
work in `fresh_beginning.md`, including the arms added by §2b (Arm A′, the
per-client logging re-run, and the controlled σ ladder).

**Run the timing pilot before relying on any of this.** The per-run figure is
uncertain by roughly 3× — see below.

## The per-run time is the weak number, and two sources disagree

| Source | Per-run estimate | Basis |
|---|---|---|
| Old COSTS.md (25 Sep) | **0.5–1 min** | extrapolated from a 25-round, 5-client gate run on cached features |
| `p3_byzantine.py` own comment | **~2.5 min** | *"A full 3x7x3 grid would be ~8 hr"* → 63 configs × 3 trials = 189 runs |

The second is more trustworthy — it was written while looking at P3 itself — and
it is **2.5–5× slower**. Everything below uses the range **1–2.5 min/run**.

**P3 wall-clock was never recorded.** `p3_byzantine.py:251` prints elapsed
minutes per config (`[%.0fm]`), but no stdout log survived and
`exp_data/p3_byzantine.json` has no timing fields. Hence the pilot.

Other measured points, for calibration:

- **GTX 1650 Super (this Linux box):** MNIST DRL training averaged **3.4
  min/episode** (episodes 11→200 took 10 h 41 min, from `DRL_Nets/` timestamps).
- **RTX 4090 (user-reported):** full 800-episode MNIST run in ~6 h.
- **RTX 5090 (user-reported):** ~12 h for ~600 episodes — *slower than the 4090*
  and anomalous. See "5090 check" below before trusting the 5090 column.

## Cost table

Original P3 grid = **33 configs × 3 trials = 99 runs**.

| Step | Runs | RTX 5090 (healthy) |
|---|---|---|
| **Timing pilot** (1 config × 1 trial) | 1 | **3–5 min** |
| Fix P3 positional detector/report split | code only | no GPU |
| Arm **A** re-run, σ=1.83, **with per-client logging** | 99 | 1.7–4.2 h |
| Arm **A′** — heterogeneous ρ, uniform over rounds | 72–99 | 1.2–4.2 h |
| Arms **B–E** — heterogeneous ρ, adaptive | 72–99 | 1.2–4.2 h |
| Arm **F** — σ-lying sweep (5 ratios × 2 attacks × 3 trials) | 30 | 0.5–1.3 h |
| Arm **G** — per-client baseline detector | 0 | **0 — post-processing of the §2b logs** |
| Controlled σ ladder (4 spread levels, for figH3/H4/H5) | 48 | 0.8–2 h |
| **Total** | **~320–380** | **≈ 6–15 h** |

That is **one long day or two evenings**, not the "3–5 h, about one evening" the
previous version claimed.

Arm A must be re-run even though its accuracy/TPR numbers already exist, because
no per-client update norms were ever logged (`fresh_beginning.md` §2b, and §6a
on figures). Without that re-run, figH1's control panel cannot be drawn.

### Not GPU, but on the critical path

- **Cache setup on the 5090:** needs `fl_workdirs/clear` — copy ~13 GB from the
  lab box, or rebuild with `stage1_clear.py` (~20–40 min GPU).
- **Code:** split fix + heterogeneous σ + σ-normalised detector + σ-lying arm +
  the §6a logging + six figure scripts. Realistically **1–2 days of writing.**
  **The code dominates the compute.** Budget accordingly.

## Run the decisive subset FIRST — the thesis is testable in under an hour

Do not start with the full grid. The reframe stands or falls on predictions 1–3
(`fresh_beginning.md` §6), and those need 15 runs:

Use **Arm A′** (heterogeneous budgets, *uniform over rounds*) as the
heterogeneous condition here — **not** the adaptive arm. The claim is that budget
heterogeneity opens the channel (§2b C3), so A′ tests it with no adaptive
confound. If A′ alone widens the band and defeats detection, the result
implicates the literature's plain baseline, which is the stronger finding.
Adaptive is then the *aggravating* condition, tested afterwards.

| Subset | Runs | Tests |
|---|---|---|
| **A** σ=1.83, RMD, Norm-raw, m=3 | 3 | baseline TPR (expect 1.00) |
| **A′** heterogeneous, RMD, Norm-raw, m=3 | 3 | **prediction 1** — does TPR drop? |
| **A and A′**, RMD, **no defence**, m=3 | 6 | **prediction 2** — is damage worse than −0.017? |
| **A′** heterogeneous, RMD, Norm-**normalised**, m=3 | 3 | **prediction 3** — does the fix work? |

**15 runs ≈ 30–60 min.** If prediction 1 or 2 fails, the channel does not open in
practice, the reframe fails, and you have saved ~14 h of GPU and an abstract
rewrite. Only run the full grid once this subset comes back the right way.

Log per-client norms in this subset too. That gives a draft **figH1** (the figure
the whole claim rests on) and, for free, a first read on **Arm G** — the
per-client historical baseline detector (`fresh_beginning.md` §5a), which is pure
post-processing over these same logs and needs **no additional GPU runs at all**.

## 5090 check — do this before trusting any 5090 timing

The 5090 previously ran at half a 4090's speed. Most likely a PyTorch build
without native Blackwell kernels:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(), torch.cuda.get_arch_list())"
```

If **`sm_120` is not listed**, install a build that includes it — otherwise
**double every GPU number above.**

## Why P3 cannot run on the 1650 Super

`gate_DB_sweep.py:48-51` loads **all** cached features onto the GPU at once.

- One frame's feature map: 128 ch × 32 × 40 × fp16 = **320 KB**.
- Clear split: 29 folders, ~40,000 frames → **~13 GB** (estimate; the cache is
  not on this machine, so the frame count is unverified).
- 1650 Super: **4 GB VRAM**. This box: **15 GB RAM, ~11 GB free.**

It does not fit in VRAM and barely fits in RAM. A streaming loader would work but
would read from the shared drive, pushing the run into tens of hours.

- **RTX 5090 (32 GB):** fits the cache as-is.
- **Lab RTX PRO 6000 (96 GB):** fits, and **already has the cache** — P3
  originally ran there, so it is the easiest place for P3-H. Re-apply
  `nvidia-smi -pl 300` after any reboot (see `CLAUDE.md`).

## Planning notes

1. **MNIST training is DONE** (completed 2026-09-27, 800 episodes, best actor
   668, snapshots every 50 in `DRL_Nets/snapshots/`). The old "finish MNIST
   212→800" row is removed. No further DRL training is planned — the track is
   closed (`fresh_beginning.md` §2a step 2).
2. **CIFAR was never run** and is not scheduled. `test_CIFAR10_paired.py` exists
   but has never executed (§2b C9).
3. **Record which machine produced which result.** Results have already been seen
   to change across machines in this project, so note the card in every output.
4. The old note that a full 1650-Super MNIST run is ~45 h (not 26 h) is retained
   for the record but is now moot.

## Suggested allocation

- **1650 Super:** nothing further. It cannot run P3.
- **5090 or lab card:** 5090 check → timing pilot → **decisive 15-run subset** →
  (only if it passes) split fix re-run of Arm A with logging → A′ → B–E → F →
  σ ladder → figures.
