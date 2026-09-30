# Figure guide for `thesis_rewrite.tex`

All figures are in **`plots/`**, each as both `.png` and `.pdf`. Use the `.pdf`
in LaTeX (vector, scales cleanly). Put this in your preamble:

```latex
\usepackage{graphicx,amsmath,amssymb,booktabs,siunitx,hyperref}
\graphicspath{{plots/}}
```

Then `\includegraphics{figC_allocation_variance.pdf}` resolves without a path.

## Figures used in the rewrite, in order of appearance

| # | File (in `plots/`) | Goes in | Label | What it shows |
|---|---|---|---|---|
| 1 | `figE_operating_curve.pdf` | Ch. 4, §"Locating the Informative Regime", right after Table~\ref{tab:noise} | `fig:operating` | IoU.25 vs σ with the no-DP ceiling (0.769), trivial-predictor floor (0.021), and the source paper's σ∈[1,20] band shaded. **Proves the regime is noise-sensitive** — the precondition for everything else. |
| 2 | `figF_adam_vs_sgd.pdf` | Ch. 4, §"Model: Public Trunk, Private Head", beside the SGD-not-Adam bullet | `fig:adam` | Two panels: under Adam 16× noise moves accuracy 0.016; under SGD, 0.064 and monotone. **The optimiser pitfall.** |
| 3 | `figC_allocation_variance.pdf` | Ch. 5 §5.1.4 | `fig:variance` | **The main theory figure.** All 13 allocations vs Σσ²; Spearman −0.767, p=0.0022. Uniform at the minimum. |
| 4 | `figD_budget_crossover.pdf` | Ch. 5 §5.1.5 | `fig:crossover` | Two panels: penalty shrinks toward the cliff but never flips; noise grows faster than the effect. |
| 5 | `figP1_defence_comparison.pdf` | Ch. 5 §5.2.3 | `fig:p1` | Bar chart, all 7 defences × 3 attacks at m=30%. |
| 6 | `figP2_detection.pdf` | Ch. 5 §5.2.3 | `fig:p2` | TPR vs malicious ratio + honest-client FPR (their Figs 5–7). |
| 7 | `figP3_gamma.pdf` | Ch. 5 §5.2.3 | `fig:p3` | γ over rounds — attacker escalates/retreats (their Fig 8). |
| 8 | `figP4_robust_agg.pdf` | Ch. 5 §5.2.3 | `fig:p4` | Trajectories with/without Krum and median (their Fig 10). |

Figures 1 and 2 are **not yet cited** in `thesis_rewrite.tex` — I generated them
after writing it. Drop these blocks in at the marked places:

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.85\textwidth]{figE_operating_curve.pdf}
\caption{The calibrated operating regime. Accuracy is genuinely sensitive to the
noise multiplier across the source paper's operating range, which is the
precondition for any allocation study. Floor and ceiling are marked.}
\label{fig:operating}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figF_adam_vs_sgd.pdf}
\caption{Adaptive optimisers normalise the privacy parameter away. A
sixteen-fold change in $\sigma$ moves final accuracy by $0.016$ under Adam and
$0.064$ under SGD, with the SGD degradation monotone in both noise and rounds.}
\label{fig:adam}
\end{figure}
```

## Tables (already written into the `.tex`, no external files needed)

| Table | Label | Content |
|---|---|---|
| Noise sensitivity | `tab:noise` | σ = 0…32 → IoU.25 0.769…0.047 |
| Power-law schedules | `tab:headroom` | k = −1…+3, n=20, CIs and p-values |
| Tight-budget ladder | `tab:tight` | ρ = 0.5/2/8 × T′ = 3/6/12/25, n=10 |
| Update envelope | `tab:envelope` | honest norms and relative s.d. vs σ |
| Byzantine resilience | `tab:p3` | 7 defences × 3 attacks, TPR, FPR |

## Figures from the OLD thesis you should drop

* `Figure 3.5 Confusion Matrix of all methods` — built from the invalid ~0.99 mAP
  runs (the pretrained-weights bug).
* `Figure 3.6 Privacy Utility Tradeoff Trajectory` and
  `Figure 3.7 Noise level Allocation` — same provenance.
* `Table 3.6` (DRL 0.9312 / Uniform 0.9791 / Exponential 0.9927 / Arithmetic
  0.9776) — **must go.** These are the numbers your supervisor flagged as
  overfitting, and they show DRL losing anyway.

Figures 1.1 (Process Flow) and 1.2 (Data Pipeline) can stay if you redraw them to
match the actual architecture: public trunk frozen → cached features → DP-trained
head → FedAvg with optional detection layer.

## Regenerating everything

```bash
python make_setup_figures.py      # figE, figF
python make_allocation_figure.py  # figC
python make_crossover_figure.py   # figD
python make_p3_figures.py         # figP1–figP4
```

All read from `exp_data/*.json`; no GPU needed, a few seconds each.
