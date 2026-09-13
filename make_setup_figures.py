# Two setup figures the rewritten thesis needs:
#   figE  the calibrated operating curve (IoU.25 vs sigma) with floor/ceiling
#   figF  the Adam pitfall: Adam washes the noise scale out, SGD transmits it
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── figE: operating curve on the clear split ────────────────────────────────
d = json.load(open('./exp_data/gateDB_B128_clear.json'))
sig = sorted(float(k) for k in d)
acc = [d[str(s)] if str(s) in d else d['%g' % s] for s in sig]

fig, ax = plt.subplots(figsize=(7.2, 4.5))
x = [s if s > 0 else 0.7 for s in sig]          # place sigma=0 on the log axis
ax.plot(x, acc, marker='o', ms=7, lw=1.9, color='#2b6cb0', zorder=3)
ax.axhline(0.021, color='#a0aec0', ls=':', lw=1.5)
ax.annotate('trivial constant predictor (0.021)', (0.75, 0.021),
            textcoords='offset points', xytext=(4, 6), fontsize=8, color='#4a5568')
ax.axhline(acc[0], color='#2f855a', ls='--', lw=1.3)
ax.annotate('no-DP ceiling (%.3f)' % acc[0], (0.75, acc[0]),
            textcoords='offset points', xytext=(4, -12), fontsize=8, color='#2f855a')
ax.axvspan(1, 20, color='#f6e05e', alpha=0.20, zorder=0)
ax.annotate("source paper's operating range\n$\\sigma \\in [1,20]$", (4.5, 0.68),
            fontsize=8.5, color='#975a16', ha='center')
for s, a in zip(x, acc):
    ax.annotate('%.3f' % a, (s, a), textcoords='offset points', xytext=(0, 9),
                fontsize=7.5, ha='center', color='#2b6cb0')
ax.set_xscale('log')
ax.set_xticks([0.7, 2, 4, 8, 16, 32])
ax.set_xticklabels(['0\n(no DP)', '2', '4', '8', '16', '32'])
ax.set_xlabel('noise multiplier  $\\sigma$')
ax.set_ylabel('IoU$_{.25}$')
ax.set_title('Calibrated operating regime: accuracy is genuinely noise-sensitive\n'
             '$D$=566,789, $B$=128, 5 clients, 25 rounds', fontsize=10.5)
ax.grid(alpha=0.25)
ax.set_ylim(0, 0.86)
fig.tight_layout()
for e in ('png', 'pdf'):
    fig.savefig('./plots/figE_operating_curve.%s' % e, dpi=170)
print('figE written')

# ── figF: Adam vs SGD ───────────────────────────────────────────────────────
c = json.load(open('./exp_data/diag_gate1_causes.json'))


def traj(tag):
    return [(r, i25) for r, _, i25, _ in c[tag]] if tag in c else None


pairs = [('adam DP s=0.5 C=1', 'adam DP s=8.0 C=1', 'Adam', '#c53030'),
         ('sgd  DP s=0.5 C=1 lr=.05', 'sgd  DP s=8.0 C=1 lr=.05', 'SGD', '#2b6cb0')]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
for ax, (lo, hi, name, col) in zip((a1, a2), pairs):
    for tag, ls, lab in ((lo, '-', '$\\sigma=0.5$'), (hi, '--', '$\\sigma=8$')):
        t = traj(tag)
        if not t:
            continue
        ax.plot([r for r, _ in t], [v for _, v in t], ls, marker='o', ms=6,
                lw=1.8, color=col, label=lab,
                alpha=1.0 if ls == '-' else 0.6)
    t_lo, t_hi = traj(lo), traj(hi)
    if t_lo and t_hi:
        gap = abs(t_lo[-1][1] - t_hi[-1][1])
        ax.set_title('%s — 16$\\times$ noise changes accuracy by %.3f'
                     % (name, gap), fontsize=10.5)
    ax.set_xlabel('federated round'); ax.grid(alpha=0.25); ax.legend(fontsize=9)
a1.set_ylabel('IoU$_{.25}$')
fig.suptitle('Adaptive optimisers normalise the privacy parameter away', fontsize=11.5)
fig.tight_layout()
for e in ('png', 'pdf'):
    fig.savefig('./plots/figF_adam_vs_sgd.%s' % e, dpi=170)
print('figF written')
