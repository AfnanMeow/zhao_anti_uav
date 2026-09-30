# =============================================================================
# P3 figures. Analogues of exploiting_dp.pdf Figs 5-8 and 10, on the anti-UAV
# detector. Tolerates partial data so it can be run while the grid is still going.
#
#   figP1  defence comparison   -- final IoU.25 per defence, per attack (m=30%)
#   figP2  detection accuracy   -- TPR/FPR vs malicious ratio (their Figs 5-7)
#   figP3  degree of poisoning  -- gamma over rounds (their Fig 8)
#   figP4  with / without Krum  -- accuracy trajectories, attack vs none (their Fig 10)
# =============================================================================
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = json.load(open('./exp_data/p3_byzantine.json'))
R = D['results']
NC = D['n_clients']
ROUNDS = D['rounds']

ATT_LABEL = {'none': 'no attack', 'rmd': 'RMD', 'mpelm': r'$\alpha$-MPELM'}
ATT_COL = {'none': '#2f855a', 'rmd': '#2b6cb0', 'mpelm': '#c53030'}
DEF_ORDER = ['none', 'norm', 'accuracy', 'mix', 'krum', 'median', 'trimmed']
DEF_LABEL = {'none': 'no defence', 'norm': 'Norm', 'accuracy': 'Accuracy',
             'mix': 'Mix', 'krum': 'Krum', 'median': 'median', 'trimmed': 'trimmed mean'}


def get(at, df, m):
    return R.get('%s|%s|%d' % (at, df, m))


def final_acc(hs, last=3):
    v = [np.mean(h['acc'][-last:]) for h in hs]
    return float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def rates(hs, m):
    tpr = np.mean([np.sum(h['tp']) / max(1, m * len(h['tp'])) for h in hs]) if m else np.nan
    fpr = np.mean([np.sum(h['fp']) / max(1, (NC - m) * len(h['fp'])) for h in hs])
    return tpr, fpr


# ── figP1: defence comparison at m = 30% ────────────────────────────────────
base = get('none', 'none', 0)
if base:
    b_mu, _ = final_acc(base)
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    w, xs = 0.26, np.arange(len(DEF_ORDER))
    for j, at in enumerate(['none', 'rmd', 'mpelm']):
        mus, ses = [], []
        for df in DEF_ORDER:
            hs = get(at, df, 0 if at == 'none' else 3)
            if hs:
                mu, se = final_acc(hs)
            else:
                mu, se = np.nan, 0
            mus.append(mu); ses.append(se)
        ax.bar(xs + (j - 1) * w, mus, w, yerr=ses, capsize=3,
               color=ATT_COL[at], label=ATT_LABEL[at], alpha=0.9)
    ax.axhline(b_mu, color='#2f855a', ls='--', lw=1.4)
    ax.annotate('no-attack, no-defence baseline (%.3f)' % b_mu, (len(DEF_ORDER) - 0.6, b_mu),
                fontsize=8, color='#2f855a', ha='right', va='bottom')
    ax.set_xticks(xs); ax.set_xticklabels([DEF_LABEL[d] for d in DEF_ORDER])
    ax.set_ylabel('IoU.25 (final 3 rounds)')
    ax.set_title('Defence comparison under poisoning, m = 30%% of %d clients' % NC, fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    for e in ('png', 'pdf'):
        fig.savefig('./plots/figP1_defence_comparison.%s' % e, dpi=170)
    print('figP1 written')

# ── figP2: detection accuracy vs malicious ratio ────────────────────────────
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))
ms = [1, 2, 3]
for at in ('rmd', 'mpelm'):
    for df, ls in (('norm', '-'), ('mix', '--')):
        tprs = []
        for m in ms:
            hs = get(at, df, m)
            tprs.append(rates(hs, m)[0] * 100 if hs else np.nan)
        a1.plot([m * 100 // NC for m in ms], tprs, ls, marker='o', ms=6, lw=1.7,
                color=ATT_COL[at], alpha=1.0 if df == 'norm' else 0.55,
                label='%s / %s' % (ATT_LABEL[at], DEF_LABEL[df]))
a1.set_xlabel('malicious clients  m  (%% of %d)' % NC)
a1.set_ylabel('detection rate (TPR, %)')
a1.set_title('Detection of malicious updates', fontsize=10.5)
a1.set_ylim(-4, 104); a1.legend(fontsize=8); a1.grid(alpha=0.25)
a1.set_xticks([m * 100 // NC for m in ms])

for at in ('none', 'rmd', 'mpelm'):
    fprs = []
    for m in ms:
        hs = get(at, 'mix', 0 if at == 'none' else m)
        fprs.append(rates(hs, 0 if at == 'none' else m)[1] * 100 if hs else np.nan)
    a2.plot([m * 100 // NC for m in ms], fprs, marker='s', ms=6, lw=1.7,
            color=ATT_COL[at], label=ATT_LABEL[at])
a2.set_xlabel('malicious clients  m  (%% of %d)' % NC)
a2.set_ylabel('false-positive rate on honest clients (%)')
a2.set_title('Cost of the Mix detector to honest clients', fontsize=10.5)
a2.legend(fontsize=8); a2.grid(alpha=0.25)
a2.set_xticks([m * 100 // NC for m in ms])
fig.tight_layout()
for e in ('png', 'pdf'):
    fig.savefig('./plots/figP2_detection.%s' % e, dpi=170)
print('figP2 written')

# ── figP3: degree of poisoning over rounds ──────────────────────────────────
fig, ax = plt.subplots(figsize=(7.4, 4.3))
for df, c in (('none', '#718096'), ('norm', '#2b6cb0'), ('mix', '#c05621'),
              ('krum', '#6b46c1')):
    hs = get('mpelm', df, 3)
    if not hs:
        continue
    g = np.array([h['gamma'] for h in hs])
    mu, se = g.mean(0), g.std(0, ddof=1) / np.sqrt(len(g))
    r = np.arange(1, len(mu) + 1)
    ax.plot(r, mu, lw=1.8, color=c, label='vs %s' % DEF_LABEL[df])
    ax.fill_between(r, mu - se, mu + se, color=c, alpha=0.18)
ax.set_xlabel('round'); ax.set_ylabel(r'degree of poisoning  $\gamma$')
ax.set_title(r'$\alpha$-MPELM adapts its poisoning to the defence' + '\n'
             '(escalates when undetected, backs off when caught)', fontsize=10.5)
ax.legend(fontsize=9); ax.grid(alpha=0.25)
fig.tight_layout()
for e in ('png', 'pdf'):
    fig.savefig('./plots/figP3_gamma.%s' % e, dpi=170)
print('figP3 written')

# ── figP4: with / without a robust aggregator ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), sharey=True)
for ax, df in zip(axes, ['none', 'krum', 'median']):
    for at in ('none', 'rmd', 'mpelm'):
        hs = get(at, df, 0 if at == 'none' else 3)
        if not hs:
            continue
        A = np.array([h['acc'] for h in hs])
        mu, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(len(A))
        r = np.arange(1, len(mu) + 1)
        ax.plot(r, mu, lw=1.8, color=ATT_COL[at], label=ATT_LABEL[at])
        ax.fill_between(r, mu - se, mu + se, color=ATT_COL[at], alpha=0.18)
    ax.set_title('%s' % DEF_LABEL[df], fontsize=10.5)
    ax.set_xlabel('round'); ax.grid(alpha=0.25)
axes[0].set_ylabel('IoU.25')
axes[0].legend(fontsize=9)
fig.suptitle('Attack impact with and without a Byzantine-robust aggregator (m = 30%)',
             fontsize=11)
fig.tight_layout()
for e in ('png', 'pdf'):
    fig.savefig('./plots/figP4_robust_agg.%s' % e, dpi=170)
print('figP4 written')

# ── summary table to stdout ─────────────────────────────────────────────────
print('\n%-10s %-12s %8s %8s %8s' % ('attack', 'defence', 'IoU.25', 'TPR', 'FPR'))
for at in ('none', 'rmd', 'mpelm'):
    for df in DEF_ORDER:
        m = 0 if at == 'none' else 3
        hs = get(at, df, m)
        if not hs:
            continue
        mu, se = final_acc(hs)
        tpr, fpr = rates(hs, m)
        print('%-10s %-12s %8.4f %8s %8.2f'
              % (ATT_LABEL[at], DEF_LABEL[df], mu,
                 '-' if np.isnan(tpr) else '%.2f' % tpr, fpr))
