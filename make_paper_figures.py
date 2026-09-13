# =============================================================================
# Figures from the held-out-evaluation comparison (lightweight CNN pipeline).
#
# Fig A: mAP@0.5 vs communication round -- DRL / Uniform / Exponential /
#        Arithmetic, mean over 8 shared-init paired trials, +/-1 std band.
# Fig B: noise level sigma_k^t vs communication round -- what the DRL agent
#        actually learned, against the three fixed schedules (the source
#        paper's Fig. 2 analogue).
#
# Source data: exp_data/lightcnn_pub_fig23_data.json, produced by
# lightcnn_publication_plots.py -- every point scored on
# single_test_coco/annotations/val.json, which no client ever trains on.
#
# NOTE ON HONESTY: an earlier run showed DRL beating Uniform 5/5, but that was
# scored on each client's OWN TRAINING SHARD. Re-scored on held-out data the
# same policy lost (0.823 vs 0.906). These figures use the held-out numbers, so
# Fig A does not show DRL leading. That is the real measurement.
# =============================================================================
import os, json
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = './exp_data/lightcnn_pub_fig23_data.json'
OUT  = './plots'
os.makedirs(OUT, exist_ok=True)

# validated categorical palette, fixed order, never cycled
COLORS  = {'DRL': '#2a78d6', 'Uniform': '#eb6834', 'Exponential': '#1baf7a', 'Arithmetic': '#eda100'}
STYLES  = {'DRL': '-', 'Uniform': ':', 'Exponential': '--', 'Arithmetic': '-.'}
MARKERS = {'DRL': 'o', 'Uniform': 's', 'Exponential': '^', 'Arithmetic': 'D'}
ORDER   = ['DRL', 'Uniform', 'Exponential', 'Arithmetic']

d = json.load(open(DATA))
accs, sigmas, rhos = d['all_accs'], d['all_sigmas'], d['all_rhos']
n_trials = len(accs['DRL'])
CR = len(accs['DRL'][0]) - 1
print(f'{n_trials} paired trials, {CR} communication rounds', flush=True)

# ── Fig A: mAP vs communication round ──────────────────────────────────────
x = list(range(CR + 1))
plt.figure(figsize=(8, 5))
for name in ORDER:
    a = np.array(accs[name]); m, s = a.mean(0), a.std(0)
    plt.plot(x, m, linestyle=STYLES[name], color=COLORS[name], linewidth=2,
             marker=MARKERS[name], markersize=5, label=name)
    plt.fill_between(x, m - s, m + s, color=COLORS[name], alpha=0.12)
plt.xticks(x, fontsize=12); plt.yticks(fontsize=12)
plt.xlabel('Communication Round $t$', fontsize=15)
plt.ylabel('mAP@0.5 (held-out val)', fontsize=15)
plt.title(f'Detection accuracy vs communication round (mean $\\pm$ 1 s.d., {n_trials} trials)',
          fontsize=13)
plt.legend(fontsize=12); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUT, 'figA_map_vs_rounds.pdf'))
plt.savefig(os.path.join(OUT, 'figA_map_vs_rounds.png'), dpi=150)
plt.close()
print('saved figA_map_vs_rounds', flush=True)

# ── Fig B: noise level vs communication round ──────────────────────────────
# sigmas[name][trial][client][round]; average over trials & clients per method
xr = list(range(CR))
plt.figure(figsize=(8, 5))
for name in ORDER:
    arr = np.array(sigmas[name])            # (trials, clients, rounds)
    m = arr.mean(axis=(0, 1)); s = arr.std(axis=(0, 1))
    plt.plot(xr, m, linestyle=STYLES[name], color=COLORS[name], linewidth=2,
             marker=MARKERS[name], markersize=5, label=name)
    plt.fill_between(xr, m - s, m + s, color=COLORS[name], alpha=0.10)
plt.xticks(xr, fontsize=12); plt.yticks(fontsize=12)
plt.xlabel('Communication Round $t$', fontsize=15)
plt.ylabel('Noise Level $\\sigma_k^t$', fontsize=15)
plt.title('Noise level allocated per round, by method (mean over trials & clients)', fontsize=13)
plt.legend(fontsize=12); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUT, 'figB_noise_vs_rounds.pdf'))
plt.savefig(os.path.join(OUT, 'figB_noise_vs_rounds.png'), dpi=150)
plt.close()
print('saved figB_noise_vs_rounds', flush=True)

# ── Fig B2: DRL's schedule per client, split by privacy budget (paper's Fig 2)
ref = rhos[0]
order = sorted(range(len(ref)), key=lambda i: ref[i])
picks = [order[0], order[len(order) // 2], order[-1]]
drl = np.array(sigmas['DRL'])               # (trials, clients, rounds)
plt.figure(figsize=(8, 5))
for j, ci in enumerate(picks):
    m = drl[:, ci, :].mean(0)
    plt.plot(xr, m, linewidth=2, marker='o', markersize=5,
             color=[COLORS['DRL'], COLORS['Uniform'], COLORS['Exponential']][j],
             linestyle=['-', '--', ':'][j],
             label=f'Client with $\\rho_k=${ref[ci]:.1f}')
plt.xticks(xr, fontsize=12); plt.yticks(fontsize=12)
plt.xlabel('Communication Round $t$', fontsize=15)
plt.ylabel('DRL Noise Level $\\sigma_k^t$', fontsize=15)
plt.title('Learned DRL noise schedule, by client privacy budget', fontsize=13)
plt.legend(fontsize=12); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUT, 'figB2_drl_noise_by_budget.pdf'))
plt.savefig(os.path.join(OUT, 'figB2_drl_noise_by_budget.png'), dpi=150)
plt.close()
print('saved figB2_drl_noise_by_budget', flush=True)

# ── numbers behind the figures ─────────────────────────────────────────────
print('\nfinal-round mAP (held-out), mean over trials:', flush=True)
for name in ORDER:
    a = np.array(accs[name])[:, -1]
    print(f'  {name:12s} {a.mean():.4f} +/- {a.std():.4f}', flush=True)
print('\nmean sigma by method (early rounds 0-2 vs late rounds -3:):', flush=True)
for name in ORDER:
    arr = np.array(sigmas[name])
    print(f'  {name:12s} early={arr[:, :, :3].mean():.2f}  late={arr[:, :, -3:].mean():.2f}',
          flush=True)
