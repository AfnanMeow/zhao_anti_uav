"""
plot_privacy_utility.py

Generates all graphs that only need all_test_results.json:
  1. Privacy vs Utility scatter plot (main contribution graph)
  2. Sigma evolution per method (policy comparison)
  3. Budget utilization verification
  4. Cumulative budget spent per round

Run from your project directory:
    python plot_privacy_utility.py

No checkpoint files needed — all data comes from all_test_results.json
"""

import json
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('./plots', exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with open('./exp_data/all_test_results.json') as f:
    data = json.load(f)

results      = data['results']        # mAP curves per method
rhos         = data['rhos']           # [rho_0, rho_1, rho_2]
sigmas_drl   = data['sigmas_drl']     # [3 clients][10 rounds]
sigmas_uni   = data['sigmas_uni']
sigmas_exp   = data['sigmas_exp']
sigmas_arith = data['sigmas_arith']

el = 5   # epoch_local — must match your training script

methods = {
    'DRL (ours)':         (results['DRL'],         sigmas_drl,   'blue',    'o',  '-'),
    'Uniform':            (results['Uniform'],      sigmas_uni,   'green',   '^',  '--'),
    'Exponential':        (results['Exponential'],  sigmas_exp,   'magenta', 'D',  ':'),
    'Arithmetic sequence':(results['Arithmetic'],   sigmas_arith, 'red',     's',  '-.'),
}

CR_Total = len(results['DRL'])
x_rounds = list(range(1, CR_Total + 1))
N        = len(rhos)

# ── Helper: compute average sigma per round (across clients) ──────────────────
def avg_sigma_per_round(sigmas):
    return [np.mean([sigmas[i][cr] for i in range(N)]) for cr in range(CR_Total)]

# ── Helper: compute cumulative rho spent per round ────────────────────────────
def cumulative_rho(sigmas, client_idx):
    cum = []
    total = 0.0
    for cr in range(CR_Total):
        total += 2 * el / (sigmas[client_idx][cr] ** 2)
        cum.append(total)
    return cum

# =============================================================================
# GRAPH 1 — Privacy vs Utility (main contribution graph)
# X-axis: average sigma (higher = more noise = STRONGER privacy)
# Y-axis: mean mAP@0.5 across rounds (HIGHER = better utility)
# Each method = one point. Shows the privacy-utility tradeoff clearly.
# =============================================================================
plt.figure(figsize=(8, 6))

for name, (accs, sigmas, color, marker, ls) in methods.items():
    mean_map   = np.mean(accs)
    mean_sigma = np.mean([np.mean(sigmas[i]) for i in range(N)])
    plt.scatter(mean_sigma, mean_map,
                color=color, marker=marker, s=180, zorder=5, label=name)
    # label offset so points don't overlap
    plt.annotate(name,
                 xy=(mean_sigma, mean_map),
                 xytext=(8, 4), textcoords='offset points',
                 fontsize=10, color=color)

plt.xlabel('Average Noise Level $\\bar{\\sigma}$ (Privacy Strength ↑)', fontsize=13)
plt.ylabel('Mean mAP@0.5 (Utility ↑)', fontsize=13)
plt.title('Privacy–Utility Tradeoff\nunder Different Budget Allocation Schemes', fontsize=13)
plt.legend(fontsize=10, loc='lower left')
plt.grid(True, alpha=0.3)

# Add annotation explaining axes
plt.annotate('← Less Privacy\n   More Utility',
             xy=(0.02, 0.15), xycoords='axes fraction', fontsize=9,
             color='gray', style='italic')
plt.annotate('More Privacy →\nLess Utility',
             xy=(0.75, 0.05), xycoords='axes fraction', fontsize=9,
             color='gray', style='italic')

plt.tight_layout()
plt.savefig('./plots/privacy_vs_utility_REAL.pdf')
plt.savefig('./plots/privacy_vs_utility_REAL.png', dpi=150)
plt.close()
print('Saved: privacy_vs_utility_REAL')

# =============================================================================
# GRAPH 2 — Privacy vs Utility CURVE (not just points)
# Shows how each method trades off privacy vs utility ACROSS rounds
# X-axis: avg sigma at that round, Y-axis: mAP at that round
# Each method = a curve showing how the tradeoff evolves round by round
# =============================================================================
plt.figure(figsize=(8, 6))

for name, (accs, sigmas, color, marker, ls) in methods.items():
    avg_sigs = avg_sigma_per_round(sigmas)
    plt.plot(avg_sigs, accs,
             color=color, marker=marker, linestyle=ls,
             linewidth=2, markersize=5, label=name)
    # mark round 1 and round 10
    plt.annotate('R1', xy=(avg_sigs[0],  accs[0]),
                 xytext=(4, 4), textcoords='offset points', fontsize=7, color=color)
    plt.annotate('R10', xy=(avg_sigs[-1], accs[-1]),
                 xytext=(4, -10), textcoords='offset points', fontsize=7, color=color)

plt.xlabel('Average Noise Level $\\bar{\\sigma}_t$ per Round (Privacy ↑)', fontsize=13)
plt.ylabel('mAP@0.5 (Utility ↑)', fontsize=13)
plt.title('Privacy–Utility Tradeoff Trajectory\nacross Communication Rounds', fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./plots/privacy_utility_curve_REAL.pdf')
plt.savefig('./plots/privacy_utility_curve_REAL.png', dpi=150)
plt.close()
print('Saved: privacy_utility_curve_REAL')

# =============================================================================
# GRAPH 3 — Sigma evolution per method (paper's Fig 2 extended)
# Shows what noise schedule each method applies — key difference between methods
# Use client 0 (lowest rho) to match paper's convention
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: DRL sigma schedule for all 3 clients (paper's Fig 2 exact)
sorted_idx = sorted(range(N), key=lambda i: rhos[i])
c0, c1, c2 = sorted_idx[0], sorted_idx[1], sorted_idx[2]
ls_list    = ['-', '--', '-.']
for j, ci in enumerate([c0, c1, c2]):
    axes[0].plot(list(range(CR_Total)), sigmas_drl[ci],
                 linestyle=ls_list[j], linewidth=2, marker='o', markersize=5,
                 label=f'Client $\\rho_k$={rhos[ci]:.3f}')
axes[0].set_xlabel('Communication Round $t$', fontsize=12)
axes[0].set_ylabel('Noise Level $\\sigma_k^t$', fontsize=12)
axes[0].set_title('(a) DRL — Noise Levels per Client', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Right: All methods for client 0 — shows policy comparison
for name, (accs, sigmas, color, marker, ls) in methods.items():
    axes[1].plot(list(range(CR_Total)), sigmas[c0],
                 color=color, marker=marker, linestyle=ls,
                 linewidth=2, markersize=5, label=name)
axes[1].set_xlabel('Communication Round $t$', fontsize=12)
axes[1].set_ylabel('Noise Level $\\sigma_k^t$', fontsize=12)
axes[1].set_title(f'(b) All Methods — Client 0 ($\\rho_k$={rhos[c0]:.3f})', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Noise Level Allocation across Communication Rounds', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('./plots/sigma_evolution_REAL.pdf', bbox_inches='tight')
plt.savefig('./plots/sigma_evolution_REAL.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: sigma_evolution_REAL')

# =============================================================================
# GRAPH 4 — Budget utilization verification
# Shows cumulative rho spent vs total budget for each client — proves constraint
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

method_sigmas = {
    'DRL':         sigmas_drl,
    'Uniform':     sigmas_uni,
    'Exponential': sigmas_exp,
    'Arithmetic':  sigmas_arith,
}
colors_budget = ['blue', 'green', 'magenta', 'red']

for ci in range(N):
    ax = axes[ci]
    for (mname, sigs), color in zip(method_sigmas.items(), colors_budget):
        cum = cumulative_rho(sigs, ci)
        ax.plot(x_rounds, cum, color=color, linewidth=2, label=mname)
    # budget ceiling line
    ax.axhline(y=rhos[ci], color='black', linestyle='--',
               linewidth=2, label=f'Budget ρ={rhos[ci]:.2f}')
    ax.set_xlabel('Communication Round $t$', fontsize=11)
    ax.set_ylabel('Cumulative ρ Spent', fontsize=11)
    ax.set_title(f'Client {ci} (ρ={rhos[ci]:.2f})', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Cumulative Privacy Budget Expenditure per Client\n(dashed line = total budget ceiling)',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('./plots/budget_utilization_REAL.pdf', bbox_inches='tight')
plt.savefig('./plots/budget_utilization_REAL.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: budget_utilization_REAL')

# =============================================================================
# PRINT: Budget compliance table (for thesis)
# =============================================================================
print('\n── Budget Compliance Table ─────────────────────────────────')
print(f'{"Method":<15} {"Client 0":>12} {"Client 1":>12} {"Client 2":>12} {"Satisfied":>10}')
print('-' * 65)
for mname, sigs in method_sigmas.items():
    row = []
    all_ok = True
    for ci in range(N):
        spent = sum(2*el / (sigs[ci][cr]**2) for cr in range(CR_Total))
        util  = spent / rhos[ci] * 100
        ok    = spent <= rhos[ci] + 1e-6
        row.append(f'{util:.2f}%')
        if not ok:
            all_ok = False
    print(f'{mname:<15} {row[0]:>12} {row[1]:>12} {row[2]:>12} {"✓" if all_ok else "✗":>10}')
print('-' * 65)

# =============================================================================
# PRINT: Final results summary table (for thesis Table)
# =============================================================================
print('\n── Results Summary Table ───────────────────────────────────')
print(f'{"Method":<22} {"Mean mAP":>10} {"Std":>8} {"Final R10":>10} {"Avg σ":>8}')
print('-' * 62)
for name, (accs, sigmas, color, marker, ls) in methods.items():
    mean_map   = np.mean(accs)
    std_map    = np.std(accs)
    final_map  = accs[-1]
    mean_sigma = np.mean([np.mean(sigmas[i]) for i in range(N)])
    print(f'{name:<22} {mean_map:>10.4f} {std_map:>8.4f} {final_map:>10.4f} {mean_sigma:>8.4f}')
print('-' * 62)
print('\nAll plots saved to ./plots/')