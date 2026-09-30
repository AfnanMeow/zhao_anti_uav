import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator

d = json.load(open('./exp_data/tight_budget_ladder.json'))
r, Ts, rhos = d['results'], d['T_values'], d['rho_values']
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.5))
cols = {12: '#2b6cb0', 6: '#c05621', 3: '#6b46c1'}

for T in Ts:
    if T == 25:
        continue
    mu, se = [], []
    for rho in rhos:
        b = np.array(r[str(rho)]['25'])
        dd = np.array(r[str(rho)][str(T)]) - b
        mu.append(dd.mean()); se.append(dd.std(ddof=1) / np.sqrt(len(dd)))
    a1.errorbar(rhos, mu, yerr=se, marker='o', ms=6, capsize=4, lw=1.6,
                color=cols[T], label='T=%d rounds (lower sigma each)' % T)
a1.axhline(0, color='k', lw=1.2)
a1.set_xscale('log'); a1.set_xticks(rhos); a1.set_xticklabels([str(x) for x in rhos])
a1.set_xlabel("privacy budget  rho   (tighter <--)")
a1.set_ylabel('IoU.25  minus uniform')
a1.set_title('Truncation vs uniform across budgets\npenalty shrinks toward the cliff, never flips', fontsize=10)
a1.legend(fontsize=8); a1.grid(alpha=0.25)
a1.annotate('uniform optimal', (8, 0.006), fontsize=8, color='#2f855a')

sds, unif = [], []
for rho in rhos:
    b = np.array(r[str(rho)]['25'])
    sds.append(np.mean([np.std(np.array(r[str(rho)][str(T)]) - b, ddof=1)
                        for T in Ts if T != 25]))
    unif.append(b.mean())
a2.plot(rhos, sds, marker='s', ms=7, lw=1.8, color='#b83280', label='sd of paired differences')
a2.plot(rhos, [0.0404] * 3, ls='--', lw=1.4, color='#718096',
        label='best candidate effect (+0.040)')
a2.set_xscale('log'); a2.set_xticks(rhos); a2.set_xticklabels([str(x) for x in rhos])
a2.set_xlabel("privacy budget  rho   (tighter <--)")
a2.set_ylabel('sd of paired difference')
a2.set_title('...because noise grows faster than the effect\nany gain at the cliff is unlearnable', fontsize=10)
a2.legend(fontsize=8); a2.grid(alpha=0.25)
for x, y in zip(rhos, sds):
    a2.annotate('%.3f' % y, (x, y), textcoords='offset points', xytext=(0, 8),
                fontsize=8, ha='center')
for ax in (a1, a2):
    ax.xaxis.set_minor_locator(NullLocator())     # log minor ticks collided with labels
    ax.xaxis.set_minor_formatter(NullFormatter())
fig.tight_layout()
for e in ('png', 'pdf'):
    fig.savefig('./plots/figD_budget_crossover.%s' % e, dpi=170)
print('wrote plots/figD_budget_crossover.png/.pdf')
