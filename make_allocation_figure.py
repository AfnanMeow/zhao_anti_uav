# Figure: accumulated noise variance predicts allocation performance.
# One scalar, sum_t sigma_t^2, orders every allocation we measured -- across
# power schedules, bang-bang, and random Dirichlet draws. Uniform minimises it.
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

CR, el, RHO = 25, 2, 30.0


def clipsig(s):
    return np.clip(s, 1.0, 20.0)


def power(k):
    w = np.array([(t + 1) ** k for t in range(CR)], float)
    return clipsig(np.sqrt(2 * el / (w / w.sum() * RHO)))


def bang(m, front):
    rem = RHO - 4.0 * m
    sr = np.sqrt(2 * el / (rem / (CR - m)))
    o = np.full(CR, sr)
    if front:
        o[:m] = 1.0
    else:
        o[-m:] = 1.0
    return clipsig(o)


def rnd(sd):
    rho = np.random.default_rng(sd).dirichlet(np.ones(CR) * 0.5) * RHO
    return clipsig(np.sqrt(2 * el / np.maximum(rho, 1e-9)))


base_nc = float(np.sum(power(0.0) ** 2))
pts = []

d = json.load(open('./exp_data/headroom_clear.json'))
r = d['results']; b = np.array(r['0.0'])
for k in d['k_values']:
    if k == 0.0:
        continue
    x = np.array(r[str(k)]) - b
    pts.append(('k=%+g' % k, float(np.sum(power(k) ** 2)) / base_nc,
                x.mean(), x.std(ddof=1) / np.sqrt(len(x)), 'power'))

d = json.load(open('./exp_data/fine_negk.json'))
r = d['results']; b = np.array(r['0.0'])
for k in d['k_values']:
    if k == 0.0:
        continue
    x = np.array(r[str(k)]) - b
    pts.append(('k=%+g' % k, float(np.sum(power(k) ** 2)) / base_nc,
                x.mean(), x.std(ddof=1) / np.sqrt(len(x)), 'power'))

d = json.load(open('./exp_data/allocation_bound.json'))
r = d['results']; b = np.array(r['uniform'])
for nm, sg, st in [('front-clean', bang(7, True), 'bang'),
                   ('back-clean', bang(7, False), 'bang'),
                   ('rand-A', rnd(7), 'random'),
                   ('rand-B', rnd(8), 'random')]:
    x = np.array(r[nm]) - b
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    pts.append((nm, float(np.sum(sg ** 2)) / base_nc, x.mean(), se, st))

fig, ax = plt.subplots(figsize=(7.4, 4.9))
style = {'power':  ('o', '#2b6cb0', r'power  $\rho_t \propto (t{+}1)^k$'),
         'bang':   ('s', '#c05621', 'bang-bang (7 clean rounds)'),
         'random': ('^', '#6b46c1', 'random (Dirichlet)')}
seen = set()
for nm, pen, mu, se, st in pts:
    m, c, lab = style[st]
    ax.errorbar(pen, mu, yerr=se, fmt=m, color=c, ms=7, capsize=3, lw=1.2,
                label=lab if st not in seen else None, zorder=3)
    seen.add(st)
    if abs(mu) > 0.004:
        dx = -46 if pen > 15 else 8
        ax.annotate(nm, (pen, mu), textcoords='offset points', xytext=(dx, -3),
                    fontsize=7.5, color='#444')

ax.axhline(0, color='k', lw=1.1, zorder=2)
ax.axvline(1.0, color='#2f855a', ls='--', lw=1.3, zorder=2)
ax.annotate('uniform = analytic minimum\n' + r'of $\sum_t \sigma_t^2$',
            (1.0, -0.088), textcoords='offset points', xytext=(9, 0),
            fontsize=8.5, color='#2f855a')

x = np.array([p[1] for p in pts]); y = np.array([p[2] for p in pts])
sr, sp = stats.spearmanr(x, y)
sl, ic = np.polyfit(np.log(x), y, 1)
xs = np.linspace(x.min() * 0.95, x.max() * 1.05, 200)
ax.plot(xs, sl * np.log(xs) + ic, color='#999', ls=':', lw=1.4, zorder=1)

ax.set_xscale('log')
ax.set_xlabel(r'accumulated noise variance  $\sum_t \sigma_t^2$   (relative to uniform)')
ax.set_ylabel(r'IoU$_{.25}$  minus uniform')
ax.set_title('Uniform is optimal: one scalar orders every allocation\n'
             + r'Spearman $\rho$=%+.2f (p=%.4f), %d allocations, 3 families'
             % (sr, sp, len(pts)), fontsize=10.5)
ax.legend(fontsize=8.5, loc='lower left', framealpha=0.95)
ax.grid(alpha=0.25, which='both')
ax.set_xlim(0.92, 26)
# the near-uniform cluster is too dense to label in place; call it out as a group
ax.annotate('k = -0.75 ... +0.5\n(all within 0.002 of uniform)', (1.25, 0.012),
            fontsize=7.5, color='#2b6cb0', ha='center')
ax.set_xticks([1, 2, 5, 10, 20])
ax.set_xticklabels(['1x', '2x', '5x', '10x', '20x'])
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig('./plots/figC_allocation_variance.%s' % ext, dpi=170)
print('wrote plots/figC_allocation_variance.png/.pdf')
print('Spearman %+.3f p=%.5f   slope(log) %+.4f' % (sr, sp, sl))
