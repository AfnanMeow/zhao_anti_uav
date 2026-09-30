# =============================================================================
# Analyse a paired MNIST allocation run and compare against the stored
# actor-114 baseline.
#
#   python analyse_mnist_paired.py [path_to_exp_data_*.json] [--actor 612]
#
# Prints: per-arm means/sd, paired differences vs every baseline with 95% CI,
# win counts and p-values, the variance-penalty ordering, and the
# Spearman(accumulated variance, accuracy) that decides whether the thesis's
# variance-only model holds in this regime.
# =============================================================================
import json, sys, argparse
import numpy as np
from scipy import stats

NAMES = ['DRL', 'Uniform', 'Exponential', 'Arithmetic']

# actor 114, corrected conversion, n=10 -- from FINDINGS_regime_dependence.md
# (raw JSON was lost when exp_data/ was replaced; these are the reported values)
REF_114 = {'DRL': 0.9061, 'Exponential': 0.9032,
           'Arithmetic': 0.8976, 'Uniform': 0.8834}
REF_114_DIFF = {'Uniform': (+0.0227, 9, 0.004),
                'Exponential': (+0.0029, 7, 0.557),
                'Arithmetic': (+0.0085, 6, 0.123)}


def load(path):
    raw = json.load(open(path))
    d = json.loads(raw) if isinstance(raw, str) else raw
    acc = {n: np.array([t[-1] for t in d[i]], float) for i, n in enumerate(NAMES)}
    pen = {}
    for off, n in zip(range(8, 12), NAMES):
        v = []
        for tr in d[off]:
            s = np.array(tr, float)
            m = np.atleast_1d(s.mean(0) if s.ndim > 1 else s)
            v.append(float((m ** 2).sum()))
        pen[n] = float(np.mean(v))
    sched = {}
    for off, n in zip(range(8, 12), NAMES):
        s = np.array(d[off][0], float)
        sched[n] = np.atleast_1d(s.mean(0) if s.ndim > 1 else s)
    return acc, pen, sched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', nargs='?',
                    default='Author/Code/exp_data/exp_data_MNIST_consistent_paired.json')
    ap.add_argument('--actor', default='612')
    a = ap.parse_args()

    acc, pen, sched = load(a.path)
    n = len(acc['DRL'])

    print(f'=== actor {a.actor}, corrected conversion, n={n} paired ===\n')
    print(f'{"arm":<13}{"mean":>9}{"sd":>9}')
    for k in NAMES:
        sd = acc[k].std(ddof=1) if n > 1 else 0.0
        print(f'{k:<13}{acc[k].mean():>9.4f}{sd:>9.4f}')

    print(f'\n{"DRL vs":<13}{"mean diff":>11}{"95% CI":>24}{"wins":>8}{"p":>9}')
    for k in NAMES[1:]:
        d = acc['DRL'] - acc[k]
        if n > 1:
            ci = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=stats.sem(d))
            p = float(stats.ttest_rel(acc['DRL'], acc[k])[1])
        else:
            ci, p = (float('nan'), float('nan')), float('nan')
        flag = ' *' if p < 0.05 else ''
        print(f'{k:<13}{d.mean():>+11.4f}   [{ci[0]:+.4f},{ci[1]:+.4f}]{int((d>0).sum()):>5}/{n:<2}{p:>9.3f}{flag}')

    base = pen['Uniform']
    print(f'\n{"arm":<13}{"penalty":>10}{"accuracy":>11}   (sorted by variance)')
    rows = sorted((pen[k] / base, acc[k].mean(), k) for k in NAMES)
    for pv, av, k in rows:
        print(f'{k:<13}{pv:>9.4f}x{av:>11.4f}')
    sp = stats.spearmanr([r[0] for r in rows], [r[1] for r in rows])
    print(f'\nSpearman(accumulated variance, accuracy) = {sp.statistic:+.3f}')
    print('  thesis reports -0.767 (drone, sigma~1.83); actor 114 gave +1.000 here')
    if sp.statistic > 0:
        print('  -> POSITIVE: more noise variance = better accuracy. Variance-only')
        print('     model inverted; consistent with a steep a_t at high sigma.')

    print('\n=== learned schedule (trial 1) ===')
    for k in NAMES:
        s = sched[k]
        print(f'  {k:<13}{s[0]:7.3f} -> {s[-1]:7.3f}   '
              f'{"FLAT" if abs(s[0]-s[-1]) < 0.05*max(abs(s[0]),1e-9) else "decreasing"}')

    print('\n=== vs stored actor-114 result (n=10, same configuration) ===')
    print(f'{"arm":<13}{"actor 114":>11}{"actor "+a.actor:>11}{"delta":>10}')
    for k in NAMES:
        print(f'{k:<13}{REF_114[k]:>11.4f}{acc[k].mean():>11.4f}{acc[k].mean()-REF_114[k]:>+10.4f}')
    print(f'\n{"DRL vs":<13}{"114 diff":>10}{"114 wins":>10}{"  |":>4}{a.actor+" diff":>10}{"wins":>8}')
    for k in NAMES[1:]:
        d = acc['DRL'] - acc[k]
        md, wn, _ = REF_114_DIFF[k]
        print(f'{k:<13}{md:>+10.4f}{wn:>8}/10   |{d.mean():>+10.4f}{int((d>0).sum()):>6}/{n}')


if __name__ == '__main__':
    main()
