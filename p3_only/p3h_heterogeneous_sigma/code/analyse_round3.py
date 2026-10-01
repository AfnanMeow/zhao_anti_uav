# =============================================================================
# Round-3 verdicts, applying ../PREREG_round3.md exactly. Verdicts print first.
#   python analyse_round3.py
# =============================================================================
import os, json
import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
MODES = ('uniform', 'hetero', 'adaptive')


def load(tag):
    p = os.path.join(RES, 'p3h_%s.json' % tag)
    return json.load(open(p))['results'] if os.path.exists(p) else {}


CORE, ARMG, LADDER, MIXS = load('core'), load('armG'), load('ladder'), load('mixsplit')


def get(res, mode, att, dfn, extra=''):
    k = '%s|%s|%s|m%d|lie1|cap0%s' % (mode, att, dfn, 0 if att == 'none' else 3, extra)
    return res.get(k)


def fin(r):
    return float(np.mean([x['acc'] for x in r['rounds'][-3:]]))


def rate(recs, field='rejected', start=0):
    tp = fp = pos = neg = 0
    for r in recs:
        for x in r['rounds'][start:]:
            rej = ({i for i, b in enumerate(x['rejected']) if b} if field == 'rejected'
                   else set(x[field] or []))
            for i, mal in enumerate(x['malicious']):
                pos += mal; neg += not mal
                tp += mal and i in rej; fp += (not mal) and i in rej
    return (tp / pos if pos else float('nan')), (fp / neg if neg else float('nan'))


def damage(res, mode, att, dfn, clean_res=None, extra=''):
    a = get(res, mode, att, dfn, extra)
    c = get(clean_res if clean_res is not None else res, mode, 'none', 'none')
    if not a or not c:
        return None
    n = min(len(a), len(c))
    return float(np.mean([fin(a[t]) - fin(c[t]) for t in range(n)]))


def V(p):
    return 'MISSING' if p is None else ('HELD' if p else 'FAILED')


def allm(f):
    v = [f(m) for m in MODES]
    return None if None in v else all(v)


# P9 -- cos catches alpha-MPELM
def p9(m):
    r = get(CORE, m, 'mpelm', 'cos')
    if not r:
        return None
    t, f = rate(r)
    return t >= 0.70 and f <= 0.10

# P10 -- cos halves alpha-MPELM damage
def p10(m):
    dc, dn = damage(CORE, m, 'mpelm', 'cos'), damage(CORE, m, 'mpelm', 'none')
    return None if dc is None or dn is None else abs(dc) <= 0.5 * abs(dn)

# P11 -- cos costs <= 0.010 on clean runs
def p11(m):
    a, c = get(CORE, m, 'none', 'cos'), get(CORE, m, 'none', 'none')
    if not a or not c:
        return None
    n = min(len(a), len(c))
    return np.mean([fin(c[t]) - fin(a[t]) for t in range(n)]) <= 0.010

# P12 -- cos_sn no regression vs best round-2 norm defence on the band attacker
def p12(m):
    d = damage(CORE, m, 'mpelm_scale', 'cos_sn')
    ref = [damage(CORE, m, 'mpelm_scale', x) for x in ('norm', 'norm_sn', 'mix', 'mix_sn')]
    ref = [x for x in ref if x is not None]
    return None if d is None or not ref else abs(d) <= min(abs(x) for x in ref) + 0.005

# P13 -- median costs >= 0.050 with no attack
def p13(m):
    a, c = get(CORE, m, 'none', 'median'), get(CORE, m, 'none', 'none')
    if not a or not c:
        return None
    n = min(len(a), len(c))
    return np.mean([fin(c[t]) - fin(a[t]) for t in range(n)]) >= 0.050


# R3-5 decision
mix_clean = MIXS.get('uniform|none|mix|m0|lie1|cap0')
if mix_clean:
    _, acc_fpr = rate(mix_clean, 'accuracy_rej')
    r35 = ('LEAK (positional split causes the honest rejections)' if acc_fpr >= 0.10 else
           'NOT THE LEAK (trunk / accuracy level)' if acc_fpr < 0.02 else 'INCONCLUSIVE')
else:
    acc_fpr, r35 = float('nan'), 'MISSING'

print('=' * 78)
print('ROUND-3 VERDICTS (rules: PREREG_round3.md)')
print('  P9  cos catches alpha-MPELM (TPR>=0.70, FPR<=0.10), all sigma : %s' % V(allm(p9)))
print('  P10 cos halves alpha-MPELM damage, all sigma                  : %s' % V(allm(p10)))
print('  P11 cos costs <= 0.010 IoU on clean runs, all sigma           : %s' % V(allm(p11)))
print('  P12 cos_sn no worse than best norm defence vs band attacker   : %s' % V(allm(p12)))
print('  P13 median costs >= 0.050 with no attack (P3 replication)     : %s' % V(allm(p13)))
print('  R3-5 Mix discrepancy: accuracy-detector honest FPR on positional split = %.3f -> %s'
      % (acc_fpr, r35))
print('=' * 78, '\n')

# ── detail ──────────────────────────────────────────────────────────────────
print('Per-setting detail (damage = IoU.25 vs clean run, same sigma setting and trial):')
for att in ('none', 'rmd', 'mpelm', 'mpelm_scale'):
    print('\n  attack = %s' % att)
    for dfn in ('none', 'norm', 'norm_sn', 'mix', 'mix_sn', 'cos', 'cos_sn', 'krum', 'median', 'trimmed'):
        cells = []
        for m in MODES:
            r = get(CORE, m, att, dfn)
            if not r:
                cells.append('%26s' % '-'); continue
            if att == 'none':
                c = get(CORE, m, 'none', 'none')
                d = np.mean([fin(r[t]) - fin(c[t]) for t in range(min(len(r), len(c)))])
                t, f = rate(r)
                cells.append('%+.4f   FPR %.2f      ' % (d, f))
            else:
                d = damage(CORE, m, att, dfn)
                t, f = rate(r)
                cells.append('%+.4f TPR %.2f FPR %.2f' % (d, t, f) if dfn != 'none'
                             else '%+.4f                ' % d)
        print('    %-8s %s' % (dfn, ' | '.join(cells)))
print('    columns: %s' % ' | '.join(MODES))

print('\nCosine statistic, honest vs malicious (mean c_i over all rounds, cos runs):')
for m in MODES:
    for att in ('rmd', 'mpelm', 'mpelm_scale'):
        r = get(CORE, m, att, 'cos')
        if r:
            h = [c for x in (y for rr in r for y in rr['rounds']) for c, b in zip(x['cos'], x['malicious']) if not b]
            a = [c for x in (y for rr in r for y in rr['rounds']) for c, b in zip(x['cos'], x['malicious']) if b]
            print('  %-9s %-12s honest %.4f +- %.4f | malicious %.4f +- %.4f'
                  % (m, att, np.mean(h), np.std(h), np.mean(a), np.std(a)))

print('\nR3-4 Arm G vs clean-window length (hetero):')
for k in sorted(ARMG):
    if k.startswith('hetero|'):
        on = int(k.split('|on')[1].split('|')[0])
        recs = ARMG[k]
        tp = fp = pos = neg = 0
        for r in recs:
            W = np.array([x['norm'] for x in r['rounds'][:on]])
            base, rsd = np.median(W, 0), float(np.median(W.std(0) / W.mean(0)))
            for x in r['rounds'][on:]:
                flag = np.abs(np.array(x['norm']) / base - 1) > 3 * rsd
                for i, mal in enumerate(x['malicious']):
                    pos += mal; neg += not mal
                    tp += bool(mal and flag[i]); fp += bool((not mal) and flag[i])
        print('  %-45s onset %d: TPR %.2f FPR %.2f' % (k, on, tp / max(pos, 1), fp / max(neg, 1)))

print('\nR3-3 damage vs sigma spread (band attacker; damage vs clean at the same level):')
lv = sorted({k.split('|')[0] for k in LADDER}, key=lambda s: float(s.split(':')[1]))
for l in lv:
    row = []
    for dfn in ('none', 'norm', 'norm_sn'):
        d = damage(LADDER, l, 'mpelm_scale', dfn)
        row.append('%s %s' % (dfn, '%+.4f' % d if d is not None else '-'))
    print('  %-12s %s' % (l, ' | '.join(row)))

print('\nR3-5 detail (positional split, uniform):')
for k, v in sorted(MIXS.items()):
    print('  %-40s IoU %.4f  active TPR/FPR %.2f/%.2f  accuracy-detector alone TPR/FPR %.2f/%.2f'
          % (k, np.mean([fin(r) for r in v]), *rate(v), *rate(v, 'accuracy_rej')))
