# =============================================================================
# Verdict on the decisive subset (results/p3h_decisive.json), predictions 1-3
# of fresh_beginning.md §6, stated before any interpretation.
#
#   1. hetero Norm-raw TPR vs RMD drops well below uniform's (~1.00)
#   2. hetero undefended damage (rmd|none - none|none, paired by trial)
#      exceeds the uniform-sigma worst case of -0.017 IoU.25
#   3. hetero Norm-sigma-normalised restores TPR toward 1.00
#
# "Well below" / "toward" are not numerically fixed in the plan. The thresholds
# used here are printed with the verdict so they can be challenged: prediction 1
# holds if TPR <= 0.5, prediction 3 if TPR >= 0.9.
# =============================================================================
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'results', 'p3h_decisive.json')
REF_DAMAGE = -0.017
P1_MAX_TPR, P3_MIN_TPR = 0.5, 0.9

d = json.load(open(PATH))
res, meta = d['results'], d['meta']


def get(mode, att, dfn, m):
    for k, v in res.items():
        if k.startswith('%s|%s|%s|m%d|' % (mode, att, dfn, m)):
            return v
    return None


def fin(r):                       # final accuracy, as P3: mean of last 3 rounds
    return float(np.mean([x['acc'] for x in r['rounds'][-3:]]))


def rate(recs, key):
    """TPR, FPR of a rejection list/flag field over all trials x rounds."""
    tp = fp = pos = neg = 0
    for r in recs:
        for x in r['rounds']:
            rej = set(x[key]) if key != 'rejected' else {i for i, b in enumerate(x['rejected']) if b}
            for i, mal in enumerate(x['malicious']):
                pos += mal; neg += not mal
                tp += mal and i in rej; fp += (not mal) and i in rej
    return tp / max(pos, 1), fp / max(neg, 1)


def honest_rsd(recs, normalise):
    out = []
    for r in recs:
        for x in r['rounds']:
            h = [n / (s if normalise else 1.0) for n, s, mal
                 in zip(x['norm'], x['sigma_declared'], x['malicious']) if not mal]
            out.append(np.std(h) / np.mean(h))
    return float(np.mean(out))


print('file   :', os.path.abspath(PATH))
print('machine: %s  %s  torch %s  power limit %s W' % (meta['machine'], meta['gpu'], meta['torch'],
                                                        meta['power_limit_w']))
print('split  : detector %s (%d frames) | report %s (%d frames)'
      % (meta['split']['detector_sessions'], meta['split']['n_det'],
         meta['split']['report_sessions'], meta['split']['n_rep']))
print('trials per config:', {k: len(v) for k, v in res.items()}, '\n')

tpr = {}
for mode, dfn in [('uniform', 'norm'), ('hetero', 'norm'), ('hetero', 'norm_sn')]:
    recs = get(mode, 'rmd', dfn, 3)
    tpr[(mode, dfn)] = rate(recs, 'rejected') if recs else (np.nan, np.nan)

dmg, rel = {}, {}
for mode in ('uniform', 'hetero'):
    a, c = get(mode, 'rmd', 'none', 3), get(mode, 'none', 'none', 0)
    if a and c:
        n = min(len(a), len(c))
        dmg[mode] = [fin(a[t]) - fin(c[t]) for t in range(n)]
        rel[mode] = [(fin(a[t]) - fin(c[t])) / fin(c[t]) for t in range(n)]

# ── verdict first ───────────────────────────────────────────────────────────
print('=' * 72)
print('VERDICT (thresholds: P1 TPR <= %.2f, P3 TPR >= %.2f, P2 damage < %.3f)'
      % (P1_MAX_TPR, P3_MIN_TPR, REF_DAMAGE))
t_u, t_h, t_s = tpr[('uniform', 'norm')][0], tpr[('hetero', 'norm')][0], tpr[('hetero', 'norm_sn')][0]
p1 = None if np.isnan(t_h) else t_h <= P1_MAX_TPR
p2 = None if 'hetero' not in dmg else np.mean(dmg['hetero']) < REF_DAMAGE
p3 = None if np.isnan(t_s) else t_s >= P3_MIN_TPR
V = lambda p: 'MISSING' if p is None else ('HELD' if p else 'FAILED')
print('  Prediction 1 (channel opens)      : %s   Norm-raw TPR uniform %.2f -> hetero %.2f'
      % (V(p1), t_u, t_h))
print('  Prediction 2 (damage > 0.017)     : %s   hetero damage %s'
      % (V(p2),
         '%+.4f' % np.mean(dmg['hetero']) if 'hetero' in dmg else 'n/a'))
print('  Prediction 3 (sigma-norm restores): %s   Norm-sn TPR hetero %.2f'
      % (V(p3), t_s))
if p1 is None or p2 is None:
    print('  -> Data for prediction 1 or 2 is missing: no verdict.')
elif not (p1 and p2):
    print('  -> Prediction 1 or 2 failed: per the pre-registered rule the reframe FAILS;'
          ' do not run the full grid or rewrite the abstract.')
else:
    print('  -> Predictions 1 and 2 held: proceed to the full grid.')
print('=' * 72, '\n')

# ── detail ──────────────────────────────────────────────────────────────────
print('%-36s %8s %6s %6s %9s %9s' % ('config', 'acc', 'TPR', 'FPR', 'rsd raw', 'rsd /sig'))
for k, recs in res.items():
    m = int(k.split('|')[3][1:])
    t, f = rate(recs, 'rejected')
    print('%-36s %8.4f %6s %6.2f %8.3f%% %8.3f%%'
          % (k, np.mean([fin(r) for r in recs]), '%.2f' % t if m else '-', f,
             100 * honest_rsd(recs, False), 100 * honest_rsd(recs, True)))

print('\nundefended damage per trial (rmd|none - none|none), ref %.3f:' % REF_DAMAGE)
for mode, v in dmg.items():
    print('  %-8s %s  mean %+.4f  (relative %+.1f%%)'
          % (mode, ' '.join('%+.4f' % x for x in v), np.mean(v), 100 * np.mean(rel[mode])))
print('  original P3 reference: -0.017 absolute at a ~0.65 baseline = -2.7% relative.\n'
      '  This cache sits at a different accuracy level, so compare the RELATIVE figure too.')

print('\nshadow detector rates on the SAME updates (every RMD run, whatever defence was active):')
for k, recs in res.items():
    if '|rmd|' in k:
        (a1, b1), (a2, b2) = rate(recs, 'shadow_norm_rej'), rate(recs, 'shadow_norm_sn_rej')
        print('  %-36s Norm-raw TPR %.2f FPR %.2f | Norm-sn TPR %.2f FPR %.2f' % (k, a1, b1, a2, b2))
