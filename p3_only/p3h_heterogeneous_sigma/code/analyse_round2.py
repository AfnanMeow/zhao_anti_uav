# =============================================================================
# Round-2 verdicts, applying ../PREREG_round2.md exactly (thresholds fixed there
# before any round-2 run). Verdicts print first; tables after.
#   python analyse_round2.py            (reads ../results/p3h_{core,armG,armF,ladder}.json)
# =============================================================================
import os, json
import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
MODES = ('uniform', 'hetero', 'adaptive')


def load(tag):
    p = os.path.join(RES, 'p3h_%s.json' % tag)
    return json.load(open(p))['results'] if os.path.exists(p) else {}


CORE, ARMG, ARMF, LADDER = load('core'), load('armG'), load('armF'), load('ladder')


def get(res, prefix, full=False):
    """Records for the config whose key starts with prefix (exact key if full)."""
    for k, v in res.items():
        if (k == prefix) if full else k.startswith(prefix + '|'):
            return v
    return None


def fin(r):
    return float(np.mean([x['acc'] for x in r['rounds'][-3:]]))


def rate(recs, field='rejected', start=0):
    """TPR, FPR over trials x rounds >= start. field: 'rejected' (active defence)
    or a shadow list field."""
    tp = fp = pos = neg = 0
    for r in recs:
        for x in r['rounds'][start:]:
            rej = ({i for i, b in enumerate(x['rejected']) if b} if field == 'rejected'
                   else set(x[field]))
            for i, mal in enumerate(x['malicious']):
                pos += mal; neg += not mal
                tp += mal and i in rej; fp += (not mal) and i in rej
    return (tp / pos if pos else float('nan')), (fp / neg if neg else float('nan'))


def honest_rsd(recs, normalise=False):
    v = []
    for r in recs:
        for x in r['rounds']:
            h = [n / (s if normalise else 1) for n, s, m
                 in zip(x['norm'], x['sigma_declared'], x['malicious']) if not m]
            v.append(np.std(h) / np.mean(h))
    return float(np.mean(v))


def damage(mode, att, dfn):
    a, c = get(CORE, '%s|%s|%s|m3' % (mode, att, dfn)), get(CORE, '%s|none|none|m0' % mode)
    if not a or not c:
        return None
    n = min(len(a), len(c))
    return [fin(a[t]) - fin(c[t]) for t in range(n)]


def V(p):
    return 'MISSING' if p is None else ('HELD' if p else 'FAILED')


# ── Arm G: per-client historical baseline (post-processing) ──────────────────
def arm_g(recs, onset=5, k=3.0):
    tp = fp = pos = neg = 0
    for r in recs:
        R = r['rounds']
        W = np.array([x['norm'] for x in R[:onset]])               # onset x N
        base = np.median(W, 0)
        rsd = float(np.median(W.std(0) / W.mean(0)))
        for x in R[onset:]:
            flag = np.abs(np.array(x['norm']) / base - 1) > k * rsd
            for i, mal in enumerate(x['malicious']):
                pos += mal; neg += not mal
                tp += bool(mal and flag[i]); fp += bool((not mal) and flag[i])
    return tp / max(pos, 1), fp / max(neg, 1)


verdict = []

# P4 -- alpha-MPELM: norm detectors blind in hetero/adaptive
p4 = []
for mode in ('hetero', 'adaptive'):
    for dfn in ('norm', 'norm_sn'):
        recs = get(CORE, '%s|mpelm|%s|m3' % (mode, dfn))
        p4.append(None if not recs else rate(recs)[0] <= 0.5)
P4 = None if None in p4 else all(p4)

# P8 / P8b -- band-exploiting attacker damage under defence
def paired_diff(dfn):
    du, dh = damage('uniform', 'mpelm_scale', dfn), damage('hetero', 'mpelm_scale', dfn)
    if du is None or dh is None:
        return None
    n = min(len(du), len(dh))
    return [dh[t] - du[t] for t in range(n)]

d8, d8b = paired_diff('norm'), paired_diff('norm_sn')
P8 = None if d8 is None else (np.mean(d8) <= -0.010 and sum(x < 0 for x in d8) >= 2)
P8b = None if d8b is None else abs(np.mean(d8b)) <= 0.005

# P6 / P7 -- Arm G
g = {m: (arm_g(get(ARMG, '%s|rmd|none|m3' % m)) if get(ARMG, '%s|rmd|none|m3' % m) else None)
     for m in MODES}
P6 = None if g['hetero'] is None else (g['hetero'][0] >= 0.90 and g['hetero'][1] <= 0.10)
P7 = (None if g['hetero'] is None or g['adaptive'] is None else
      (g['hetero'][0] - g['adaptive'][0] >= 0.30 or g['adaptive'][1] - g['hetero'][1] >= 0.20))

# P5a / P5b -- sigma lying
f09 = get(ARMF, 'adaptive|rmd|norm_sn|m3|lie0.9|cap0', full=True)
ft0 = get(ARMF, 'adaptive|mpelm_scale|norm_sn|m3|lie1|cap0|track', full=True)
ft1 = get(ARMF, 'adaptive|mpelm_scale|norm_sn|m3|lie1|cap1|track', full=True)
P5a = None if not f09 else rate(f09)[0] <= 0.5
P5b = (None if not (ft0 and ft1) else
       (rate(ft0)[0] <= 0.5 and rate(ft1)[0] - rate(ft0)[0] >= 0.30))

print('=' * 78)
print('ROUND-2 VERDICTS (rules: PREREG_round2.md)')
print('  P4  alpha-MPELM invisible to Norm & Norm-sn (hetero, adaptive)  : %s' % V(P4))
print('  P5a lie=0.9 defeats Norm-sn (adaptive, no cap)                  : %s' % V(P5a))
print('  P5b tracking liar defeats Norm-sn; cap restores by >=0.30       : %s' % V(P5b))
print('  P6  Arm G works under hetero (TPR>=0.90, FPR<=0.10)              : %s' % V(P6))
print('  P7  Arm G degrades under adaptive                                : %s' % V(P7))
print('  P8  band attacker: extra damage under Norm-raw (<=-0.010, 2/3)   : %s' % V(P8))
print('  P8b Norm-sn removes the extra damage (|diff|<=0.005)             : %s' % V(P8b))
print('=' * 78, '\n')

# ── detail ──────────────────────────────────────────────────────────────────
print('P8 paired damage diff hetero-uniform, Norm-raw :', d8 and np.round(d8, 4).tolist(),
      ' mean %+.4f' % np.mean(d8) if d8 else '')
print('P8b                                   Norm-sn  :', d8b and np.round(d8b, 4).tolist(),
      ' mean %+.4f' % np.mean(d8b) if d8b else '')
print('\nArm G (per-client history, onset 5) TPR / FPR, vs cross-client detectors on the same runs:')
for m in MODES:
    recs = get(ARMG, '%s|rmd|none|m3' % m)
    if recs:
        a, b = rate(recs, 'shadow_norm_rej', 5), rate(recs, 'shadow_norm_sn_rej', 5)
        print('  %-9s ArmG %.2f / %.2f | Norm-raw %.2f / %.2f | Norm-sn %.2f / %.2f'
              % (m, *g[m], *a, *b))

print('\nDefence grid (core): IoU.25, damage vs clean, TPR, FPR, honest-norm rsd raw / after /sigma')
print('%-44s %7s %8s %5s %5s %8s %8s' % ('config', 'acc', 'damage', 'TPR', 'FPR', 'rsd', 'rsd/sig'))
for k in sorted(CORE):
    recs = CORE[k]
    mode, att, dfn, m = k.split('|')[:4]
    d = damage(mode, att, dfn) if att != 'none' else None
    t, f = rate(recs)
    print('%-44s %7.4f %8s %5s %5.2f %7.2f%% %7.2f%%'
          % (k, np.mean([fin(r) for r in recs]), '%+.4f' % np.mean(d) if d else '-',
             '%.2f' % t if att != 'none' else '-', f, 100 * honest_rsd(recs),
             100 * honest_rsd(recs, True)))

print('\nArm F (adaptive, Norm-sn):')
for k in sorted(ARMF):
    t, f = rate(ARMF[k])
    print('  %-52s acc %.4f  TPR %.2f  FPR %.2f' % (k, np.mean([fin(r) for r in ARMF[k]]), t, f))

print('\nSigma ladder (RMD, Norm-raw active; Norm-sn shadow):')
for k in sorted(LADDER, key=lambda s: float(s.split('|')[0].split(':')[1])):
    recs = LADDER[k]
    sg = np.mean([np.std(r['rounds'][0]['sigma_true']) / np.mean(r['rounds'][0]['sigma_true'])
                  for r in recs])
    a, b = rate(recs, 'shadow_norm_rej'), rate(recs, 'shadow_norm_sn_rej')
    print('  %-10s sigma rsd %5.1f%% | honest-norm rsd %6.2f%% | Norm-raw TPR %.2f FPR %.2f'
          ' | Norm-sn TPR %.2f FPR %.2f'
          % (k.split('|')[0], 100 * sg, 100 * honest_rsd(recs), *a, *b))
