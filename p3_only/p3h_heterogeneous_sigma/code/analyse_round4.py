# =============================================================================
# Round-4 verdicts, applying ../PREREG_round4.md exactly. Verdicts print first.
# Clean reference = hetero|none|none from p3h_core.json (same seeds and split).
# =============================================================================
import os, json
import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
R4 = json.load(open(os.path.join(RES, 'p3h_round4.json')))['results']
CLEAN = json.load(open(os.path.join(RES, 'p3h_core.json')))['results']['hetero|none|none|m0|lie1|cap0']


def key(att, dfn, onset):
    return 'hetero|%s|%s|m3|lie1|cap0%s' % (att, dfn, '|on%d' % onset if onset else '')


def fin(r):
    return float(np.mean([x['acc'] for x in r['rounds'][-3:]]))


def damage(att, dfn, onset):
    a = R4.get(key(att, dfn, onset))
    if not a:
        return None
    return [fin(a[t]) - fin(CLEAN[t]) for t in range(min(len(a), len(CLEAN)))]


def rate(recs):
    tp = fp = pos = neg = 0
    for r in recs:
        for x in r['rounds']:
            if not any(x['malicious']):
                continue                       # only rounds with active attackers
            for i, mal in enumerate(x['malicious']):
                rj = x['rejected'][i]
                pos += mal; neg += not mal; tp += mal and rj; fp += (not mal) and rj
    return tp / max(pos, 1), fp / max(neg, 1)


def halves(att, onset):
    dc, dn = damage(att, 'cos', onset), damage(att, 'none', onset)
    return None if dc is None or dn is None else abs(np.mean(dc)) <= 0.5 * abs(np.mean(dn))


def V(p):
    return 'MISSING' if p is None else ('HELD' if p else 'FAILED')


r14 = R4.get(key('mpelm_fixed', 'cos', 0))
P14 = None if not r14 else (rate(r14)[0] >= 0.70 and rate(r14)[1] <= 0.10)
P15, P16, P17 = halves('mpelm_fixed', 0), halves('mpelm', 5), halves('mpelm_fixed', 5)

print('=' * 78)
print('ROUND-4 VERDICTS (rules: PREREG_round4.md)')
print('  P14 cos catches non-backing-off flip (TPR>=0.70, FPR<=0.10)   : %s' % V(P14))
print('  P15 cos halves damage, non-backing-off flip, onset 0          : %s' % V(P15))
print('  P16 cos halves damage, adaptive alpha-MPELM, onset 5          : %s' % V(P16))
print('  P17 cos halves damage, non-backing-off flip, onset 5          : %s' % V(P17))
if None not in (P15, P16, P17):
    print('  -> %s' % ('P15-P17 all FAILED: cosine only deters an early, backing-off attacker; '
                       'direction attacks under DP remain open.' if not (P15 or P16 or P17) else
                       'at least one held: report as a partial defence, with its conditions.'))
print('=' * 78, '\n')

print('%-10s %-6s %-9s %10s %18s %6s %6s' % ('attack', 'onset', 'defence', 'damage', 'per trial', 'TPR', 'FPR'))
for att, on in (('mpelm_fixed', 0), ('mpelm', 5), ('mpelm_fixed', 5)):
    for dfn in ('none', 'norm_sn', 'cos', 'cos_sn'):
        d = damage(att, dfn, on)
        r = R4.get(key(att, dfn, on))
        if d is None:
            continue
        t, f = rate(r)
        print('%-10s %-6d %-9s %+10.4f %18s %6s %6.2f' % (att, on, dfn, np.mean(d),
              ' '.join('%+.3f' % x for x in d), '%.2f' % t if dfn != 'none' else '-', f))

print('\nCosine detector, caught attackers per round (mean over trials):')
for att, on in (('mpelm_fixed', 0), ('mpelm', 5), ('mpelm_fixed', 5)):
    r = R4.get(key(att, 'cos', on))
    if r:
        tp = np.mean([[x['tp'] for x in t['rounds']] for t in r], 0)
        print('  %-11s onset %d: %s' % (att, on, ' '.join('%.1f' % v for v in tp)))
