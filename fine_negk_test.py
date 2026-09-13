# =============================================================================
# THE SWEET SPOT: can a MILD front-load beat uniform?
#
# Two effects govern allocation here, and the earlier sweeps confounded them:
#
#  (1) VARIANCE PENALTY (dominant). Noise accumulated into the model over T
#      rounds ~ sum_t sigma_t^2 = sum_t 2*el/rho_t. Clipping bounds each round's
#      SIGNAL contribution at C regardless of sigma_t, so allocation changes only
#      the noise term. Minimising sum(1/rho_t) s.t. sum(rho_t)=rho gives
#      rho_t = const by Jensen (1/x convex) -- uniform is the analytic optimum.
#      This is why k=+2 (10.2x penalty) and k=+3 (21.0x) lose so badly.
#
#  (2) TIMING BENEFIT (secondary). k=-1 and k=+1 carry near-identical penalties
#      (1.99x vs 1.98x) yet measured +0.0023 and -0.0045. At equal noise cost,
#      clean EARLY rounds beat clean late ones -- the head is learned from
#      scratch, so early corruption never recovers.
#
# The headroom sweep jumped k=-1 -> k=0 and never sampled between, where (2) is
# bought at almost no (1):
#      k=-0.25 -> 1.04x penalty      k=-0.75 -> 1.47x
#      k=-0.50 -> 1.18x              k=-1.00 -> 1.99x
# If the timing effect is real, the optimum is a mild front-load, and a learned
# policy searching sigma_t continuously would find it while uniform cannot.
#
# Paired on init + common random numbers. Powered for a small effect: the k=-1
# arm had paired sd ~0.0076, so n=16 gives SE ~0.0019 and resolves ~+0.005.
# =============================================================================
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_DB_sweep as G

DATA = './fl_workdirs/clear'
PUBLIC = './fl_workdirs/clear/public_pretrained.pt'
OUT_PATH = './exp_data/fine_negk.json'

CR_TOTAL, EPOCH_LOCAL, B, LR = 25, 2, 128, 0.003
RHO_FIXED = 30.0
SIGMA_MIN, SIGMA_MAX = 1.0, 20.0
SCORE_LAST = 6
K_VALUES = [0.0, -0.25, -0.5, -0.75, -1.0]     # 0.0 = uniform control
N_TRIALS = 16
SEED_BASE = 550000


def sigma_schedule(k):
    w = np.array([(t + 1) ** k for t in range(CR_TOTAL)], dtype=float)
    return np.clip(np.sqrt(2 * EPOCH_LOCAL / (w / w.sum() * RHO_FIXED)), SIGMA_MIN, SIGMA_MAX)


def run_arm(init, feats, tgts, vf, vo, vb, sigmas, seed):
    torch.manual_seed(seed)                       # common random numbers
    st = {k: v.clone() for k, v in init.items()}
    tail = []
    for r in range(CR_TOTAL):
        st = G.fedavg([G.client_round(st, feats[i], tgts[i][0], tgts[i][1],
                                      float(sigmas[r]), True, B, LR)
                       for i in range(G.N_CLIENTS)])
        if r >= CR_TOTAL - SCORE_LAST:
            tail.append(G.evaluate(st, vf, vo, vb))
    return float(np.mean(tail))


if __name__ == '__main__':
    G.DATA = DATA
    G.TGT = DATA
    print('loading cached features...', flush=True)
    feats, tgts, vf, (vo, vb) = G.load_cut()
    scheds = {k: sigma_schedule(k) for k in K_VALUES}
    b = float(np.sum(scheds[0.0] ** 2))
    print('rho=%.1f  B=%d  CRN=on  n/client=%d\n' % (RHO_FIXED, B, feats[0].shape[0]), flush=True)
    for k in K_VALUES:
        s = scheds[k]
        print('  k=%+.2f  sigma r1=%.2f r25=%.2f | var penalty %.2fx | rho spent %.1f/%.1f'
              % (k, s[0], s[-1], float(np.sum(s ** 2)) / b,
                 float(np.sum(2 * EPOCH_LOCAL / s ** 2)), RHO_FIXED), flush=True)
    print('\n%5s | %s' % ('trial', '  '.join('k=%-7.2f' % k for k in K_VALUES)), flush=True)

    res = {str(k): [] for k in K_VALUES}
    t0 = time.time()
    try:
        from scipy import stats
    except Exception:
        stats = None
    for t in range(N_TRIALS):
        seed = SEED_BASE + t
        init = G.make_init('public_tail', seed, PUBLIC)
        row = []
        for k in K_VALUES:
            v = run_arm(init, feats, tgts, vf, vo, vb, scheds[k], seed)
            res[str(k)].append(v)
            row.append(v)
        print('%5d | %s   [%.0fm]'
              % (t + 1, '  '.join('%-9.4f' % v for v in row), (time.time() - t0) / 60), flush=True)
        json.dump({'rho': RHO_FIXED, 'seed_base': SEED_BASE, 'crn': True,
                   'k_values': K_VALUES, 'results': res}, open(OUT_PATH, 'w'), indent=1)
        if t >= 1:
            base = np.array(res['0.0'])
            parts = []
            for k in K_VALUES:
                if k == 0.0:
                    continue
                d = np.array(res[str(k)]) - base
                pv = stats.ttest_rel(np.array(res[str(k)]), base).pvalue if (stats and len(d) > 1) else float('nan')
                parts.append('k%+.2f %+.4f(%d/%d,p=%.3f)' % (k, d.mean(), int((d > 0).sum()), len(d), pv))
            print('        ' + '  '.join(parts), flush=True)

    base = np.array(res['0.0'])
    print('\n=== %d paired trials vs uniform (mean=%.4f) ===' % (N_TRIALS, base.mean()), flush=True)
    print('%7s %11s %9s %22s %8s %9s' % ('k', 'mean diff', 'SE', '95% CI', 'wins', 'p'), flush=True)
    for k in K_VALUES:
        if k == 0.0:
            continue
        x = np.array(res[str(k)]); d = x - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        if stats:
            lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=se)
            p = stats.ttest_rel(x, base).pvalue
        else:
            lo = hi = p = float('nan')
        print('%7.2f %+11.4f %9.4f  [%+.4f,%+.4f] %5d/%-3d %9.4f'
              % (k, d.mean(), se, lo, hi, int((d > 0).sum()), len(d), p), flush=True)
    print('FINE NEGK DONE', flush=True)
