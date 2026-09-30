# =============================================================================
# Does ANY fixed allocation beat uniform? (broader than the power family)
#
# headroom_sweep.py tested rho_t ~ (t+1)^k and found uniform optimal, with the
# ordering monotone in k and INVERTED relative to the source paper: the only arm
# above uniform was k=-1 (low noise early, rising later). Two gaps in that test:
#
#   1. It was dense on the paper's side (k>0) and sampled the winning side once.
#   2. Pushing further negative does not work inside the paper's sigma in [1,20]
#      convention -- k=-2/-3/-4 UNDERSPEND the budget (14.6 / 9.1 / 6.4 of 30)
#      because concentrating budget early needs sigma<1, which the floor forbids.
#      Those arms are handicapped, not informative.
#
# So concentration is tested here with explicit BANG-BANG schedules that spend
# the budget exactly: m rounds at sigma=1 (the cleanest the convention allows)
# and the remaining budget spread over the rest. m=7 is the most extreme split
# that still spends rho=30 exactly (7*4 + 18*0.111 = 30.0).
#
#   FRONT-clean : sigma 1.0 for rounds 1-7,  then 6.00   <- concentrate EARLY
#   BACK-clean  : sigma 6.00 for rounds 1-18, then 1.0   <- concentrate LATE (paper's side)
#
# Plus random draws from the budget simplex (Dirichlet), which explore shapes
# outside any parametric family. If none of these beats uniform either, the
# "uniform is optimal" claim covers all fixed allocations, not just power ones.
#
# Paired + common random numbers throughout (CRN cut sd 0.0466 -> 0.0171).
# =============================================================================
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_DB_sweep as G

DATA = './fl_workdirs/clear'
PUBLIC = './fl_workdirs/clear/public_pretrained.pt'
OUT_PATH = './exp_data/allocation_bound.json'

CR_TOTAL, EPOCH_LOCAL, B, LR = 25, 2, 128, 0.003
RHO_FIXED = 30.0
SIGMA_MIN, SIGMA_MAX = 1.0, 20.0
SCORE_LAST = 6
N_TRIALS = 12
SEED_BASE = 330000            # disjoint from 110000 (n=30) and 220000 (headroom)


def clip(s):
    return np.clip(s, SIGMA_MIN, SIGMA_MAX)


def sched_power(k):
    w = np.array([(t + 1) ** k for t in range(CR_TOTAL)], dtype=float)
    return clip(np.sqrt(2 * EPOCH_LOCAL / (w / w.sum() * RHO_FIXED)))


def sched_bang(m, front):
    """m rounds at sigma=SIGMA_MIN, remaining budget spread over the rest."""
    rem = RHO_FIXED - (2 * EPOCH_LOCAL / SIGMA_MIN ** 2) * m
    s_rest = np.sqrt(2 * EPOCH_LOCAL / (rem / (CR_TOTAL - m)))
    out = np.full(CR_TOTAL, s_rest)
    if front:
        out[:m] = SIGMA_MIN
    else:
        out[-m:] = SIGMA_MIN
    return clip(out)


def sched_random(seed):
    rho = np.random.default_rng(seed).dirichlet(np.ones(CR_TOTAL) * 0.5) * RHO_FIXED
    return clip(np.sqrt(2 * EPOCH_LOCAL / np.maximum(rho, 1e-9)))


ARMS = [
    ('uniform',     sched_power(0.0)),
    ('k=-1',        sched_power(-1.0)),
    ('front-clean', sched_bang(7, True)),
    ('back-clean',  sched_bang(7, False)),
    ('rand-A',      sched_random(7)),
    ('rand-B',      sched_random(8)),
]


def run_arm(init, feats, tgts, vf, vo, vb, sigmas, seed):
    torch.manual_seed(seed)                     # common random numbers
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
    print('rho=%.1f  %d rounds  B=%d  CRN=on  n/client=%d\n'
          % (RHO_FIXED, CR_TOTAL, B, feats[0].shape[0]), flush=True)
    for name, s in ARMS:
        print('  %-12s sigma r1=%5.2f r13=%5.2f r25=%5.2f | mean=%5.2f  rho spent=%.1f/%.1f'
              % (name, s[0], s[12], s[-1], s.mean(),
                 float(np.sum(2 * EPOCH_LOCAL / s ** 2)), RHO_FIXED), flush=True)
    print('\n%5s | %s' % ('trial', '  '.join('%-11s' % n for n, _ in ARMS)), flush=True)

    res = {n: [] for n, _ in ARMS}
    t0 = time.time()
    for t in range(N_TRIALS):
        seed = SEED_BASE + t
        init = G.make_init('public_tail', seed, PUBLIC)
        row = []
        for name, s in ARMS:
            v = run_arm(init, feats, tgts, vf, vo, vb, s, seed)
            res[name].append(v)
            row.append(v)
        print('%5d | %s   [%.0fm]'
              % (t + 1, '  '.join('%-11.4f' % v for v in row), (time.time() - t0) / 60),
              flush=True)
        json.dump({'rho': RHO_FIXED, 'seed_base': SEED_BASE, 'crn': True,
                   'schedules': {n: s.tolist() for n, s in ARMS}, 'results': res},
                  open(OUT_PATH, 'w'), indent=1)
        if t >= 1:
            base = np.array(res['uniform'])
            print('        vs uniform: ' + '  '.join(
                '%s %+.4f(%d/%d)' % (n, (np.array(res[n]) - base).mean(),
                                     int((np.array(res[n]) - base > 0).sum()), len(base))
                for n, _ in ARMS if n != 'uniform'), flush=True)

    base = np.array(res['uniform'])
    print('\n=== %d paired trials vs uniform (mean=%.4f) ===' % (N_TRIALS, base.mean()), flush=True)
    print('%-12s %10s %9s %8s' % ('arm', 'mean diff', 'SE', 'wins'), flush=True)
    for name, _ in ARMS:
        if name == 'uniform':
            continue
        d = np.array(res[name]) - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        print('%-12s %+10.4f %9.4f %5d/%-3d' % (name, d.mean(), se, int((d > 0).sum()), len(d)),
              flush=True)
    print('ALLOCATION BOUND DONE', flush=True)
