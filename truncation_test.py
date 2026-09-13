# =============================================================================
# EARLY STOPPING AS BUDGET ALLOCATION -- the one strategy inside the agent's
# reach that none of the previous sweeps could express.
#
# headroom_sweep.py and allocation_bound.py both searched schedules that spend
# something in all 25 rounds: rho_t ~ (t+1)^k, bang-bang, Dirichlet. None beat
# uniform. But the power family CANNOT zero out a round, whereas the DDPG agent
# CAN -- pushing sigma to sigma_max late is effectively "stop training", and a
# round that is not trained costs essentially no budget. So truncation lies
# inside the paper's own action space and outside everything tested so far.
#
# It is also the strategy the trajectories keep pointing at: accuracy peaks
# early and then decays under continued noisy updates (clearly visible in
# knee_sigma_sweep.json, e.g. sigma=1024 running 0.358 -> 0.414 -> 0.265 -> 0.135).
# Spending the same rho over fewer rounds buys a LOWER sigma in each of them:
#
#   T=25 (uniform) sigma=1.826      T=12  sigma=1.265
#   T=20           sigma=1.633      T=10  sigma=1.155
#   T=15           sigma=1.414      T= 8  sigma=1.033
#
# Every arm spends exactly rho=30.0, and every sigma stays above the sigma>=1
# floor, so no arm is handicapped by clipping. Paired on init + common random
# numbers, as in headroom_sweep.py.
#
# Score = mean IoU.25 over the last 6 TRAINED rounds, i.e. the model actually
# deployed. Truncated arms are cheaper to run, so this sweep is fast.
# =============================================================================
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_DB_sweep as G

DATA = './fl_workdirs/clear'
PUBLIC = './fl_workdirs/clear/public_pretrained.pt'
OUT_PATH = './exp_data/truncation_test.json'

EPOCH_LOCAL, B, LR = 2, 128, 0.003
RHO_FIXED = 30.0
SCORE_LAST = 6
T_VALUES = [8, 10, 12, 15, 20, 25]        # 25 = uniform baseline
N_TRIALS = 12
SEED_BASE = 440000


def sigma_for(T):
    return float(np.sqrt(2 * EPOCH_LOCAL / (RHO_FIXED / T)))


def run_arm(init, feats, tgts, vf, vo, vb, T, sigma, seed):
    torch.manual_seed(seed)                       # common random numbers
    st = {k: v.clone() for k, v in init.items()}
    tail = []
    for r in range(T):
        st = G.fedavg([G.client_round(st, feats[i], tgts[i][0], tgts[i][1],
                                      sigma, True, B, LR)
                       for i in range(G.N_CLIENTS)])
        if r >= T - SCORE_LAST:
            tail.append(G.evaluate(st, vf, vo, vb))
    return float(np.mean(tail))


if __name__ == '__main__':
    G.DATA = DATA
    G.TGT = DATA
    print('loading cached features...', flush=True)
    feats, tgts, vf, (vo, vb) = G.load_cut()
    print('rho=%.1f fixed, B=%d, CRN=on, n/client=%d' % (RHO_FIXED, B, feats[0].shape[0]),
          flush=True)
    for T in T_VALUES:
        s = sigma_for(T)
        print('  T=%2d rounds  sigma=%.3f  rho spent=%.1f/%.1f%s'
              % (T, s, T * 2 * EPOCH_LOCAL / s ** 2, RHO_FIXED,
                 '   <- uniform baseline' if T == 25 else ''), flush=True)
    print('\n%5s | %s' % ('trial', '  '.join('T=%-6d' % T for T in T_VALUES)), flush=True)

    res = {str(T): [] for T in T_VALUES}
    t0 = time.time()
    for t in range(N_TRIALS):
        seed = SEED_BASE + t
        init = G.make_init('public_tail', seed, PUBLIC)
        row = []
        for T in T_VALUES:
            v = run_arm(init, feats, tgts, vf, vo, vb, T, sigma_for(T), seed)
            res[str(T)].append(v)
            row.append(v)
        print('%5d | %s   [%.0fm]'
              % (t + 1, '  '.join('%-8.4f' % v for v in row), (time.time() - t0) / 60),
              flush=True)
        json.dump({'rho': RHO_FIXED, 'seed_base': SEED_BASE, 'crn': True,
                   'T_values': T_VALUES,
                   'sigmas': {str(T): sigma_for(T) for T in T_VALUES},
                   'results': res}, open(OUT_PATH, 'w'), indent=1)
        if t >= 1:
            base = np.array(res['25'])
            print('        vs uniform(T=25): ' + '  '.join(
                'T%d %+.4f(%d/%d)' % (T, (np.array(res[str(T)]) - base).mean(),
                                      int((np.array(res[str(T)]) - base > 0).sum()), len(base))
                for T in T_VALUES if T != 25), flush=True)

    base = np.array(res['25'])
    print('\n=== %d paired trials vs uniform T=25 (mean=%.4f) ===' % (N_TRIALS, base.mean()),
          flush=True)
    print('%6s %8s %11s %9s %8s %9s' % ('T', 'sigma', 'mean diff', 'SE', 'wins', 'p'), flush=True)
    try:
        from scipy import stats
    except Exception:
        stats = None
    for T in T_VALUES:
        if T == 25:
            continue
        x = np.array(res[str(T)]); d = x - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        p = stats.ttest_rel(x, base).pvalue if stats else float('nan')
        print('%6d %8.3f %+11.4f %9.4f %5d/%-3d %9.4f'
              % (T, sigma_for(T), d.mean(), se, int((d > 0).sum()), len(d), p), flush=True)
    print('TRUNCATION DONE', flush=True)
