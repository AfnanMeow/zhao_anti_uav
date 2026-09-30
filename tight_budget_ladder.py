# =============================================================================
# DOOR 1: does allocation matter at TIGHT privacy budgets?
#
# Everything so far ran at rho=30, where uniform sigma=1.83 sits in a smooth part
# of the noise curve. There the analytic result holds cleanly: accumulated noise
# ~ sum_t sigma_t^2, clipping pins the per-round signal at C, and Jensen makes
# uniform the argmin (confirmed, Spearman -0.767 over 13 allocations).
#
# That derivation assumes accuracy is a smooth monotone function of sum sigma_t^2.
# Near the collapse cliff (sigma=16 -> 0.371, sigma=32 -> 0.047) that assumption
# is at its weakest: with a threshold response, total variance stops being the
# right summary statistic and surviving SOME rounds may beat spreading thin.
# This is the one remaining door with a real argument behind it.
#
# TEST: spend the whole budget over the first T rounds, then stop. sigma = sqrt(4T/rho).
#   rho=0.5 : T=25 sigma=14.14 (on the cliff for all 25 rounds) ... T=3 sigma=4.90
#   rho=2   : T=25 sigma= 7.07                                  ... T=3 sigma=2.45
#   rho=8   : T=25 sigma= 3.54                                  ... T=3 sigma=1.22
# Every arm spends rho exactly and no sigma touches the [1,20] clip.
#
# Theory predicts sum sigma_t^2 = 4T^2/rho, i.e. FEWER rounds carry less noise
# (T=3 is 69x quieter than T=25) but buy less learning. At rho=30 learning won
# (T=8 scored -0.058 vs uniform). If noise wins at rho=0.5, the optimum moves off
# uniform and there is a regime where allocation matters.
#
# PRE-SPECIFIED: three budgets, four arms, all reported regardless of outcome.
# Sweeping rho until one came out significant would manufacture the result.
#
# Full per-round trajectories are saved for every arm, not just final scores, so
# the data supports trajectory plots as well as the summary comparison.
# =============================================================================
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_DB_sweep as G
from gpu_guard import GUARD

DATA = './fl_workdirs/clear'
PUBLIC = './fl_workdirs/clear/public_pretrained.pt'
OUT_PATH = './exp_data/tight_budget_ladder.json'

EPOCH_LOCAL, B, LR = 2, 128, 0.003
RHO_VALUES = [0.5, 2.0, 8.0]
T_VALUES = [25, 12, 6, 3]            # 25 = uniform baseline at that budget
SCORE_LAST = 3                        # short arms only have 3 rounds
N_TRIALS = 10
SEED_BASE = 660000


def sigma_for(rho, T):
    return float(np.sqrt(2 * EPOCH_LOCAL * T / rho))


def run_arm(init, feats, tgts, vf, vo, vb, T, sigma, seed):
    """Returns (score, full per-round trajectory)."""
    torch.manual_seed(seed)                       # common random numbers
    st = {k: v.clone() for k, v in init.items()}
    traj = []
    for r in range(T):
        st = G.fedavg([G.client_round(st, feats[i], tgts[i][0], tgts[i][1],
                                      sigma, True, B, LR)
                       for i in range(G.N_CLIENTS)])
        traj.append(G.evaluate(st, vf, vo, vb))
    return float(np.mean(traj[-SCORE_LAST:])), traj


if __name__ == '__main__':
    G.DATA = DATA
    G.TGT = DATA
    print('loading cached features...', flush=True)
    feats, tgts, vf, (vo, vb) = G.load_cut()
    print('B=%d lr=%s n/client=%d  CRN=on  score=mean IoU.25 over last %d rounds\n'
          % (B, LR, feats[0].shape[0], SCORE_LAST), flush=True)
    for rho in RHO_VALUES:
        line = '  rho=%5.1f : ' % rho
        for T in T_VALUES:
            s = sigma_for(rho, T)
            line += 'T=%-2d sigma=%5.2f (rho %.2f)  ' % (T, s, T * 2 * EPOCH_LOCAL / s ** 2)
        print(line, flush=True)

    try:
        from scipy import stats
    except Exception:
        stats = None

    res = {str(rho): {str(T): [] for T in T_VALUES} for rho in RHO_VALUES}
    trajs = {str(rho): {str(T): [] for T in T_VALUES} for rho in RHO_VALUES}
    t0 = time.time()
    for t in range(N_TRIALS):
        seed = SEED_BASE + t
        init = G.make_init('public_tail', seed, PUBLIC)
        for rho in RHO_VALUES:
            row = []
            for T in T_VALUES:
                sc, tr = run_arm(init, feats, tgts, vf, vo, vb, T,
                                 sigma_for(rho, T), seed)
                res[str(rho)][str(T)].append(sc)
                trajs[str(rho)][str(T)].append(tr)
                row.append(sc)
            base = res[str(rho)]['25']
            msg = '  rho=%-5.1f ' % rho + '  '.join(
                'T%d %.4f' % (T, v) for T, v in zip(T_VALUES, row))
            if len(base) > 1:
                msg += '  | vs uniform: ' + '  '.join(
                    'T%d %+.4f' % (T, np.mean(np.array(res[str(rho)][str(T)]) - np.array(base)))
                    for T in T_VALUES if T != 25)
            print(msg, flush=True)
        print('trial %d/%d  [%.0fm]  %s' % (t + 1, N_TRIALS, (time.time() - t0) / 60,
                                            GUARD.summary()), flush=True)
        json.dump({'rho_values': RHO_VALUES, 'T_values': T_VALUES,
                   'seed_base': SEED_BASE, 'crn': True, 'score_last': SCORE_LAST,
                   'sigmas': {str(r): {str(T): sigma_for(r, T) for T in T_VALUES}
                              for r in RHO_VALUES},
                   'results': res, 'trajectories': trajs},
                  open(OUT_PATH, 'w'), indent=1)

    print('\n=== tight-budget ladder, %d paired trials ===' % N_TRIALS, flush=True)
    for rho in RHO_VALUES:
        base = np.array(res[str(rho)]['25'])
        print('\nrho=%.1f   uniform(T=25) mean = %.4f   sigma=%.2f'
              % (rho, base.mean(), sigma_for(rho, 25)), flush=True)
        print('%6s %8s %11s %9s %8s %9s' % ('T', 'sigma', 'mean diff', 'SE', 'wins', 'p'),
              flush=True)
        for T in T_VALUES:
            if T == 25:
                continue
            x = np.array(res[str(rho)][str(T)]); d = x - base
            se = d.std(ddof=1) / np.sqrt(len(d))
            p = stats.ttest_rel(x, base).pvalue if stats else float('nan')
            print('%6d %8.2f %+11.4f %9.4f %5d/%-3d %9.4f'
                  % (T, sigma_for(rho, T), d.mean(), se, int((d > 0).sum()), len(d), p),
                  flush=True)
    print('\n' + GUARD.summary(), flush=True)
    print('TIGHT BUDGET DONE', flush=True)
