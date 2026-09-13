# =============================================================================
# HEADROOM: how much can ANY allocation beat uniform in the live regime?
#
# This bounds the entire DRL phase before it is run. A learned policy cannot
# score better than the best allocation available to it, so if the best fixed
# schedule is worth only ~1pp, the RL is chasing a signal smaller than its own
# reward noise (run-to-run sd ~4.7pp) and will fail for a second, different
# reason. If the best schedule is worth 3-5pp, the RL has real room.
#
# Family: rho_t proportional to (t+1)^k, normalised to spend exactly rho_total.
#   k<0  more budget early -> noise RISES over rounds
#   k=0  uniform (the control every arm is differenced against)
#   k>0  more budget late  -> noise FALLS over rounds (the paper's prediction)
# This is the same family as verify_uniform_optimality.py, but run in the regime
# calibrated in FINDINGS_gate1.md rather than the inert one.
#
# VARIANCE REDUCTION -- two, on top of the paired design:
#   1. All schedules in a trial share ONE init (paired, as before).
#   2. COMMON RANDOM NUMBERS: every arm re-seeds to the same trial seed, so the
#      arms draw the same underlying normals for DP noise and the same shuffling
#      order. Magnitudes still differ (they are scaled by sigma_t), but the draws
#      are correlated across arms, which shrinks the variance of the DIFFERENCE.
#      The n=30 run did NOT do this -- its arms ran off independent streams.
# =============================================================================
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_DB_sweep as G
from stage1b_recache import DPTail

DATA = './fl_workdirs/clear'
PUBLIC = './fl_workdirs/clear/public_pretrained.pt'
OUT_PATH = './exp_data/headroom_clear.json'

CR_TOTAL, EPOCH_LOCAL, B, LR = 25, 2, 128, 0.003
RHO_FIXED = 30.0
SIGMA_MIN, SIGMA_MAX = 1.0, 20.0
SCORE_LAST = 6
K_VALUES = [-1.0, 0.0, 0.5, 1.0, 2.0, 3.0]
N_TRIALS = 20
SEED_BASE = 220000            # disjoint from the n=30 run (110000) and all earlier runs


def sigma_schedule(k):
    w = np.array([(t + 1) ** k for t in range(CR_TOTAL)], dtype=float)
    rho = w / w.sum() * RHO_FIXED
    return np.clip(np.sqrt(2 * EPOCH_LOCAL / np.maximum(rho, 1e-9)), SIGMA_MIN, SIGMA_MAX)


def run_arm(init, feats, tgts, vf, vo, vb, sigmas, seed):
    torch.manual_seed(seed)          # <- common random numbers across arms
    st = {k: v.clone() for k, v in init.items()}
    tail = []
    for r in range(CR_TOTAL):
        st = G.fedavg([G.client_round(st, feats[i], tgts[i][0], tgts[i][1],
                                      float(sigmas[r]), True, B, LR)
                       for i in range(G.N_CLIENTS)])
        if r >= CR_TOTAL - SCORE_LAST:
            tail.append(G.evaluate(st, vf, vo, vb))
    return float(np.mean(tail))


def summarise(res):
    base = np.array(res['0.0'])
    out = []
    for k in K_VALUES:
        if k == 0.0:
            continue
        d = np.array(res[str(k)]) - base
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan')
        t = d.mean() / se if len(d) > 1 and se > 0 else 0.0
        out.append((k, d.mean(), se, int((d > 0).sum()), len(d), t))
    return out


if __name__ == '__main__':
    G.DATA = DATA
    G.TGT = DATA
    print('loading cached features...', flush=True)
    feats, tgts, vf, (vo, vb) = G.load_cut()
    print('D=%d  B=%d lr=%s  %d rounds  n/client=%d  rho=%.1f  CRN=on'
          % (sum(p.numel() for p in DPTail().parameters()), B, LR, CR_TOTAL,
             feats[0].shape[0], RHO_FIXED), flush=True)
    scheds = {k: sigma_schedule(k) for k in K_VALUES}
    # Effective budget actually spent: rho_t = 2*el/sigma_t^2. Clipping sigma DOWN
    # to SIGMA_MAX means less noise than allocated, i.e. OVERSPENDING rho. Arms
    # that overspend are not a fair comparison against uniform -- flagged here so
    # the headroom number is not read as if every arm paid the same price.
    for k in K_VALUES:
        s = scheds[k]
        eff = float(np.sum(2 * EPOCH_LOCAL / s ** 2))
        flag = '  <-- OVERSPENDS budget (sigma clipped at max)' if eff > RHO_FIXED * 1.01 else ''
        print('  k=%+.1f  sigma r1=%.2f r13=%.2f r25=%.2f  mean=%.2f | rho spent=%.1f/%.1f%s'
              % (k, s[0], s[12], s[-1], s.mean(), eff, RHO_FIXED, flag), flush=True)
    print('\nscore = mean IoU.25 over last %d rounds; all arms share init + RNG stream'
          % SCORE_LAST, flush=True)
    print('%5s | %s' % ('trial', '  '.join('k=%+.1f' % k for k in K_VALUES)), flush=True)

    res = {str(k): [] for k in K_VALUES}
    t0 = time.time()
    for t in range(N_TRIALS):
        seed = SEED_BASE + t
        init = G.make_init('public_tail', seed, PUBLIC)
        row = []
        for k in K_VALUES:
            v = run_arm(init, feats, tgts, vf, vo, vb, scheds[k], seed)
            res[str(k)].append(v)
            row.append(v)
        print('%5d | %s   [%.0fm]'
              % (t + 1, '  '.join('%6.4f' % v for v in row), (time.time() - t0) / 60), flush=True)
        json.dump({'rho': RHO_FIXED, 'seed_base': SEED_BASE, 'crn': True,
                   'k_values': K_VALUES, 'results': res}, open(OUT_PATH, 'w'), indent=1)
        if t >= 1:
            print('        vs uniform: ' + '  '.join(
                'k%+.1f %+.4f(%d/%d)' % (k, m, w, n) for k, m, se, w, n, tt in summarise(res)),
                flush=True)

    print('\n=== headroom, %d paired trials (vs k=0 uniform) ===' % N_TRIALS, flush=True)
    print('%6s %10s %9s %8s %7s' % ('k', 'mean diff', 'SE', 'wins', 't'), flush=True)
    for k, m, se, w, n, tt in summarise(res):
        print('%6.1f %+10.4f %9.4f %5d/%-3d %+7.2f' % (k, m, se, w, n, tt), flush=True)
    base = np.array(res['0.0'])
    print('\nuniform mean = %.4f   sd of paired diffs (CRN) = %.4f'
          % (base.mean(), np.std(np.array(res['1.0']) - base, ddof=1)), flush=True)
    print('HEADROOM DONE', flush=True)
