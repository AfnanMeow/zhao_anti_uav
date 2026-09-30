# =============================================================================
# THE THESIS EXPERIMENT: does non-uniform noise allocation beat uniform?
#
# Pre-registered, single hypothesis: k=+1 (rho_t ~ (t+1), i.e. LESS noise in
# later rounds -- the source paper's predicted optimum) vs k=0 (uniform).
# Nothing else is tested, so there is no multiple-comparison inflation.
#
# Run in the regime calibrated in FINDINGS_gate1.md, which is the first setup in
# this project where accuracy actually depends on sigma:
#   clear-sky session-disjoint split, frozen public s1-s3, DP-trained k1+k2+head
#   (D=566,789), SGD lr=0.003, B=128, 25 rounds x 2 local epochs, 5 x 3808 frames
#   measured: sigma 0/2/4/8/16/32 -> IoU.25 0.769/0.557/0.499/0.492/0.371/0.047
#
# Budget: sigma_t = sqrt(2*el/rho_t) with el=2, so rho=30 (the authors' own
# CIFAR-10 midpoint) gives uniform sigma=1.83, and k=+1 sweeps sigma 6.58 -> 1.32.
# Both arms stay inside the measured-sensitive band and never hit the sigma>=1 clip.
#
# Two variance reductions, both carried over from dpsgd_final_test.py:
#   1. PAIRING -- both arms of a trial share one init and one rho, so init luck
#      cancels. Essential here: unpaired seed spread ran 0.325 vs 0.098 at sigma=16.
#   2. Score = mean IoU.25 over the last SCORE_LAST rounds, not a single round.
# =============================================================================
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_DB_sweep as G
from stage1b_recache import DPTail

DATA = './fl_workdirs/clear'
PUBLIC = './fl_workdirs/clear/public_pretrained.pt'
OUT_PATH = './exp_data/paired_schedule_clear.json'

CR_TOTAL, EPOCH_LOCAL, B, LR = 25, 2, 128, 0.003
RHO_FIXED = 30.0                  # authors' CIFAR-10 budget midpoint (they used [20,40])
SIGMA_MIN, SIGMA_MAX = 1.0, 20.0
K_VALUES = [0.0, 1.0]             # control, and the single pre-specified hypothesis
SCORE_LAST = 6
N_TRIALS = 30
SEED_BASE = 110000                # disjoint from every prior run in this project


def power_allocation(rho_total, T, k):
    w = np.array([(t + 1) ** k for t in range(T)], dtype=float)
    return (w / w.sum() * rho_total)


def sigma_schedule(k):
    rho = power_allocation(RHO_FIXED, CR_TOTAL, k)
    return np.clip(np.sqrt(2 * EPOCH_LOCAL / np.maximum(rho, 1e-9)), SIGMA_MIN, SIGMA_MAX)


def run_arm(init, feats, tgts, vf, vo, vb, sigmas):
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
    D = sum(p.numel() for p in DPTail().parameters())
    print('D=%d  B=%d lr=%s  %d rounds x %d ep  n/client=%d  rho=%.1f (fixed)'
          % (D, B, LR, CR_TOTAL, EPOCH_LOCAL, feats[0].shape[0], RHO_FIXED), flush=True)
    for k in K_VALUES:
        sg = sigma_schedule(k)
        print('  k=%+.1f sigma: %s ... %s   (mean %.2f)'
              % (k, np.round(sg[:3], 2), np.round(sg[-3:], 2), sg.mean()), flush=True)
    print('\nscore = mean IoU.25 over last %d rounds; paired on init+rho\n' % SCORE_LAST,
          flush=True)
    print('%5s | %8s %8s | %9s | %s' % ('trial', 'k=0', 'k=+1', 'diff', 'running'), flush=True)

    scheds = {k: sigma_schedule(k) for k in K_VALUES}
    res = {str(k): [] for k in K_VALUES}
    t0 = time.time()
    for t in range(N_TRIALS):
        seed = SEED_BASE + t
        init = G.make_init('public_tail', seed, PUBLIC)     # SAME init for both arms
        row = {}
        for k in K_VALUES:
            row[k] = run_arm(init, feats, tgts, vf, vo, vb, scheds[k])
            res[str(k)].append(row[k])
        d = np.array(res['1.0']) - np.array(res['0.0'])
        wins = int((d > 0).sum())
        tstat = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 1 and d.std() > 0 else 0.0
        print('%5d | %8.4f %8.4f | %+9.4f | mean%+.4f %2d/%-2d t=%+.2f  [%.0fm]'
              % (t + 1, row[0.0], row[1.0], row[1.0] - row[0.0],
                 d.mean(), wins, len(d), tstat, (time.time() - t0) / 60), flush=True)
        json.dump({'rho': RHO_FIXED, 'seed_base': SEED_BASE, 'score_last': SCORE_LAST,
                   'sigma_schedules': {str(k): scheds[k].tolist() for k in K_VALUES},
                   'results': res}, open(OUT_PATH, 'w'), indent=1)

    d = np.array(res['1.0']) - np.array(res['0.0'])
    print('\n%d paired trials: k=+1 minus k=0 = %+.4f +- %.4f (sd), %d/%d wins'
          % (len(d), d.mean(), d.std(ddof=1), int((d > 0).sum()), len(d)), flush=True)
    try:
        from scipy import stats
        ts, p = stats.ttest_rel(res['1.0'], res['0.0'])
        lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(),
                                  scale=d.std(ddof=1) / np.sqrt(len(d)))
        print('paired t=%+.3f  p=%.4f  95%%CI [%+.4f, %+.4f]' % (ts, p, lo, hi), flush=True)
    except Exception as e:
        print('(scipy unavailable: %s)' % e, flush=True)
    print('PAIRED TEST DONE', flush=True)
