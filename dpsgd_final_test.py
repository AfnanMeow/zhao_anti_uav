# =============================================================================
# Properly-powered test: does k=+1 (less noise late, the paper's predicted
# optimum) beat uniform under DP-SGD at the authors' round count?
#
# Prior evidence:
#   DP-FedAvg (the thesis port's mechanism), 630+ paired runs: uniform optimal,
#     premise provably inert (clipping normalizes gradient magnitude away).
#   DP-SGD (the authors' actual mechanism), 8 rounds,  n=25: k=+1 is -6.2% (the
#     model never reaches a fine-tuning phase; front-loaded noise kills it early).
#   DP-SGD, 25 rounds (authors' count), n=20: k=+1 is +1.7%, 13/20, p=0.58 --
#     right direction, underpowered.
#
# Two variance reductions so a real ~1-2% effect is detectable at feasible n
# (the naive design needed n~237):
#   1. FIXED rho for every client and trial -- removes budget-draw variance
#      entirely. Only init + DP noise realizations vary.
#   2. Score = mean mAP over the LAST 3 rounds, not a single final round, which
#      is a noisy one-shot measurement of a stochastic trajectory.
# Only k=0 vs k=+1 is tested (single pre-specified hypothesis, no multiple-
# comparison inflation), on seeds disjoint from every prior run.
# =============================================================================
import os, sys, json, random, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fl_dpsgd_final import make_small_dp_model, DPClient, fedavg, quiet_eval
from fl_lightcnn_final import ShardDataset

N, CR_TOTAL = 5, 25
EPOCH_LOCAL, BATCH_SIZE, LR, C_CLIP = 20, 64, 0.5, 1.0
RHO_FIXED = 650.0                  # same for every client & trial (variance reduction 1)
SIGMA_MIN, SIGMA_MAX = 1.0, 20.0
SCORE_LAST_N = 3                   # variance reduction 2

N_TRIALS = 60
K_VALUES = [0.0, 1.0]
SEED_BASE = 63000                  # disjoint from 41000 / 52000 runs

SHARD_DIR = './fl_workdirs/lightcnn_shards'
VAL_ANN   = 'single_test_coco/annotations/val.json'
OUT_PATH  = './exp_data/dpsgd_final_test.json'


def power_allocation(rho_total, T, k):
    w = np.array([(t + 1) ** k for t in range(T)], dtype=float)
    return (w / w.sum() * rho_total).tolist()


def run_allocation(clients, allocs, init_sd, val_ds):
    gsd = {k: v.clone() for k, v in init_sd.items()}
    tail = []
    for CR in range(CR_TOTAL):
        local = []
        for i, c in enumerate(clients):
            sigma = float(np.clip(np.sqrt(2 * c.el / max(allocs[i][CR], 1e-9)),
                                   SIGMA_MIN, SIGMA_MAX))
            local.append(c.train_one_round(gsd, sigma))
        gsd = fedavg(local)
        if CR >= CR_TOTAL - SCORE_LAST_N:
            tail.append(quiet_eval(gsd, val_ds, model_fn=make_small_dp_model))
    return float(np.mean(tail))


if __name__ == '__main__':
    shard_paths = [os.path.join(SHARD_DIR, f'shard_{i}.json') for i in range(N)]
    clients = [DPClient(i, shard_paths[i], C=C_CLIP, bs=BATCH_SIZE, el=EPOCH_LOCAL,
                         lr=LR, model_fn=make_small_dp_model) for i in range(N)]
    val_ds = ShardDataset(VAL_ANN, device='cuda:0')
    rhos = [RHO_FIXED] * N
    print(f'DP-SGD  {CR_TOTAL} rounds  rho={RHO_FIXED} (fixed)  score=mean(last {SCORE_LAST_N})',
          flush=True)
    print(f'k=+1.0 vs uniform, n={N_TRIALS}, seeds {SEED_BASE}+\n', flush=True)
    for k in K_VALUES:
        a = power_allocation(RHO_FIXED, CR_TOTAL, k)
        sg = [round(float(np.clip(np.sqrt(2 * EPOCH_LOCAL / x), SIGMA_MIN, SIGMA_MAX)), 2) for x in a]
        print(f'  k={k:+.1f} sigma schedule: {sg[:4]} ... {sg[-4:]}', flush=True)
    print('', flush=True)

    res = {f'{k}': [] for k in K_VALUES}
    for t in range(N_TRIALS):
        random.seed(SEED_BASE + t); np.random.seed(SEED_BASE + t)
        torch.manual_seed(SEED_BASE + t)
        init_sd = make_small_dp_model().state_dict()
        line = []
        for k in K_VALUES:
            allocs = [power_allocation(rhos[i], CR_TOTAL, k) for i in range(N)]
            acc = run_allocation(clients, allocs, init_sd, val_ds)
            res[f'{k}'].append(acc)
            line.append(f'k={k:+.1f}:{acc:.3f}')
        d = res['1.0'][-1] - res['0.0'][-1]
        print(f'[Trial {t+1:2d}] ' + '  '.join(line) + f'  diff={d:+.3f}', flush=True)
        with open(OUT_PATH, 'w') as f:
            json.dump(res, f, indent=1)

    uni = np.array(res['0.0']); k1 = np.array(res['1.0']); d = k1 - uni
    se = d.std(ddof=1) / np.sqrt(len(d)); tst = d.mean() / se
    try:
        from scipy import stats
        p = 2 * (1 - stats.t.cdf(abs(tst), df=len(d) - 1))
        wp = stats.wilcoxon(k1, uni).pvalue
    except Exception:
        p = wp = float('nan')
    print(f'\n=== FINAL (DP-SGD, {CR_TOTAL} rounds, n={N_TRIALS}) ===', flush=True)
    print(f'uniform  mean={uni.mean():.4f}   k=+1.0  mean={k1.mean():.4f}', flush=True)
    print(f'diff={d.mean():+.4f}  95%CI=[{d.mean()-1.96*se:+.4f},{d.mean()+1.96*se:+.4f}]  '
          f'beats={sum(d>0)}/{len(d)}  t={tst:+.2f}  p={p:.4f}  Wilcoxon p={wp:.4f}', flush=True)
    print(f'paired sd={d.std(ddof=1):.4f}  (was 0.135 before variance reduction)', flush=True)
