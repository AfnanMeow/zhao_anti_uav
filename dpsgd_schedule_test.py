# =============================================================================
# Does adaptive privacy-budget allocation beat uniform under DP-SGD (the
# authors' ACTUAL mechanism)?
#
# Under the thesis port's DP-FedAvg, uniform was unbeatable across 630+ paired
# runs -- because clean local epochs healed each round's perturbation, so WHEN
# the budget was spent barely mattered. DP-SGD removes that healing: noise
# corrupts every gradient step and the damage compounds. Calibration confirms
# the resulting noise sensitivity is strong (sigma 1.26 -> 2.19 costs 32 mAP
# points), which is the asymmetry the paper's adaptive allocation exploits.
#
# Schedules: rho_t proportional to (t+1)^k, normalized to spend exactly rho_total.
#   k = 0  -> uniform (control)
#   k > 0  -> more budget late = LESS noise late = the paper's predicted optimum
#   k < 0  -> the reverse (negative control)
# Paired design: identical init + identical per-client budgets within each trial.
# =============================================================================
import os, sys, json, random, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fl_dpsgd_final import make_small_dp_model, DPClient, fedavg, quiet_eval
from fl_lightcnn_final import ShardDataset

N, CR_TOTAL = 5, 25   # 25 rounds = the authors' setting; at 8 rounds the model
                      # never reaches a fine-tuning phase for the premise to protect
EPOCH_LOCAL, BATCH_SIZE, LR, C_CLIP = 20, 64, 0.5, 1.0
RHO_MIN, RHO_MAX = 500.0, 800.0   # scaled with round count to keep uniform's
                                  # per-round sigma near the calibrated ~1.2-1.4
SIGMA_MIN, SIGMA_MAX = 1.0, 20.0

N_TRIALS = 20
K_VALUES = [-1.0, 0.0, 1.0]
SEED_BASE = 52000

SHARD_DIR = './fl_workdirs/lightcnn_shards'
VAL_ANN   = 'single_test_coco/annotations/val.json'
OUT_PATH  = './exp_data/dpsgd_schedule_test_25r.json'


def power_allocation(rho_total, T, k):
    w = np.array([(t + 1) ** k for t in range(T)], dtype=float)
    return (w / w.sum() * rho_total).tolist()


def run_allocation(clients, allocs, init_sd, val_ds):
    gsd = {k: v.clone() for k, v in init_sd.items()}
    for CR in range(CR_TOTAL):
        local = []
        for i, c in enumerate(clients):
            sigma = float(np.clip(np.sqrt(2 * c.el / max(allocs[i][CR], 1e-9)),
                                   SIGMA_MIN, SIGMA_MAX))
            local.append(c.train_one_round(gsd, sigma))
        gsd = fedavg(local)
    return quiet_eval(gsd, val_ds, model_fn=make_small_dp_model)


if __name__ == '__main__':
    shard_paths = [os.path.join(SHARD_DIR, f'shard_{i}.json') for i in range(N)]
    clients = [DPClient(i, shard_paths[i], C=C_CLIP, bs=BATCH_SIZE, el=EPOCH_LOCAL,
                         lr=LR, model_fn=make_small_dp_model) for i in range(N)]
    val_ds = ShardDataset(VAL_ANN, device='cuda:0')
    print(f'DP-SGD (opacus)  lr={LR} el={EPOCH_LOCAL} B={BATCH_SIZE} C={C_CLIP} '
          f'rho in [{RHO_MIN},{RHO_MAX}]  {N_TRIALS} paired trials', flush=True)
    print('k>0 = less noise late = paper\'s predicted optimum\n', flush=True)

    res = {f'{k}': [] for k in K_VALUES}
    for t in range(N_TRIALS):
        rng = np.random.RandomState(SEED_BASE + t)
        random.seed(SEED_BASE + t); np.random.seed(SEED_BASE + t)
        rhos = [float(rng.uniform(RHO_MIN, RHO_MAX)) for _ in range(N)]
        init_sd = make_small_dp_model().state_dict()
        line = []
        for k in K_VALUES:
            allocs = [power_allocation(rhos[i], CR_TOTAL, k) for i in range(N)]
            acc = run_allocation(clients, allocs, init_sd, val_ds)
            res[f'{k}'].append(acc)
            line.append(f'k={k:+.1f}:{acc:.3f}')
        print(f'[Trial {t+1:2d}] ' + '  '.join(line), flush=True)
        with open(OUT_PATH, 'w') as f:
            json.dump(res, f, indent=1)

    uni = np.array(res['0.0'])
    print(f'\n=== n={N_TRIALS} paired vs uniform (DP-SGD) ===', flush=True)
    print(f'{"uniform (k=0)":16s} mean={uni.mean():.4f}', flush=True)
    try:
        from scipy import stats
    except Exception:
        stats = None
    for k in K_VALUES:
        if k == 0.0:
            continue
        arr = np.array(res[f'{k}']); d = arr - uni
        se = d.std(ddof=1) / np.sqrt(len(d)); tst = d.mean() / se
        p = 2 * (1 - stats.t.cdf(abs(tst), df=len(d) - 1)) if stats else float('nan')
        print(f'k={k:<+13.1f} mean={arr.mean():.4f}  diff={d.mean():+.4f}  '
              f'95%CI=[{d.mean()-1.96*se:+.4f},{d.mean()+1.96*se:+.4f}]  '
              f'beats={sum(d>0)}/{len(d)}  t={tst:+.2f}  p={p:.4f}', flush=True)
