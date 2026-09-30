# =============================================================================
# Focused high-n test: can ANY small positive-k schedule beat uniform by the
# 1-2% that would still be a publishable edge?
#
# The broad sweep (verify_uniform_optimality.py) already ruled out k<0
# (significantly harmful) and random reallocation (5 of 6 lost). It left the
# k>0 direction -- the source paper's predicted optimum, noise decreasing over
# rounds -- statistically indistinguishable from uniform at n=10, which can
# only resolve ~5% effects. This narrows to the k>0 family and raises n so a
# genuine 1-2% edge would actually show up.
#
# Paired design: within each trial, every schedule shares one initialization
# and one set of per-client budgets, so the paired differences cancel the
# initialization variance that dominates the raw scores.
# =============================================================================
import os, sys, json, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fl_lightcnn_final import TinyDetector, ShardDataset, Client, fedavg, evaluate_map

N, CR_TOTAL, EPOCH_LOCAL, BATCH_SIZE, LR, C_CLIP = 5, 8, 7, 8, 1e-3, 1.0
RHO_MIN, RHO_MAX = 20.0, 40.0
SIGMA_MIN, SIGMA_MAX = 1.0, 20.0

N_TRIALS = 40
K_VALUES = [0.0, 0.25, 0.5, 1.0]        # 0.0 is uniform (the control)

SHARD_DIR = './fl_workdirs/lightcnn_shards'
VAL_ANN   = 'single_test_coco/annotations/val.json'
OUT_PATH  = './exp_data/small_edge_check.json'


def power_allocation(rho_total, T, k):
    w = np.array([(t + 1) ** k for t in range(T)], dtype=float)
    return (w / w.sum() * rho_total).tolist()


def quiet_eval(model, ds):
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return evaluate_map(model, ds)


def run_allocation(clients, allocs, init_sd, val_ds):
    global_sd = {k: v.clone() for k, v in init_sd.items()}
    model = TinyDetector().cuda()
    for CR in range(CR_TOTAL):
        local_sds = []
        for i, c in enumerate(clients):
            sigma = float(np.clip(np.sqrt(2 * c.el / max(allocs[i][CR], 1e-9)), SIGMA_MIN, SIGMA_MAX))
            local_sds.append(c.train_one_round(global_sd, sigma))
        global_sd = fedavg(local_sds)
    model.load_state_dict(global_sd)
    return quiet_eval(model, val_ds)


if __name__ == '__main__':
    shard_paths = [os.path.join(SHARD_DIR, f'shard_{i}.json') for i in range(N)]
    clients = [Client(i, shard_paths[i], C=C_CLIP, bs=BATCH_SIZE, el=EPOCH_LOCAL, lr=LR)
               for i in range(N)]
    val_ds = ShardDataset(VAL_ANN, device='cuda:0')
    print(f'Focused test: {N_TRIALS} paired trials, k in {K_VALUES} (k=0 is uniform)\n', flush=True)

    res = {f'{k}': [] for k in K_VALUES}
    for t in range(N_TRIALS):
        rng = np.random.RandomState(20000 + t)
        random.seed(20000 + t); np.random.seed(20000 + t)
        rhos = [float(rng.uniform(RHO_MIN, RHO_MAX)) for _ in range(N)]
        init_sd = TinyDetector().cuda().state_dict()
        line = []
        for k in K_VALUES:
            allocs = [power_allocation(rhos[i], CR_TOTAL, k) for i in range(N)]
            acc = run_allocation(clients, allocs, init_sd, val_ds)
            res[f'{k}'].append(acc)
            line.append(f'k={k}:{acc:.3f}')
        print(f'[Trial {t+1:2d}] ' + '  '.join(line), flush=True)
        with open(OUT_PATH, 'w') as f:
            json.dump(res, f, indent=1)

    uni = np.array(res['0.0'])
    print(f'\n=== n={N_TRIALS} paired vs uniform ===', flush=True)
    print(f'{"uniform (k=0)":18s} mean={uni.mean():.4f}', flush=True)
    for k in K_VALUES[1:]:
        arr = np.array(res[f'{k}'])
        diff = arr - uni
        sd = diff.std(ddof=1)
        se = sd / np.sqrt(len(diff))
        tstat = diff.mean() / se
        ci = (diff.mean() - 1.96 * se, diff.mean() + 1.96 * se)
        print(f'k={k:<16} mean={arr.mean():.4f}  diff={diff.mean():+.4f}  '
              f'95%CI=[{ci[0]:+.4f},{ci[1]:+.4f}]  beats={sum(diff>0)}/{len(diff)}  t={tstat:+.2f}',
              flush=True)
    print(f'\n(A real +0.01 to +0.02 edge would show a CI excluding 0.)', flush=True)
