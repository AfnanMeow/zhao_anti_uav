# =============================================================================
# CONFIRMATORY test, pre-registered hypothesis, fresh seeds.
#
# Exploratory run (verify_paper_premise.py, n=30, SGD regime) found k=+1.0 --
# i.e. rho_t proportional to (t+1), meaning LESS noise in later rounds, the
# source paper's predicted direction -- beat uniform by +2.27% (20/30 wins,
# t=+1.65, 95%CI [-0.004,+0.050]). Suggestive but not conclusive, and it was
# one of 4 schedules tested, so it carries multiple-comparison risk.
#
# This script tests ONLY that single pre-specified hypothesis (k=+1.0 vs
# k=0.0), on SEEDS DISJOINT from the exploratory run (31000-31029 there,
# 77000+ here), at n=60. Nothing else is tested, so there is no multiple-
# comparison inflation and the p-value means what it says.
#
# Regime (calibrated, see conversation): SGD lr=0.005 (update norm decays with
# convergence, unlike Adam), C_clip=3.0 (far above late-round norms ~0.3-0.4,
# so clipping does NOT bind late and per-round SNR tracks the real update
# magnitude -> ~8-10x SNR asymmetry early-vs-late = the paper's premise), and
# rho in [150,250] so sigma ~1.0 and the model still trains (~0.90 mAP).
# =============================================================================
import os, sys, json, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fl_lightcnn_final import TinyDetector, ShardDataset, Client, fedavg, evaluate_map

N, CR_TOTAL, EPOCH_LOCAL, BATCH_SIZE = 5, 8, 7, 8
SGD_LR, C_CLIP = 0.005, 3.0
RHO_MIN, RHO_MAX = 150.0, 250.0
SIGMA_MIN, SIGMA_MAX = 1.0, 20.0

N_TRIALS = 250
K_VALUES = [0.0, 1.0]        # control, and the single pre-specified hypothesis
SEED_BASE = 90000            # disjoint from BOTH the exploratory run (31000-31029)
                             # and the first n=60 confirmation (77000-77059)

SHARD_DIR = './fl_workdirs/lightcnn_shards'
VAL_ANN   = 'single_test_coco/annotations/val.json'
OUT_PATH  = './exp_data/confirm_k1_powered.json'


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
        local = []
        for i, c in enumerate(clients):
            sigma = float(np.clip(np.sqrt(2 * c.el / max(allocs[i][CR], 1e-9)),
                                   SIGMA_MIN, SIGMA_MAX))
            local.append(c.train_one_round(global_sd, sigma))
        global_sd = fedavg(local)
    model.load_state_dict(global_sd)
    return quiet_eval(model, val_ds)


if __name__ == '__main__':
    shard_paths = [os.path.join(SHARD_DIR, f'shard_{i}.json') for i in range(N)]
    clients = [Client(i, shard_paths[i], C=C_CLIP, bs=BATCH_SIZE, el=EPOCH_LOCAL,
                       lr=SGD_LR, optimizer='sgd') for i in range(N)]
    val_ds = ShardDataset(VAL_ANN, device='cuda:0')
    print(f'WELL-POWERED CONFIRMATORY: k=+1.0 vs uniform, n={N_TRIALS}, fresh seeds ({SEED_BASE}+)', flush=True)
    print(f'(n=250 gives ~80%% power to detect the +1.09%% observed at n=60)', flush=True)
    print(f'SGD lr={SGD_LR}  C_clip={C_CLIP}  rho in [{RHO_MIN},{RHO_MAX}]\n', flush=True)

    res = {f'{k}': [] for k in K_VALUES}
    for t in range(N_TRIALS):
        rng = np.random.RandomState(SEED_BASE + t)
        random.seed(SEED_BASE + t); np.random.seed(SEED_BASE + t)
        rhos = [float(rng.uniform(RHO_MIN, RHO_MAX)) for _ in range(N)]
        init_sd = TinyDetector().cuda().state_dict()
        line = []
        for k in K_VALUES:
            allocs = [power_allocation(rhos[i], CR_TOTAL, k) for i in range(N)]
            acc = run_allocation(clients, allocs, init_sd, val_ds)
            res[f'{k}'].append(acc)
            line.append(f'k={k:+.1f}:{acc:.3f}')
        d = res['1.0'][-1] - res['0.0'][-1]
        print(f'[Trial {t+1:2d}] ' + '  '.join(line) + f'   diff={d:+.3f}', flush=True)
        with open(OUT_PATH, 'w') as f:
            json.dump(res, f, indent=1)

    uni = np.array(res['0.0']); k1 = np.array(res['1.0'])
    d = k1 - uni
    se = d.std(ddof=1) / np.sqrt(len(d))
    t_stat = d.mean() / se
    # exact-ish two-sided p from the t distribution
    try:
        from scipy import stats
        p = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(d) - 1))
        w = stats.wilcoxon(k1, uni)
        extra = f'  |  Wilcoxon p={w.pvalue:.4f}'
    except Exception:
        p = float('nan'); extra = ''
    print(f'\n=== CONFIRMATORY RESULT (n={N_TRIALS}, fresh seeds) ===', flush=True)
    print(f'uniform  mean={uni.mean():.4f}', flush=True)
    print(f'k=+1.0   mean={k1.mean():.4f}', flush=True)
    print(f'diff={d.mean():+.4f}  95%CI=[{d.mean()-1.96*se:+.4f},{d.mean()+1.96*se:+.4f}]  '
          f'beats={sum(d>0)}/{len(d)}  t={t_stat:+.2f}  p={p:.4f}{extra}', flush=True)
