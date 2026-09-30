# =============================================================================
# THE decisive test of the source paper's premise, in a regime where that
# premise can actually manifest.
#
# Background: the paper's whole argument is "early rounds have big gradients
# and tolerate more noise; late rounds are fine-tuning with small gradients
# and are fragile -- so spend less privacy budget early, more late." Measured
# live, the earlier Adam-based setup violated BOTH preconditions for that:
#   (a) Adam's step size is ~scale-invariant to gradient magnitude, so the
#       client update norm stayed flat (~2.1) across 25 rounds even after the
#       model fully converged -- late rounds were never "small gradient".
#   (b) C_clip=1.0 sat BELOW that norm, so clipping bound every round, which
#       pins post-clip SNR at exactly 1/sigma independent of gradient size.
# Under (a)+(b) the premise is mathematically impossible, uniform allocation
# is optimal by Jensen, and no policy (RL or otherwise) can beat it -- which
# is exactly what 290 controlled paired runs found.
#
# This script re-runs the same paired schedule sweep under SGD (update norm
# DOES decay with convergence: 60.5 -> 0.27 over 25 rounds) with C_clip set
# ABOVE the late-round norms so clipping does not bind and SNR tracks the real
# signal. If the paper's premise is real, k>0 (less noise late) should now beat
# uniform (k=0). If it still doesn't, the negative result holds in the paper's
# own regime too.
# =============================================================================
import os, sys, json, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fl_lightcnn_final import TinyDetector, ShardDataset, Client, fedavg, evaluate_map

N            = 5
CR_TOTAL     = 8
EPOCH_LOCAL  = 7
BATCH_SIZE   = 8
SGD_LR       = 0.005      # swept: trains to mAP 0.963 AND shows decaying update norms
C_CLIP       = 3.0        # calibrated: trains to ~0.91 mAP AND sits far above late-round
                          # norms (~0.3-0.4) so clipping does NOT bind late -> ~8-10x SNR
                          # asymmetry early-vs-late, i.e. the paper's premise in force
RHO_MIN, RHO_MAX = 150.0, 250.0   # budget raised so sigma~1.0; at rho 20-40 the
                                  # model cannot train at all in this C_clip regime
SIGMA_MIN, SIGMA_MAX = 1.0, 20.0

N_TRIALS = 30
K_VALUES = [-1.0, 0.0, 0.5, 1.0, 2.0]     # 0.0 == uniform (control); k>0 == paper's prediction

SHARD_DIR = './fl_workdirs/lightcnn_shards'
VAL_ANN   = 'single_test_coco/annotations/val.json'
OUT_PATH  = './exp_data/paper_premise_check_v2.json'


def power_allocation(rho_total, T, k):
    w = np.array([(t + 1) ** k for t in range(T)], dtype=float)
    return (w / w.sum() * rho_total).tolist()


def quiet_eval(model, ds):
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return evaluate_map(model, ds)


def run_allocation(clients, allocs, init_sd, val_ds, track=False):
    global_sd = {k: v.clone() for k, v in init_sd.items()}
    model = TinyDetector().cuda()
    binds = 0; tot = 0
    for CR in range(CR_TOTAL):
        local_sds = []
        for i, c in enumerate(clients):
            sigma = float(np.clip(np.sqrt(2 * c.el / max(allocs[i][CR], 1e-9)),
                                   SIGMA_MIN, SIGMA_MAX))
            local_sds.append(c.train_one_round(global_sd, sigma))
            tot += 1
        global_sd = fedavg(local_sds)
    model.load_state_dict(global_sd)
    return quiet_eval(model, val_ds)


if __name__ == '__main__':
    shard_paths = [os.path.join(SHARD_DIR, f'shard_{i}.json') for i in range(N)]
    clients = [Client(i, shard_paths[i], C=C_CLIP, bs=BATCH_SIZE, el=EPOCH_LOCAL,
                       lr=SGD_LR, optimizer='sgd') for i in range(N)]
    val_ds = ShardDataset(VAL_ANN, device='cuda:0')
    print(f'SGD lr={SGD_LR}  C_clip={C_CLIP}  {CR_TOTAL} rounds  {N_TRIALS} paired trials', flush=True)
    print(f'k>0 = LESS noise in late rounds = the paper\'s predicted optimum\n', flush=True)

    res = {f'{k}': [] for k in K_VALUES}
    for t in range(N_TRIALS):
        rng = np.random.RandomState(31000 + t)
        random.seed(31000 + t); np.random.seed(31000 + t)
        rhos = [float(rng.uniform(RHO_MIN, RHO_MAX)) for _ in range(N)]
        init_sd = TinyDetector().cuda().state_dict()
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
    print(f'\n=== n={N_TRIALS} paired vs uniform (SGD regime) ===', flush=True)
    print(f'{"uniform (k=0)":18s} mean={uni.mean():.4f}', flush=True)
    for k in K_VALUES:
        if k == 0.0:
            continue
        arr = np.array(res[f'{k}']); d = arr - uni
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f'k={k:<+15.1f} mean={arr.mean():.4f}  diff={d.mean():+.4f}  '
              f'95%CI=[{d.mean()-1.96*se:+.4f},{d.mean()+1.96*se:+.4f}]  '
              f'beats={sum(d>0)}/{len(d)}  t={d.mean()/se:+.2f}', flush=True)
