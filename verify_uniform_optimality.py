# =============================================================================
# Is uniform noise allocation actually beatable in this setup?
#
# Bypasses RL entirely. Instead of asking "can a learned policy beat uniform",
# asks the prior question: does ANY non-uniform allocation of the privacy
# budget across rounds beat uniform here? If none does, no RL can ever win
# and the negative result is a property of the PROBLEM, not of the agent.
# If some do, there is real headroom and the RL simply failed to find it.
#
# Schedule family: rho_t proportional to (t+1)^k, normalized to spend exactly
# rho_total.  k=0 IS uniform.  k>0 spends more budget late  => noise DECREASES
# over rounds (the source paper's predicted optimum).  k<0 is the reverse.
# Plus random draws from the budget simplex for coverage outside that family.
#
# Every schedule is paired against uniform on an IDENTICAL initialization and
# identical per-client budgets within each trial, and scored on the same
# held-out val.json used everywhere else.
# =============================================================================
import os, sys, json, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fl_lightcnn_final import TinyDetector, ShardDataset, Client, fedavg, evaluate_map

N           = 5
CR_TOTAL    = 8
EPOCH_LOCAL = 7
BATCH_SIZE  = 8
LR          = 1e-3
C_CLIP      = 1.0
RHO_MIN, RHO_MAX = 20.0, 40.0
SIGMA_MIN, SIGMA_MAX = 1.0, 20.0

N_TRIALS   = 10
K_VALUES   = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
N_RANDOM   = 6

SHARD_DIR = './fl_workdirs/lightcnn_shards'
VAL_ANN   = 'single_test_coco/annotations/val.json'
OUT_PATH  = './exp_data/uniform_optimality_check.json'


def rho_to_sigma(rho_t, el):
    return float(np.clip(np.sqrt(2 * el / max(rho_t, 1e-9)), SIGMA_MIN, SIGMA_MAX))


def power_allocation(rho_total, T, k):
    """rho_t proportional to (t+1)^k, summing to exactly rho_total."""
    w = np.array([(t + 1) ** k for t in range(T)], dtype=float)
    w = w / w.sum()
    return (w * rho_total).tolist()


def random_allocation(rho_total, T, rng):
    """Uniform draw from the simplex (Dirichlet(1,...,1))."""
    w = rng.dirichlet(np.ones(T))
    return (w * rho_total).tolist()


def quiet_eval(model, ds):
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return evaluate_map(model, ds)


def run_allocation(clients, allocs_per_client, init_sd, val_ds):
    """allocs_per_client[i][t] = rho spent by client i in round t."""
    global_sd = {k: v.clone() for k, v in init_sd.items()}
    model = TinyDetector().cuda()
    model.load_state_dict(global_sd)
    accs = [quiet_eval(model, val_ds)]
    clipped = 0
    total = 0
    for CR in range(CR_TOTAL):
        local_sds = []
        for i, c in enumerate(clients):
            raw = np.sqrt(2 * c.el / max(allocs_per_client[i][CR], 1e-9))
            total += 1
            if raw < SIGMA_MIN or raw > SIGMA_MAX:
                clipped += 1
            sigma = float(np.clip(raw, SIGMA_MIN, SIGMA_MAX))
            local_sds.append(c.train_one_round(global_sd, sigma))
        global_sd = fedavg(local_sds)
        model.load_state_dict(global_sd)
        accs.append(quiet_eval(model, val_ds))
    return accs, clipped, total


if __name__ == '__main__':
    shard_paths = [os.path.join(SHARD_DIR, f'shard_{i}.json') for i in range(N)]
    assert all(os.path.exists(p) for p in shard_paths), 'run fl_lightcnn_final.py first'
    clients = [Client(i, shard_paths[i], C=C_CLIP, bs=BATCH_SIZE, el=EPOCH_LOCAL, lr=LR)
               for i in range(N)]
    val_ds = ShardDataset(VAL_ANN, device='cuda:0')
    print(f'Held-out val: {val_ds.n} images | {N} clients | {CR_TOTAL} rounds', flush=True)
    print(f'{N_TRIALS} trials x ({len(K_VALUES)} power-law schedules + {N_RANDOM} random)\n', flush=True)

    results = {}   # label -> list of final mAP, one per trial
    labels = [f'k={k:+.1f}' + ('  (UNIFORM)' if k == 0.0 else '') for k in K_VALUES]
    labels += [f'random_{j}' for j in range(N_RANDOM)]
    for lab in labels:
        results[lab] = []
    clip_report = []

    for t in range(N_TRIALS):
        rng = np.random.RandomState(9000 + t)
        random.seed(9000 + t); np.random.seed(9000 + t)
        rhos = [float(rng.uniform(RHO_MIN, RHO_MAX)) for _ in range(N)]
        init_sd = TinyDetector().cuda().state_dict()   # SHARED across every schedule in this trial

        line = []
        for k, lab in zip(K_VALUES, labels[:len(K_VALUES)]):
            allocs = [power_allocation(rhos[i], CR_TOTAL, k) for i in range(N)]
            accs, cl, tot = run_allocation(clients, allocs, init_sd, val_ds)
            results[lab].append(accs[-1])
            clip_report.append((lab, cl, tot))
            line.append(f'{lab}={accs[-1]:.3f}')

        for j in range(N_RANDOM):
            lab = f'random_{j}'
            allocs = [random_allocation(rhos[i], CR_TOTAL, rng) for i in range(N)]
            accs, cl, tot = run_allocation(clients, allocs, init_sd, val_ds)
            results[lab].append(accs[-1])
            clip_report.append((lab, cl, tot))
            line.append(f'{lab}={accs[-1]:.3f}')

        print(f'[Trial {t+1}] ' + ' '.join(line), flush=True)

    uni_lab = [l for l in labels if 'UNIFORM' in l][0]
    uni = np.array(results[uni_lab])

    print('\n=== Paired vs UNIFORM (same init + same budgets within each trial) ===', flush=True)
    summary = []
    for lab in labels:
        arr = np.array(results[lab])
        diff = arr - uni
        sd = diff.std(ddof=1) if len(diff) > 1 else 0.0
        # paired t statistic
        tstat = diff.mean() / (sd / np.sqrt(len(diff))) if sd > 1e-12 else 0.0
        summary.append((lab, arr.mean(), diff.mean(), sum(diff > 0), len(diff), tstat))
        print(f'{lab:22s} mean={arr.mean():.4f}  vs_uniform={diff.mean():+.4f}  '
              f'beats_uniform={sum(diff>0)}/{len(diff)}  t={tstat:+.2f}', flush=True)

    best = max((s for s in summary if 'UNIFORM' not in s[0]), key=lambda s: s[2])
    print(f'\nBest non-uniform schedule: {best[0]}  ({best[2]:+.4f} vs uniform, '
          f'beats it {best[3]}/{best[4]} trials, t={best[5]:+.2f})', flush=True)
    nclip = sum(c for _, c, _ in clip_report); ntot = sum(tt for _, _, tt in clip_report)
    print(f'sigma clipping bound in {nclip}/{ntot} client-rounds ({100*nclip/max(ntot,1):.1f}%)', flush=True)

    with open(OUT_PATH, 'w') as f:
        json.dump({'results': results, 'uniform_label': uni_lab}, f, indent=1)
    print(f'\nSaved {OUT_PATH}', flush=True)
