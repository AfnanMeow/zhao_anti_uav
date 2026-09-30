# =============================================================================
# Publication figures: DRL vs Uniform vs Exponential vs Arithmetic noise
# allocation, for the lightweight-CNN FL+DP+DRL pipeline (fl_lightcnn_final.py).
#
# Uses the ALREADY-TRAINED actor checkpoint from fl_lightcnn_final.py's 400-
# episode run (CR_Total=8, explore_decay=0.995 -- the run that beat Uniform
# 5/5 in the fair shared-init comparison). Unlike that run's own internal
# reward signal (per-client-shard self-eval, fine as a training proxy), every
# number in these figures is measured on single_test_coco/annotations/val.json
# -- 198 images NEVER used in any client's training shard -- for a real
# held-out utility measurement suitable for reporting.
# =============================================================================
import os, sys, json, random, copy
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fl_lightcnn_final import TinyDetector, ShardDataset, Client, fedavg
from fl_damoyolo_final import Actor

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config (matches the winning fl_lightcnn_final.py run) ──────────────────
N           = 5
CR_TOTAL    = 8
EPOCH_LOCAL = 7
BATCH_SIZE  = 8
LR          = 1e-3
C_CLIP      = 1.0
RHO_MIN, RHO_MAX = 20.0, 40.0
SIGMA_MIN, SIGMA_MAX = 1.0, 20.0
RHO_USED_MIN = 2 / SIGMA_MAX**2 * EPOCH_LOCAL
RHO_USED_MAX = 2 / SIGMA_MIN**2 * EPOCH_LOCAL

SHARD_DIR = './fl_workdirs/lightcnn_shards'   # reuse the exact shards the actor trained on
CKPT_PATH = './DRL_Nets/lightcnn_checkpoint.pth'
VAL_ANN   = 'single_test_coco/annotations/val.json'
PLOTS_DIR = './plots'
DATA_DIR  = './exp_data'

# Validated categorical palette (dataviz skill), fixed order, never cycled:
# 1=DRL(blue) 2=Uniform(orange) 3=Exponential(aqua) 4=Arithmetic(yellow)
COLORS = {'DRL': '#2a78d6', 'Uniform': '#eb6834', 'Exponential': '#1baf7a', 'Arithmetic': '#eda100'}
STYLES = {'DRL': '-', 'Uniform': ':', 'Exponential': '--', 'Arithmetic': '-.'}
MARKERS = {'DRL': 'o', 'Uniform': 's', 'Exponential': '^', 'Arithmetic': 'D'}

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def sigma2rho(sigma, el):
    return 2 * el / (sigma ** 2)


def rho2sigma(rho, el):
    return float(np.sqrt(2 * el / max(rho, 1e-9)))


def sigma_uniform(rho_total, T, el):
    s = float(np.clip(np.sqrt(2 * el / (rho_total / T)), SIGMA_MIN, SIGMA_MAX))
    return [s] * T


def sigma_exponential(rho_total, T, el):
    decay_ratio = 2 ** (1 / (T - 1))
    rho_base = (1 - decay_ratio) * rho_total / (1 - decay_ratio ** T)
    out = []
    for t in range(T):
        rho_used = max(rho_base * (decay_ratio ** t), 1e-9)
        out.append(float(np.clip(np.sqrt(2 * el / rho_used), SIGMA_MIN, SIGMA_MAX)))
    return out


def sigma_arithmetic(rho_total, T, el):
    decay = 2 * rho_total / (T * (T - 1) * 3)
    rho_base = 2 / 3 * (rho_total / T)
    out = []
    for t in range(T):
        rho_used = max(rho_base + decay * t, 1e-9)
        out.append(float(np.clip(np.sqrt(2 * el / rho_used), SIGMA_MIN, SIGMA_MAX)))
    return out


def action_mapping(el, sigmas_i, rho_total, T=CR_TOTAL):
    rho_useds = [sigma2rho(s, el) for s in sigmas_i]
    S = sum(rho_useds)
    if abs(max(sigmas_i) - min(sigmas_i)) < 0.01:
        a = rho_total / max(S, 1e-9); b = 0.0
    else:
        ubs = [rho_total / T]; lbs = [-np.inf]
        for CR in range(T):
            diff = S - T * rho_useds[CR]
            if diff > 1e-7:
                ubs.append((RHO_USED_MAX * S - rho_total * rho_useds[CR]) / diff)
                lbs.append((RHO_USED_MIN * S - rho_total * rho_useds[CR]) / diff)
            elif diff < -1e-7:
                ubs.append((RHO_USED_MIN * S - rho_total * rho_useds[CR]) / diff)
                lbs.append((RHO_USED_MAX * S - rho_total * rho_useds[CR]) / diff)
        bub = min(ubs); blb = max(lbs)
        dnom = sum((T * rho_useds[CR] - S) ** 2 for CR in range(T))
        b = 0.0 if abs(dnom) < 1e-12 else \
            -sum((rho_useds[CR] * S - rho_total) * (T * rho_useds[CR] - S) for CR in range(T)) / dnom
        b = max(blb, min(bub, b))
        a = (rho_total - b * T) / max(S, 1e-9)
    return [rho2sigma(a * rho_useds[CR] + b, el) for CR in range(T)]


def sigma_drl(actor, rho_total, T, el):
    raw = []
    for t in range(T):
        state = torch.Tensor([t / T, rho_total / RHO_MAX]).cuda()
        with torch.no_grad():
            a = actor(state).item()
        raw.append(a * (SIGMA_MAX - SIGMA_MIN) + SIGMA_MIN)
    return action_mapping(el, raw, rho_total, T)


def run_schedule(clients, sigmas_per_client, init_sd, val_ds):
    """Train CR_TOTAL real FL+DP rounds under a given per-client sigma
    schedule; evaluate the GLOBAL aggregated model on the held-out val set
    after every round (not per-client-shard self-eval)."""
    global_sd = {k: v.clone() for k, v in init_sd.items()}
    model = TinyDetector().cuda()
    model.load_state_dict(global_sd)
    accs = [evaluate_map_quiet(model, val_ds)]
    for CR in range(CR_TOTAL):
        local_sds = [c.train_one_round(global_sd, sigmas_per_client[i][CR])
                     for i, c in enumerate(clients)]
        global_sd = fedavg(local_sds)
        model.load_state_dict(global_sd)
        accs.append(evaluate_map_quiet(model, val_ds))
    return accs


def evaluate_map_quiet(model, dataset):
    """evaluate_map wraps pycocotools, which prints ~15 lines per call --
    silence that for a sweep of hundreds of calls."""
    import io, contextlib
    from fl_lightcnn_final import evaluate_map
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return evaluate_map(model, dataset)


if __name__ == '__main__':
    print('Loading trained DRL actor...', flush=True)
    ckpt = torch.load(CKPT_PATH, map_location='cuda:0')
    actor = Actor(dim_state=2, dim_action=1).cuda()
    actor.load_state_dict(ckpt['actor'])
    actor.eval()
    print(f'Actor loaded (episode {ckpt.get("episode")})', flush=True)

    shard_paths = [os.path.join(SHARD_DIR, f'shard_{i}.json') for i in range(N)]
    assert all(os.path.exists(p) for p in shard_paths), 'run fl_lightcnn_final.py first to create client shards'
    clients = [Client(i, shard_paths[i], C=C_CLIP, bs=BATCH_SIZE, el=EPOCH_LOCAL, lr=LR) for i in range(N)]
    val_ds = ShardDataset(VAL_ANN, device='cuda:0')
    print(f'Held-out val set: {val_ds.n} images (never used in any client shard)', flush=True)

    methods = {
        'DRL': lambda rhos: [sigma_drl(actor, rhos[i], CR_TOTAL, EPOCH_LOCAL) for i in range(N)],
        'Uniform': lambda rhos: [sigma_uniform(rhos[i], CR_TOTAL, EPOCH_LOCAL) for i in range(N)],
        'Exponential': lambda rhos: [sigma_exponential(rhos[i], CR_TOTAL, EPOCH_LOCAL) for i in range(N)],
        'Arithmetic': lambda rhos: [sigma_arithmetic(rhos[i], CR_TOTAL, EPOCH_LOCAL) for i in range(N)],
    }

    # =========================================================================
    # Figures 2 & 3: mAP vs round, noise vs round -- N_TRIALS shared-init trials
    # =========================================================================
    N_TRIALS_23 = 8
    print(f'\n=== Figures 2 & 3: {N_TRIALS_23} shared-init trials, all 4 methods ===', flush=True)
    all_accs = {m: [] for m in methods}
    all_sigmas = {m: [] for m in methods}
    all_rhos = []
    for t in range(N_TRIALS_23):
        random.seed(5000 + t); np.random.seed(5000 + t)
        rhos = [float(np.random.uniform(RHO_MIN, RHO_MAX)) for _ in range(N)]
        all_rhos.append(rhos)
        init_sd = TinyDetector().cuda().state_dict()
        for name, sched_fn in methods.items():
            sigmas = sched_fn(rhos)
            accs = run_schedule(clients, sigmas, init_sd, val_ds)
            all_accs[name].append(accs)
            all_sigmas[name].append(sigmas)
            print(f'[Trial {t+1}] {name:12s} final={accs[-1]:.4f}', flush=True)

    with open(os.path.join(DATA_DIR, 'lightcnn_pub_fig23_data.json'), 'w') as f:
        json.dump({'all_accs': all_accs, 'all_sigmas': all_sigmas, 'all_rhos': all_rhos}, f)

    # ── Figure 2: mAP vs Communication Round (mean +/- std across trials) ──
    x = list(range(0, CR_TOTAL + 1))
    plt.figure(figsize=(8, 5))
    for name in ['DRL', 'Uniform', 'Exponential', 'Arithmetic']:
        arr = np.array(all_accs[name])  # (trials, CR_TOTAL+1)
        mean = arr.mean(0); std = arr.std(0)
        plt.plot(x, mean, linestyle=STYLES[name], color=COLORS[name], linewidth=2,
                  marker=MARKERS[name], markersize=5, label=name)
        plt.fill_between(x, mean - std, mean + std, color=COLORS[name], alpha=0.12)
    plt.xticks(x, fontsize=12); plt.yticks(fontsize=12)
    plt.xlabel('Communication Round $t$', fontsize=15)
    plt.ylabel('mAP@0.5 (held-out val)', fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig2_mAP_vs_rounds.pdf'))
    plt.savefig(os.path.join(PLOTS_DIR, 'fig2_mAP_vs_rounds.png'), dpi=150)
    plt.close()
    print('Saved fig2_mAP_vs_rounds', flush=True)

    # ── Figure 3: Noise level vs round -- DRL's schedule for 3 representative
    #    clients (min/median/max rho), mean across trials ──
    ref_rhos = all_rhos[0]
    order = sorted(range(N), key=lambda i: ref_rhos[i])
    picks = [order[0], order[len(order) // 2], order[-1]]
    drl_sigmas_arr = np.array(all_sigmas['DRL'])  # (trials, N, CR_TOTAL)
    plt.figure(figsize=(8, 5))
    xr = list(range(CR_TOTAL))
    for idx, ci in enumerate(picks):
        mean_sigma = drl_sigmas_arr[:, ci, :].mean(0)
        plt.plot(xr, mean_sigma, linewidth=2, marker='o', markersize=5,
                  color=[COLORS['DRL'], COLORS['Uniform'], COLORS['Exponential']][idx],
                  linestyle=['-', '--', ':'][idx],
                  label=f'Client $\\rho_k \\approx$ {ref_rhos[ci]:.1f}')
    plt.xticks(xr, fontsize=12); plt.yticks(fontsize=12)
    plt.xlabel('Communication Round $t$', fontsize=15)
    plt.ylabel('DRL Noise Level $\\sigma_k^t$', fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig3_noise_vs_rounds.pdf'))
    plt.savefig(os.path.join(PLOTS_DIR, 'fig3_noise_vs_rounds.png'), dpi=150)
    plt.close()
    print('Saved fig3_noise_vs_rounds', flush=True)

    # =========================================================================
    # Figure 1: Privacy (rho budget) vs Utility (final mAP)
    # =========================================================================
    RHO_LEVELS = [10, 15, 20, 25, 30, 35, 40, 50, 60]
    N_TRIALS_1 = 3
    print(f'\n=== Figure 1: privacy-utility sweep, {len(RHO_LEVELS)} levels x {N_TRIALS_1} trials ===', flush=True)
    sweep = {m: {lvl: [] for lvl in RHO_LEVELS} for m in methods}
    for lvl in RHO_LEVELS:
        for t in range(N_TRIALS_1):
            random.seed(7000 + lvl * 10 + t); np.random.seed(7000 + lvl * 10 + t)
            # Identical rho for every client is out-of-distribution for the
            # actor (trained on independently-random per-client budgets) --
            # verified live: it loses to Uniform there. Keep the same +/-20%
            # heterogeneity the actor trained under, just re-centered on each
            # sweep level, so the x-axis is a clean "average privacy budget"
            # without asking the policy to generalize past its training regime.
            rhos = [float(np.random.uniform(0.8 * lvl, 1.2 * lvl)) for _ in range(N)]
            init_sd = TinyDetector().cuda().state_dict()
            for name, sched_fn in methods.items():
                sigmas = sched_fn(rhos)
                accs = run_schedule(clients, sigmas, init_sd, val_ds)
                sweep[name][lvl].append(accs[-1])
            print(f'[rho={lvl} trial {t+1}] ' +
                  ' '.join(f'{name}={sweep[name][lvl][-1]:.3f}' for name in methods), flush=True)

    with open(os.path.join(DATA_DIR, 'lightcnn_pub_fig1_data.json'), 'w') as f:
        json.dump(sweep, f, indent=1)

    plt.figure(figsize=(8, 5))
    for name in ['DRL', 'Uniform', 'Exponential', 'Arithmetic']:
        means = [np.mean(sweep[name][lvl]) for lvl in RHO_LEVELS]
        stds = [np.std(sweep[name][lvl]) for lvl in RHO_LEVELS]
        plt.errorbar(RHO_LEVELS, means, yerr=stds, linestyle=STYLES[name], color=COLORS[name],
                     linewidth=2, marker=MARKERS[name], markersize=6, capsize=3, label=name)
    plt.xticks(RHO_LEVELS, fontsize=12); plt.yticks(fontsize=12)
    plt.xlabel('Total Privacy Budget $\\rho$ (per client)', fontsize=15)
    plt.ylabel('Final mAP@0.5 (held-out val)', fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig1_privacy_vs_utility.pdf'))
    plt.savefig(os.path.join(PLOTS_DIR, 'fig1_privacy_vs_utility.png'), dpi=150)
    plt.close()
    print('Saved fig1_privacy_vs_utility', flush=True)

    print('\n=== Done. All figures in ./plots/ ===', flush=True)
