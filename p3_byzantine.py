# =============================================================================
# P3: Byzantine resilience of the DP-FL setup.
#
# Reproduces and extends the attack/defence analysis of exploiting_dp.pdf
# (Hossain et al., IEEE TAI 6(11), 2025) on the anti-UAV detector.
#
# ATTACKS
#   RMD       random malicious device -- the conventional baseline: a malicious
#             client returns a random update, making no attempt to hide.
#   a-MPELM   adaptive model poisoning with episodic loss memorisation -- the
#             paper's attack. It adapts a poisoning degree gamma each round from
#             whether it was caught last round, and hides inside the legitimate
#             DP-noise envelope.
#
#             OUR TWIST, forced by measurement (p3_envelope.py): in this model
#             the honest update norm is astonishingly tight -- relative sd 0.10%
#             at sigma=8, matching the 1/sqrt(2D) concentration of a Gaussian
#             norm in D=566,789 dimensions. The paper's "hide in the noise
#             magnitude" room therefore does NOT exist here (2% at sigma=1.83,
#             <1% above). What does exist is direction: a norm detector is blind
#             to it, and in 566k dimensions a rotated update is essentially
#             orthogonal to the honest one. So a-MPELM is implemented as a
#             NORM-PRESERVING DIRECTION ATTACK, the strongest form available.
#
# DEFENCES
#   norm      reject updates whose norm deviates > beta1 MADs from the median
#   accuracy  server scores each update on a small held-out set, rejects those
#             that degrade it by more than beta2
#   mix       either of the above fires
#   krum      select the single most geometrically central update (Blanchard+)
#   median    coordinate-wise median
#   trimmed   coordinate-wise trimmed mean
#   The last three are NOT in the paper; they see direction, which the paper's
#   detectors do not, and are what turn this from a replication into a result.
#
# The server's detector validation set is DISJOINT from the reported metric set
# (first 300 val frames for detection, last 1200 for reporting).
# =============================================================================
import os, sys, json, time, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_DB_sweep as G
from gpu_guard import GUARD

DATA = './fl_workdirs/clear'
PUBLIC = './fl_workdirs/clear/public_pretrained.pt'
OUT_PATH = './exp_data/p3_byzantine.json'

N_CLIENTS = 10                 # split from the 5 cached shards -> m = 10/20/30%
ROUNDS, B, LR = 25, 128, 0.003
SIGMA = 1.83                   # uniform at rho=30, the canonical operating point
BETA1 = 3.0                    # norm detector: MADs from the median
BETA2 = 0.02                   # accuracy detector: tolerated IoU.25 drop
TRIM = 0.2
N_DET = 300                    # server detector val frames (disjoint from report)
SEED_BASE = 770000


# ── data: split the 5 cached shards into 10 clients ─────────────────────────
def load_10(device=G.DEV):
    G.DATA, G.TGT = DATA, DATA
    feats, tgts, vf, (vo, vb) = G.load_cut()
    f10, t10 = [], []
    for i in range(len(feats)):
        n = feats[i].shape[0] // 2
        f10 += [feats[i][:n], feats[i][n:2 * n]]
        t10 += [(tgts[i][0][:n], tgts[i][1][:n]),
                (tgts[i][0][n:2 * n], tgts[i][1][n:2 * n])]
    return f10, t10, vf, vo, vb


def flat(sd, ref):
    return torch.cat([(sd[k].float() - ref[k].float()).flatten() for k in ref])


def unflat(vec, ref):
    out, o = {}, 0
    for k in ref:
        n = ref[k].numel()
        out[k] = (ref[k].float() + vec[o:o + n].view(ref[k].shape)).to(ref[k].dtype)
        o += n
    return out


# ── attacks ─────────────────────────────────────────────────────────────────
def attack_rmd(delta, gamma, rng):
    """Random malicious device: a random direction, scaled by gamma. Makes no
    attempt to match the honest envelope, so a norm detector should see it."""
    r = torch.randn(delta.shape, generator=rng, device=delta.device)
    r = r / r.norm() * delta.norm()
    return (1 - gamma) * delta + gamma * r * (1.0 + gamma)


def attack_mpelm(delta, gamma, rng):
    """a-MPELM: steer against the honest update, then RESCALE to the honest norm
    so the magnitude is indistinguishable. Damage is carried by direction, which
    the norm detector cannot observe."""
    r = torch.randn(delta.shape, generator=rng, device=delta.device)
    r = r / r.norm() * delta.norm()
    adv = (1 - gamma) * delta + gamma * (-delta + 0.3 * r)
    return adv / adv.norm().clamp(min=1e-12) * delta.norm()      # norm-preserving


# ── detectors: return the set of client indices to REJECT ───────────────────
def detect_norm(deltas, beta1):
    n = torch.tensor([float(d.norm()) for d in deltas])
    med = n.median()
    mad = (n - med).abs().median().clamp(min=1e-9)
    return {i for i in range(len(deltas)) if abs(float(n[i] - med)) > beta1 * mad}


def detect_accuracy(deltas, gsd, vf, vo, vb, beta2, base_score):
    rej = set()
    for i, d in enumerate(deltas):
        s = G.evaluate(unflat(d, gsd), vf, vo, vb)
        GUARD.tick()          # these evals bypass client_round's throttle
        if s < base_score - beta2:
            rej.add(i)
    return rej


# ── robust aggregators ──────────────────────────────────────────────────────
def agg_mean(ds):
    return torch.stack(ds).mean(0)


def agg_median(ds):
    return torch.stack(ds).median(0).values


def agg_trimmed(ds, frac=TRIM):
    S = torch.stack(ds)
    k = max(1, int(len(ds) * frac))
    S, _ = S.sort(0)
    return S[k:len(ds) - k].mean(0) if len(ds) - 2 * k > 0 else S.mean(0)


def agg_krum(ds, n_mal):
    S = torch.stack(ds)
    n = len(ds)
    d2 = torch.cdist(S.unsqueeze(0), S.unsqueeze(0)).squeeze(0) ** 2
    GUARD.tick()
    k = max(1, n - n_mal - 2)
    scores = [d2[i].sort().values[1:k + 1].sum() for i in range(n)]
    return ds[int(torch.tensor(scores).argmin())]


# ── one federated run ───────────────────────────────────────────────────────
def run(attack, defence, m, seed, feats, tgts, vf_det, vo_det, vb_det,
        vf_rep, vo_rep, vb_rep, rounds=ROUNDS):
    init = G.make_init('public_tail', seed, PUBLIC)
    gsd = {k: v.clone() for k, v in init.items()}
    rng = torch.Generator(device=G.DEV); rng.manual_seed(seed + 5)
    mal = set(range(m))
    gamma = 0.5
    hist = {'acc': [], 'gamma': [], 'tp': [], 'fp': [], 'detected': []}
    for r in range(rounds):
        torch.manual_seed(seed * 31 + r)
        locals_ = [G.client_round(gsd, feats[i], tgts[i][0], tgts[i][1],
                                  SIGMA, True, B, LR) for i in range(N_CLIENTS)]
        deltas = [flat(lo, gsd) for lo in locals_]
        for i in sorted(mal):
            if attack == 'rmd':
                deltas[i] = attack_rmd(deltas[i], gamma, rng)
            elif attack == 'mpelm':
                deltas[i] = attack_mpelm(deltas[i], gamma, rng)

        rej = set()
        if defence in ('norm', 'mix'):
            rej |= detect_norm(deltas, BETA1)
        if defence in ('accuracy', 'mix'):
            base = G.evaluate(gsd, vf_det, vo_det, vb_det)
            rej |= detect_accuracy(deltas, gsd, vf_det, vo_det, vb_det, BETA2, base)

        tp = len(rej & mal); fp = len(rej - mal)
        keep = [deltas[i] for i in range(N_CLIENTS) if i not in rej] or deltas

        if defence == 'krum':
            agg = agg_krum(keep, max(m, 1))
        elif defence == 'median':
            agg = agg_median(keep)
        elif defence == 'trimmed':
            agg = agg_trimmed(keep)
        else:
            agg = agg_mean(keep)
        gsd = unflat(agg, gsd)

        # episodic loss memorisation: back off when caught, press when not
        if attack == 'mpelm':
            gamma = min(1.0, gamma * 1.15) if tp == 0 else max(0.05, gamma * 0.6)

        hist['acc'].append(G.evaluate(gsd, vf_rep, vo_rep, vb_rep))
        GUARD.tick()
        hist['gamma'].append(float(gamma))
        hist['tp'].append(tp); hist['fp'].append(fp)
        hist['detected'].append(sorted(rej))
    return hist


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=ROUNDS)
    a = ap.parse_args()

    print('loading + splitting to %d clients...' % N_CLIENTS, flush=True)
    feats, tgts, vf, vo, vb = load_10()
    vf_det, vo_det, vb_det = vf[:N_DET], vo[:N_DET], vb[:N_DET]
    vf_rep, vo_rep, vb_rep = vf[N_DET:], vo[N_DET:], vb[N_DET:]
    print('  %d clients x %d frames | detector val %d | report val %d | sigma=%.2f\n'
          % (N_CLIENTS, feats[0].shape[0], vf_det.shape[0], vf_rep.shape[0], SIGMA),
          flush=True)

    ATT = ['none', 'rmd', 'mpelm']
    DEF = ['none', 'norm', 'accuracy', 'mix', 'krum', 'median', 'trimmed']
    MS = [1, 2, 3]                       # 10%, 20%, 30% of 10 clients

    # Full defence comparison at the hardest ratio (m=3, 30%), plus an m-sweep
    # on the three defences the figures need. A full 3x7x3 grid would be ~8 hr.
    grid = [('none', d, 0) for d in DEF]
    for at in ('rmd', 'mpelm'):
        for d in DEF:
            grid.append((at, d, 3))
    for at in ('rmd', 'mpelm'):
        for d in ('norm', 'mix', 'krum'):
            for m in (1, 2):
                grid.append((at, d, m))
    print('%d configs x %d trials\n' % (len(grid), a.trials), flush=True)

    res = {}
    if os.path.exists(OUT_PATH):
        try:
            res = json.load(open(OUT_PATH)).get('results', {})
            print('resuming: %d configs already complete\n' % len(res), flush=True)
        except Exception:
            res = {}
    t0 = time.time()
    for gi, (at, df, m) in enumerate(grid):
        key = '%s|%s|%d' % (at, df, m)
        if key in res and len(res[key]) >= a.trials:
            continue
        res[key] = []
        for t in range(a.trials):
            h = run(at, df, m, SEED_BASE + t, feats, tgts,
                    vf_det, vo_det, vb_det, vf_rep, vo_rep, vb_rep, a.rounds)
            res[key].append(h)
        fin = np.mean([np.mean(h['acc'][-3:]) for h in res[key]])
        tpr = np.mean([np.sum(h['tp']) / max(1, m * a.rounds) for h in res[key]]) if m else 0.0
        fpr = np.mean([np.sum(h['fp']) / max(1, (N_CLIENTS - m) * a.rounds) for h in res[key]])
        print('[%2d/%2d] %-6s %-9s m=%d | acc=%.4f  TPR=%.2f FPR=%.2f  [%.0fm]'
              % (gi + 1, len(grid), at, df, m, fin, tpr, fpr, (time.time() - t0) / 60),
              flush=True)
        json.dump({'n_clients': N_CLIENTS, 'sigma': SIGMA, 'rounds': a.rounds,
                   'beta1': BETA1, 'beta2': BETA2, 'trials': a.trials,
                   'seed_base': SEED_BASE, 'results': res}, open(OUT_PATH, 'w'), indent=1)
    print('\nP3 BYZANTINE DONE', flush=True)
