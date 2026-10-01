# =============================================================================
# P3-H: Byzantine resilience under HETEROGENEOUS per-client sigma.
#
# Plan: p3_only/fresh_beginning.md §2b, §5, §5a, §6.  Runbook: NEXT_EXPERIMENTS.md.
#
# Identical to p3_byzantine.py (attacks, detectors, 10 clients, B=128, lr=0.003,
# 15 rounds, BETA1/BETA2, seeds) except for four things:
#
#   1. SIGMA MODES (--sigma)
#        uniform    every client sigma=1.826 every round          (Arm A, control)
#        hetero     rho_i ~ U(20,40) per client, constant over rounds (Arm A')
#        adaptive   rho_i ~ U(20,40), spent over rounds on a per-client
#                   exponential schedule rho_it ~ exp(k_i t/(R-1)), k_i ~ U(-1,1)
#                   (Arms B-F). A stand-in for a learned per-client policy: it
#                   makes sigma_i,t non-stationary, which is the property §2b C3
#                   says adaptive allocation adds. Mean per-round rho = A'.
#        ladder:S   deterministic geometric spread centred on 1.826:
#                   sigma_i = 1.826 * exp(S * z_i), z evenly spaced in [-1,1]
#                   (controlled sigma ladder for figH3/H4/H5)
#      sigma = sqrt(2*EL*T/rho) with EL=2, T=25, so rho=30 -> 1.826 (§2b C6).
#      The rho draw depends only on the trial seed, so all arms of one trial see
#      the same budgets (common random numbers).
#
#   2. DETECTORS (--defences): none / norm / norm_sn / accuracy / mix / mix_sn.
#      norm_sn divides each norm by the client's DECLARED sigma (§5). Both norm
#      detectors are evaluated as SHADOWS every round whatever the active
#      defence is, and their thresholds logged, so figures never need a re-run.
#
#   3. SIGMA LYING (--lie R, Arm F): malicious clients declare R*sigma_true.
#      --cap clamps declared sigma to the range an honest client could have.
#
#   4. SESSION-DISJOINT detector/report split (NEXT_EXPERIMENTS.md §4), from the
#      per-frame session labels stage1_clear.py now writes to manifest.json. The
#      original positional 300/1200 cut landed inside a session.
#
# Per trial x round x client it logs norm, sigma_true, sigma_declared, malicious,
# rejected (§2b), plus both detectors' thresholds. Outputs go ONLY to
# p3_only/p3h_heterogeneous_sigma/results/, never exp_data/.
# =============================================================================
import os, sys, gc, json, time, argparse, platform, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
os.chdir(ROOT)                          # the shared modules use ./fl_workdirs paths
sys.path.insert(0, ROOT)
import gate_DB_sweep as G
import p3_byzantine as P               # attacks, detectors, split-to-10 (unchanged)
from gpu_guard import GUARD

# On Windows (WDDM) an over-full GPU does not raise OOM: the driver silently pages
# allocations into system RAM and every step crawls over PCIe (seen 2026-09-30:
# 3.1 GB spilled, trials 5-10x slower). Capping the allocator makes it release
# its cached blocks before that point instead. Numerically neutral.
torch.cuda.set_per_process_memory_fraction(float(os.environ.get('P3H_MEM_FRAC', 0.85)))

RESULTS = os.path.join(ROOT, 'p3_only', 'p3h_heterogeneous_sigma', 'results')
MANIFEST = os.path.join(P.DATA, 'manifest.json')

N = P.N_CLIENTS
ROUNDS = 15                             # what the stored P3 baseline ran
EL, T_BUDGET = 2.0, 25                  # sigma accounting constants (§2b C6)
RHO_LO, RHO_HI, RHO_MID = 20.0, 40.0, 30.0
SIG_LO, SIG_HI = 1.0, 20.0              # the paper's clamp; must never bind
DET_FRAC = 0.2                          # target share of val frames for the detector
SEED_BASE = P.SEED_BASE                 # 770000, same as P3


def sigma_of(rho_per_round):
    return float(np.sqrt(2 * EL / rho_per_round))


SIGMA0 = sigma_of(RHO_MID / T_BUDGET)   # 1.826


# ── sigma plans: [rounds][clients] true sigma ────────────────────────────────
def sigma_plan(mode, seed, rounds):
    rng = np.random.default_rng(seed + 11)          # depends on the trial only
    rhos = rng.uniform(RHO_LO, RHO_HI, N)
    ks = rng.uniform(-1.0, 1.0, N)
    perm = rng.permutation(N)
    info = {'rho_total': rhos.tolist(), 'k': None}
    if mode == 'uniform':
        S = np.full((rounds, N), SIGMA0)
        info['rho_total'] = [RHO_MID] * N
    elif mode == 'hetero':
        S = np.tile([sigma_of(r / T_BUDGET) for r in rhos], (rounds, 1))
    elif mode == 'adaptive':
        t = np.arange(rounds) / max(rounds - 1, 1)
        S = np.zeros((rounds, N))
        for i in range(N):
            w = np.exp(ks[i] * t); w = w / w.mean()          # mean per-round rho kept
            S[:, i] = [sigma_of(rhos[i] / T_BUDGET * wt) for wt in w]
        info['k'] = ks.tolist()
    elif mode.startswith('ladder:'):
        s = float(mode.split(':')[1])
        z = np.linspace(-1, 1, N)[perm]
        S = np.tile(SIGMA0 * np.exp(s * z), (rounds, 1))
        info['rho_total'] = None
    else:
        raise ValueError(mode)
    info['clamp_binds'] = bool((S < SIG_LO).any() or (S > SIG_HI).any())
    return S, info


def honest_sigma_range(mode, rounds):
    """What an honest client could possibly declare: the server's cap for Arm F."""
    if mode == 'uniform':
        return SIGMA0, SIGMA0
    lo, hi = sigma_of(RHO_HI / T_BUDGET), sigma_of(RHO_LO / T_BUDGET)
    if mode == 'adaptive':
        w_hi = np.exp(1.0) / np.mean(np.exp(np.linspace(0, 1, rounds)))
        lo, hi = lo / np.sqrt(w_hi), hi * np.sqrt(w_hi)
    if mode.startswith('ladder:'):
        s = float(mode.split(':')[1]); lo, hi = SIGMA0 * np.exp(-s), SIGMA0 * np.exp(s)
    return float(lo), float(hi)


# ── session-disjoint detector / report split ────────────────────────────────
N_DET_POSITIONAL = 300                  # the ORIGINAL P3 split (PREREG_round3 R3-5 only)


def split_val(vf, vo, vb, mode='session'):
    if mode == 'positional':
        n = N_DET_POSITIONAL
        return ((vf[:n], vo[:n], vb[:n]), (vf[n:], vo[n:], vb[n:]),
                {'mode': 'positional', 'n_det': n, 'n_rep': int(vf.shape[0] - n),
                 'detector_sessions': 'first %d frames' % n, 'report_sessions': 'the rest'})
    man = json.load(open(MANIFEST))
    lab = man.get('val_frame_sessions')
    if lab is None or len(lab) != vf.shape[0]:
        raise SystemExit('manifest.json has no per-frame val session labels -- rebuild '
                         'the cache with the patched stage1_clear.py')
    sess = sorted(set(lab))
    cnt = {s: lab.count(s) for s in sess}
    target = DET_FRAC * len(lab)
    det = min(sess, key=lambda s: (abs(cnt[s] - target), s))       # one whole session
    dmask = torch.tensor([s == det for s in lab], device=vf.device)
    split = {'mode': 'session', 'detector_sessions': [det], 'report_sessions': [s for s in sess if s != det],
             'frames_per_session': cnt, 'n_det': int(dmask.sum()), 'n_rep': int((~dmask).sum())}
    return (vf[dmask], vo[dmask], vb[dmask]), (vf[~dmask], vo[~dmask], vb[~dmask]), split


# ── robust-statistic thresholds (same rule as P.detect_norm) ─────────────────
def mad_detect(stat, beta):
    t = torch.tensor(stat, dtype=torch.float64)
    med = t.median(); mad = (t - med).abs().median().clamp(min=1e-12)
    rej = [i for i in range(len(stat)) if abs(float(t[i] - med)) > beta * float(mad)]
    return rej, {'median': float(med), 'mad': float(mad),
                 'lo': float(med - beta * mad), 'hi': float(med + beta * mad)}


def cos_loo(deltas):
    """Cosine of each update with the mean of the OTHER updates (PREREG_round3 R3-1)."""
    S = torch.stack(deltas).double()
    tot = S.sum(0)
    out = []
    for i in range(S.shape[0]):
        o = (tot - S[i]) / (S.shape[0] - 1)
        out.append(float(S[i] @ o / (S[i].norm() * o.norm()).clamp(min=1e-30)))
    return out


def low_detect(stat, beta):
    """One-sided MAD rule: reject only values BELOW median - beta*MAD."""
    t = torch.tensor(stat, dtype=torch.float64)
    med = t.median(); mad = (t - med).abs().median().clamp(min=1e-15)
    rej = [i for i in range(len(stat)) if float(t[i]) < float(med - beta * mad)]
    return rej, {'median': float(med), 'mad': float(mad), 'lo': float(med - beta * mad)}


def krum_index(ds, n_mal):
    """Index of the update P.agg_krum would pick (same scoring)."""
    S = torch.stack(ds)
    d2 = torch.cdist(S.unsqueeze(0), S.unsqueeze(0)).squeeze(0) ** 2
    k = max(1, len(ds) - n_mal - 2)
    return int(torch.stack([d2[i].sort().values[1:k + 1].sum() for i in range(len(ds))]).argmin())


# ── crash-safe saving ───────────────────────────────────────────────────────
# A power cut mid-write must never corrupt the results file: write a temp file,
# fsync it, then atomically replace. The per-round checkpoint means a restart
# resumes inside the interrupted trial, losing at most one round. Every round
# reseeds torch (seed*31+r) and the attacker RNG state is saved, so a resumed
# trial follows the same random stream as an uninterrupted one.
def _atomic(write, path):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as fh:
        write(fh)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)


def save_json(obj, path):
    _atomic(lambda fh: fh.write(json.dumps(obj).encode()), path)


def save_ckpt(obj, path):
    _atomic(lambda fh: torch.save(obj, fh), path)


# ── one federated run ───────────────────────────────────────────────────────
def run(mode, attack, defence, m, seed, lie, cap, data, rounds, onset=0,
        ckpt=None, ckey=None, lie_track=False):
    feats, tgts, det_set, rep_set = data
    S, info = sigma_plan(mode, seed, rounds)
    cap_lo, cap_hi = honest_sigma_range(mode, rounds)
    init = G.make_init('public_tail', seed, P.PUBLIC)
    gsd = {k: v.clone() for k, v in init.items()}
    rng = torch.Generator(device=G.DEV); rng.manual_seed(seed + 5)
    mal = set(range(m))
    gamma = 1.0 if attack == 'mpelm_fixed' else 0.5
    scale = {i: 1.0 for i in mal}       # mpelm_scale: per-attacker norm inflation (PREREG_round2 G4)
    rec = {'seed': seed, 'sigma_info': info, 'cap_range': [cap_lo, cap_hi], 'rounds': []}
    start, spent = 0, 0.0
    if ckpt and os.path.exists(ckpt):
        c = torch.load(ckpt, map_location=G.DEV, weights_only=False)
        if c['ckey'] == ckey:
            gsd, gamma, rec = c['gsd'], c['gamma'], c['rec']
            scale = c.get('scale', scale)
            rng.set_state(c['rng'].cpu())
            start, spent = c['next_round'], c['seconds']
            print('    resuming %s at round %d/%d' % (ckey, start + 1, rounds), flush=True)
    t0 = time.time() - spent
    g0 = (GUARD.slept_s, GUARD.paused_s)       # throttle accounting for this session
    for r in range(start, rounds):
        torch.manual_seed(seed * 31 + r)
        s_true = [float(S[r, i]) for i in range(N)]
        locals_ = [G.client_round(gsd, feats[i], tgts[i][0], tgts[i][1],
                                  s_true[i], True, P.B, P.LR) for i in range(N)]
        deltas = [P.flat(lo, gsd) for lo in locals_]
        act = mal if r >= onset else set()      # clients actually attacking this round
        for i in sorted(act):
            if attack == 'rmd':
                deltas[i] = P.attack_rmd(deltas[i], gamma, rng)
            elif attack == 'mpelm':
                deltas[i] = P.attack_mpelm(deltas[i], gamma, rng)
            elif attack == 'mpelm_fixed':        # PREREG_round4: gamma pinned at 1, never backs off
                deltas[i] = P.attack_mpelm(deltas[i], 1.0, rng)
            elif attack == 'mpelm_scale':
                deltas[i] = P.attack_mpelm(deltas[i], gamma, rng) * scale[i]

        # lie_track (Arm F2): the attacker declares sigma inflated by its own scale
        s_decl = [s * ((scale[i] if lie_track else lie) if i in act else 1.0)
                  for i, s in enumerate(s_true)]
        if cap:
            s_decl = [min(max(s, cap_lo), cap_hi) for s in s_decl]
        norms = [float(d.norm()) for d in deltas]
        raw_rej, raw_thr = mad_detect(norms, P.BETA1)
        sn_rej, sn_thr = mad_detect([n / s for n, s in zip(norms, s_decl)], P.BETA1)
        coss = cos_loo(deltas)
        cos_rej, cos_thr = low_detect(coss, P.BETA1)

        rej, acc_rej = set(), None
        if defence in ('norm', 'mix'):
            rej |= set(raw_rej)
        if defence in ('norm_sn', 'mix_sn', 'cos_sn'):
            rej |= set(sn_rej)
        if defence in ('cos', 'cos_sn'):
            rej |= set(cos_rej)
        if defence == 'krum':
            pick = krum_index(deltas, max(m, 1))
            rej = set(range(N)) - {pick}
        if defence in ('accuracy', 'mix', 'mix_sn'):
            base = G.evaluate(gsd, *det_set)
            acc_rej = sorted(P.detect_accuracy(deltas, gsd, *det_set, P.BETA2, base))
            rej |= set(acc_rej)

        tp = len(rej & act); fp = len(rej - act)
        keep = [deltas[i] for i in range(N) if i not in rej] or deltas
        agg = {'median': P.agg_median, 'trimmed': P.agg_trimmed}.get(defence, P.agg_mean)
        gsd = P.unflat(agg(keep), gsd)
        if attack in ('mpelm', 'mpelm_scale'):
            gamma = min(1.0, gamma * 1.15) if tp == 0 else max(0.05, gamma * 0.6)
        scale_used = {i: scale[i] for i in act}
        if attack == 'mpelm_scale':
            for i in act:
                scale[i] = max(1.0, 0.7 * scale[i]) if i in rej else min(3.0, 1.1 * scale[i])
        acc = G.evaluate(gsd, *rep_set)
        GUARD.tick()
        rec['rounds'].append({
            'acc': acc, 'gamma': float(gamma), 'tp': tp, 'fp': fp,
            'norm': norms, 'sigma_true': s_true, 'sigma_declared': s_decl,
            'malicious': [i in act for i in range(N)],
            'rejected': [i in rej for i in range(N)],
            'shadow_norm_rej': raw_rej, 'shadow_norm_sn_rej': sn_rej,
            'accuracy_rej': acc_rej, 'thr_norm': raw_thr, 'thr_norm_sn': sn_thr,
            'cos': coss, 'shadow_cos_rej': cos_rej, 'thr_cos': cos_thr,
            'scale': {str(k): v for k, v in scale_used.items()}})
        if ckpt:
            save_ckpt({'ckey': ckey, 'next_round': r + 1, 'gsd': gsd, 'gamma': gamma, 'scale': scale,
                       'rec': rec, 'rng': rng.get_state(), 'seconds': time.time() - t0}, ckpt)
            if os.environ.get('P3H_TEST_CRASH_AFTER') == str(r + 1):   # resume test only
                os._exit(3)
    rec['seconds'] = time.time() - t0
    rec['gpu_mem_peak_gb'] = torch.cuda.max_memory_reserved() / 2**30
    rec['guard'] = {'slept_s': GUARD.slept_s - g0[0], 'paused_s': GUARD.paused_s - g0[1],
                    'duty_end': GUARD.duty, 'peak_power': GUARD.peak_power,
                    'peak_temp': GUARD.peak_temp, 'resumed_from_round': start}
    return rec


def summarise(recs, m):
    fin = np.mean([np.mean([x['acc'] for x in r['rounds'][-3:]]) for r in recs])
    pos = lambda r: sum(sum(x['malicious']) for x in r['rounds'])
    tpr = np.mean([sum(x['tp'] for x in r['rounds']) / max(1, pos(r)) for r in recs]) if m else 0.0
    fpr = np.mean([sum(x['fp'] for x in r['rounds']) / max(1, N * len(r['rounds']) - pos(r))
                   for r in recs])
    hon = [np.std(h) / np.mean(h) for r in recs for x in r['rounds']
           for h in [[n for n, b in zip(x['norm'], x['malicious']) if not b]]]
    return fin, tpr, fpr, float(np.mean(hon))


PRESETS = {
    # timing pilot: one run
    'pilot':    [('uniform', 'rmd', 'norm', 3)],
    # the decisive subset (COSTS.md), plus the two clean baselines that
    # prediction 2 needs (damage = attacked - clean)
    'decisive': [('uniform', 'none', 'none', 0), ('hetero', 'none', 'none', 0),
                 ('uniform', 'rmd', 'norm', 3), ('hetero', 'rmd', 'norm', 3),
                 ('uniform', 'rmd', 'none', 3), ('hetero', 'rmd', 'none', 3),
                 ('hetero', 'rmd', 'norm_sn', 3)],
}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', choices=sorted(PRESETS))
    ap.add_argument('--sigma', default='uniform', help='uniform | hetero | adaptive | ladder:S')
    ap.add_argument('--attacks', nargs='+', default=['rmd'])
    ap.add_argument('--defences', nargs='+', default=['norm'])
    ap.add_argument('--ms', nargs='+', type=int, default=[3])
    ap.add_argument('--lie', type=float, default=1.0)
    ap.add_argument('--cap', action='store_true')
    ap.add_argument('--split', default='session', choices=['session', 'positional'],
                    help='positional = the original P3 300/1200 cut (R3-5 only)')
    ap.add_argument('--lie-track', action='store_true',
                    help='Arm F2: attackers declare sigma * their own inflation factor')
    ap.add_argument('--onset', type=int, default=0,
                    help='first attacking round; >0 gives Arm G a clean baseline window')
    ap.add_argument('--trials', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=ROUNDS)
    ap.add_argument('--tag', default='', help='output file: results/p3h_<tag>.json')
    a = ap.parse_args()

    if a.preset:
        grid = PRESETS[a.preset]
        tag = a.tag or a.preset
        if a.preset == 'pilot':
            a.trials = 1
    else:
        grid = [(a.sigma, at, d, (m if at != 'none' else 0))
                for at in a.attacks for d in a.defences for m in a.ms]
        grid = list(dict.fromkeys(grid))
        tag = a.tag or a.sigma.replace(':', '')
    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, 'p3h_%s.json' % tag)
    ckpt_path = os.path.join(RESULTS, 'ckpt_%s.pt' % tag)

    print('loading + splitting to %d clients...' % N, flush=True)
    feats, tgts, vf, vo, vb = P.load_10()
    det_set, rep_set, split = split_val(vf, vo, vb, a.split)
    print('  %d clients x %d frames | detector %s (%d) | report %s (%d)'
          % (N, feats[0].shape[0], split['detector_sessions'], split['n_det'],
             split['report_sessions'], split['n_rep']), flush=True)
    print('  %d configs x %d trials, %d rounds -> %s\n'
          % (len(grid), a.trials, a.rounds, out_path), flush=True)
    data = (feats, tgts, det_set, rep_set)

    meta = {'n_clients': N, 'rounds': a.rounds, 'beta1': P.BETA1, 'beta2': P.BETA2,
            'B': P.B, 'lr': P.LR, 'sigma0': SIGMA0, 'rho_range': [RHO_LO, RHO_HI],
            'el': EL, 't_budget': T_BUDGET, 'seed_base': SEED_BASE, 'split': split,
            'machine': platform.node(), 'gpu': torch.cuda.get_device_name(0),
            'torch': torch.__version__, 'power_limit_w': __import__('gpu_guard').power_limit_w()}
    res = {}
    if os.path.exists(out_path):
        old = json.load(open(out_path))
        res = old.get('results', {})
        if old.get('meta', {}).get('rounds') != a.rounds:
            raise SystemExit('existing %s used a different round count' % out_path)
        if old.get('meta', {}).get('split', {}).get('mode', 'session') != split['mode']:
            raise SystemExit('existing %s used a different val split' % out_path)
        print('resuming: %d configs present\n' % len(res), flush=True)

    t_all = time.time()
    for gi, (mode, at, df, m) in enumerate(grid):
        key = '%s|%s|%s|m%d|lie%g|cap%d' % (mode, at, df, m, a.lie, a.cap)
        if a.onset:
            key += '|on%d' % a.onset
        if a.lie_track:
            key += '|track'
        recs = res.setdefault(key, [])
        for t in range(len(recs), a.trials):
            rec = run(mode, at, df, m, SEED_BASE + t, a.lie, a.cap, data, a.rounds, a.onset,
                      ckpt=ckpt_path, ckey='%s|trial%d' % (key, t), lie_track=a.lie_track)
            recs.append(rec)
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            if rec['sigma_info']['clamp_binds']:
                print('  !! sigma clamp [1,20] binds in %s trial %d' % (key, t), flush=True)
            print('    %s trial %d: %.1f min  (gpu mem peak %.1f GB; guard slept %.0fs paused %.0fs duty %.2f, peak %.0fW %.0fC)'
                  % (key, t, rec['seconds'] / 60, rec['gpu_mem_peak_gb'], rec['guard']['slept_s'], rec['guard']['paused_s'],
                     rec['guard']['duty_end'], rec['guard']['peak_power'], rec['guard']['peak_temp']),
                  flush=True)
            save_json({'meta': meta, 'results': res}, out_path)
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)          # trial is safely in the JSON now
        fin, tpr, fpr, hon = summarise(recs, m)
        print('[%2d/%2d] %-40s acc=%.4f TPR=%.2f FPR=%.2f honest-norm rsd=%.3f%%  [%.0fm]'
              % (gi + 1, len(grid), key, fin, tpr, fpr, 100 * hon, (time.time() - t_all) / 60),
              flush=True)
    print('\nP3-H DONE -> %s' % out_path, flush=True)
