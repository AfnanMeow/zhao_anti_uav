# =============================================================================
# E1 -- measure the weight profile a_t.
#
# The uniform-optimality theorem assumes every round's noise reaches the final
# model with EQUAL weight, so V = sum_t sigma_t^2. If the weights a_t are not
# flat the objective is V = sum_t a_t sigma_t^2 and the optimum is
# rho_t ~ sqrt(a_t), which is NON-uniform -- in the direction the source paper
# claims. Clipping is not what drives the theorem; equal weighting is.
#
# Method: hold every round at the uniform operating point, add a single extra
# noise pulse at exactly one round t*, and read the drop in final IoU.25.
#
#     a_hat(t*) = [ IoU_base - IoU_pulse(t*) ] / (sigma_pulse^2 - sigma_ref^2)
#
# Only the SHAPE matters; report a_hat normalised by its mean.
#
# Pairing: baseline and every pulse arm inside a trial share one init AND one
# RNG stream (common random numbers). Without CRN the paired sd is 0.0466
# instead of 0.0171 and this experiment needs ~7x the trials.
#
# NOTE: E1 deliberately does NOT hold sum_t rho_t fixed. We are measuring a
# local derivative of utility w.r.t. injected variance at round t, not
# comparing schedules under a budget. Say so in the write-up.
#
#   pilot : python e1_weight_profile.py --mode pilot
#   full  : python e1_weight_profile.py --mode full --n 20
#
# Requires stage1_clear.py to have been run once -- it builds the session-disjoint
# split, pretrains the public trunk, and caches the cut_s3 features, all into
# ./fl_workdirs/clear. (stage1b_recache.py is for the OLDER bigdata path and is
# NOT needed here.)
# Re-apply the GPU power cap first:  nvidia-smi -pl 300   (Administrator shell)
# =============================================================================
import os, sys, json, time, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_DB_sweep as G
from gate_DB_sweep import make_init, load_cut, evaluate, client_round
from stage2_gate import fedavg, N_CLIENTS
from stage1b_recache import DPTail

# canonical live regime (FINDINGS_gate1.md "Canonical config")
ROUNDS, EPOCHS, B, LR = 25, 2, 128, 0.003
RHO, EL = 30.0, 2
SIGMA_REF = float(np.sqrt(2 * EL * ROUNDS / RHO))     # 1.8257 -- uniform at rho=30
SCORE_LAST = 6                                        # mean IoU.25 over rounds 20-25


def run_sched(init, feats, tgts, vf, vo, vb, sigmas, seed):
    """One federated run with a per-round sigma schedule.
    `seed` fixes the RNG stream so paired arms see identical shuffling and
    identical DP noise draws wherever their sigmas agree."""
    assert len(sigmas) == ROUNDS
    torch.manual_seed(seed)
    st = {k: v.clone() for k, v in init.items()}
    tail = []
    for r in range(1, ROUNDS + 1):
        s = float(sigmas[r - 1])
        st = fedavg([client_round(st, feats[i], tgts[i][0], tgts[i][1],
                                  s, s > 0, B, LR) for i in range(N_CLIENTS)])
        if r > ROUNDS - SCORE_LAST:
            tail.append(evaluate(st, vf, vo, vb))
    return float(np.mean(tail))


def pulse_schedule(t_star, sigma_pulse):
    s = [SIGMA_REF] * ROUNDS
    s[t_star - 1] = sigma_pulse
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['pilot', 'full'], default='pilot')
    ap.add_argument('--n', type=int, default=0, help='trials (default 4 pilot / 20 full)')
    ap.add_argument('--sigma-pulse', type=float, default=6.0)
    ap.add_argument('--probes', type=int, nargs='+', default=[])
    ap.add_argument('--seed-base', type=int, default=770000)
    # stage1_clear.py writes features AND targets to ./fl_workdirs/clear
    # (it does its own cut_s3 caching -- stage1b_recache.py is for the older
    # bigdata path and is NOT needed for the clear-sky regime).
    ap.add_argument('--data', default='./fl_workdirs/clear')
    ap.add_argument('--public', default='')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    if a.data:
        G.DATA = a.data; G.TGT = a.data
    G.ROUNDS, G.EPOCHS, G.C = ROUNDS, EPOCHS, 1.0

    # gate_DB_sweep.make_init defaults to the OLD bigdata path
    # (./fl_workdirs/stage1/public_pretrained.pt); the clear-sky run writes its
    # trunk beside its features. Resolve it from --data unless told otherwise.
    public = a.public or os.path.join(a.data, 'public_pretrained.pt')
    if not os.path.exists(public):
        sys.exit(f'public trunk not found: {public}\n'
                 f'Run  python stage1_clear.py  first -- it builds the split, '
                 f'pretrains the trunk and caches the cut_s3 features.')

    if a.mode == 'pilot':
        n = a.n or 4
        probes = a.probes or [1, 13, 25]
        lin_sigmas = [4.0, 6.0, 8.0]          # linearity check at the middle probe
        out = a.out or './exp_data/E1_pilot.json'
    else:
        n = a.n or 20
        probes = a.probes or [1, 4, 7, 10, 13, 16, 19, 22, 25]
        lin_sigmas = []
        out = a.out or './exp_data/E1_weight_profile.json'

    feats, tgts, vf, (vo, vb) = load_cut()
    D = sum(p.numel() for p in DPTail().parameters())
    print(f'E1 [{a.mode}]  D={D}  B={B}  lr={LR}  rounds={ROUNDS}  '
          f'sigma_ref={SIGMA_REF:.4f}  sigma_pulse={a.sigma_pulse}  n={n}', flush=True)
    print(f'probes t*={probes}   delta_var={a.sigma_pulse**2 - SIGMA_REF**2:.2f}\n', flush=True)

    res = {'mode': a.mode, 'rounds': ROUNDS, 'B': B, 'lr': LR, 'rho': RHO,
           'sigma_ref': SIGMA_REF, 'sigma_pulse': a.sigma_pulse, 'crn': True,
           'seed_base': a.seed_base, 'probes': probes,
           'baseline': [], 'pulse': {str(t): [] for t in probes},
           'linearity': {str(s): [] for s in lin_sigmas}}

    t0 = time.time()
    for trial in range(n):
        seed = a.seed_base + trial
        init = make_init('public_tail', seed, public)

        base = run_sched(init, feats, tgts, vf, vo, vb, [SIGMA_REF] * ROUNDS, seed)
        res['baseline'].append(base)
        print(f'trial {trial}  baseline={base:.4f}', flush=True)

        for t in probes:
            sc = run_sched(init, feats, tgts, vf, vo, vb,
                           pulse_schedule(t, a.sigma_pulse), seed)
            res['pulse'][str(t)].append(sc)
            dv = a.sigma_pulse ** 2 - SIGMA_REF ** 2
            print(f'   t*={t:>2}  pulse={sc:.4f}  drop={base - sc:+.4f}  '
                  f'a_hat={(base - sc) / dv:+.6f}', flush=True)

        for sp in lin_sigmas:                  # linearity: same t*, different pulse size
            t_mid = probes[len(probes) // 2]
            sc = run_sched(init, feats, tgts, vf, vo, vb,
                           pulse_schedule(t_mid, sp), seed)
            res['linearity'][str(sp)].append(sc)
            dv = sp ** 2 - SIGMA_REF ** 2
            print(f'   lin t*={t_mid} sigma={sp}  {sc:.4f}  a_hat={(base - sc) / dv:+.6f}',
                  flush=True)

        os.makedirs('./exp_data', exist_ok=True)
        json.dump(res, open(out, 'w'), indent=1)

    # ── summary ────────────────────────────────────────────────────────────
    basearr = np.array(res['baseline'])
    dv = a.sigma_pulse ** 2 - SIGMA_REF ** 2
    print(f'\n{"t*":>4} {"a_hat":>10} {"sd":>9} {"drop":>9}   (n={n})', flush=True)
    ah_mean = []
    for t in probes:
        p = np.array(res['pulse'][str(t)])
        ah = (basearr - p) / dv
        ah_mean.append(ah.mean())
        print(f'{t:>4} {ah.mean():>10.6f} {ah.std(ddof=1) if n > 1 else 0:>9.6f} '
              f'{(basearr - p).mean():>9.4f}', flush=True)

    ah_mean = np.array(ah_mean)
    if ah_mean.mean() != 0:
        norm = ah_mean / ah_mean.mean()
        print('\nnormalised a_hat:', np.round(norm, 3), flush=True)
        print(f'range (max/min)  : {norm.max() / max(norm.min(), 1e-9):.2f}x', flush=True)
        if n > 2:
            sl, ic = np.polyfit(probes, ah_mean, 1)
            print(f'slope vs t       : {sl:+.3e}  (flat => ~0)', flush=True)

    if lin_sigmas:
        print('\nlinearity check (a_hat must be stable across pulse size):', flush=True)
        for sp in lin_sigmas:
            p = np.array(res['linearity'][str(sp)])
            print(f'  sigma_pulse={sp}: a_hat={((basearr - p) / (sp**2 - SIGMA_REF**2)).mean():+.6f}',
                  flush=True)
        print('  drifting a_hat => out of the linear regime, lower sigma_pulse.', flush=True)

    print(f'\nwrote {out}   ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
