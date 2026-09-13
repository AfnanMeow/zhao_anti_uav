# =============================================================================
# P3 step 1: measure the HONEST update envelope.
#
# Every defence in this phase needs to know what a legitimate client update looks
# like, because that is what an attacker must imitate. A Norm detector accepts a
# band of update norms; the band has to be wide enough for honest DP-noised
# updates or it rejects everyone, and exactly that width is the attacker's hiding
# room (Fig. 2 of exploiting_dp.pdf).
#
# Two quantities per noise level:
#   ||delta||     total norm of a client's round update (signal + DP noise)
#   spread        sd of that norm across clients and rounds
# Plus the noise-only component, obtained by differencing against a sigma=0 run
# with the same seed, which is what sets how much adversarial perturbation can be
# buried inside the legitimate envelope.
#
# Run at the P3 operating point: the live regime from FINDINGS_gate1.md.
# =============================================================================
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_DB_sweep as G

DATA = './fl_workdirs/clear'
PUBLIC = './fl_workdirs/clear/public_pretrained.pt'
OUT = './exp_data/p3_envelope.json'
B, LR = 128, 0.003
SIGMAS = [0.0, 1.83, 4.0, 8.0]     # 1.83 = uniform at rho=30, the canonical point
ROUNDS = 4
SEED = 900000


def flat(sd_new, sd_old):
    return torch.cat([(sd_new[k].float() - sd_old[k].float()).flatten() for k in sd_old])


if __name__ == '__main__':
    G.DATA = DATA
    G.TGT = DATA
    print('loading cached features...', flush=True)
    feats, tgts, vf, (vo, vb) = G.load_cut()
    init = G.make_init('public_tail', SEED, PUBLIC)
    print('operating point: D=566,789  B=%d  lr=%s  %d clients\n' % (B, LR, G.N_CLIENTS),
          flush=True)

    out = {}
    print('%8s | %-28s %-24s' % ('sigma', 'client update norm', 'noise-only norm'), flush=True)
    print('%8s | %-28s %-24s' % ('', 'mean +- sd  [min, max]', 'mean (vs sigma=0)'), flush=True)
    clean_updates = None
    for sig in SIGMAS:
        st = {k: v.clone() for k, v in init.items()}
        norms, noise_norms = [], []
        per_round_clean = []
        for r in range(ROUNDS):
            torch.manual_seed(SEED + r)          # same stream across sigmas
            locals_ = [G.client_round(st, feats[i], tgts[i][0], tgts[i][1],
                                      sig, sig > 0, B, LR) for i in range(G.N_CLIENTS)]
            for i, lo in enumerate(locals_):
                d = flat(lo, st)
                norms.append(float(d.norm()))
                if clean_updates is not None:
                    noise_norms.append(float((d - clean_updates[r][i]).norm()))
            if sig == 0.0:
                per_round_clean.append([flat(lo, st) for lo in locals_])
            st = G.fedavg(locals_)
        if sig == 0.0:
            clean_updates = per_round_clean
        n = np.array(norms)
        nn = np.array(noise_norms) if noise_norms else np.array([0.0])
        out[str(sig)] = {'norms': norms, 'noise_norms': noise_norms,
                         'mean': float(n.mean()), 'sd': float(n.std(ddof=1)),
                         'noise_mean': float(nn.mean())}
        print('%8.2f | %.4f +- %.4f  [%.4f, %.4f]   %.4f'
              % (sig, n.mean(), n.std(ddof=1), n.min(), n.max(), nn.mean()), flush=True)

    json.dump(out, open(OUT, 'w'), indent=1)

    # What a Norm detector must accept, and therefore what it must also let through
    print('\nNorm-detector accept band (mu +- k*sd) and the attacker room it implies:', flush=True)
    for sig in SIGMAS:
        if sig == 0.0:
            continue
        m, s = out[str(sig)]['mean'], out[str(sig)]['sd']
        for k in (2, 3):
            hi = m + k * s
            print('  sigma=%.2f  k=%d: accept up to %.4f  -> an attacker may add up to '
                  '%.4f (%.0f%% of a legitimate update) undetected'
                  % (sig, k, hi, hi - m, 100 * (hi - m) / m), flush=True)
    print('\nP3 ENVELOPE DONE', flush=True)
