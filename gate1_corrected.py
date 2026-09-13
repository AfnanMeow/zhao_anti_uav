# CORRECTED GATE 1 -- noise-survival curve with real dynamic range.
#
# The original gate seeded the DP head from public pretraining, which left only
# 0.376->0.434 IoU.25 of headroom for the whole federated phase; every DP run
# simply sat at its init and the curve came out flat. That flatness was an
# artifact of the init, not a property of DP.
#
# Here the head starts RANDOM (floor 0.002) and is learned from scratch under DP
# on the private shards, trunk frozen and public. Ceiling (no DP) is 0.399.
# Optimizer is SGD, not Adam: Adam's per-coordinate normalization was shown to
# wash the noise SCALE out entirely (16x sigma -> no change), while SGD
# transmits it (see diag_gate1_causes.json).
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
sys.path.insert(0, os.path.abspath('.'))
from stage1_public_pretrain import Head
from stage2_gate import load_all, loss_fn, FeatDS, fedavg, evaluate, N_CLIENTS, DEV
from opacus import PrivacyEngine

ROUNDS, EPOCHS, B, LR, C = 25, 2, 2048, 1e-2, 1.0
SIGMAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

def client_round(state, feat, obj, box, sigma, dp):
    h = Head().to(DEV); h.load_state_dict(state); h.train()
    dl = torch.utils.data.DataLoader(FeatDS(feat, obj, box), batch_size=B, shuffle=True)
    opt = torch.optim.SGD(h.parameters(), lr=LR, momentum=0.9)
    if dp:
        pe = PrivacyEngine()
        h, opt, dl = pe.make_private(module=h, optimizer=opt, data_loader=dl,
                                     noise_multiplier=sigma, max_grad_norm=C)
    for _ in range(EPOCHS):
        for f, o, b in dl:
            opt.zero_grad()
            loss_fn(h(f.float()), o, b).backward()
            opt.step()
    inner = h._module if hasattr(h, '_module') else h
    return {k: v.detach().clone() for k, v in inner.state_dict().items()}

if __name__ == '__main__':
    print('loading cached features...', flush=True)
    feats, tgts, vf, (vo, vb) = load_all()
    D = sum(p.numel() for p in Head().parameters())
    torch.manual_seed(1234)
    init = {k: v.detach().clone() for k, v in Head().to(DEV).state_dict().items()}
    c0, i0, _ = evaluate(init, vf, vo, vb)
    print(f'random-head floor: cell={c0:.3f} IoU.25={i0:.3f}   D={D} sqrt(D)={D**0.5:.0f}')
    print(f'config: SGD lr={LR} mom=0.9  B={B}  C={C}  {ROUNDS} rounds x {EPOCHS} ep\n', flush=True)

    print(f'{"sigma":>6} {"SNR":>7} | ' + ' '.join(f'r{r:<2}' for r in (1,3,5,10,15,20,25))
          + '   peak(round)', flush=True)
    out = {}
    for sigma in SIGMAS:
        dp = sigma > 0
        st = {k: v.clone() for k, v in init.items()}
        traj = []
        t0 = time.time()
        for r in range(1, ROUNDS + 1):
            st = fedavg([client_round(st, feats[i], tgts[i][0], tgts[i][1], sigma, dp)
                         for i in range(N_CLIENTS)])
            _, i25, _ = evaluate(st, vf, vo, vb)
            traj.append(i25)
        out[str(sigma)] = traj
        snr = B / (max(sigma, 1e-9) * D ** 0.5)
        tag = 'clip' if sigma == 0 else f'{snr:7.2f}'
        pk = int(np.argmax(traj)) + 1
        cells = ' '.join(f'{traj[r-1]:.3f}' for r in (1,3,5,10,15,20,25))
        print(f'{sigma:6.2f} {tag:>7} | {cells}   {max(traj):.3f}(r{pk}) [{time.time()-t0:.0f}s]',
              flush=True)
        json.dump(out, open('./exp_data/gate1_corrected.json', 'w'), indent=1)
    print('\nCORRECTED GATE 1 DONE', flush=True)
