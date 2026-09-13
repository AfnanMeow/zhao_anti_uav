# Diagnostic 2: restore dynamic range (random head) + calibrate the DP optimizer.
# Gate 1 seeded the head from public pretraining, leaving only 0.376->0.434 of
# headroom for the federated phase -- too narrow to resolve any noise curve.
# Random head widens it to 0.000->0.434. Then find a DP config that actually
# learns, since Adam was shown to wash sigma out entirely.
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
sys.path.insert(0, os.path.abspath('.'))
from stage1_public_pretrain import Head
from stage2_gate import load_all, loss_fn, FeatDS, fedavg, evaluate, N_CLIENTS, DEV
from opacus import PrivacyEngine

def client_round(state, feat, obj, box, epochs, B, lr, sigma, C, dp, optname):
    h = Head().to(DEV); h.load_state_dict(state); h.train()
    dl = torch.utils.data.DataLoader(FeatDS(feat, obj, box), batch_size=B, shuffle=True)
    opt = (torch.optim.Adam(h.parameters(), lr=lr) if optname == 'adam'
           else torch.optim.SGD(h.parameters(), lr=lr, momentum=0.9))
    if dp:
        pe = PrivacyEngine()
        h, opt, dl = pe.make_private(module=h, optimizer=opt, data_loader=dl,
                                     noise_multiplier=sigma, max_grad_norm=C)
    for _ in range(epochs):
        for f, o, b in dl:
            opt.zero_grad()
            loss_fn(h(f.float()), o, b).backward()
            opt.step()
    inner = h._module if hasattr(h, '_module') else h
    return {k: v.detach().clone() for k, v in inner.state_dict().items()}

def run(tag, init, feats, tgts, vf, vo, vb, rounds, epochs, B, lr, sigma, C, dp, optname,
        probe=(5, 15, 25)):
    st = {k: v.clone() for k, v in init.items()}
    t0 = time.time(); traj = []
    for r in range(1, rounds + 1):
        st = fedavg([client_round(st, feats[i], tgts[i][0], tgts[i][1], epochs, B,
                                  lr, sigma, C, dp, optname) for i in range(N_CLIENTS)])
        if r in probe:
            c, i25, i50 = evaluate(st, vf, vo, vb)
            traj.append((r, c, i25, i50))
            print(f'   r{r:>3}: cell={c:.3f} IoU.25={i25:.3f} IoU.50={i50:.3f}', flush=True)
    print(f'  [{tag}] {time.time()-t0:.0f}s', flush=True)
    return traj

if __name__ == '__main__':
    print('loading cached features...', flush=True)
    feats, tgts, vf, (vo, vb) = load_all()
    torch.manual_seed(1234)
    rand_init = {k: v.detach().clone() for k, v in Head().to(DEV).state_dict().items()}
    c, i25, i50 = evaluate(rand_init, vf, vo, vb)
    print(f'RANDOM-HEAD START: cell={c:.3f} IoU.25={i25:.3f} IoU.50={i50:.3f}\n', flush=True)

    #  tag                          rounds ep   B     lr     sigma  C    dp     opt
    cfgs = [
        ('CEILING adam no-DP B512',     25,  2,  512, 1e-3,  0.0,  1.0, False, 'adam'),
        ('adam  DP s=1 B512',           25,  2,  512, 1e-3,  1.0,  1.0, True,  'adam'),
        ('sgd   DP s=1 B512  lr.01',    25,  2,  512, 1e-2,  1.0,  1.0, True,  'sgd'),
        ('sgd   DP s=1 B2048 lr.01',    25,  2, 2048, 1e-2,  1.0,  1.0, True,  'sgd'),
        ('sgd   DP s=1 B2048 lr.05 e8', 25,  8, 2048, 5e-2,  1.0,  1.0, True,  'sgd'),
        ('sgd   DP s=1 B4096 lr.05 e8', 25,  8, 4096, 5e-2,  1.0,  1.0, True,  'sgd'),
        ('sgd   noDP  B2048 lr.05 e8',  25,  8, 2048, 5e-2,  0.0,  1.0, False, 'sgd'),
    ]
    out = {}
    for tag, rounds, ep, B, lr, sigma, C, dp, optname in cfgs:
        print(f'>> {tag}', flush=True)
        out[tag] = run(tag, rand_init, feats, tgts, vf, vo, vb,
                       rounds, ep, B, lr, sigma, C, dp, optname)
        json.dump(out, open('./exp_data/diag_gate2_randomhead.json', 'w'), indent=1)
    print('\nDIAG2 DONE', flush=True)
