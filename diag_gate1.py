# Isolated diagnostic: WHY is Gate 1's curve flat?
# Separates three candidate causes -- gradient clipping, noise magnitude,
# and Adam's scale-invariance -- by varying one at a time on cached features.
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

def run(tag, init, feats, tgts, vf, vo, vb, rounds=25, epochs=2, B=512,
        lr=1e-3, sigma=0.0, C=1.0, dp=True, optname='adam', probe=(5, 10, 25)):
    st = {k: v.clone() for k, v in init.items()}
    t0 = time.time(); traj = []
    for r in range(1, rounds + 1):
        st = fedavg([client_round(st, feats[i], tgts[i][0], tgts[i][1], epochs, B,
                                  lr, sigma, C, dp, optname) for i in range(N_CLIENTS)])
        if r in probe:
            c, i25, i50 = evaluate(st, vf, vo, vb)
            traj.append((r, c, i25, i50))
            print(f'   r{r:>3}: cell={c:.3f} IoU.25={i25:.3f} IoU.50={i50:.3f}', flush=True)
    print(f'  [{tag}] done in {time.time()-t0:.0f}s', flush=True)
    return traj

if __name__ == '__main__':
    print('loading cached features...', flush=True)
    feats, tgts, vf, (vo, vb) = load_all()
    pre = torch.load('./fl_workdirs/stage1/public_pretrained.pt', map_location=DEV)
    init = pre['head']
    c, i25, i50 = evaluate(init, vf, vo, vb)
    print(f'INIT: cell={c:.3f} IoU.25={i25:.3f} IoU.50={i50:.3f}\n', flush=True)

    cfgs = [
        # tag                          dp     sigma  C     opt     lr
        ('adam no-DP (control)',       False, 0.0,   1.0,  'adam', 1e-3),
        ('adam CLIP-ONLY C=1',         True,  0.0,   1.0,  'adam', 1e-3),
        ('adam CLIP-ONLY C=10',        True,  0.0,   10.0, 'adam', 1e-3),
        ('adam DP s=0.5 C=1',          True,  0.5,   1.0,  'adam', 1e-3),
        ('adam DP s=8.0 C=1',          True,  8.0,   1.0,  'adam', 1e-3),
        ('sgd  no-DP lr=.05',          False, 0.0,   1.0,  'sgd',  5e-2),
        ('sgd  CLIP-ONLY C=1 lr=.05',  True,  0.0,   1.0,  'sgd',  5e-2),
        ('sgd  DP s=0.5 C=1 lr=.05',   True,  0.5,   1.0,  'sgd',  5e-2),
        ('sgd  DP s=2.0 C=1 lr=.05',   True,  2.0,   1.0,  'sgd',  5e-2),
        ('sgd  DP s=8.0 C=1 lr=.05',   True,  8.0,   1.0,  'sgd',  5e-2),
    ]
    out = {}
    for tag, dp, sigma, C, optname, lr in cfgs:
        print(f'>> {tag}', flush=True)
        out[tag] = run(tag, init, feats, tgts, vf, vo, vb,
                       sigma=sigma, C=C, dp=dp, optname=optname, lr=lr)
        json.dump(out, open('./exp_data/diag_gate1_causes.json', 'w'), indent=1)
    print('\nDIAG DONE', flush=True)
