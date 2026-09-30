# Where does accuracy ACTUALLY start to degrade with noise?
# Gate 1 (corrected) showed sigma in [0.25,16] is entirely inert. If the knee sits
# far above the source paper's sigma range [1,20], then no allocation policy --
# uniform, scheduled, or RL -- can differ within that range, and that is the
# finding. Reported alongside the real (eps, delta) each sigma buys.
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
sys.path.insert(0, os.path.abspath('.'))
from stage1_public_pretrain import Head
from stage2_gate import load_all, loss_fn, FeatDS, fedavg, evaluate, N_CLIENTS, DEV
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier

ROUNDS, EPOCHS, B, LR, C = 25, 2, 2048, 1e-2, 1.0
NCLIENT = 8000
DELTA = 1e-5
SIGMAS = [0.0, 4.0, 16.0, 64.0, 256.0, 512.0, 1024.0]

def eps_for(sigma):
    """Total eps for one client over all rounds at this sigma."""
    if sigma <= 0: return float('inf')
    q = B / NCLIENT
    steps = ROUNDS * EPOCHS * int(np.ceil(NCLIENT / B))
    acc = RDPAccountant()
    for _ in range(steps):
        acc.step(noise_multiplier=sigma, sample_rate=q)
    try:    return acc.get_epsilon(delta=DELTA)
    except Exception: return float('nan')

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
            opt.zero_grad(); loss_fn(h(f.float()), o, b).backward(); opt.step()
    inner = h._module if hasattr(h, '_module') else h
    return {k: v.detach().clone() for k, v in inner.state_dict().items()}

if __name__ == '__main__':
    print('loading cached features...', flush=True)
    feats, tgts, vf, (vo, vb) = load_all()
    D = sum(p.numel() for p in Head().parameters())
    torch.manual_seed(1234)
    init = {k: v.detach().clone() for k, v in Head().to(DEV).state_dict().items()}
    print(f'D={D}  B={B}  {ROUNDS}r x {EPOCHS}ep  C={C}  lr={LR}\n', flush=True)
    SEEDS = [1234, 5678, 9012]
    print(f'{"sigma":>7} {"eps":>9} | ' + '  '.join(f'seed{i}' for i in range(len(SEEDS)))
          + ' |   mean +- sd   (score = mean IoU.25 over r20-25)', flush=True)
    out = {}
    for sigma in SIGMAS:
        dp = sigma > 0
        scores = []
        for sd_i, sd in enumerate(SEEDS):
            torch.manual_seed(sd)
            st = {k: v.detach().clone() for k, v in Head().to(DEV).state_dict().items()}
            traj = []
            for r in range(1, ROUNDS + 1):
                st = fedavg([client_round(st, feats[i], tgts[i][0], tgts[i][1], sigma, dp)
                             for i in range(N_CLIENTS)])
                if r > ROUNDS - 6:
                    _, i25, _ = evaluate(st, vf, vo, vb)
                    traj.append(i25)
            scores.append(float(np.mean(traj)))
        e = eps_for(sigma)
        out[str(sigma)] = {'scores': scores, 'eps': e}
        es = 'no-DP' if sigma == 0 else f'{e:9.3f}'
        cells = '  '.join(f'{s:.3f}' for s in scores)
        print(f'{sigma:7.1f} {es:>9} | {cells} |  {np.mean(scores):.3f} +- {np.std(scores):.3f}',
              flush=True)
        json.dump(out, open('./exp_data/knee_seeds.json', 'w'), indent=1)
    print('\nKNEE SEEDS DONE', flush=True)
