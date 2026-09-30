# Is the DP noise actually reaching the weights? Measure, don't reason.
import os, sys, warnings
warnings.filterwarnings('ignore')
import torch
sys.path.insert(0, os.path.abspath('.'))
from stage1_public_pretrain import Head
from stage2_gate import load_all, loss_fn, FeatDS, evaluate, DEV
from opacus import PrivacyEngine

B, LR, C, EPOCHS = 2048, 1e-2, 1.0, 2

feats, tgts, vf, (vo, vb) = load_all()
feat, obj, box = feats[0], tgts[0][0], tgts[0][1]
torch.manual_seed(1234)
init = {k: v.detach().clone() for k, v in Head().to(DEV).state_dict().items()}
D = sum(p.numel() for p in Head().parameters())
print(f'D={D} sqrt(D)={D**0.5:.1f}  B={B} C={C} lr={LR}  n_client={feat.shape[0]}')

def one_round(sigma, seed, momentum=0.9):
    torch.manual_seed(seed)
    h = Head().to(DEV); h.load_state_dict(init); h.train()
    dl = torch.utils.data.DataLoader(FeatDS(feat, obj, box), batch_size=B, shuffle=True)
    opt = torch.optim.SGD(h.parameters(), lr=LR, momentum=momentum)
    pe = PrivacyEngine()
    h, opt, dl = pe.make_private(module=h, optimizer=opt, data_loader=dl,
                                 noise_multiplier=sigma, max_grad_norm=C)
    nsteps = 0
    gnorms = []
    for _ in range(EPOCHS):
        for f, o, b in dl:
            opt.zero_grad()
            loss_fn(h(f.float()), o, b).backward()
            # per-sample grad norms BEFORE clipping (opacus stores them)
            ps = torch.cat([p.grad_sample.reshape(p.grad_sample.shape[0], -1)
                            for p in h.parameters() if hasattr(p, 'grad_sample')], dim=1)
            gnorms.append(ps.norm(2, dim=1).median().item())
            opt.step(); nsteps += 1
    inner = h._module if hasattr(h, '_module') else h
    sd = inner.state_dict()
    delta = torch.cat([(sd[k].float() - init[k].float()).flatten() for k in init])
    return delta, nsteps, sum(gnorms)/len(gnorms)

print(f'\n{"sigma":>7} {"steps":>6} {"med|g_i|":>9} {"||delta||":>10}  {"pairwise diff":>13}')
prev = {}
for sigma in [0.0, 1.0, 16.0, 64.0]:
    d1, ns, gn = one_round(sigma, seed=11)
    d2, _, _ = one_round(sigma, seed=22)      # different noise draw, same data order seed base
    print(f'{sigma:7.2f} {ns:6d} {gn:9.3f} {d1.norm().item():10.4f}  '
          f'{(d1-d2).norm().item():13.4f}')
    prev[sigma] = d1

print('\n||delta(sigma) - delta(0)||  -- how far noise moves the endpoint:')
for s in [1.0, 16.0, 64.0]:
    print(f'  sigma={s:5.1f}: {(prev[s]-prev[0.0]).norm().item():.4f}')

# predicted injected-noise norm per step, after averaging over the batch
for s in [1.0, 16.0, 64.0]:
    print(f'  predicted per-step noise norm sigma={s:5.1f}: {s*C*(D**0.5)/B:.4f}   '
          f'(signal norm after clipping <= {C})')
