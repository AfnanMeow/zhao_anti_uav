# =============================================================================
# THE BOUNDARY MEASUREMENT: does per-sample clipping BIND?
#
# Theorem (thesis_rewrite.tex, Sec. "Optimality of Uniform Allocation") assumes
# per-sample clipping binds, so each round contributes at most C of signal
# regardless of sigma. That assumption is what removes the round-to-round signal
# variation the adaptive-allocation literature relies on.
#
# If it holds, uniform is optimal. If clipping does NOT bind, the update
# magnitude still tracks the true gradient -- large early, small late -- the
# premise is live, and a non-uniform schedule can win.
#
# This makes a single number the explanatory variable for the whole disagreement:
#
#   anti-UAV detector : median per-sample grad norm 26.8 vs C=1.0  -> binds 27x
#   CIFAR-10 toy CNN  : measured here             vs C=6.0  -> ?
#
# Measured at an EARLY and a LATE point in training, because the premise is
# specifically about the gradient shrinking as the model converges.
# =============================================================================
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opacus

from test_CIFAR10_improved import Net

DEV = 'cuda:0'
C = 6.0                 # their setting
BS = 200                # their batch_size
EL = 4                  # their epoch_local
OUT = './exp_data/boundary_clipping.json'


def per_sample_norms(net, loader, crit, n_batches=6):
    """Median per-sample gradient norm BEFORE clipping."""
    net.train()
    got = []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        x, y = x.to(DEV), y.to(DEV)
        net.zero_grad()
        crit(net(x), y).backward()
        ps = torch.cat([p.grad_sample.reshape(p.grad_sample.shape[0], -1)
                        for p in net.parameters() if hasattr(p, 'grad_sample')], dim=1)
        got.append(ps.norm(2, dim=1).detach().cpu())
    v = torch.cat(got)
    return float(v.median()), float(v.mean()), float((v > C).float().mean())


if __name__ == '__main__':
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    print('loading CIFAR-10...', flush=True)
    ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                      transform=tf)
    sub = torch.utils.data.Subset(ds, list(range(10000)))   # one client's shard
    loader = torch.utils.data.DataLoader(sub, batch_size=BS, shuffle=True)

    net = Net().to(DEV)
    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(net.parameters(), lr=0.1)
    pe = opacus.PrivacyEngine()
    net, opt, loader = pe.make_private(module=net, optimizer=opt, data_loader=loader,
                                       noise_multiplier=1.0, max_grad_norm=C)
    net = net.to(DEV)

    D = sum(p.numel() for p in net.parameters())
    print('CIFAR-10 toy CNN: D=%d   C=%.1f   batch=%d\n' % (D, C, BS), flush=True)

    res = {}
    print('%-22s %12s %12s %14s %10s' %
          ('training stage', 'median|g_i|', 'mean|g_i|', 'frac > C', 'binds?'))

    med, mean, frac = per_sample_norms(net, loader, crit)
    res['round_0'] = dict(median=med, mean=mean, frac_clipped=frac)
    print('%-22s %12.3f %12.3f %13.1f%% %10s'
          % ('start (round 0)', med, mean, 100 * frac, 'YES' if frac > 0.5 else 'no'))

    # train the way a client does, then re-measure once converged
    for stage, rounds in (('early (after 5 rounds)', 5), ('late (after 25 rounds)', 20)):
        for _ in range(rounds):
            for ep in range(EL):
                for x, y in loader:
                    x, y = x.to(DEV), y.to(DEV)
                    opt.zero_grad()
                    crit(net(x), y).backward()
                    opt.step()
        med, mean, frac = per_sample_norms(net, loader, crit)
        res[stage] = dict(median=med, mean=mean, frac_clipped=frac)
        print('%-22s %12.3f %12.3f %13.1f%% %10s'
              % (stage, med, mean, 100 * frac, 'YES' if frac > 0.5 else 'no'))

    json.dump({'C': C, 'D': D, 'batch': BS, 'results': res,
               'detector_reference': {'median_grad_norm': 26.8, 'C': 1.0,
                                      'binds_factor': 26.8}},
              open(OUT, 'w'), indent=1)

    print('\nReference, anti-UAV detector: median 26.8 vs C=1.0  -> binds by 26.8x')
    late = res['late (after 25 rounds)']
    print('CIFAR-10 toy CNN at convergence: median %.3f vs C=%.1f -> %s'
          % (late['median'], C,
             'binds by %.1fx' % (late['median'] / C) if late['median'] > C
             else 'DOES NOT BIND (%.0f%% of samples clipped)' % (100 * late['frac_clipped'])))
    print('BOUNDARY PROBE DONE', flush=True)
