# =============================================================================
# STAGE 2 + GATE 1: head-only DP-SGD on cached frozen-trunk features.
#
# GATE 1 asks the one question that decides whether this project can continue:
# does accuracy degrade SMOOTHLY as noise rises (0.20 -> 0.15 -> 0.10 ...), or
# does it collapse to zero the way it did when all 648K params were privatized?
#   * smooth  -> noise placement can matter -> schedule/DRL comparison is live
#   * collapse-> DP is impractical for this task; that is the finding, stop.
#
# Everything trains on CACHED features, so the frozen trunk is never re-run.
# A 25-round federated run costs seconds instead of ~10 minutes, which is what
# makes the later RL phase feasible.
# =============================================================================
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage1_public_pretrain import Head, FGW, FGH, FS
from opacus import PrivacyEngine

STAGE1 = './fl_workdirs/stage1'
N_CLIENTS = 5
DEV = 'cuda:0'


def load_all(device=DEV):
    feats, tgts = [], []
    for i in range(N_CLIENTS):
        f = torch.load(os.path.join(STAGE1, f'feat_client_{i}.pt'), map_location='cpu')
        t = torch.load(os.path.join(STAGE1, f'tgt_client_{i}.pt'), map_location='cpu')
        feats.append(f.to(device))
        tgts.append((t['obj'].to(device), t['box'].to(device)))
    vf = torch.load(os.path.join(STAGE1, 'feat_val.pt'), map_location='cpu').to(device)
    vt = torch.load(os.path.join(STAGE1, 'tgt_val.pt'), map_location='cpu')
    return feats, tgts, vf, (vt['obj'].to(device), vt['box'].to(device))


def loss_fn(pred, obj_t, box_t):
    ol = F.binary_cross_entropy_with_logits(pred[:, 0], obj_t)
    pm = obj_t.unsqueeze(1)
    npos = pm.sum().clamp(min=1.0)
    bl = (F.smooth_l1_loss(pred[:, 1:5], box_t, reduction='none') * pm).sum() / npos
    return ol + 5.0 * bl


class FeatDS(torch.utils.data.Dataset):
    def __init__(self, f, o, b):
        self.f, self.o, self.b = f, o, b

    def __len__(self):
        return self.f.shape[0]

    def __getitem__(self, i):
        return self.f[i], self.o[i], self.b[i]


def head_round(state, feat, obj, box, epochs, B, lr, sigma, C, dp=True):
    """One client's local round: DP-SGD on the head only, over cached features."""
    h = Head().to(DEV)
    h.load_state_dict(state)
    h.train()
    dl = torch.utils.data.DataLoader(FeatDS(feat, obj, box), batch_size=B, shuffle=True)
    opt = torch.optim.Adam(h.parameters(), lr=lr)
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


def fedavg(states):
    out, ref = {}, states[0]
    for k in ref:
        out[k] = torch.stack([s[k].float() for s in states]).mean(0)
    return out


@torch.no_grad()
def evaluate(state, vf, vo, vb, bs=400):
    h = Head().to(DEV); h.load_state_dict(state); h.eval()
    cell = h25 = h50 = tot = 0
    for s in range(0, vf.shape[0], bs):
        p = h(vf[s:s + bs].float())
        o = torch.sigmoid(p[:, 0])
        f = o.flatten(1).argmax(1); gj = f // FGW; gi = f % FGW
        idx = torch.arange(p.shape[0], device=p.device)
        cx = (gi.float() + torch.sigmoid(p[idx, 1, gj, gi])) * FS
        cy = (gj.float() + torch.sigmoid(p[idx, 2, gj, gi])) * FS
        w = torch.exp(p[idx, 3, gj, gi]) * FS; hh = torch.exp(p[idx, 4, gj, gi]) * FS
        go = vo[s:s + bs]; gb = vb[s:s + bs]
        gf = go.flatten(1).argmax(1); ggj = gf // FGW; ggi = gf % FGW
        has = go.flatten(1).max(1).values > 0
        gcx = (ggi.float() + gb[idx, 0, ggj, ggi]) * FS
        gcy = (ggj.float() + gb[idx, 1, ggj, ggi]) * FS
        gw = torch.exp(gb[idx, 2, ggj, ggi]) * FS; gh = torch.exp(gb[idx, 3, ggj, ggi]) * FS
        ix1 = torch.maximum(cx - w / 2, gcx - gw / 2); iy1 = torch.maximum(cy - hh / 2, gcy - gh / 2)
        ix2 = torch.minimum(cx + w / 2, gcx + gw / 2); iy2 = torch.minimum(cy + hh / 2, gcy + gh / 2)
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        iou = inter / (w * hh + gw * gh - inter).clamp(min=1e-9)
        cell += int(((gj == ggj) & (gi == ggi) & has).sum())
        h25 += int(((iou >= 0.25) & has).sum()); h50 += int(((iou >= 0.5) & has).sum())
        tot += int(has.sum())
    return cell / max(tot, 1), h25 / max(tot, 1), h50 / max(tot, 1)


def run_fl(feats, tgts, vf, vo, vb, init, rounds, sigmas, epochs, B, lr, C, dp=True):
    """sigmas: either a scalar (same every round) or list per round."""
    st = {k: v.clone() for k, v in init.items()}
    for r in range(rounds):
        sg = sigmas[r] if isinstance(sigmas, (list, tuple, np.ndarray)) else sigmas
        st = fedavg([head_round(st, feats[i], tgts[i][0], tgts[i][1], epochs, B, lr, sg, C, dp)
                     for i in range(N_CLIENTS)])
    return st, evaluate(st, vf, vo, vb)


if __name__ == '__main__':
    print('loading cached features...', flush=True)
    feats, tgts, vf, (vo, vb) = load_all()
    print(f'  clients {[f.shape[0] for f in feats]} | val {vf.shape[0]} | feat {tuple(feats[0].shape[1:])}',
          flush=True)
    D = sum(p.numel() for p in Head().parameters())
    print(f'  DP-trained head: D={D}, sqrt(D)={D**0.5:.0f}\n', flush=True)

    pre = torch.load(os.path.join(STAGE1, 'public_pretrained.pt'), map_location=DEV)
    init = pre['head']
    c, i25, i50 = evaluate(init, vf, vo, vb)
    print(f'starting point (public-pretrained head): cell={c:.3f} IoU.25={i25:.3f} IoU.50={i50:.3f}\n',
          flush=True)

    ROUNDS, EPOCHS, B, LR, C = 25, 2, 512, 1e-3, 1.0
    print(f'GATE 1 -- noise survival curve  ({ROUNDS} rounds, B={B}, C={C})', flush=True)
    print(f'{"sigma":>7} {"SNR":>6} | cell   IoU.25  IoU.50   time', flush=True)
    results = {}
    for sigma in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
        t0 = time.time()
        dp = sigma > 0
        _, (c, i25, i50) = run_fl(feats, tgts, vf, vo, vb, init, ROUNDS,
                                   sigma if dp else 0.0, EPOCHS, B, LR, C, dp=dp)
        snr = B / (max(sigma, 1e-9) * D ** 0.5)
        results[sigma] = (c, i25, i50)
        tag = 'no-DP' if sigma == 0 else f'{snr:6.2f}'
        print(f'{sigma:7.2f} {tag:>6} | {c:.3f}  {i25:.3f}  {i50:.3f}   ({time.time()-t0:.0f}s)',
              flush=True)
    json.dump({str(k): v for k, v in results.items()},
              open('./exp_data/gate1_noise_curve.json', 'w'), indent=1)
    print('\nGATE 1 verdict: smooth degradation => schedule comparison is live.', flush=True)
