# =============================================================================
# Place the noise knee inside the source paper's sigma range by tuning D and B.
#
# Head-only (D=12,677, B=2048) had its knee near sigma=512 -- effective SNR
#   eff = (B/(sigma*sqrt(D))) * sqrt(steps*clients) ~= 1.11 at the knee.
# Cutting the trunk after s3 raises D to 566,789, and eff ~ sqrt(B)/(sigma*sqrt(D))
# predicts  sigma_knee ~= 1.69*sqrt(B):   B=512 -> 42,  B=128 -> 19,  B=32 -> 10.
# This script tests that prediction. If it holds, the SNR model can be used to
# pick B directly instead of sweeping blindly.
# =============================================================================
import os, sys, json, time, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage1_public_pretrain import Head, FGW, FS
from stage1b_recache import DPTail, CUT_OUT
from stage2_gate import loss_fn, FeatDS, fedavg, N_CLIENTS, DEV
from opacus import PrivacyEngine
from gpu_guard import GUARD

ROUNDS, EPOCHS, C = 25, 2, 1.0


def make_init(mode, seed, public='./fl_workdirs/stage1/public_pretrained.pt'):
    """k1/k2 seeded from the PUBLIC trunk so the deeper stack optimises at all
    (from scratch it collapses to all-background at every lr tried); the head
    stays random so the federated phase still has real dynamic range. Both are
    DP-trained either way -- D is what is privatised, not what is random."""
    torch.manual_seed(seed)
    m = DPTail().to(DEV)
    if mode == 'public_tail':
        pre = torch.load(public, map_location=DEV)
        sd = m.state_dict()
        for k in list(sd):
            if k.startswith(('k1.', 'k2.')):
                sd[k] = pre['trunk'][k].clone()
        m.load_state_dict(sd)
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


DATA = CUT_OUT          # overridden by --data; targets live beside the features
TGT = './fl_workdirs/stage1'


def load_cut(device=DEV):
    feats, tgts = [], []
    for i in range(N_CLIENTS):
        feats.append(torch.load(os.path.join(DATA, f'feat_client_{i}.pt'), map_location='cpu').to(device))
        t = torch.load(os.path.join(TGT, f'tgt_client_{i}.pt'), map_location='cpu')
        tgts.append((t['obj'].to(device), t['box'].to(device)))
    vf = torch.load(os.path.join(DATA, 'feat_val.pt'), map_location='cpu').to(device)
    vt = torch.load(os.path.join(TGT, 'tgt_val.pt'), map_location='cpu')
    return feats, tgts, vf, (vt['obj'].to(device), vt['box'].to(device))


@torch.no_grad()
def evaluate(state, vf, vo, vb, bs=256):
    m = DPTail().to(DEV); m.load_state_dict(state); m.eval()
    h25 = tot = 0
    for s in range(0, vf.shape[0], bs):
        p = m(vf[s:s + bs].float())
        o = torch.sigmoid(p[:, 0]); f = o.flatten(1).argmax(1)
        gj = f // FGW; gi = f % FGW; idx = torch.arange(p.shape[0], device=p.device)
        cx = (gi.float() + torch.sigmoid(p[idx, 1, gj, gi])) * FS
        cy = (gj.float() + torch.sigmoid(p[idx, 2, gj, gi])) * FS
        w = torch.exp(p[idx, 3, gj, gi]) * FS; hh = torch.exp(p[idx, 4, gj, gi]) * FS
        go = vo[s:s + bs]; gb = vb[s:s + bs]
        gf = go.flatten(1).argmax(1); qj = gf // FGW; qi = gf % FGW
        has = go.flatten(1).max(1).values > 0
        gcx = (qi.float() + gb[idx, 0, qj, qi]) * FS; gcy = (qj.float() + gb[idx, 1, qj, qi]) * FS
        gw = torch.exp(gb[idx, 2, qj, qi]) * FS; gh = torch.exp(gb[idx, 3, qj, qi]) * FS
        x1 = torch.maximum(cx - w / 2, gcx - gw / 2); y1 = torch.maximum(cy - hh / 2, gcy - gh / 2)
        x2 = torch.minimum(cx + w / 2, gcx + gw / 2); y2 = torch.minimum(cy + hh / 2, gcy + gh / 2)
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        iou = inter / (w * hh + gw * gh - inter).clamp(min=1e-9)
        h25 += int(((iou >= 0.25) & has).sum()); tot += int(has.sum())
    return h25 / max(tot, 1)


def client_round(state, feat, obj, box, sigma, dp, B, lr):
    m = DPTail().to(DEV); m.load_state_dict(state); m.train()
    dl = torch.utils.data.DataLoader(FeatDS(feat, obj, box), batch_size=B, shuffle=True)
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)
    if dp:
        pe = PrivacyEngine()
        m, opt, dl = pe.make_private(module=m, optimizer=opt, data_loader=dl,
                                     noise_multiplier=sigma, max_grad_norm=C)
    for _ in range(EPOCHS):
        for f, o, b in dl:
            opt.zero_grad(); loss_fn(m(f.float()), o, b).backward(); opt.step()
            GUARD.tick()      # per-step throttle: keeps peak power off the PSU limit
    inner = m._module if hasattr(m, '_module') else m
    return {k: v.detach().clone() for k, v in inner.state_dict().items()}


def run(init, feats, tgts, vf, vo, vb, sigma, B, lr, score_last=6):
    dp = sigma > 0
    st = {k: v.clone() for k, v in init.items()}
    tail = []
    for r in range(1, ROUNDS + 1):
        st = fedavg([client_round(st, feats[i], tgts[i][0], tgts[i][1], sigma, dp, B, lr)
                     for i in range(N_CLIENTS)])
        if r > ROUNDS - score_last:
            tail.append(evaluate(st, vf, vo, vb))
    return float(np.mean(tail))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, required=True)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--sigmas', type=float, nargs='+', required=True)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--tag', default='')
    ap.add_argument('--init', default='public_tail', choices=['public_tail', 'random'])
    ap.add_argument('--rounds', type=int, default=0)
    ap.add_argument('--data', default='', help='feature dir; targets read from the same dir')
    ap.add_argument('--public', default='./fl_workdirs/stage1/public_pretrained.pt')
    a = ap.parse_args()

    if a.data:
        globals()['DATA'] = a.data
        globals()['TGT'] = a.data
    feats, tgts, vf, (vo, vb) = load_cut()
    D = sum(p.numel() for p in DPTail().parameters())
    if a.rounds:
        globals()['ROUNDS'] = a.rounds
    init = make_init(a.init, a.seed, a.public)
    n_per = feats[0].shape[0]
    steps = ROUNDS * EPOCHS * int(np.ceil(n_per / a.B))
    pred = 1.11 * a.B / (D ** 0.5) * (steps * N_CLIENTS) ** 0.5 / 1.11
    print(f'D={D} sqrt(D)={D**0.5:.0f}  B={a.B} lr={a.lr} init={a.init} rounds={ROUNDS}  '
          f'n/client={n_per} steps/run={steps}', flush=True)
    print(f'random-tail floor: {evaluate(init, vf, vo, vb):.3f}\n', flush=True)
    print(f'{"sigma":>8} | {"IoU.25 (r20-25)":>15}   time', flush=True)
    out = {}
    for s in a.sigmas:
        t0 = time.time()
        sc = run(init, feats, tgts, vf, vo, vb, s, a.B, a.lr)
        out[str(s)] = sc
        print(f'{s:8.1f} | {sc:15.3f}   ({time.time()-t0:.0f}s)', flush=True)
        json.dump(out, open(f'./exp_data/gateDB_B{a.B}{a.tag}.json', 'w'), indent=1)
    print('SWEEP DONE', flush=True)
