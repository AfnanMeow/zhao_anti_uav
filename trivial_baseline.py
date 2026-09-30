# What does a DEGENERATE predictor score on this metric?
# If a constant-output head already scores ~0.3 IoU.25, then every number in the
# gate is near-trivial and the metric cannot support any conclusion.
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
sys.path.insert(0, os.path.abspath('.'))
from stage1_public_pretrain import Head, FGW, FGH, FS
from stage2_gate import STAGE1, DEV, evaluate

vf = torch.load(os.path.join(STAGE1,'feat_val.pt'), map_location='cpu').to(DEV)
vt = torch.load(os.path.join(STAGE1,'tgt_val.pt'), map_location='cpu')
vo, vb = vt['obj'].to(DEV), vt['box'].to(DEV)

# --- where are the targets, and how big? ---
n = vo.shape[0]
flat = vo.flatten(1)
has = flat.max(1).values > 0
gf = flat.argmax(1); gj = gf // FGW; gi = gf % FGW
idx = torch.arange(n, device=DEV)
gw = torch.exp(vb[idx,2,gj,gi])*FS; gh = torch.exp(vb[idx,3,gj,gi])*FS
print(f'val frames={n}  with-target={int(has.sum())}')
print(f'target cell   : gi mean={gi.float().mean():.1f} sd={gi.float().std():.1f} (grid W={FGW})')
print(f'                gj mean={gj.float().mean():.1f} sd={gj.float().std():.1f} (grid H={FGH})')
print(f'target size px: w mean={gw.mean():.1f} sd={gw.std():.1f} | h mean={gh.mean():.1f} sd={gh.std():.1f}')
mode_cell = torch.bincount(gf[has], minlength=FGW*FGH).argmax().item()
print(f'most common cell {mode_cell} -> {(gf[has]==mode_cell).float().mean()*100:.2f}% of frames')

# --- CONSTANT head: ignores features entirely (zero all weights, set biases) ---
# objectness bias uniform -> argmax lands on a fixed cell; box biases = dataset mean.
class ConstHead(Head):
    def forward(self, f):
        b = self.c2.bias.view(1,5,1,1).expand(f.shape[0],5,FGH,FGW).contiguous()
        return b

mw = float(torch.log(gw.mean()/FS)); mh = float(torch.log(gh.mean()/FS))
best = None
for cell in [mode_cell, int(gf[has].float().median())]:
    cj, ci = cell // FGW, cell % FGW
    h = ConstHead().to(DEV)
    with torch.no_grad():
        h.c2.bias.copy_(torch.tensor([0.0, 0.0, 0.0, mw, mh], device=DEV))
    # force argmax cell by making that objectness cell win: emulate via eval on a
    # head whose output is constant -> argmax picks index 0; instead measure directly.
    class Fixed(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.p = torch.zeros(1,5,FGH,FGW, device=DEV)
            s.p[0,0] = -10.0; s.p[0,0,cj,ci] = 10.0
            s.p[0,1] = 0.0; s.p[0,2] = 0.0; s.p[0,3] = mw; s.p[0,4] = mh
        def forward(s, f): return s.p.expand(f.shape[0],-1,-1,-1)
    fx = Fixed()
    # inline eval (same math as stage2_gate.evaluate)
    cell_acc = h25 = h50 = tot = 0
    for s0 in range(0, n, 400):
        p = fx(vf[s0:s0+400]); B0 = p.shape[0]
        o = torch.sigmoid(p[:,0]); f2 = o.flatten(1).argmax(1)
        pj = f2 // FGW; pi = f2 % FGW; ii = torch.arange(B0, device=DEV)
        cx = (pi.float()+torch.sigmoid(p[ii,1,pj,pi]))*FS; cy = (pj.float()+torch.sigmoid(p[ii,2,pj,pi]))*FS
        w = torch.exp(p[ii,3,pj,pi])*FS; hh = torch.exp(p[ii,4,pj,pi])*FS
        go = vo[s0:s0+400]; gb = vb[s0:s0+400]
        g2 = go.flatten(1).argmax(1); qj = g2//FGW; qi = g2%FGW
        hs = go.flatten(1).max(1).values > 0
        gcx = (qi.float()+gb[ii,0,qj,qi])*FS; gcy = (qj.float()+gb[ii,1,qj,qi])*FS
        gww = torch.exp(gb[ii,2,qj,qi])*FS; ghh = torch.exp(gb[ii,3,qj,qi])*FS
        x1 = torch.maximum(cx-w/2, gcx-gww/2); y1 = torch.maximum(cy-hh/2, gcy-ghh/2)
        x2 = torch.minimum(cx+w/2, gcx+gww/2); y2 = torch.minimum(cy+hh/2, gcy+ghh/2)
        inter = (x2-x1).clamp(min=0)*(y2-y1).clamp(min=0)
        iou = inter/(w*hh+gww*ghh-inter).clamp(min=1e-9)
        cell_acc += int(((pj==qj)&(pi==qi)&hs).sum())
        h25 += int(((iou>=0.25)&hs).sum()); h50 += int(((iou>=0.5)&hs).sum()); tot += int(hs.sum())
    r = (cell_acc/max(tot,1), h25/max(tot,1), h50/max(tot,1))
    print(f'CONSTANT head @cell({cj},{ci}) meanbox: cell={r[0]:.3f} IoU.25={r[1]:.3f} IoU.50={r[2]:.3f}')
