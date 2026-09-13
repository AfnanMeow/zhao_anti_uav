# =============================================================================
# Does a finer output grid fix localization?
# Diagnosis: box SIZE is accurate (pred/gt ratio 1.06-1.07) but the model picks
# the right grid cell only 52% of the time -> localization, not regression, is
# the bottleneck. At 320x256 with stride 16 the grid is just 20x16, so each cell
# spans 16px while a drone is ~50px. This re-encodes targets at stride 8 (40x32)
# from the stored box coordinates -- no JPEG re-decoding needed -- and adds a
# matching stride-8 detector head.
# =============================================================================
import os, sys, warnings, time
warnings.filterwarnings('ignore')
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dpsgd_bigdata_test import CachedShard, BIGDATA
from fl_dpsgd_final import fedavg
from opacus.validators import ModuleValidator
import torch.nn.functional as F

IMG_W, IMG_H = 320, 256
FS = 8                      # finer stride
FGW, FGH = IMG_W // FS, IMG_H // FS      # 40 x 32


def reencode(shard, stride=FS):
    """Rebuild grid targets at a finer stride from the stored box metadata."""
    n = shard.n
    gw, gh = IMG_W // stride, IMG_H // stride
    obj = torch.zeros(n, gh, gw, device=shard.images_u8.device)
    box = torch.zeros(n, 4, gh, gw, device=shard.images_u8.device)
    for i, (p, gt) in enumerate(shard.meta):
        if gt is None:
            continue
        # meta boxes are in ORIGINAL image coords; shards were resized to 320x256
        import cv2
        # original Anti-UAV frames are 640x512
        sx, sy = IMG_W / 640.0, IMG_H / 512.0
        x, y, w, h = gt
        cx, cy = (x + w / 2) * sx, (y + h / 2) * sy
        rw, rh = max(w * sx, 1e-3), max(h * sy, 1e-3)
        gi = min(max(int(cx // stride), 0), gw - 1)
        gj = min(max(int(cy // stride), 0), gh - 1)
        obj[i, gj, gi] = 1.0
        box[i, :, gj, gi] = torch.tensor([(cx - gi * stride) / stride,
                                           (cy - gj * stride) / stride,
                                           float(np.log(rw / stride)),
                                           float(np.log(rh / stride))],
                                          device=obj.device)
    return obj, box


# NOTE: ReLU must be inplace=False -- Opacus's per-sample gradient hooks raise
# "Output 0 of BackwardHookFunctionBackward is a view and is being modified
# inplace" on in-place activations.
class FineDetector(nn.Module):
    """Same trunk, but output at stride 8 (40x32) instead of 16 (20x16)."""

    def __init__(self, gn=8):
        super().__init__()

        def blk(i, o, s=2):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(gn, o), nn.ReLU(inplace=False))

        self.s1 = blk(3, 32)       # 160x128
        self.s2 = blk(32, 64)      # 80x64
        self.s3 = blk(64, 128)     # 40x32   <- stride 8, keep this resolution
        self.k1 = nn.Sequential(nn.Conv2d(128, 192, 3, 1, 1), nn.GroupNorm(gn, 192), nn.ReLU(inplace=False))
        self.k2 = nn.Sequential(nn.Conv2d(192, 192, 3, 1, 1), nn.GroupNorm(gn, 192), nn.ReLU(inplace=False))
        self.head = nn.Conv2d(192, 5, 1)

    def forward(self, x):
        x = self.s1(x); x = self.s2(x); x = self.s3(x)
        return self.head(self.k2(self.k1(x)))


def make_fine(device='cuda:0'):
    return ModuleValidator.fix(FineDetector()).to(device)


def loss_fn(pred, obj_t, box_t):
    ol = F.binary_cross_entropy_with_logits(pred[:, 0], obj_t)
    pm = obj_t.unsqueeze(1)
    npos = pm.sum().clamp(min=1.0)
    bl = (F.smooth_l1_loss(pred[:, 1:5], box_t, reduction='none') * pm).sum() / npos
    return ol + 5.0 * bl


def batches(imgs_u8, obj, box, bs):
    idx = torch.randperm(imgs_u8.shape[0], device=imgs_u8.device)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        yield imgs_u8[b].float() / 255.0, obj[b], box[b]


@torch.no_grad()
def evaluate(sd, vi, vo, vb):
    m = make_fine(); m.load_state_dict(sd); m.eval()
    cell = hit50 = hit25 = tot = 0
    for s in range(0, vi.shape[0], 400):
        imgs = vi[s:s + 400].float() / 255.0
        p = m(imgs); o = torch.sigmoid(p[:, 0])
        f = o.flatten(1).argmax(1); gj = f // FGW; gi = f % FGW
        idx = torch.arange(p.shape[0], device=p.device)
        cx = (gi.float() + torch.sigmoid(p[idx, 1, gj, gi])) * FS
        cy = (gj.float() + torch.sigmoid(p[idx, 2, gj, gi])) * FS
        w = torch.exp(p[idx, 3, gj, gi]) * FS; h = torch.exp(p[idx, 4, gj, gi]) * FS
        go = vo[s:s + 400]; gb = vb[s:s + 400]
        gf = go.flatten(1).argmax(1); ggj = gf // FGW; ggi = gf % FGW
        has = go.flatten(1).max(1).values > 0
        gcx = (ggi.float() + gb[idx, 0, ggj, ggi]) * FS
        gcy = (ggj.float() + gb[idx, 1, ggj, ggi]) * FS
        gw = torch.exp(gb[idx, 2, ggj, ggi]) * FS; gh = torch.exp(gb[idx, 3, ggj, ggi]) * FS
        ix1 = torch.maximum(cx - w / 2, gcx - gw / 2); iy1 = torch.maximum(cy - h / 2, gcy - gh / 2)
        ix2 = torch.minimum(cx + w / 2, gcx + gw / 2); iy2 = torch.minimum(cy + h / 2, gcy + gh / 2)
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        iou = inter / (w * h + gw * gh - inter).clamp(min=1e-9)
        cell += int(((gj == ggj) & (gi == ggi) & has).sum())
        hit50 += int(((iou >= 0.5) & has).sum()); hit25 += int(((iou >= 0.25) & has).sum())
        tot += int(has.sum())
    return cell / tot, hit25 / tot, hit50 / tot


if __name__ == '__main__':
    N, CR = 5, 40
    shards = [CachedShard(os.path.join(BIGDATA, f'client_{i}.pt')) for i in range(N)]
    val = CachedShard(os.path.join(BIGDATA, 'val.pt'))
    print('re-encoding targets at stride 8 (40x32 grid)...', flush=True)
    tgts = [reencode(s) for s in shards]
    vo, vb = reencode(val)
    print(f'clients {[s.n for s in shards]} | val {val.n} (disjoint seqs)\n', flush=True)

    torch.manual_seed(0)
    gsd = make_fine().state_dict()
    t0 = time.time()
    for r in range(CR):
        locals_ = []
        for s, (o, b) in zip(shards, tgts):
            m = make_fine(); m.load_state_dict(gsd); m.train()
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            for _ in range(3):
                for im, ob, bx in batches(s.images_u8, o, b, 32):
                    opt.zero_grad(); loss_fn(m(im), ob, bx).backward(); opt.step()
            locals_.append({k: v.detach().clone() for k, v in m.state_dict().items()})
        gsd = fedavg(locals_)
        if r in (4, 9, 14, 19, 24, 29, 34, 39):
            c, i25, i50 = evaluate(gsd, val.images_u8, vo, vb)
            print(f'  round {r+1:2d}: cell={c:.3f}  IoU>=0.25={i25:.3f}  IoU>=0.5={i50:.3f}  '
                  f'({time.time()-t0:.0f}s)', flush=True)
    torch.save(gsd, './fl_workdirs/bigdata/fine_clean.pt')
    print('saved ./fl_workdirs/bigdata/fine_clean.pt', flush=True)
