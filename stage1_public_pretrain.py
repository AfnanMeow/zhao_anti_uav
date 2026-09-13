# =============================================================================
# STAGE 1: public pretraining + frozen-trunk feature cache.
#
# Why: DP-SGD noise scales as sigma*C*sqrt(D)/B. Training all 648K detector
# params under DP collapsed accuracy 0.41 -> <=0.08 at every usable noise level,
# because detection needs capacity but DP punishes it. Standard fix (how DP is
# made to work at scale in practice): pretrain a backbone on PUBLIC data, freeze
# it, and spend the privacy budget only on a small head.
#
#   trunk (frozen, public)  ~636K params  -> no privacy cost, never sees client data
#   head  (DP-trained)      ~13K params   -> sqrt(D) 805 -> 113, ~7x better SNR
#
# The 150 Anti-UAV sequences split three ways with NO overlap:
#   public   : pretraining only          (never used as client or val data)
#   clients  : 5 x 8000 private frames   (fl_workdirs/bigdata/client_*.pt)
#   val      : 12 disjoint sequences     (fl_workdirs/bigdata/val.pt)
#
# Because the trunk is frozen, its output per image is FIXED -- so we cache
# features once. Head-only DP training then runs on cached tensors, turning a
# 25-round FL run from ~10 min into seconds, which is what makes the later RL
# phase feasible at all.
# =============================================================================
import os, sys, json, glob, random, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMG_W, IMG_H = 320, 256
FS = 8                                   # stride-8 grid (fixed the localization bottleneck)
FGW, FGH = IMG_W // FS, IMG_H // FS      # 40 x 32
TRAIN_ROOT = './3rd_Anti-UAV_train_val/train'
BIGDATA    = './fl_workdirs/bigdata'
OUT        = './fl_workdirs/stage1'
PUBLIC_FOLDERS = 20
PUBLIC_EPOCHS  = 8
SEED = 12345


# ── model, split explicitly into frozen trunk and DP-trained head ───────────
class Trunk(nn.Module):
    """Public-pretrained feature extractor. Frozen during federated training,
    so it costs no privacy budget and its outputs can be cached."""

    def __init__(self, gn=8):
        super().__init__()

        def blk(i, o, s=2):
            # inplace=False: Opacus per-sample grad hooks fail on in-place ops
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(gn, o), nn.ReLU(inplace=False))

        self.s1 = blk(3, 32)       # 160x128
        self.s2 = blk(32, 64)      # 80x64
        self.s3 = blk(64, 128)     # 40x32  <- stride 8, keep resolution
        self.k1 = nn.Sequential(nn.Conv2d(128, 192, 3, 1, 1), nn.GroupNorm(gn, 192), nn.ReLU(inplace=False))
        self.k2 = nn.Sequential(nn.Conv2d(192, 192, 3, 1, 1), nn.GroupNorm(gn, 192), nn.ReLU(inplace=False))

    def forward(self, x):
        return self.k2(self.k1(self.s3(self.s2(self.s1(x)))))


class Head(nn.Module):
    """The ONLY part trained under DP. Deliberately small: D drives the noise."""

    def __init__(self, cin=192, mid=64):
        super().__init__()
        self.c1 = nn.Conv2d(cin, mid, 1)
        self.c2 = nn.Conv2d(mid, 5, 1)

    def forward(self, f):
        return self.c2(F.relu(self.c1(f)))


class FullNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = Trunk()
        self.head = Head()

    def forward(self, x):
        return self.head(self.trunk(x))


def loss_fn(pred, obj_t, box_t):
    ol = F.binary_cross_entropy_with_logits(pred[:, 0], obj_t)
    pm = obj_t.unsqueeze(1)
    npos = pm.sum().clamp(min=1.0)
    bl = (F.smooth_l1_loss(pred[:, 1:5], box_t, reduction='none') * pm).sum() / npos
    return ol + 5.0 * bl


# ── data ────────────────────────────────────────────────────────────────────
def load_folder(folder):
    lp = os.path.join(folder, 'IR_label.json')
    if not os.path.exists(lp):
        return []
    lbl = json.load(open(lp))
    rects, exists = lbl.get('gt_rect', []), lbl.get('exist', [])
    imgs = sorted(glob.glob(os.path.join(folder, '*.jpg')))
    out = []
    for p, gt, ex in zip(imgs, rects, exists):
        ok = ex and gt and len(gt) == 4 and gt[2] > 0 and gt[3] > 0
        out.append((p, gt if ok else None))
    return out


def encode(items):
    ims, objs, boxes = [], [], []
    for p, gt in items:
        im = cv2.imread(p)
        if im is None:
            continue
        H0, W0 = im.shape[:2]
        im = cv2.cvtColor(cv2.resize(im, (IMG_W, IMG_H)), cv2.COLOR_BGR2RGB)
        ims.append(torch.from_numpy(im).permute(2, 0, 1).contiguous())
        o = torch.zeros(FGH, FGW); b = torch.zeros(4, FGH, FGW)
        if gt is not None:
            sx, sy = IMG_W / W0, IMG_H / H0
            x, y, w, h = gt
            cx, cy = (x + w / 2) * sx, (y + h / 2) * sy
            rw, rh = max(w * sx, 1e-3), max(h * sy, 1e-3)
            gi = min(max(int(cx // FS), 0), FGW - 1); gj = min(max(int(cy // FS), 0), FGH - 1)
            o[gj, gi] = 1.0
            b[:, gj, gi] = torch.tensor([(cx - gi * FS) / FS, (cy - gj * FS) / FS,
                                          float(np.log(rw / FS)), float(np.log(rh / FS))])
        objs.append(o); boxes.append(b)
    return torch.stack(ims), torch.stack(objs), torch.stack(boxes)


def reencode_stride8(shard_path, device='cuda:0'):
    """Targets for an existing client/val shard at stride 8, from stored boxes."""
    d = torch.load(shard_path, map_location='cpu')
    meta = d['meta']; n = len(meta)
    o = torch.zeros(n, FGH, FGW); b = torch.zeros(n, 4, FGH, FGW)
    for i, (p, gt) in enumerate(meta):
        if gt is None:
            continue
        sx, sy = IMG_W / 640.0, IMG_H / 512.0
        x, y, w, h = gt
        cx, cy = (x + w / 2) * sx, (y + h / 2) * sy
        rw, rh = max(w * sx, 1e-3), max(h * sy, 1e-3)
        gi = min(max(int(cx // FS), 0), FGW - 1); gj = min(max(int(cy // FS), 0), FGH - 1)
        o[i, gj, gi] = 1.0
        b[i, :, gj, gi] = torch.tensor([(cx - gi * FS) / FS, (cy - gj * FS) / FS,
                                         float(np.log(rw / FS)), float(np.log(rh / FS))])
    return d['images'], o, b


@torch.no_grad()
def eval_full(trunk, head, imgs_u8, obj, box, bs=400):
    trunk.eval(); head.eval()
    cell = h25 = h50 = tot = 0
    for s in range(0, imgs_u8.shape[0], bs):
        x = imgs_u8[s:s + bs].to('cuda:0').float() / 255.0
        p = head(trunk(x))
        o = torch.sigmoid(p[:, 0])
        f = o.flatten(1).argmax(1); gj = f // FGW; gi = f % FGW
        idx = torch.arange(p.shape[0], device=p.device)
        cx = (gi.float() + torch.sigmoid(p[idx, 1, gj, gi])) * FS
        cy = (gj.float() + torch.sigmoid(p[idx, 2, gj, gi])) * FS
        w = torch.exp(p[idx, 3, gj, gi]) * FS; h = torch.exp(p[idx, 4, gj, gi]) * FS
        go = obj[s:s + bs].to('cuda:0'); gb = box[s:s + bs].to('cuda:0')
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
        h25 += int(((iou >= 0.25) & has).sum()); h50 += int(((iou >= 0.5) & has).sum())
        tot += int(has.sum())
    return cell / max(tot, 1), h25 / max(tot, 1), h50 / max(tot, 1)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    dev = 'cuda:0'
    man = json.load(open(os.path.join(BIGDATA, 'manifest.json')))
    used = set(man['train_folders']) | set(man['val_folders'])
    allf = sorted([d for d in os.listdir(TRAIN_ROOT) if os.path.isdir(os.path.join(TRAIN_ROOT, d))])
    free = [f for f in allf if f not in used]
    random.seed(SEED); random.shuffle(free)
    pub = free[:PUBLIC_FOLDERS]
    print(f'public pretraining on {len(pub)} sequences (disjoint from all clients & val)', flush=True)

    items = []
    for f in pub:
        items.extend(load_folder(os.path.join(TRAIN_ROOT, f)))
    random.shuffle(items)
    print(f'  {len(items)} public frames; encoding...', flush=True)
    t0 = time.time()
    pim, pobj, pbox = encode(items)
    print(f'  encoded in {time.time()-t0:.0f}s -> {tuple(pim.shape)}', flush=True)
    pim = pim.to(dev); pobj = pobj.to(dev); pbox = pbox.to(dev)

    net = FullNet().to(dev)
    ntr = sum(p.numel() for p in net.trunk.parameters())
    nhd = sum(p.numel() for p in net.head.parameters())
    print(f'  trunk={ntr} params (frozen later)   head={nhd} params (DP-trained), '
          f'sqrt(D_head)={nhd**0.5:.0f}', flush=True)

    # ── public pretraining (clean, centralized -- public data, no privacy cost)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    B = 128
    vimg, vobj, vbox = reencode_stride8(os.path.join(BIGDATA, 'val.pt'))
    t0 = time.time()
    for ep in range(PUBLIC_EPOCHS):
        net.train()
        idx = torch.randperm(pim.shape[0], device=dev)
        for i in range(0, len(idx), B):
            j = idx[i:i + B]
            opt.zero_grad()
            loss_fn(net(pim[j].float() / 255.0), pobj[j], pbox[j]).backward()
            opt.step()
        c, i25, i50 = eval_full(net.trunk, net.head, vimg, vobj, vbox)
        print(f'  public epoch {ep+1}/{PUBLIC_EPOCHS}: val cell={c:.3f} IoU.25={i25:.3f} '
              f'IoU.50={i50:.3f} ({time.time()-t0:.0f}s)', flush=True)

    torch.save({'trunk': net.trunk.state_dict(), 'head': net.head.state_dict()},
               os.path.join(OUT, 'public_pretrained.pt'))
    print(f'saved {OUT}/public_pretrained.pt', flush=True)

    # ── cache frozen-trunk features for clients + val ───────────────────────
    net.trunk.eval()
    for p in net.trunk.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def cache(imgs_u8, tag):
        feats = []
        for s in range(0, imgs_u8.shape[0], 256):
            x = imgs_u8[s:s + 256].to(dev).float() / 255.0
            feats.append(net.trunk(x).half().cpu())
        f = torch.cat(feats)
        torch.save(f, os.path.join(OUT, f'feat_{tag}.pt'))
        return f.shape

    for i in range(5):
        im, o, b = reencode_stride8(os.path.join(BIGDATA, f'client_{i}.pt'))
        shp = cache(im, f'client_{i}')
        torch.save({'obj': o, 'box': b}, os.path.join(OUT, f'tgt_client_{i}.pt'))
        print(f'  cached client {i}: features {tuple(shp)}', flush=True)
    shp = cache(vimg, 'val')
    torch.save({'obj': vobj, 'box': vbox}, os.path.join(OUT, 'tgt_val.pt'))
    print(f'  cached val: features {tuple(shp)}', flush=True)
    print('STAGE 1 DONE', flush=True)
