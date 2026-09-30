# =============================================================================
# DP-SGD schedule test at the SOURCE AUTHORS' DATA SCALE.
#
# Everything else has been held constant and already ruled out:
#   * mechanism  -- now real DP-SGD via opacus (the port used DP-FedAvg and never
#                   called opacus at all), noise on per-sample grads every step
#   * round count-- 25, the authors' setting
#   * model size -- SmallDetector (25K params) so per-step SNR = B/(sigma*sqrt(D))
#                   lands in the authors' regime rather than 30x below it
#   * data       -- THIS FILE: 5000 images/client, matching the authors exactly.
#                   Every earlier run had 158/client, where per-run sd ~0.13
#                   swamps the ~1%% effect size the authors themselves report.
#
# Evaluation is on 600 frames from 4 sequence folders DISJOINT from all 22
# training folders -- generalization to unseen scenes, not held-out frames of a
# sequence the model already saw.
#
# Schedules: rho_t proportional to (t+1)^k, normalized to spend exactly rho_total.
#   k=0 uniform (control) | k>0 less noise late (paper's prediction) | k<0 control
# =============================================================================
import os, sys, json, random, io, contextlib, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fl_dpsgd_final import make_small_dp_model, fedavg, train_local_dpsgd
from fl_lightcnn_final import compute_loss, GRID_W, STRIDE, SCALE_X, SCALE_Y

BIGDATA   = './fl_workdirs/bigdata'
N         = 5
CR_TOTAL  = 25
EPOCH_LOCAL, BATCH_SIZE, LR, C_CLIP = 4, 200, 0.5, 1.0   # el/B match the authors
RHO_MIN, RHO_MAX = 20.0, 40.0                            # authors' CIFAR-10 budget
SIGMA_MIN, SIGMA_MAX = 1.0, 20.0
N_TRIALS  = 12
K_VALUES  = [-1.0, 0.0, 1.0]
SEED_BASE = 84000
OUT_PATH  = './exp_data/dpsgd_bigdata_test.json'


class CachedShard:
    """Wraps a preprocessed .pt shard. Images cached as uint8 (4x less memory
    than float32: 25k imgs ~6GB not ~25GB); normalized per batch on the fly."""

    def __init__(self, path, device='cuda:0'):
        d = torch.load(path, map_location='cpu')
        self.images_u8 = d['images'].to(device)
        self.obj_targets = d['obj'].to(device)
        self.box_targets = d['box'].to(device)
        self.meta = d['meta']
        self.n = self.images_u8.shape[0]
        self.device = device

    @property
    def images(self):
        return self.images_u8

    def float_images(self, idx):
        return self.images_u8[idx].float() / 255.0


class _TensorDS(torch.utils.data.Dataset):
    def __init__(self, shard):
        self.s = shard

    def __len__(self):
        return self.s.n

    def __getitem__(self, i):
        return self.s.images_u8[i], self.s.obj_targets[i], self.s.box_targets[i]


def loader_for(shard, batch_size):
    return torch.utils.data.DataLoader(_TensorDS(shard), batch_size=batch_size,
                                        shuffle=True, num_workers=0)


def train_round(state_dict, shard, epochs, batch_size, lr, sigma, C, device='cuda:0'):
    from opacus import PrivacyEngine
    model = make_small_dp_model(device)
    model.load_state_dict(state_dict)
    model.train()
    loader = loader_for(shard, batch_size)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    pe = PrivacyEngine()
    model, opt, loader = pe.make_private(module=model, optimizer=opt, data_loader=loader,
                                          noise_multiplier=sigma, max_grad_norm=C)
    for _ in range(epochs):
        for imgs_u8, obj_t, box_t in loader:
            opt.zero_grad()
            loss, _, _ = compute_loss(model(imgs_u8.float() / 255.0), obj_t, box_t)
            loss.backward()
            opt.step()
    inner = model._module if hasattr(model, '_module') else model
    return {k: v.detach().clone() for k, v in inner.state_dict().items()}


@torch.no_grad()
def eval_map(state_dict, val_shard, device='cuda:0'):
    """Top-1 box per image vs ground truth, IoU>=0.5 -> detection rate.
    (COCOeval needs a COCO object; at this scale a direct IoU@0.5 hit-rate over
    the held-out sequences is equivalent for ranking schedules and much faster.)"""
    model = make_small_dp_model(device)
    model.load_state_dict(state_dict)
    model.eval()
    hits = 0; total = 0
    B = 500
    for s in range(0, val_shard.n, B):
        imgs = val_shard.images_u8[s:s + B].float() / 255.0
        pred = model(imgs)
        obj = torch.sigmoid(pred[:, 0])
        flat = obj.flatten(1).argmax(1)
        gj = (flat // GRID_W); gi = (flat % GRID_W)
        idx = torch.arange(pred.shape[0], device=pred.device)
        tx = torch.sigmoid(pred[idx, 1, gj, gi]); ty = torch.sigmoid(pred[idx, 2, gj, gi])
        tw = pred[idx, 3, gj, gi]; th = pred[idx, 4, gj, gi]
        cx = (gi.float() + tx) * STRIDE; cy = (gj.float() + ty) * STRIDE
        w = torch.exp(tw) * STRIDE;      h = torch.exp(th) * STRIDE
        px1, py1, px2, py2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

        gt_obj = val_shard.obj_targets[s:s + B]
        gt_box = val_shard.box_targets[s:s + B]
        gflat = gt_obj.flatten(1).argmax(1)
        ggj = (gflat // GRID_W); ggi = (gflat % GRID_W)
        has = gt_obj.flatten(1).max(1).values > 0
        gtx = gt_box[idx, 0, ggj, ggi]; gty = gt_box[idx, 1, ggj, ggi]
        gtw = gt_box[idx, 2, ggj, ggi]; gth = gt_box[idx, 3, ggj, ggi]
        gcx = (ggi.float() + gtx) * STRIDE; gcy = (ggj.float() + gty) * STRIDE
        gw = torch.exp(gtw) * STRIDE;       gh = torch.exp(gth) * STRIDE
        gx1, gy1, gx2, gy2 = gcx - gw / 2, gcy - gh / 2, gcx + gw / 2, gcy + gh / 2

        ix1 = torch.maximum(px1, gx1); iy1 = torch.maximum(py1, gy1)
        ix2 = torch.minimum(px2, gx2); iy2 = torch.minimum(py2, gy2)
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        union = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0) + \
                (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0) - inter
        iou = inter / union.clamp(min=1e-9)
        hits += int(((iou >= 0.5) & has).sum().item())
        total += int(has.sum().item())
    return hits / max(total, 1)


def power_allocation(rho_total, T, k):
    w = np.array([(t + 1) ** k for t in range(T)], dtype=float)
    return (w / w.sum() * rho_total).tolist()


def run_allocation(shards, allocs, init_sd, val_shard):
    gsd = {k: v.clone() for k, v in init_sd.items()}
    for CR in range(CR_TOTAL):
        local = []
        for i, sh in enumerate(shards):
            sigma = float(np.clip(np.sqrt(2 * EPOCH_LOCAL / max(allocs[i][CR], 1e-9)),
                                   SIGMA_MIN, SIGMA_MAX))
            local.append(train_round(gsd, sh, EPOCH_LOCAL, BATCH_SIZE, LR, sigma, C_CLIP))
        gsd = fedavg(local)
    return eval_map(gsd, val_shard)


if __name__ == '__main__':
    print('loading cached shards (5 x 5000 imgs)...', flush=True)
    shards = [CachedShard(os.path.join(BIGDATA, f'client_{i}.pt')) for i in range(N)]
    val_shard = CachedShard(os.path.join(BIGDATA, 'val.pt'))
    print(f'clients: {[s.n for s in shards]}   val: {val_shard.n} (unseen sequences)', flush=True)
    print(f'DP-SGD  el={EPOCH_LOCAL} B={BATCH_SIZE} C={C_CLIP} lr={LR}  {CR_TOTAL} rounds  '
          f'rho in [{RHO_MIN},{RHO_MAX}] (authors\' budget)  n={N_TRIALS}\n', flush=True)

    res = {f'{k}': [] for k in K_VALUES}
    for t in range(N_TRIALS):
        rng = np.random.RandomState(SEED_BASE + t)
        random.seed(SEED_BASE + t); np.random.seed(SEED_BASE + t); torch.manual_seed(SEED_BASE + t)
        rhos = [float(rng.uniform(RHO_MIN, RHO_MAX)) for _ in range(N)]
        init_sd = make_small_dp_model().state_dict()
        line = []
        for k in K_VALUES:
            allocs = [power_allocation(rhos[i], CR_TOTAL, k) for i in range(N)]
            acc = run_allocation(shards, allocs, init_sd, val_shard)
            res[f'{k}'].append(acc)
            line.append(f'k={k:+.1f}:{acc:.3f}')
        print(f'[Trial {t+1:2d}] ' + '  '.join(line), flush=True)
        with open(OUT_PATH, 'w') as f:
            json.dump(res, f, indent=1)

    uni = np.array(res['0.0'])
    print(f'\n=== n={N_TRIALS} paired vs uniform (DP-SGD, authors\' data scale) ===', flush=True)
    print(f'{"uniform (k=0)":16s} mean={uni.mean():.4f}', flush=True)
    try:
        from scipy import stats
    except Exception:
        stats = None
    for k in K_VALUES:
        if k == 0.0:
            continue
        arr = np.array(res[f'{k}']); d = arr - uni
        se = d.std(ddof=1) / np.sqrt(len(d)); tst = d.mean() / se
        p = 2 * (1 - stats.t.cdf(abs(tst), df=len(d) - 1)) if stats else float('nan')
        print(f'k={k:<+13.1f} mean={arr.mean():.4f}  diff={d.mean():+.4f}  '
              f'95%CI=[{d.mean()-1.96*se:+.4f},{d.mean()+1.96*se:+.4f}]  '
              f'beats={sum(d>0)}/{len(d)}  t={tst:+.2f}  p={p:.4f}', flush=True)
