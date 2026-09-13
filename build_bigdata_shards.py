# =============================================================================
# Build client shards at the SOURCE AUTHORS' data scale, from multiple
# Anti-UAV sequence folders (lightweight-CNN pipeline only -- no DAMO-YOLO).
#
# Why: across ~1,400 controlled paired runs, adaptive privacy-budget allocation
# never separated from uniform -- but the authors have 5000 images PER CLIENT
# (CIFAR-10, 10 clients) while single_test_coco gave us 158. Their own reported
# margins are ~1%% (MNIST) / ~0.2%% (CIFAR-10), i.e. genuinely small effects that
# need that data volume to resolve; at 158 img/client the per-run sd is ~0.13,
# which swamps a 1%% effect. Data volume is the last structural gap.
#
# Also upgrades the evaluation: val frames come from sequence folders DISJOINT
# from every training folder, so it measures generalization to unseen scenes
# rather than held-out frames of a sequence the model already trained on.
#
# Caches preprocessed uint8 tensors to .pt so the many trial runs don't re-decode
# JPEGs (uint8 not float32: 4x less memory, 25k images ~6GB not ~25GB).
# =============================================================================
import os, sys, json, glob, random
import numpy as np
import cv2
import torch

IMG_W, IMG_H = 320, 256
GRID_W, GRID_H, STRIDE = 20, 16, 16

TRAIN_ROOT   = './3rd_Anti-UAV_train_val/train'
OUT_DIR      = './fl_workdirs/bigdata'
N_CLIENTS    = 5
IMGS_PER_CLIENT = 8000       # exceeds the authors' 5000/client
VAL_IMAGES   = 1200          # from SEQUENCE-DISJOINT folders
SEED         = 12345


def load_folder(folder):
    """Return list of (img_path, bbox_or_None) for one sequence folder."""
    lbl_path = os.path.join(folder, 'IR_label.json')
    if not os.path.exists(lbl_path):
        return []
    with open(lbl_path) as f:
        lbl = json.load(f)
    rects = lbl.get('gt_rect', [])
    exists = lbl.get('exist', [])
    imgs = sorted(glob.glob(os.path.join(folder, '*.jpg')))
    out = []
    for p, gt, ex in zip(imgs, rects, exists):
        if ex and gt and len(gt) == 4 and gt[2] > 0 and gt[3] > 0:
            out.append((p, gt))
        else:
            out.append((p, None))
    return out


def encode(items):
    """Preprocess to uint8 image tensor + grid targets."""
    imgs, objs, boxes, keep = [], [], [], []
    for p, gt in items:
        im = cv2.imread(p)
        if im is None:
            continue
        H0, W0 = im.shape[:2]
        im = cv2.resize(im, (IMG_W, IMG_H))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        imgs.append(torch.from_numpy(im).permute(2, 0, 1).contiguous())  # uint8 CHW
        o = torch.zeros(GRID_H, GRID_W)
        b = torch.zeros(4, GRID_H, GRID_W)
        if gt is not None:
            sx, sy = IMG_W / W0, IMG_H / H0
            x, y, w, h = gt
            cx, cy = (x + w / 2) * sx, (y + h / 2) * sy
            rw, rh = max(w * sx, 1e-3), max(h * sy, 1e-3)
            gi = min(max(int(cx // STRIDE), 0), GRID_W - 1)
            gj = min(max(int(cy // STRIDE), 0), GRID_H - 1)
            o[gj, gi] = 1.0
            b[:, gj, gi] = torch.tensor([(cx - gi * STRIDE) / STRIDE,
                                          (cy - gj * STRIDE) / STRIDE,
                                          float(np.log(rw / STRIDE)),
                                          float(np.log(rh / STRIDE))])
        objs.append(o); boxes.append(b); keep.append((p, gt))
    return (torch.stack(imgs), torch.stack(objs), torch.stack(boxes), keep)


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(SEED)
    folders = sorted([os.path.join(TRAIN_ROOT, d) for d in os.listdir(TRAIN_ROOT)
                      if os.path.isdir(os.path.join(TRAIN_ROOT, d))])
    random.shuffle(folders)

    need_train = N_CLIENTS * IMGS_PER_CLIENT
    # SEQUENCE-DISJOINT split. Anti-UAV frames are consecutive video frames, so
    # neighbouring frames are near-duplicates. Splitting frames randomly -- even
    # pooled across many folders -- puts near-copies on both sides and leaks:
    # single folder + random frames read 0.9+, 22 folders pooled+shuffled read
    # 0.21, and fully unseen sequences read 0.093 with the SAME model. Only the
    # last is an honest measurement, so train and val sequences never overlap.
    n_val_folders = 12
    val_folders, train_folders = folders[:n_val_folders], folders[n_val_folders:]

    train_items, used = [], []
    for f in train_folders:
        if len(train_items) >= need_train:
            break
        it = load_folder(f)
        if it:
            train_items.extend(it); used.append(os.path.basename(f))
    train_items = train_items[:need_train]

    val_pool = []
    for f in val_folders:
        val_pool.extend(load_folder(f))
    random.shuffle(val_pool)
    val_items = val_pool[:VAL_IMAGES]

    print(f'{len(folders)} folders total', flush=True)
    print(f'train: {len(train_items)} frames from {len(used)} sequences', flush=True)
    print(f'val:   {len(val_items)} frames from {n_val_folders} DISJOINT sequences', flush=True)
    random.shuffle(train_items)
    per = len(train_items) // N_CLIENTS
    for i in range(N_CLIENTS):
        chunk = train_items[i * per:(i + 1) * per]
        imgs, objs, boxes, keep = encode(chunk)
        torch.save({'images': imgs, 'obj': objs, 'box': boxes,
                    'meta': [(p, gt) for p, gt in keep]},
                   os.path.join(OUT_DIR, f'client_{i}.pt'))
        pos = int(objs.sum().item())
        print(f'  client {i}: {imgs.shape[0]} imgs ({pos} with a drone) -> client_{i}.pt', flush=True)

    imgs, objs, boxes, keep = encode(val_items)
    torch.save({'images': imgs, 'obj': objs, 'box': boxes,
                'meta': [(p, gt) for p, gt in keep]},
               os.path.join(OUT_DIR, 'val.pt'))
    print(f'  val: {imgs.shape[0]} imgs ({int(objs.sum().item())} with a drone) -> val.pt',
          flush=True)
    with open(os.path.join(OUT_DIR, 'manifest.json'), 'w') as f:
        json.dump({'train_folders': used, 'val_folders': [os.path.basename(x) for x in val_folders],
                   'n_clients': N_CLIENTS, 'imgs_per_client': per}, f, indent=1)
    print('done', flush=True)
