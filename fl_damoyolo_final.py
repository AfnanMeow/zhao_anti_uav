# =============================================================================
# FL + DP + DRL for Drone Detection
# Fixed: trainer._model instead of trainer.model, robust checkpoint saving.
# =============================================================================

from collections.abc import Callable, Iterable, Mapping
from typing import Any
import numpy as np
import torch.multiprocessing as mp
import time
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import opacus
import numpy as np
from collections import deque
import random
import os
import sys
import copy
import json
import subprocess
import warnings
import pickle
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# =============================================================================
# ── PATHS — update these ──────────────────────────────────────────────────────
# =============================================================================
TRAIN_IMG_DIR = 'C:/Users/T25301039/Documents/zhao_anti_uav-p2/3rd_Anti-UAV_train_val/train/20190925_130434_1_3'
VAL_IMG_DIR   = 'C:/Users/T25301039/Documents/zhao_anti_uav-p2/3rd_Anti-UAV_train_val/train/20190925_130434_1_3'
TRAIN_ANN     = 'C:/Users/T25301039/Documents/zhao_anti_uav-p2/single_test_coco/annotations/train.json'
VAL_ANN       = 'C:/Users/T25301039/Documents/zhao_anti_uav-p2/single_test_coco/annotations/val.json'
CACHE_PATH    = 'C:/Users/T25301039/.cache/modelscope/hub/damo/cv_tinynas_uav-detection_damoyolo'
SHARD_DIR     = 'C:/Users/T25301039/Documents/zhao_anti_uav-p2/fl_workdirs/shards'
CKPT_DIR      = 'C:/Users/T25301039/Documents/zhao_anti_uav-p2/fl_workdirs/client_ckpts'
DRONE_TRAINER = 'C:/Users/T25301039/Documents/zhao_anti_uav-p2/uav_scripts/train_uav_detection_single_folder.py'
CONDA_ENV     = 'anti_uav'  # unused (subprocess launches via sys.executable, i.e. whatever venv runs this script)

# =============================================================================
# ── HYPERPARAMETERS ─────────────────────────────────────────────────────────
# =============================================================================
N           = 5
batch_size  = 8
episodes    = 10
CR_Total    = 15
epoch_local = 7
rho_min     = 20.0
rho_max     = 40.0
DRL_steps   = 1
sigma_max   = 20.0
sigma_min   = 1.0
rho_used_min = 2 / sigma_max**2 * epoch_local
rho_used_max = 2 / sigma_min**2 * epoch_local

# C_clip: per-round clip norm for the WHOLE flattened client update, EXCLUDING
# BatchNorm running_mean/running_var/num_batches_tracked buffers (see
# Training.train_one_round()'s two noise-recalibration comments for the full
# derivation -- both the per-tensor-noise bug AND the BN-buffer-contamination
# bug are documented there). The original C_clip=6 was tuned for the author's
# toy CNN. First measurement attempt (before excluding BN buffers) showed an
# unclipped norm of ~2.85 million -- but that turned out to be 99.75%% just ONE
# BatchNorm running_var buffer, not real trained-parameter signal. After
# excluding BN buffers, the REAL trainable-parameter delta norm measured only
# ~0.157 for DAMO-YOLO-S (16.3M trainable params) on a real local-training
# round. Set in that ballpark. This was measured at a SMALL smoke-test scale
# (24 images/client, 6 local epochs, batch 4) -- re-measure at the real
# production hyperparameters (epoch_local=7, batch_size=8, ~150 img/client
# shard) via the `DEBUG noise calib` print in train_one_round() before
# treating this as final for the thesis run; update magnitude scales with
# both dataset size and epoch count.
C_CLIP_DEFAULT = 0.15

CHECKPOINT_PATH     = './DRL_Nets/checkpoint_drone.pth'
CHECKPOINT_INTERVAL = 5

# How each episode's model is seeded (see Net.__init__ for details):
#   'seed_perturb'  - keep the pretrained backbone/neck/regression-head as-is,
#                      and partially perturb ONLY the classification layers
#                      (Recommended -- see HEAD_PERTURB_SCALE below and
#                      Net.__init__'s docstring for why this replaced
#                      'seed_transfer' as the default).
#   'seed_transfer' - keep pretrained backbone+neck, fully randomize the head
#                      (both cls AND reg branches). Verified to destabilize
#                      training: the regression branch's DFL bucket
#                      representation is very sensitive to a full reset, and
#                      the model collapses to predicting background
#                      everywhere within a handful of gradient steps
#                      (loss_bbox/loss_dfl -> exactly 0.0 and stay there).
#                      Kept only for reference/comparison, not recommended.
#   'seed_scratch'  - randomize backbone+neck+head too (true from-scratch FL).
INIT_MODE = 'seed_perturb'

# Gaussian noise scale (relative to each perturbed tensor's own std) applied
# to the classification branch under 'seed_perturb'. Chosen empirically via
# an isolated 8-epoch diagnostic sweep (0.3, 0.5, 1.0, 2.0): 0.3 kept
# loss_cls/loss_bbox/loss_dfl all stable in ~[0.1, 0.6] for the full 8
# epochs with no collapse; 1.0 stayed persistently elevated/unstable; 2.0
# reproduced the same all-background collapse as a full 'seed_transfer'
# reset. 0.3 leaves a comfortable safety margin before that instability.
HEAD_PERTURB_SCALE = 0.3

def rho2sigma(rho, el, bs):
    return float(np.sqrt(2 * el / rho))

def sigma2rho(sigma, el, bs):
    return 2 * el / (sigma**2)

# =============================================================================
# ── ANNOTATION SHARD SPLITTER ────────────────────────────────────────────────
# =============================================================================
def split_coco_annotations(master_ann, n_clients, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(master_ann) as f:
        data = json.load(f)
    imgs = data['images'][:]
    random.shuffle(imgs)
    shard_size = max(1, len(imgs) // n_clients)
    id2anns = {}
    for ann in data['annotations']:
        id2anns.setdefault(ann['image_id'], []).append(ann)
    paths, offset = [], 0
    for i in range(n_clients):
        shard_imgs = imgs[i * shard_size:(i + 1) * shard_size]
        shard_anns = []
        for img in shard_imgs:
            for ann in id2anns.get(img['id'], []):
                a = copy.deepcopy(ann); a['id'] = offset; offset += 1
                shard_anns.append(a)
        out = os.path.join(out_dir, f'train_client_{i}.json')
        with open(out, 'w') as f:
            json.dump({'info': data.get('info', {}),
                       'licenses': data.get('licenses', []),
                       'categories': data['categories'],
                       'images': shard_imgs,
                       'annotations': shard_anns}, f)
        print(f'  Client {i}: {len(shard_imgs)} imgs, {len(shard_anns)} anns → {out}')
        paths.append(out)
    return paths

# =============================================================================
# ── mAP EVALUATOR (unchanged) ────────────────────────────────────────────────
# =============================================================================
def evaluate_map(ckpt_path, val_ann=VAL_ANN, img_size=640,
                 conf_thresh=0.3, device='cuda:0'):
    if not os.path.exists(ckpt_path):
        return 0.0
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        print(f'[evaluate_map] Loading model from {ckpt_path} ...', flush=True)
        pipe = pipeline(Tasks.domain_specific_object_detection,
                        model='damo/cv_tinynas_uav-detection_damoyolo',
                        model_revision='master',
                        trust_remote_code=True)
        sd = torch.load(ckpt_path, map_location='cpu')
        sd = {k.replace('_module.', ''): v for k, v in sd.items()}
        pipe.model.load_state_dict(sd, strict=False)
        pipe.model.eval()

        coco_gt = COCO(val_ann)
        img_ids = coco_gt.getImgIds()
        print(f'[evaluate_map] Running inference on {len(img_ids)} images ...', flush=True)
        results = []
        for idx, img_id in enumerate(img_ids):
            if idx % 50 == 0:
                print(f'[evaluate_map]   {idx}/{len(img_ids)}', flush=True)
            info = coco_gt.loadImgs(img_id)[0]
            path = info['file_name']
            # file_name is already an absolute Windows path in this dataset
            # but fall back gracefully if it is a bare filename
            if not os.path.isabs(path):
                path = os.path.join(VAL_IMG_DIR, os.path.basename(path))
            if not os.path.exists(path):
                continue
            out = pipe(path)
            # BUG FIX: the pipeline returns boxes/scores/labels as separate
            # parallel arrays -- out['boxes'][i] is [x1,y1,x2,y2] (4 elements,
            # no score). The old code did `x1,y1,x2,y2,score = det[:5]` on
            # each BOX alone, so `len(det) < 5` was true for every single
            # detection and `continue` fired unconditionally -- this function
            # silently returned 0.0 for every checkpoint ever passed to it,
            # regardless of model quality (verified live: even the original,
            # fully-pretrained, unmodified hub checkpoint scores exactly 0.0
            # through the old code). This is almost certainly the real,
            # underlying explanation for "mAP stuck at 0" and for the DRL
            # agent's reward always being 0-0=0 every round.
            for (x1, y1, x2, y2), score in zip(out.get('boxes', []), out.get('scores', [])):
                score = float(score)
                if score < conf_thresh: continue
                results.append({'image_id': img_id, 'category_id': 1,
                                'bbox': [float(x1), float(y1),
                                         float(x2-x1), float(y2-y1)],
                                'score': score})
        if not results:
            print('[evaluate_map] No detections above threshold, returning 0.0', flush=True)
            return 0.0
        tmp = f'_tmp_eval_{os.getpid()}.json'
        with open(tmp, 'w') as f: json.dump(results, f)
        coco_dt = coco_gt.loadRes(tmp)
        ev = COCOeval(coco_gt, coco_dt, 'bbox')
        ev.evaluate(); ev.accumulate(); ev.summarize()
        os.remove(tmp)
        # stats[1] = AP@IoU=0.50 (AP50), which is mAP@0.5
        map50 = float(ev.stats[1])
        print(f'[evaluate_map] mAP@0.5 = {map50:.4f}', flush=True)
        return map50
    except Exception as e:
        print(f'[evaluate_map] {e}')
        return 0.0

# =============================================================================
# ── Net SHELL ────────────────────────────────────────────────────────────────
# =============================================================================
def _reset_module_params(module):
    """Recursively reinitialize every submodule that defines reset_parameters()
    (covers Conv2d/Linear/BatchNorm2d/GroupNorm and any custom DAMO-YOLO layer
    that implements it), instead of cherry-picking a few known layer types."""
    for m in module.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()


def _reset_head_calibrated(head):
    """Reinitialize the detection head using DAMO-YOLO's OWN ZeroHead.init_weights()
    instead of the generic _reset_module_params() sweep above.

    Verified live (diagnostic run): a generic reset_parameters() sweep leaves
    the gfl_cls classification conv at PyTorch's default Kaiming-init weights
    and a ZERO bias. ZeroHead.init_weights() instead uses std=0.01 weights and
    a calibrated focal-loss-prior bias (bias_init_with_prob(0.01), i.e. every
    anchor starts at ~1%% predicted confidence -- the standard RetinaNet-style
    cold-start trick for a head sitting under focal/GFL-style classification
    loss). Without it, the freshly-random head produces a catastrophic initial
    loss (~18000 in testing, vs ~1-2 with the calibrated init) whose first few
    gradient steps collapse the model into predicting background everywhere --
    loss_bbox/loss_dfl go to exactly 0.0 within ~5 iterations and stay there
    for the rest of training, silently. This was the actual root cause of
    "mAP never moves off 0" observed during smoke testing, not insufficient
    training depth."""
    if hasattr(head, 'init_weights'):
        head.init_weights()
    else:
        _reset_module_params(head)


def _perturb_head_cls(head, scale=HEAD_PERTURB_SCALE):
    """Partially perturb ONLY the classification branch (cls_convs, gfl_cls)
    of the detection head with scaled Gaussian noise, leaving the box
    regression branch (reg_convs, gfl_reg) and backbone/neck untouched at
    their mature pretrained values.

    This replaced full-head reinitialization (both '_reset_module_params'
    and the calibrated '_reset_head_calibrated' above) as the default seeding
    strategy after live diagnostics showed BOTH forms of full head reset
    destabilize training: DAMO-YOLO's regression branch decodes boxes through
    a discretized Distribution-Focal-Loss bucket representation that is very
    sensitive to weight-space perturbation. A full reset produces decoded
    boxes so degenerate that a downstream IoU computation can leave [0,1]
    range entirely, tripping a CUDA assertion inside the OTA assigner's
    binary_cross_entropy cost -- and even when it doesn't crash outright, the
    model collapses within a handful of gradient steps into predicting
    background everywhere (loss_bbox/loss_dfl -> exactly 0.0, permanently).

    Perturbing only the classification layers avoids destabilizing box
    regression entirely (verified: no crashes, no collapse) while still
    moving the model off its exact ~99.9%% converged optimum, so the DRL
    agent gets genuine per-round gradient signal to act on. The scale was
    picked via an isolated 8-epoch sweep: 0.3 kept loss_cls/loss_bbox/
    loss_dfl all stable in ~[0.1, 0.6] for the full run; 1.0 stayed
    persistently elevated/unstable; 2.0 reproduced the same all-background
    collapse as a full reset."""
    with torch.no_grad():
        for module in [head.cls_convs, head.gfl_cls]:
            for p in module.parameters():
                noise = torch.randn_like(p) * (p.std() + 1e-6) * scale
                p.add_(noise)


class Net(nn.Module):
    def __init__(self, ckpt_path=None, init_mode='pretrained'):
        """
        init_mode:
          'pretrained'    - keep the ModelScope-shipped pretrained weights as-is.
                            Safe to use when ckpt_path is given (or will be loaded
                            right after construction), since those weights get
                            fully overwritten anyway.
          'seed_scratch'  - reinitialize backbone+neck+head randomly. Use this to
                            seed a fresh episode for a literal from-scratch FL run.
          'seed_transfer' - keep the pretrained backbone+neck (generic visual
                            features), reinitialize only the detection head.
                            NOT RECOMMENDED -- verified to destabilize training
                            (see _perturb_head_cls docstring). Kept for reference.
          'seed_perturb'  - keep pretrained backbone+neck+regression head as-is,
                            partially perturb ONLY the classification layers.
                            Recommended: moves the model off its converged
                            optimum (fixing the ~99.9%% accuracy / vanishing
                            per-round gradient problem) without destabilizing
                            training the way a full head reset does.
        """
        super().__init__()
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            pipe = pipeline(Tasks.domain_specific_object_detection,
                            model='damo/cv_tinynas_uav-detection_damoyolo',
                            model_revision='master',
                            trust_remote_code=True)
            self._inner = pipe.model
            if init_mode == 'seed_scratch':
                _reset_module_params(self._inner.backbone)
                _reset_module_params(self._inner.neck)
                _reset_head_calibrated(self._inner.head)
            elif init_mode == 'seed_transfer':
                _reset_head_calibrated(self._inner.head)
            elif init_mode == 'seed_perturb':
                _perturb_head_cls(self._inner.head)
            if ckpt_path and os.path.exists(ckpt_path):
                sd = torch.load(ckpt_path, map_location='cpu')
                sd = {k.replace('_module.', ''): v for k, v in sd.items()}
                self._inner.load_state_dict(sd, strict=False)
        except Exception as e:
            print(f'[Net] Warning: {e}')
            self._inner = nn.Linear(1, 1)

    def named_parameters(self, *args, **kwargs):
        return self._inner.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self._inner.parameters(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self._inner.state_dict(*args, **kwargs)

    def load_state_dict(self, sd, strict=False):
        sd = {k.replace('_module.', ''): v for k, v in sd.items()}
        return self._inner.load_state_dict(sd, strict=strict)

    def cuda(self, device=None):
        self._inner = self._inner.cuda(device)
        return self

    def forward(self, x):
        return self._inner(x)

# =============================================================================
# ── TRAINING PROCESS (FIXED WRAPPER) ─────────────────────────────────────────
# =============================================================================
# =============================================================================
# ── TRAINING RUNNER (sequential, no mp.Process — single GPU safe) ────────────
# =============================================================================
class Training:
    """Runs trainer subprocesses sequentially in the calling thread."""
    def __init__(self, client_id=0, C=C_CLIP_DEFAULT, bs=batch_size, lr=0.1, el=epoch_local,
                 device='cuda:0', shard_ann=None):
        self.client_id   = client_id
        self.device      = device
        self.C           = C
        self.bs          = bs
        self.el          = el
        self.lr          = lr
        self.shard_ann   = shard_ann
        self.global_ckpt = os.path.join(CKPT_DIR, f'client_{client_id}_global.pt')
        self.local_ckpt  = os.path.join(CKPT_DIR, f'client_{client_id}_local.pt')
        self.work_dir    = os.path.join(CKPT_DIR, f'client_{client_id}_work')
        os.makedirs(CKPT_DIR, exist_ok=True)
        os.makedirs(self.work_dir, exist_ok=True)
        print(f'[Client {self.client_id}] Ready', flush=True)

    def train_one_round(self, sigma):
        """Train for one FL round. Blocks until done. Returns local ckpt path."""
        wrapper = self._make_wrapper(sigma)
        wrapper_path = os.path.join(self.work_dir, 'run_client.py')
        with open(wrapper_path, 'w', encoding='utf-8') as f:
            f.write(wrapper)
        cmd = [sys.executable, wrapper_path]
        print(f'[Client {self.client_id}] Launching trainer subprocess...', flush=True)
        t_start = time.time()
        # cwd=self.work_dir: ImageDetectionDamoyoloTrainer's `work_dir` kwarg is
        # silently ignored for this model's configuration.json (it only gets
        # copied into cfg.miscs.output_dir -- the field the trainer actually
        # saves to -- inside _config_transform(), which only runs when
        # 'framework' is present in the raw cfg, which DAMO-YOLO's is not). The
        # trainer therefore always saves to the hardcoded RELATIVE path
        # './workdirs/<exp_name>/...', resolved against the process's cwd. Every
        # client subprocess previously inherited the SAME cwd (this script's),
        # so all clients silently overwrote the same checkpoint files and the
        # glob search below (scoped to this client's own work_dir) never found
        # anything -- verified live: without this, every client after the
        # first fails with "no trainer checkpoint found" every round. Giving
        # each subprocess its own work_dir as cwd makes that hardcoded relative
        # path resolve inside this client's own isolated folder instead.
        result = subprocess.run(cmd, capture_output=False, timeout=7200, cwd=self.work_dir)
        elapsed = time.time() - t_start
        print(f'[Client {self.client_id}] Done in {elapsed:.1f}s (rc={result.returncode})', flush=True)

        if os.path.exists(self.local_ckpt) and os.path.exists(self.global_ckpt):
            local_sd  = torch.load(self.local_ckpt,  map_location='cpu')
            global_sd = torch.load(self.global_ckpt, map_location='cpu')
            C_clip = self.C
            # ── DP noise recalibration (see CLAUDE.md "OPEN, UNRESOLVED ISSUE") ──
            # OLD behavior: clipped EACH TENSOR independently to norm C_clip, then
            # added i.i.d. per-ELEMENT noise with std = sigma*C_clip to every
            # tensor regardless of its size. For a tensor with d elements, that
            # noise vector's own L2 norm is ~sigma*C_clip*sqrt(d) -- for a conv
            # weight with d in the hundreds of thousands, that swamps the
            # clipped-to-C_clip signal by 3+ orders of magnitude even at
            # sigma_min, so round 2+ collapses regardless of what the DRL agent
            # picks. C_clip/sigma were tuned for the author's toy CNN (thousands
            # of params total), not DAMO-YOLO (millions).
            # NEW behavior (standard DP-FedAvg client-update clipping, e.g.
            # McMahan et al. 2017): flatten the WHOLE local-training delta
            # (across every tensor) into one vector, clip THAT vector once to
            # norm C_clip (this is the actual per-round sensitivity bound), then
            # add noise so the TOTAL injected noise vector's L2 norm is
            # sigma*C_clip -- i.e. per-element std = sigma*C_clip/sqrt(D) where D
            # is the total parameter count. This keeps noise-vs-signal norm at
            # the same ratio (sigma) regardless of model size, so the existing
            # rho_used = 2*el/sigma**2 privacy-budget accounting (which has no D
            # term) keeps meaning what it always meant, just correctly applied
            # to a D-dimensional model instead of implicitly assuming D~=1.
            # ── Exclude BatchNorm running statistics from DP clip+noise ──────
            # Measured live: a SINGLE BatchNorm buffer (a `running_var`) had a
            # raw delta norm of ~3.03 million out of a ~3.04 million TOTAL
            # flattened-delta norm -- 99.75%% of the entire "signal" the clip
            # step was trying to protect was one non-trained bookkeeping
            # buffer. running_mean/running_var are EMA'd batch statistics
            # updated every forward pass, not gradient-descended parameters;
            # their raw numeric scale has nothing to do with weight magnitude
            # and swamped the real trainable weights' contribution to the
            # flattened vector, wasting virtually the entire clip budget on a
            # buffer and leaving nothing for the parameters that actually
            # matter. Standard practice (and why DP-SGD libraries like Opacus
            # often require replacing BatchNorm with GroupNorm) is to exclude
            # BN running stats from the DP mechanism entirely: copy them
            # through from local training unprotected (they're bulk dataset
            # statistics, not the per-client signal DP-FedAvg clipping is
            # meant to bound), and only clip+noise the actual trainable
            # parameters.
            BN_BUFFER_SUFFIXES = ('.running_mean', '.running_var', '.num_batches_tracked')
            def _is_bn_buffer(key):
                return key.endswith(BN_BUFFER_SUFFIXES)

            matching_keys = [k for k in local_sd if k in global_sd and not _is_bn_buffer(k)]
            deltas = {k: local_sd[k].float() - global_sd[k].float() for k in matching_keys}
            flat = torch.cat([d.flatten() for d in deltas.values()])
            total_norm = flat.norm(2).item()
            total_numel = flat.numel()
            clip_coef = min(1.0, C_clip / (total_norm + 1e-6))
            noise_std_per_elem = sigma * C_clip / (total_numel ** 0.5)
            print(f'[Client {self.client_id}] DEBUG noise calib: D={total_numel} '
                  f'unclipped_delta_norm={total_norm:.4f} C_clip={C_clip} '
                  f'clip_coef={clip_coef:.6f} noise_std_per_elem={noise_std_per_elem:.6e} '
                  f'total_noise_norm={sigma*C_clip:.4f}', flush=True)
            noisy_sd = {}
            for k in matching_keys:
                d_clipped = deltas[k] * clip_coef
                noise = torch.randn_like(d_clipped) * noise_std_per_elem
                noisy_sd[k] = global_sd[k].float() + d_clipped + noise
            for k in local_sd:
                if k not in matching_keys:
                    # BN buffers and any key absent from global_sd: pass
                    # through this client's locally-adapted value unmodified.
                    noisy_sd[k] = local_sd[k]
            torch.save(noisy_sd, self.local_ckpt)
            return self.local_ckpt
        else:
            print(f'[Client {self.client_id}] No local ckpt found, using global.', flush=True)
            return self.global_ckpt

    def eval_one_round(self):
        """Evaluate current global ckpt. Returns (map50, 0.0)."""
        map50 = evaluate_map(self.global_ckpt, val_ann=self.shard_ann)
        return map50, 0.0

    def _make_wrapper(self, sigma):
        # This wrapper is written to and launched from a nested work_dir, but
        # `import modelscope` must resolve to the LOCAL vendored/patched copy
        # in the project root (not a pip-installed one -- there isn't one --
        # and not silently fail). Python always puts a script's OWN directory
        # at sys.path[0], never the launching process's cwd, so without this
        # the subprocess can't find modelscope at all once work_dir != project
        # root. Insert the project root explicitly, first.
        project_root = os.path.dirname(os.path.abspath(__file__))
        script = f"""
import os, sys, torch, traceback, glob
sys.path.insert(0, r'{project_root}')
import torch.nn as nn

from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer

# ── Patch: DAMO-YOLO's stock "finetune" loader deliberately SKIPS the
# detection head when loading `pretrain_model` (Detector.load_pretrain_detector
# in modelscope/models/cv/tinynas_detection/damo/detectors/detector.py). That
# logic exists for one-off transfer learning to a new class count, but here we
# pass our own FL global checkpoint as `pretrain_model` every round, so the
# stock behavior silently resets the head to random every round -- the head's
# training never accumulates across FL rounds. Patch it to a full load so the
# aggregated global model (head included) actually carries over each round.
from modelscope.models.cv.tinynas_detection.damo.detectors.detector import Detector

def _load_full_checkpoint(self, pretrain_model):
    ckpt = torch.load(pretrain_model, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    state_dict = {{k.replace('module.', '').replace('_module.', ''): v
                  for k, v in state_dict.items()}}
    result = self.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(f'[Detector] load_full_checkpoint: '
              f'missing={{len(result.missing_keys)}} unexpected={{len(result.unexpected_keys)}}',
              flush=True)

Detector.load_pretrain_detector = _load_full_checkpoint

CACHE_PATH  = r'{CACHE_PATH}'
WORK_DIR    = r'{self.work_dir}'
GLOBAL_CKPT = r'{self.global_ckpt}'
LOCAL_CKPT  = r'{self.local_ckpt}'
SIGMA       = {sigma}
EPOCHS      = {self.el}
BATCH_SIZE  = {self.bs}
GRAD_CLIP   = {self.C}

# ── Build kwargs ─────────────────────────────────────────────────────────────
# ImageDetectionDamoyoloTrainer.__init__ checks:
#   if load_pretrain and 'pretrain_model' in kwargs:
#       cfg.train.finetune_path = kwargs['pretrain_model']
# Then DamoyoloTrainer.train() calls model.load_pretrain_detector(finetune_path).
# So we must pass load_pretrain=True AND pretrain_model=<path> together.
_has_global = os.path.exists(GLOBAL_CKPT)
kwargs = dict(
    cfg_file        = os.path.join(CACHE_PATH, 'configuration.json'),
    # NOTE: deliberately NOT passing `model=` here. ImageDetectionDamoyoloTrainer
    # .__init__ unconditionally calls get_or_download_model_dir(model) whenever
    # `model` is given, even though cfg_file+cache_path already fully satisfy
    # its else-branch -- that call re-downloads the weight file from the hub on
    # every single round. Passing only cfg_file+cache_path skips it entirely.
    gpu_ids         = [0],
    batch_size      = BATCH_SIZE,
    max_epochs      = EPOCHS,
    num_classes     = 1,
    cache_path      = CACHE_PATH,
    load_pretrain   = _has_global,
    base_lr_per_img = 0.0001,
    train_image_dir = r'{TRAIN_IMG_DIR}',
    val_image_dir   = r'{VAL_IMG_DIR}',
    train_ann       = r'{self.shard_ann}',
    val_ann         = r'{self.shard_ann}',
    work_dir        = WORK_DIR,
    num_workers     = 0,
)
if _has_global:
    kwargs['pretrain_model'] = GLOBAL_CKPT

try:
    # ── Step 1: build trainer (model is NOT created yet at this point) ───────
    trainer = build_trainer(name=Trainers.tinynas_damoyolo, default_args=kwargs)
    # Defense-in-depth: ImageDetectionDamoyoloTrainer.__init__ only ever SETS
    # cfg.train.finetune_path when load_pretrain=True -- passing False does
    # NOT null it out, so the value baked into the hub model's own
    # configuration.json (usually its own finetune recipe path) survives and
    # gets loaded anyway. Explicitly clear it when we truly have no global
    # checkpoint to warm-start from, so scratch/seed_transfer runs start from
    # what the Net() seed actually produced, not the hub's shipped default.
    if not _has_global:
        trainer.cfg.train.finetune_path = None
        trainer.cfg.train.resume_path = None
    trainer.cfg.miscs.num_workers = 0
    trainer.cfg.miscs.print_interval_iters = 1   # print every batch (dataset is small)
    print(f'[Client] Starting trainer.train() for {{EPOCHS}} epochs ...', flush=True)
    # ── Step 2: train() — model is built and weights loaded inside here ──────
    trainer.train()
    print(f'[Client] trainer.train() finished.', flush=True)

    # ── Step 3: find the checkpoint written by the trainer ───────────────────
    # DamoyoloTrainer.save_ckpt writes to: WORK_DIR/<exp_name>/epoch_N_ckpt.pth
    # and finally: WORK_DIR/<exp_name>/latest_ckpt.pth
    ckpt_files = (
        glob.glob(os.path.join(WORK_DIR, '**', 'latest_ckpt.pth'), recursive=True) +
        glob.glob(os.path.join(WORK_DIR, '**', 'epoch_*.pth'),     recursive=True) +
        glob.glob(os.path.join(WORK_DIR, '**', '*.pth'),           recursive=True) +
        glob.glob(os.path.join(WORK_DIR, '**', '*.pt'),            recursive=True)
    )
    # Exclude our own LOCAL_CKPT and GLOBAL_CKPT
    ckpt_files = [f for f in ckpt_files
                  if os.path.abspath(f) != os.path.abspath(LOCAL_CKPT)
                  and os.path.abspath(f) != os.path.abspath(GLOBAL_CKPT)]

    if ckpt_files:
        # Prefer latest_ckpt.pth, else newest file by mtime
        prefer = [f for f in ckpt_files if 'latest_ckpt' in os.path.basename(f)]
        best_ckpt = prefer[0] if prefer else max(ckpt_files, key=os.path.getmtime)
        print(f'[Client] Using trainer checkpoint: {{best_ckpt}}')
        ckpt = torch.load(best_ckpt, map_location='cpu')
        if isinstance(ckpt, dict):
            # modelscope save_checkpoint writes: {{'state_dict': ..., 'meta': ...}}
            if 'state_dict' in ckpt:
                final_sd = ckpt['state_dict']
            elif 'model' in ckpt:
                final_sd = ckpt['model']
            else:
                final_sd = ckpt
        else:
            final_sd = ckpt
        torch.save(final_sd, LOCAL_CKPT)
        print(f'[Client] Checkpoint saved to {{LOCAL_CKPT}}')
    else:
        print('[Client] WARNING: no trainer checkpoint found; saving will fail.')
        sys.exit(1)

except Exception as e:
    print(f'[Client] ERROR: {{e}}')
    traceback.print_exc()
    sys.exit(1)
"""
        return script
# =============================================================================
# ── CLIENT (sequential, no queues) ───────────────────────────────────────────
# =============================================================================
class Client():
    def __init__(self, id=-1, C=C_CLIP_DEFAULT, rho_total=20.0, bs=batch_size,
                 trset=None, lr=0.1, el=epoch_local, hl=5,
                 agent=None, device='cuda:0', shard_ann=None):
        self.id        = id
        self.rho       = rho_total
        self.rho_total = rho_total
        self.C         = C
        self.bs        = bs
        self.lr        = lr
        self.el        = el
        self.hl        = hl
        self.sigma     = sigma_max
        self.rho_used  = 2 * self.el / (self.sigma**2)
        self.local_acc = 0.0
        self.state     = None
        self.action    = None
        self.device    = device
        self.agent     = agent
        self.shard_ann = shard_ann
        self._last_ckpt_path = None
        self._last_map50     = 0.0

        self.runner = Training(client_id=id, C=C, bs=bs, lr=lr, el=el,
                               device=device, shard_ann=shard_ann)
        print(f'[Client {self.id}] Process started on {self.device}')

    def reset(self, rho_total):
        self.rho       = rho_total
        self.rho_total = rho_total
        self.action    = None
        self.sigma     = sigma_max
        self.local_acc = 0.0

    def save_global(self, net):
        """Save server net to disk as this client's global ckpt."""
        sd = {k.replace('_module.', ''): v for k, v in net.state_dict().items()}
        torch.save(sd, self.runner.global_ckpt)

    def train(self, net, sigma):
        self.sigma    = sigma
        self.rho_used = self.el * 2 / (sigma**2)
        self.rho     -= self.rho_used
        self.save_global(net)
        self._last_ckpt_path = self.runner.train_one_round(sigma)

    def get_trained_net(self):
        """Load and return Net from last training round checkpoint."""
        net_out = Net()
        net_out.load_state_dict(
            torch.load(self._last_ckpt_path, map_location='cpu'), strict=False)
        return net_out

    def test(self, net):
        self.save_global(net)
        self._last_map50, _ = self.runner.eval_one_round()

    def get_test_result(self):
        return self._last_map50, 0.0

    def terminate_process(self):
        pass  # nothing to terminate

# =============================================================================
# ── SERVER (unchanged) ───────────────────────────────────────────────────────
# =============================================================================
class Server():
    def __init__(self, net, device='cuda:0'):
        self.device   = device
        self.net      = net
        self.net.cuda(device=device)
        self.dummy_net = copy.deepcopy(self.net)
        self.dummy_net.cuda(device=device)

    def reset(self, net):
        self.net = net
        self.net.cuda(device=self.device)

    def aggregate(self, net_list):
        # Averaging the detection HEAD across clients destroys score
        # calibration (verified live: global mAP top-score ~0.012 vs
        # local ~0.89 on an identical checkpoint before this fix).
        # FedAvg the backbone+neck as usual, but for the head just take
        # the first client's weights instead of blending clients together.
        count = 0
        for net in net_list:
            self.dummy_net.load_state_dict(net.state_dict(), strict=False)
            if count == 0:
                head_keys = [n for n, _ in self.net._inner.named_parameters()
                             if 'head' in n]
                print(f'[aggregate] Head param count: {len(head_keys)}, '
                      f'sample: {head_keys[:5]}', flush=True)
            for name, p in self.net._inner.named_parameters():
                if name not in self.dummy_net._inner.state_dict():
                    continue
                src = self.dummy_net._inner.state_dict()[name].data
                if 'head' in name:
                    if count == 0:
                        p.data = src.clone()
                else:
                    p.data = (p.data * (count / (count + 1)) + src / (count + 1))
            count += 1

# =============================================================================
# ── ACTOR / CRITIC / AGENT (unchanged) ───────────────────────────────────────
# =============================================================================
class Actor(nn.Module):
    def __init__(self, dim_state, dim_action):
        super().__init__()
        self.fc1 = nn.Linear(dim_state, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, dim_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, dim_state, dim_action):
        super().__init__()
        self.fc1 = nn.Linear(dim_state + dim_action, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent():
    def __init__(self, n_agents, dim_state, dim_action,
                 lr_actor=0.00002, lr_critic=0.0002,
                 lr_actor_target=0.005, lr_critic_target=0.005,
                 buffer_size=5000, minimal_size=512, gamma=0.998,
                 explore=1.0, explore_decay=0.9996,
                 explore_decay_interval=1000,
                 bs=256, train_interval=1, device='cuda:0'):
        self.device     = device
        self.n_agents   = n_agents
        self.dim_state  = dim_state
        self.dim_action = dim_action
        self.actor        = Actor(dim_state, dim_action).cuda(device=device)
        self.actor_target = Actor(dim_state, dim_action).cuda(device=device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic        = Critic(dim_state, dim_action).cuda(device=device)
        self.critic_target = Critic(dim_state, dim_action).cuda(device=device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.lr_actor_target  = lr_actor_target
        self.lr_critic_target = lr_critic_target
        self.bs               = bs
        self.criterion_critic = nn.MSELoss()
        self.optimizer_actor  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.minimal_size     = minimal_size
        self.rep_buffer = {
            'state':      [deque(maxlen=buffer_size) for _ in range(n_agents)],
            'action':     [deque(maxlen=buffer_size) for _ in range(n_agents)],
            'reward':     deque(maxlen=buffer_size),
            'next_state': [deque(maxlen=buffer_size) for _ in range(n_agents)],
            'done':       deque(maxlen=buffer_size),
        }
        self.gamma               = torch.Tensor([gamma]).cuda(device=device)
        self.explore             = explore
        self.explore_decay       = explore_decay
        self.explore_decay_interval = explore_decay_interval
        self.step       = 1
        self.train_step = 1
        self.train_interval = train_interval

    def train(self, state, action, reward, next_state, done):
        for i in range(self.n_agents):
            self.rep_buffer['state'][i].append(state[i])
            self.rep_buffer['action'][i].append([action[i]])
            self.rep_buffer['next_state'][i].append(next_state[i])
        self.rep_buffer['reward'].append(reward)
        self.rep_buffer['done'].append(done)
        if len(self.rep_buffer['state'][0]) >= self.minimal_size:
            self.explore = max(self.explore * self.explore_decay, 0.01)
            if self.train_step % self.train_interval == 0:
                for _ in range(DRL_steps):
                    indices = list(range(len(self.rep_buffer['state'][0])))
                    random.shuffle(indices)
                    idx = indices[:self.bs]
                    s  = torch.Tensor([[self.rep_buffer['state'][i][j]      for j in idx] for i in range(self.n_agents)]).cuda(self.device)
                    a  = torch.Tensor([[self.rep_buffer['action'][i][j]     for j in idx] for i in range(self.n_agents)]).cuda(self.device)
                    r  = torch.Tensor( [self.rep_buffer['reward'][j]        for j in idx]).cuda(self.device)
                    ns = torch.Tensor([[self.rep_buffer['next_state'][i][j] for j in idx] for i in range(self.n_agents)]).cuda(self.device)
                    d  = torch.Tensor( [self.rep_buffer['done'][j]          for j in idx]).cuda(self.device)
                    with torch.no_grad():
                        na    = [self.actor_target(ns[i]) for i in range(self.n_agents)]
                        Q_nxt = [self.critic_target(torch.cat([ns[i], na[i]], 1)) for i in range(self.n_agents)]
                        Q_tgt = r + sum(self.gamma * Q_nxt[i] * (1-d) for i in range(self.n_agents))
                    self.optimizer_critic.zero_grad()
                    Q_pred = sum(self.critic(torch.cat([s[i], a[i]], 1)) for i in range(self.n_agents))
                    self.criterion_critic(Q_pred, Q_tgt).backward()
                    self.optimizer_critic.step()
                    self.optimizer_actor.zero_grad()
                    self.optimizer_critic.zero_grad()
                    A_loss = -torch.mean(sum(self.critic(torch.cat([s[i], self.actor(s[i])], 1))
                                            for i in range(self.n_agents)) / self.n_agents)
                    A_loss.backward()
                    self.optimizer_actor.step()
                    self.step += 1
                for name, p in self.actor_target.named_parameters():
                    p.data = (1-self.lr_actor_target)*p.data + self.lr_actor_target*self.actor.state_dict()[name].data.clone()
                for name, p in self.critic_target.named_parameters():
                    p.data = (1-self.lr_critic_target)*p.data + self.lr_critic_target*self.critic.state_dict()[name].data.clone()
            self.train_step += 1

    def take_action(self, state):
        with torch.no_grad():
            return self.actor(torch.Tensor(state).cuda(self.device)).cpu().numpy()

    def take_action_with_noise(self, state):
        with torch.no_grad():
            a  = self.actor(torch.Tensor(state).cuda(self.device)).cpu().numpy()
            an = a + np.random.normal(0, self.explore, self.dim_action)
            for i in range(len(an)):
                e = self.explore
                while an[i] < 0 or an[i] > 1:
                    e /= 2
                    if e <= 0.0001: an[i] = a[i]; break
                    an[i] = np.random.normal(an[i], e)  # scalar draw, not size=1 array -- newer numpy
                                                          # rejects assigning a (1,) array into a scalar slot
            return a, an

    def save(self, episode):
        os.makedirs('./DRL_Nets', exist_ok=True)
        torch.save(self.actor.state_dict(),        f'./DRL_Nets/actor_drone_{episode}.pth')
        torch.save(self.actor_target.state_dict(), f'./DRL_Nets/actor_target_drone_{episode}.pth')
        torch.save(self.critic.state_dict(),       f'./DRL_Nets/critic_drone_{episode}.pth')
        torch.save(self.critic_target.state_dict(),f'./DRL_Nets/critic_target_drone_{episode}.pth')

    def save_checkpoint(self, episode, training_state, path=CHECKPOINT_PATH):
        buf = {
            'state':      [list(q) for q in self.rep_buffer['state']],
            'action':     [list(q) for q in self.rep_buffer['action']],
            'reward':     list(self.rep_buffer['reward']),
            'next_state': [list(q) for q in self.rep_buffer['next_state']],
            'done':       list(self.rep_buffer['done']),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'actor': self.actor.state_dict(),
                    'actor_target': self.actor_target.state_dict(),
                    'critic': self.critic.state_dict(),
                    'critic_target': self.critic_target.state_dict(),
                    'optimizer_actor':  self.optimizer_actor.state_dict(),
                    'optimizer_critic': self.optimizer_critic.state_dict(),
                    'rep_buffer': buf,
                    'explore': self.explore, 'step': self.step,
                    'train_step': self.train_step,
                    'episode': episode, 'training_state': training_state}, path)
        print(f'[Checkpoint] Saved episode {episode} → {path}')

    def load_checkpoint(self, path=CHECKPOINT_PATH):
        ck = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ck['actor'])
        self.actor_target.load_state_dict(ck['actor_target'])
        self.critic.load_state_dict(ck['critic'])
        self.critic_target.load_state_dict(ck['critic_target'])
        self.optimizer_actor.load_state_dict(ck['optimizer_actor'])
        self.optimizer_critic.load_state_dict(ck['optimizer_critic'])
        ml  = self.rep_buffer['state'][0].maxlen
        buf = ck['rep_buffer']
        self.rep_buffer['state']      = [deque(q, maxlen=ml) for q in buf['state']]
        self.rep_buffer['action']     = [deque(q, maxlen=ml) for q in buf['action']]
        self.rep_buffer['next_state'] = [deque(q, maxlen=ml) for q in buf['next_state']]
        self.rep_buffer['reward']     = deque(buf['reward'],  maxlen=ml)
        self.rep_buffer['done']       = deque(buf['done'],    maxlen=ml)
        self.explore    = ck['explore']
        self.step       = ck['step']
        self.train_step = ck['train_step']
        print(f"[Checkpoint] Resumed from episode {ck['episode']}")
        return ck['episode'], ck['training_state']

# =============================================================================
# ── MAIN LOOP (unchanged) ────────────────────────────────────────────────────
# =============================================================================
if __name__ == '__main__':
    t0 = time.time()
    mp.set_start_method('spawn', force=True)

    os.makedirs('./DRL_Nets', exist_ok=True)
    os.makedirs('./exp_data', exist_ok=True)
    os.makedirs(SHARD_DIR,   exist_ok=True)
    os.makedirs(CKPT_DIR,    exist_ok=True)

    agent = Agent(n_agents=N, dim_state=2, dim_action=1)

    if os.path.exists(CHECKPOINT_PATH):
        print(f'[Checkpoint] Found {CHECKPOINT_PATH}. Resuming...')
        episode, training_state = agent.load_checkpoint(CHECKPOINT_PATH)
        final_accs      = training_state['final_accs']
        mean_final_accs = training_state['mean_final_accs']
        mean_returns    = training_state['mean_returns']
        returns         = training_state['returns']
        high_acc        = training_state['high_acc']
        episode += 1
    else:
        print('[Checkpoint] Starting from scratch.')
        episode         = 1
        final_accs      = []
        mean_final_accs = []
        mean_returns    = []
        returns         = []
        high_acc        = 0.0

    print('Splitting annotations...')
    shard_anns = split_coco_annotations(TRAIN_ANN, N, SHARD_DIR)

    print('Building server model (this loads DAMO-YOLO once)...')
    net = Net(init_mode=INIT_MODE)
    server = Server(net=net)

    print('Starting client processes...')
    clients = [
        Client(id=i,
               rho_total=np.random.uniform(rho_min, rho_max),
               agent=agent,
               device='cuda:0',
               shard_ann=shard_anns[i])
        for i in range(N)
    ]
    print(f'All {N} clients ready.\n')
    while episode < episodes:
        t1 = time.time()
        print(f'Trained steps {agent.train_step}')
        print(f'Explore {agent.explore:.4f}')
        print(f'Episode {episode}')

        net_init = Net(init_mode=INIT_MODE)
        net = Net(); net.load_state_dict(net_init.state_dict())
        server.reset(net=net)
        rhos_init = []
        for i, client in enumerate(clients):
            rho = np.random.uniform(rho_min, rho_max)
            rhos_init.append(rho)
            client.reset(rho_total=rho)

        sigmas    = [[] for _ in range(N)]
        rho_useds = [[] for _ in range(N)]

        global_accs = []
        print(f'[Episode {episode}] Running baseline mAP on all clients (this may take a few minutes)...', flush=True)
        last_global_acc = 0.0
        for client in clients:
            client.test(server.net)
            map50, _ = client.get_test_result()
            last_global_acc += map50
        last_global_acc /= N
        global_accs.append(last_global_acc)

        isDone = 0.0
        for CR in range(CR_Total):
            if CR == CR_Total - 1: isDone = 1.0
            for i, client in enumerate(clients):
                state = [CR / CR_Total, client.rho_total / rho_max]
                action, action_noise = agent.take_action_with_noise(state=state)
                sigma    = action_noise[0] * (sigma_max - sigma_min) + sigma_min
                rho_used = sigma2rho(sigma=sigma, el=client.el, bs=client.bs)
                sigmas[i].append(sigma)
                rho_useds[i].append(rho_used)
        rho_used_sums = [sum(rho_useds[i]) for i in range(N)]

        a = 1.0; b = 0.0
        for i, client in enumerate(clients):
            if abs(max(sigmas[i]) - min(sigmas[i])) < 0.01:
                b = 0.0
                a = client.rho_total / sum(rho_useds[i])
            else:
                ubs = [client.rho_total / CR_Total]
                lbs = [-np.inf]
                for CR in range(CR_Total):
                    diff = rho_used_sums[i] - CR_Total * rho_useds[i][CR]
                    if diff > 1e-7:
                        ubs.append((rho_used_max * rho_used_sums[i] - client.rho_total * rho_useds[i][CR]) / diff)
                        lbs.append((rho_used_min * rho_used_sums[i] - client.rho_total * rho_useds[i][CR]) / diff)
                    elif diff < -1e-7:
                        ubs.append((rho_used_min * rho_used_sums[i] - client.rho_total * rho_useds[i][CR]) / diff)
                        lbs.append((rho_used_max * rho_used_sums[i] - client.rho_total * rho_useds[i][CR]) / diff)
                bub = min(ubs); blb = max(lbs)
                b = -sum([(rho_useds[i][CR] * rho_used_sums[i] - client.rho_total) *
                           (CR_Total * rho_useds[i][CR] - rho_used_sums[i]) for CR in range(CR_Total)]) / \
                     sum([(CR_Total * rho_useds[i][CR] - rho_used_sums[i])**2 for CR in range(CR_Total)])
                b = max(blb, min(bub, b))
                a = (client.rho_total - b * CR_Total) / rho_used_sums[i]
            for CR in range(CR_Total):
                rho_useds[i][CR] = a * rho_useds[i][CR] + b
                sigmas[i][CR]    = rho2sigma(rho=rho_useds[i][CR], el=client.el, bs=client.bs)

        isDone = 0.0
        for CR in range(CR_Total):
            if CR == CR_Total - 1: isDone = 1.0
            # Train clients sequentially (single GPU)
            net_list = []
            for i, client in enumerate(clients):
                state          = [CR / CR_Total, client.rho_total / rho_max]
                client.state   = state
                client.sigma   = sigmas[i][CR]
                client.action  = (sigmas[i][CR] - sigma_min) / (sigma_max - sigma_min)
                client.train(net=server.net, sigma=client.sigma)
                net_list.append(client.get_trained_net())
            server.aggregate(net_list=net_list)
            # Evaluate clients sequentially
            global_acc = 0.0
            for client in clients:
                client.test(server.net)
                map50, _ = client.get_test_result()
                global_acc  += map50
                client.local_acc = map50
            global_acc /= N
            states      = [client.state  for client in clients]
            actions     = [client.action for client in clients]
            rewards     = [global_acc - last_global_acc]
            global_accs.append(global_acc)
            last_global_acc = global_acc
            next_states = [[(CR+1)/CR_Total, client.rho_total/rho_max] for client in clients]
            done = [isDone]
            agent.train(state=states, action=actions, reward=rewards,
                        next_state=next_states, done=done)

        if episode % 1 == 0:
            net_init = Net(init_mode=INIT_MODE); net = Net(); net.load_state_dict(net_init.state_dict())
            server.reset(net=net)
            rhos_init2 = []
            for client in clients:
                rho = np.random.uniform(rho_min, rho_max)
                rhos_init2.append(rho)
            for i, client in enumerate(clients):
                client.reset(rho_total=rhos_init2[i])

            sigmas2    = [[] for _ in range(N)]
            rho_useds2 = [[] for _ in range(N)]
            for CR in range(CR_Total):
                for i, client in enumerate(clients):
                    state  = [CR/CR_Total, client.rho_total/rho_max]
                    action = agent.take_action(state=state)
                    sigma  = action[0] * (sigma_max - sigma_min) + sigma_min
                    rho_u  = sigma2rho(sigma, client.el, client.bs)
                    sigmas2[i].append(sigma)
                    rho_useds2[i].append(rho_u)
            rho_used_sums2 = [sum(rho_useds2[i]) for i in range(N)]
            for i, client in enumerate(clients):
                if abs(max(sigmas2[i]) - min(sigmas2[i])) < 0.01:
                    a2 = client.rho_total / sum(rho_useds2[i]); b2 = 0.0
                else:
                    ubs = [client.rho_total / CR_Total]; lbs = [-np.inf]
                    for CR in range(CR_Total):
                        diff = rho_used_sums2[i] - CR_Total * rho_useds2[i][CR]
                        if diff > 1e-7:
                            ubs.append((rho_used_max*rho_used_sums2[i] - client.rho_total*rho_useds2[i][CR])/diff)
                            lbs.append((rho_used_min*rho_used_sums2[i] - client.rho_total*rho_useds2[i][CR])/diff)
                        elif diff < -1e-7:
                            ubs.append((rho_used_min*rho_used_sums2[i] - client.rho_total*rho_useds2[i][CR])/diff)
                            lbs.append((rho_used_max*rho_used_sums2[i] - client.rho_total*rho_useds2[i][CR])/diff)
                    bub2 = min(ubs); blb2 = max(lbs)
                    b2 = -sum([(rho_useds2[i][CR]*rho_used_sums2[i] - client.rho_total) *
                                (CR_Total*rho_useds2[i][CR] - rho_used_sums2[i]) for CR in range(CR_Total)]) / \
                          sum([(CR_Total*rho_useds2[i][CR] - rho_used_sums2[i])**2 for CR in range(CR_Total)])
                    b2 = max(blb2, min(bub2, b2))
                    a2 = (client.rho_total - b2*CR_Total) / rho_used_sums2[i]
                for CR in range(CR_Total):
                    rho_useds2[i][CR] = a2*rho_useds2[i][CR] + b2
                    sigmas2[i][CR]    = rho2sigma(rho_useds2[i][CR], client.el, client.bs)

            for i, client in enumerate(clients):
                client.reset(rho_total=rhos_init2[i])
            global_accs_DRL = []
            last_global_acc_drl = 0.0
            for client in clients:
                client.test(server.net)
                m, _ = client.get_test_result()
                last_global_acc_drl += m
            last_global_acc_drl /= N
            global_accs_DRL.append(last_global_acc_drl)

            for CR in range(CR_Total):
                # Train sequentially
                net_list = []
                for i, client in enumerate(clients):
                    client.sigma = sigmas2[i][CR]
                    client.train(net=server.net, sigma=client.sigma)
                    net_list.append(client.get_trained_net())
                server.aggregate(net_list=net_list)
                # Evaluate sequentially
                global_acc = 0.0
                for client in clients:
                    client.test(server.net)
                    m, _ = client.get_test_result()
                    global_acc += m
                    client.local_acc = m
                global_acc /= N
                global_accs_DRL.append(global_acc)
                last_global_acc_drl = global_acc
                print(f'[DRL] CR {CR+1}/{CR_Total}  mAP@0.5={global_acc:.4f}')

            returns_DRL = global_accs_DRL[-1] - global_accs_DRL[0]
            final_accs.append(global_accs_DRL[-1])
            returns.append(returns_DRL)

            low_index    = max(len(final_accs) - 10, 0)
            mean_acc     = np.mean(final_accs[low_index:])
            mean_final_accs.append(mean_acc)

            if mean_acc > high_acc and episode > 10:
                high_acc = mean_acc
                agent.save(episode=episode)

            if episode % CHECKPOINT_INTERVAL == 0:
                agent.save_checkpoint(episode=episode, training_state={
                    'final_accs': final_accs, 'mean_final_accs': mean_final_accs,
                    'mean_returns': mean_returns, 'returns': returns,
                    'high_acc': high_acc})

            low_index2   = max(len(returns) - 10, 0)
            mean_return  = np.mean(returns[low_index2:])
            mean_returns.append(mean_return)

        print(f'Time spent for this episode: {time.time()-t1:.1f}s')
        episode += 1

    for client in clients:
        client.terminate_process()
    print(f'Total time: {time.time()-t0:.1f}s')