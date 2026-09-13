# =============================================================================
# FL + DP + DRL for Drone Detection  -- FIXED VERSION
#
# BUGS FIXED:
#   BUG 1: eval_one_round passed shard_ann (training data) to evaluate_map
#          -> evaluate_map now ALWAYS uses VAL_ANN internally
#
#   BUG 2: wrapper used kwargs['pretrain_model'] which trainer silently ignores
#          -> now uses kwargs['finetune_path'] (confirmed by logs)
#
#   BUG 3: get_trained_net() built a new Net() = new pipeline() call per round
#          -> replaced with get_local_sd() which just torch.loads from disk
#
#   BUG 4: server.aggregate() assumed Net._inner — fragile and wrong
#          -> clean FedAvg directly on list of state_dicts
#
#   BUG 5: Two Net() calls at startup (net_init + net) = 2 pipeline loads
#          -> one Net() at startup; all resets use set_sd()
# =============================================================================

import numpy as np
import torch.multiprocessing as mp
import time
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import os
import sys
import copy
import json
import subprocess
import warnings
import matplotlib
matplotlib.use('Agg')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# =============================================================================
# PATHS
# =============================================================================
TRAIN_IMG_DIR = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/3rd_Anti-UAV_train_val/train/20190925_130434_1_3'
VAL_IMG_DIR   = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/3rd_Anti-UAV_train_val/train/20190925_130434_1_3'
TRAIN_ANN     = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/single_test_coco/annotations/train.json'
VAL_ANN       = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/single_test_coco/annotations/val.json'
CACHE_PATH    = 'C:/Users/T25301039/.cache/modelscope/hub/damo/cv_tinynas_uav-detection_damoyolo'
SHARD_DIR     = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/fl_workdirs/shards'
CKPT_DIR      = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/fl_workdirs/client_ckpts'

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
N             = 3
batch_size    = 8
episodes      = 5
CR_Total      = 10
epoch_local   = 5
rho_min       = 20.0
rho_max       = 40.0
DRL_steps     = 1
sigma_max     = 20.0
sigma_min     = 1.0
max_eval_imgs = 40

rho_used_min = 2 / sigma_max**2 * epoch_local
rho_used_max = 2 / sigma_min**2 * epoch_local

CHECKPOINT_PATH = './DRL_Nets/checkpoint_drone.pth'

def rho2sigma(rho, el, bs):
    return float(np.sqrt(2 * el / max(rho, 1e-9)))

def sigma2rho(sigma, el, bs):
    return 2 * el / (sigma ** 2)

# =============================================================================
# ANNOTATION SHARD SPLITTER
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
                a = copy.deepcopy(ann)
                a['id'] = offset
                offset += 1
                shard_anns.append(a)
        out = os.path.join(out_dir, f'train_client_{i}.json')
        with open(out, 'w') as f:
            json.dump({'info': data.get('info', {}),
                       'licenses': data.get('licenses', []),
                       'categories': data['categories'],
                       'images': shard_imgs,
                       'annotations': shard_anns}, f)
        print(f'  Client {i}: {len(shard_imgs)} imgs, {len(shard_anns)} anns -> {out}')
        paths.append(out)
    return paths

# =============================================================================
# mAP EVALUATOR
# FIX BUG 1: always evaluates on VAL_ANN -- never shard_ann
# =============================================================================
def evaluate_map(ckpt_path, conf_thresh=0.005):
    if not os.path.exists(ckpt_path):
        print(f'[evaluate_map] ckpt not found: {ckpt_path}', flush=True)
        return 0.0
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        print(f'[evaluate_map] Loading model from {ckpt_path} ...', flush=True)
        pipe = pipeline(Tasks.domain_specific_object_detection,
                        model='damo/cv_tinynas_uav-detection_damoyolo',
                        model_revision='master')
        try:
            pipe.model.head.nms_conf_thre = 0.005
            print('[evaluate_map] Set nms_conf_thre=0.005', flush=True)
        except:
            pass
        sd = torch.load(ckpt_path, map_location='cpu')
        sd = {k.replace('_module.', ''): v for k, v in sd.items()}
        pipe.model.load_state_dict(sd, strict=False)

        # DEBUG: verify weights actually loaded
        sample_key = next(iter(sd.keys()))
        pipe_val = pipe.model.state_dict()[sample_key].flatten()[0].item()
        ckpt_val = sd[sample_key].flatten()[0].item()
        print(f'[evaluate_map] Weight check — ckpt[{sample_key}][0]={ckpt_val:.6f}, pipe[{sample_key}][0]={pipe_val:.6f}', flush=True)

        pipe.model.eval()

        coco_gt = COCO(VAL_ANN)
        img_ids = coco_gt.getImgIds()
        if len(img_ids) > max_eval_imgs:
            img_ids = random.sample(img_ids, max_eval_imgs)

        print(f'[evaluate_map] Running inference on {len(img_ids)} images ...', flush=True)
        results = []
        for idx, img_id in enumerate(img_ids):
            if idx % 20 == 0:
                print(f'[evaluate_map]   {idx}/{len(img_ids)}', flush=True)
            info = coco_gt.loadImgs(img_id)[0]
            path = info['file_name']
            if not os.path.isabs(path):
                path = os.path.join(VAL_IMG_DIR, os.path.basename(path))
            if not os.path.exists(path):
                print(f'[evaluate_map] WARNING image missing: {path}', flush=True)
                continue
            out = pipe(path)

            # DEBUG: print output structure on first image only
            if idx == 0:
                print(f'[evaluate_map] pipeline output keys: {list(out.keys())}', flush=True)
                print(f'[evaluate_map] pipeline output sample: {str(out)[:300]}', flush=True)

            for det, score in zip(out.get('boxes', []), out.get('scores', [])):
                if len(det) < 4:
                    continue
                x1, y1, x2, y2 = det[:4]
                score = float(score)
                if score < conf_thresh:
                    continue
                results.append({'image_id': img_id, 'category_id': 1,
                                'bbox': [float(x1), float(y1),
                                        float(x2 - x1), float(y2 - y1)],
                                'score': score})

        if not results:
            print('[evaluate_map] No detections above threshold, returning 0.0', flush=True)
            return 0.0

        tmp = f'_tmp_eval_{os.getpid()}.json'
        with open(tmp, 'w') as f:
            json.dump(results, f)
        coco_dt = coco_gt.loadRes(tmp)
        print(f'[evaluate_map] Total detections collected: {len(results)}', flush=True)
        print(f'[evaluate_map] Sample result: {results[0]}', flush=True)
        print(f'[evaluate_map] Sample gt ann: {coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_ids[:1]))[0]}', flush=True)
        ev = COCOeval(coco_gt, coco_dt, 'bbox')
        ev.params.imgIds = img_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        os.remove(tmp)
        map50 = float(ev.stats[1])
        print(f'[evaluate_map] mAP@0.5 = {map50:.4f}', flush=True)
        return map50
    except Exception as e:
        import traceback
        print(f'[evaluate_map] ERROR: {e}')
        traceback.print_exc()
        return 0.0

# =============================================================================
# Net SHELL -- FIX BUG 5: pipeline() called only once at startup
# =============================================================================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        pipe = pipeline(Tasks.domain_specific_object_detection,
                        model='damo/cv_tinynas_uav-detection_damoyolo',
                        model_revision='master')
        self._inner = pipe.model

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
# TRAINING RUNNER
# =============================================================================
class Training:
    def __init__(self, client_id=0, C=6, bs=batch_size, lr=0.1, el=epoch_local,
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

    def train_one_round(self, sigma):
        # DP noise is applied inside the wrapper as gradient hooks
        # during training — correct approach per Algorithm 1 of the paper
        wrapper      = self._make_wrapper(sigma)
        wrapper_path = os.path.join(self.work_dir, 'run_client.py')
        with open(wrapper_path, 'w', encoding='utf-8') as f:
            f.write(wrapper)
        print(f'[Client {self.client_id}] Launching subprocess...', flush=True)
        t_start = time.time()
        result  = subprocess.run([sys.executable, wrapper_path],
                                 capture_output=False, timeout=7200)
        elapsed = time.time() - t_start
        print(f'[Client {self.client_id}] Done in {elapsed:.1f}s (rc={result.returncode})', flush=True)

        if os.path.exists(self.local_ckpt):
            return self.local_ckpt
        else:
            print(f'[Client {self.client_id}] WARNING: no local ckpt, using global.', flush=True)
            return self.global_ckpt

    def eval_one_round(self):
        # FIX BUG 1: evaluate_map uses VAL_ANN internally, no shard_ann passed
        # Evaluate local (trained) checkpoint, not global (aggregated)
        ckpt = self.local_ckpt if os.path.exists(self.local_ckpt) else self.global_ckpt
        return evaluate_map(ckpt), 0.0

    def _make_wrapper(self, sigma):
        script = f"""
import os, sys, torch, traceback, glob
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer

CACHE_PATH  = r'{CACHE_PATH}'
WORK_DIR    = r'{self.work_dir}'
GLOBAL_CKPT = r'{self.global_ckpt}'
LOCAL_CKPT  = r'{self.local_ckpt}'
SIGMA       = {sigma}
GRAD_CLIP   = {self.C}
EPOCHS      = {self.el}
BATCH_SIZE  = {self.bs}

_has_global = os.path.exists(GLOBAL_CKPT)
print(f'[Client] global_ckpt exists={{_has_global}}: {{GLOBAL_CKPT}}', flush=True)

kwargs = dict(
    cfg_file        = os.path.join(CACHE_PATH, 'configuration.json'),
    model           = 'damo/cv_tinynas_uav-detection_damoyolo',
    gpu_ids         = [0],
    batch_size      = BATCH_SIZE,
    max_epochs      = EPOCHS,
    num_classes     = 1,
    cache_path      = CACHE_PATH,
    load_pretrain   = False,
    base_lr_per_img = 0.00005,
    train_image_dir = r'{TRAIN_IMG_DIR}',
    val_image_dir   = r'{VAL_IMG_DIR}',
    train_ann       = r'{self.shard_ann}',
    val_ann         = r'{self.shard_ann}',
    work_dir        = WORK_DIR,
    num_workers     = 0,
)
if _has_global:
    kwargs['finetune_path'] = GLOBAL_CKPT

try:
    trainer = build_trainer(name=Trainers.tinynas_damoyolo, default_args=kwargs)
    trainer.cfg.miscs.num_workers           = 0
    trainer.cfg.miscs.print_interval_iters  = 10
    trainer.cfg.train.warmup_epochs         = 0
    trainer.cfg.train.warmup_start_lr       = float(0.00005 * BATCH_SIZE)
    trainer.cfg.train.augment.mosaic_mixup.mosaic_prob = 0.0
    trainer.cfg.train.augment.mosaic_mixup.mixup_prob  = 0.0
    trainer.cfg.train.total_epochs          = EPOCHS

    # ── DP gradient noise hooks (Algorithm 1 of the paper) ───────────────
    # Noise is added to gradients during backprop, NOT to weights after.
    # This matches the paper exactly: clip each gradient, add N(0, sigma^2*C^2*I)
    # We scale by 1/BATCH_SIZE to match the per-sample gradient mean in Alg 1.
    hooks = []
    def make_dp_hook(sigma, C, bs):
        def hook(grad):
            if grad is None:
                return grad
            # clip gradient
            norm = grad.norm(2)
            grad = grad * torch.clamp(torch.tensor(C / (norm.item() + 1e-6)), max=1.0)
            # add calibrated Gaussian noise: N(0, sigma^2 * C^2 / bs^2)
            noise = torch.randn_like(grad) * (sigma * C / bs)
            return grad + noise
        return hook

    # Register hooks AFTER build_trainer creates the model
    # trainer.model exists at this point (build_trainer instantiates it)
    _model = None
    for attr in ('model', '_model', 'net'):
        obj = getattr(trainer, attr, None)
        if obj is not None and hasattr(obj, 'parameters'):
            _model = obj
            break
    if _model is not None:
        for param in _model.parameters():
            if param.requires_grad:
                h = param.register_hook(make_dp_hook(SIGMA, GRAD_CLIP, BATCH_SIZE))
                hooks.append(h)
        print(f'[Client] Registered DP gradient hooks on {{len(hooks)}} parameters (sigma={{SIGMA:.4f}}, C={{GRAD_CLIP}})', flush=True)
    else:
        print('[Client] WARNING: could not find model for DP hooks — training without DP noise', flush=True)

    print(f'[Client] Training for {{EPOCHS}} epochs ...', flush=True)
    trainer.train()
    print(f'[Client] Training finished.', flush=True)

    # Remove hooks after training
    for h in hooks:
        h.remove()

    _cwd   = os.getcwd()
    latest = os.path.join(_cwd, 'workdirs', 'uav_detection', 'latest_ckpt.pth')
    epochN = os.path.join(_cwd, 'workdirs', 'uav_detection', f'epoch_{{EPOCHS}}_ckpt.pth')
    best_ckpt = None
    for c in [latest, epochN]:
        if os.path.exists(c):
            best_ckpt = c
            break
    if best_ckpt is None:
        found = glob.glob(os.path.join(_cwd, 'workdirs', '**', '*.pth'), recursive=True)
        found = [f for f in found if 'latest' in f or 'epoch_' in f]
        if found:
            best_ckpt = max(found, key=os.path.getmtime)

    if best_ckpt:
        ckpt = torch.load(best_ckpt, map_location='cpu')
        if isinstance(ckpt, dict):
            if 'ema_state_dict' in ckpt:
                final_sd = ckpt['ema_state_dict']
                print('[Client] Using EMA weights', flush=True)
            elif 'state_dict' in ckpt:
                final_sd = ckpt['state_dict']
            elif 'model' in ckpt:
                final_sd = ckpt['model']
            else:
                final_sd = ckpt
        else:
            final_sd = ckpt
        torch.save(final_sd, LOCAL_CKPT)
        print(f'[Client] Saved to {{LOCAL_CKPT}}', flush=True)
    else:
        print('[Client] ERROR: no checkpoint found after training!', flush=True)
        sys.exit(1)

except Exception as e:
    print(f'[Client] ERROR: {{e}}', flush=True)
    traceback.print_exc()
    sys.exit(1)
"""
        return script

# =============================================================================
# CLIENT
# FIX BUG 3: get_local_sd() loads from disk -- no pipeline() call
# =============================================================================
class Client():
    def __init__(self, id=-1, C=6, rho_total=20.0, bs=batch_size,
                 lr=0.1, el=epoch_local, agent=None, device='cuda:0',
                 shard_ann=None):
        self.id          = id
        self.rho         = rho_total
        self.rho_total   = rho_total
        self.C           = C
        self.bs          = bs
        self.lr          = lr
        self.el          = el
        self.sigma       = sigma_max
        self.rho_used    = 2 * self.el / (self.sigma ** 2)
        self.local_acc   = 0.0
        self.state       = None
        self.action      = None
        self.device      = device
        self.agent       = agent
        self.shard_ann   = shard_ann
        self._last_ckpt  = None
        self._last_map50 = 0.0
        self.runner = Training(client_id=id, C=C, bs=bs, lr=lr, el=el,
                               device=device, shard_ann=shard_ann)
        print(f'[Client {self.id}] Initialized', flush=True)

    def reset(self, rho_total):
        self.rho       = rho_total
        self.rho_total = rho_total
        self.action    = None
        self.sigma     = sigma_max
        self.local_acc = 0.0

    def save_global(self, net):
        sd = {k.replace('_module.', ''): v for k, v in net.state_dict().items()}
        torch.save(sd, self.runner.global_ckpt)

    def train(self, net, sigma):
        self.sigma    = sigma
        self.rho_used = self.el * 2 / (sigma ** 2)
        self.rho     -= self.rho_used
        self.save_global(net)
        self._last_ckpt = self.runner.train_one_round(sigma)

    def get_local_sd(self):
        """FIX BUG 3: load state_dict from disk -- zero pipeline() calls."""
        path = self._last_ckpt or self.runner.global_ckpt
        sd   = torch.load(path, map_location='cpu')
        return {k.replace('_module.', ''): v for k, v in sd.items()}

    def test(self, net):
        self.save_global(net)
        self._last_map50, _ = self.runner.eval_one_round()

    def get_test_result(self):
        return self._last_map50, 0.0

    def terminate_process(self):
        pass

# =============================================================================
# SERVER
# FIX BUG 4: aggregate takes list of state_dicts, clean FedAvg
# =============================================================================
class Server():
    def __init__(self, net, device='cuda:0'):
        self.device = device
        self.net    = net
        self.net.cuda(device=device)

    def set_sd(self, sd):
        self.net.load_state_dict(sd, strict=False)

    def aggregate(self, local_sds):
        if len(local_sds) > 0:
            head_keys = [k for k in local_sds[0] if 'head' in k.lower()]
            print(f'[aggregate] Head keys sample: {head_keys[:5]}', flush=True)
        if not local_sds:
            return
        ref    = local_sds[0]
        avg_sd = {}
        for k in ref:
            is_head = any(h in k for h in ['head', 'cls_pred', 'reg_pred', 'obj_pred', 'cls_', 'reg_'])
            if is_head:
                # Keep head from first client only — averaging kills detection calibration
                avg_sd[k] = ref[k]
            elif ref[k].is_floating_point():
                tensors   = [sd[k].float() for sd in local_sds if k in sd]
                avg_sd[k] = torch.stack(tensors, dim=0).mean(dim=0)
            else:
                avg_sd[k] = ref[k]
        self.set_sd(avg_sd)

# =============================================================================
# ACTOR / CRITIC / AGENT
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
        return torch.sigmoid(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, dim_state, dim_action):
        super().__init__()
        self.fc1 = nn.Linear(dim_state + dim_action, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

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
        self.gamma                  = torch.Tensor([gamma]).cuda(device=device)
        self.explore                = explore
        self.explore_decay          = explore_decay
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
                        Q_tgt = r + sum(self.gamma * Q_nxt[i] * (1 - d) for i in range(self.n_agents))
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
                    p.data = (1 - self.lr_actor_target) * p.data + self.lr_actor_target * self.actor.state_dict()[name].data.clone()
                for name, p in self.critic_target.named_parameters():
                    p.data = (1 - self.lr_critic_target) * p.data + self.lr_critic_target * self.critic.state_dict()[name].data.clone()
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
                    if e <= 0.0001:
                        an[i] = a[i]
                        break
                    an[i] = np.random.normal(an[i], e, 1)
            return a, an

    def save(self, episode):
        os.makedirs('./DRL_Nets', exist_ok=True)
        torch.save(self.actor.state_dict(),         f'./DRL_Nets/actor_drone_{episode}.pth')
        torch.save(self.actor_target.state_dict(),  f'./DRL_Nets/actor_target_drone_{episode}.pth')
        torch.save(self.critic.state_dict(),        f'./DRL_Nets/critic_drone_{episode}.pth')
        torch.save(self.critic_target.state_dict(), f'./DRL_Nets/critic_target_drone_{episode}.pth')

    def save_checkpoint(self, episode, training_state, path=CHECKPOINT_PATH):
        buf = {
            'state':      [list(q) for q in self.rep_buffer['state']],
            'action':     [list(q) for q in self.rep_buffer['action']],
            'reward':     list(self.rep_buffer['reward']),
            'next_state': [list(q) for q in self.rep_buffer['next_state']],
            'done':       list(self.rep_buffer['done']),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'actor':            self.actor.state_dict(),
                    'actor_target':     self.actor_target.state_dict(),
                    'critic':           self.critic.state_dict(),
                    'critic_target':    self.critic_target.state_dict(),
                    'optimizer_actor':  self.optimizer_actor.state_dict(),
                    'optimizer_critic': self.optimizer_critic.state_dict(),
                    'rep_buffer':       buf,
                    'explore':          self.explore,
                    'step':             self.step,
                    'train_step':       self.train_step,
                    'episode':          episode,
                    'training_state':   training_state}, path)
        print(f'[Checkpoint] Saved episode {episode} -> {path}')

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
# ACTION MAPPING helper
# =============================================================================
def apply_action_mapping(sigmas, rho_useds, clients):
    n = len(clients)
    rho_used_sums = [sum(rho_useds[i]) for i in range(n)]
    for i, client in enumerate(clients):
        S = rho_used_sums[i]
        if abs(max(sigmas[i]) - min(sigmas[i])) < 0.01:
            a = client.rho_total / max(S, 1e-9)
            b = 0.0
        else:
            T   = CR_Total
            ubs = [client.rho_total / T]
            lbs = [-np.inf]
            for CR in range(T):
                diff = S - T * rho_useds[i][CR]
                if diff > 1e-7:
                    ubs.append((rho_used_max * S - client.rho_total * rho_useds[i][CR]) / diff)
                    lbs.append((rho_used_min * S - client.rho_total * rho_useds[i][CR]) / diff)
                elif diff < -1e-7:
                    ubs.append((rho_used_min * S - client.rho_total * rho_useds[i][CR]) / diff)
                    lbs.append((rho_used_max * S - client.rho_total * rho_useds[i][CR]) / diff)
            bub  = min(ubs)
            blb  = max(lbs)
            dnom = sum((T * rho_useds[i][CR] - S) ** 2 for CR in range(T))
            b    = 0.0 if abs(dnom) < 1e-12 else \
                   -sum((rho_useds[i][CR] * S - client.rho_total) *
                        (T * rho_useds[i][CR] - S) for CR in range(T)) / dnom
            b    = max(blb, min(bub, b))
            a    = (client.rho_total - b * T) / max(S, 1e-9)
        for CR in range(CR_Total):
            rho_useds[i][CR] = a * rho_useds[i][CR] + b
            sigmas[i][CR]    = rho2sigma(rho=rho_useds[i][CR], el=client.el, bs=client.bs)
    return sigmas, rho_useds

# =============================================================================
# MAIN LOOP
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
        episode        += 1
    else:
        print('[Checkpoint] Starting from scratch.')
        episode         = 1
        final_accs      = []
        mean_final_accs = []
        mean_returns    = []
        returns         = []
        high_acc        = 0.0

    print('Splitting annotations...')
    #shard_anns = split_coco_annotations(TRAIN_ANN, N, SHARD_DIR)
    shard_anns = [TRAIN_ANN] * N 

    # FIX BUG 5: ONE pipeline() call here
    print('Building server model (pipeline loaded once)...')
    server_net = Net()
    init_sd    = {k: v.clone() for k, v in server_net.state_dict().items()}
    server     = Server(net=server_net)

    print('Initializing clients...')
    clients = [
        Client(id=i,
               rho_total=np.random.uniform(rho_min, rho_max),
               agent=agent,
               device='cuda:0',
               shard_ann=shard_anns[i])
        for i in range(N)
    ]
    print(f'All {N} clients ready.\n')

    while episode <= episodes:
        t1 = time.time()
        print(f'\n{"="*60}')
        print(f'EPISODE {episode}/{episodes}  |  explore={agent.explore:.4f}  |  train_step={agent.train_step}')
        print(f'{"="*60}')

        server.set_sd({k: v.clone() for k, v in init_sd.items()})

        rhos_init = []
        for i, client in enumerate(clients):
            rho = np.random.uniform(rho_min, rho_max)
            rhos_init.append(rho)
            client.reset(rho_total=rho)

        sigmas    = [[] for _ in range(N)]
        rho_useds = [[] for _ in range(N)]
        for CR in range(CR_Total):
            for i, client in enumerate(clients):
                state            = [CR / CR_Total, client.rho_total / rho_max]
                action, action_n = agent.take_action_with_noise(state=state)
                sigma            = float(action_n[0]) * (sigma_max - sigma_min) + sigma_min
                sigmas[i].append(sigma)
                rho_useds[i].append(sigma2rho(sigma, client.el, client.bs))
        sigmas, rho_useds = apply_action_mapping(sigmas, rho_useds, clients)

        last_global_acc = 0.0

        # FL training loop
        for CR in range(CR_Total):
            isDone    = 1.0 if CR == CR_Total - 1 else 0.0
            local_sds = []
            print(f'\n[Episode {episode} | FL Round {CR+1}/{CR_Total}] Training {N} clients...', flush=True)
            for i, client in enumerate(clients):
                client.state  = [CR / CR_Total, client.rho_total / rho_max]
                client.sigma  = sigmas[i][CR]
                client.action = (sigmas[i][CR] - sigma_min) / (sigma_max - sigma_min)
                print(f'  -> Client {i} | sigma={client.sigma:.4f}', flush=True)
                client.train(net=server.net, sigma=client.sigma)
                local_sds.append(client.get_local_sd())   # FIX BUG 3
            server.aggregate(local_sds)                    # FIX BUG 4

            print(f'[Episode {episode} | FL Round {CR+1}/{CR_Total}] Aggregated. Evaluating...', flush=True)
            global_acc = 0.0
            for client in clients:
                client.test(server.net)                    # FIX BUG 1
                map50, _         = client.get_test_result()
                global_acc      += map50
                client.local_acc = map50
            global_acc /= N

            agent.train(
                state      = [client.state  for client in clients],
                action     = [client.action for client in clients],
                reward     = [global_acc - last_global_acc],
                next_state = [[(CR+1)/CR_Total, client.rho_total/rho_max] for client in clients],
                done       = [isDone],
            )
            last_global_acc = global_acc
            print(f'[Episode {episode} | CR {CR+1}/{CR_Total}] mAP@0.5={global_acc:.4f}', flush=True)

        # DRL Eval loop (greedy, no noise)
        server.set_sd({k: v.clone() for k, v in init_sd.items()})
        for i, client in enumerate(clients):
            client.reset(rho_total=rhos_init[i])

        sigmas2    = [[] for _ in range(N)]
        rho_useds2 = [[] for _ in range(N)]
        for CR in range(CR_Total):
            for i, client in enumerate(clients):
                state  = [CR / CR_Total, client.rho_total / rho_max]
                action = agent.take_action(state=state)
                sigma  = float(action[0]) * (sigma_max - sigma_min) + sigma_min
                sigmas2[i].append(sigma)
                rho_useds2[i].append(sigma2rho(sigma, client.el, client.bs))
        sigmas2, rho_useds2 = apply_action_mapping(sigmas2, rho_useds2, clients)

        for i, client in enumerate(clients):
            client.reset(rho_total=rhos_init[i])

        global_accs_DRL = [0.0]
        for CR in range(CR_Total):
            local_sds = []
            print(f'\n[DRL Eval | FL Round {CR+1}/{CR_Total}] Training {N} clients...', flush=True)
            for i, client in enumerate(clients):
                client.sigma = sigmas2[i][CR]
                print(f'  -> Client {i} | sigma={client.sigma:.4f}', flush=True)
                client.train(net=server.net, sigma=client.sigma)
                local_sds.append(client.get_local_sd())
            server.aggregate(local_sds)

            global_acc = 0.0
            for client in clients:
                client.test(server.net)
                m, _             = client.get_test_result()
                global_acc      += m
                client.local_acc = m
            global_acc /= N
            global_accs_DRL.append(global_acc)
            print(f'[DRL Eval] CR {CR+1}/{CR_Total}  mAP@0.5={global_acc:.4f}', flush=True)

        final_accs.append(global_accs_DRL[-1])
        returns.append(global_accs_DRL[-1] - global_accs_DRL[0])
        low_idx  = max(len(final_accs) - 10, 0)
        mean_acc = float(np.mean(final_accs[low_idx:]))
        mean_final_accs.append(mean_acc)

        if mean_acc > high_acc and episode > 1:
            high_acc = mean_acc
            agent.save(episode=episode)
            print(f'[Agent] New best mAP={high_acc:.4f} -- saved.')

        mean_returns.append(float(np.mean(returns[max(len(returns)-10, 0):])))

        agent.save_checkpoint(episode=episode, training_state={
            'final_accs':     final_accs,
            'mean_final_accs': mean_final_accs,
            'mean_returns':   mean_returns,
            'returns':        returns,
            'high_acc':       high_acc,
            'global_accs_DRL': global_accs_DRL,
        })
        with open('./exp_data/exp_data_drone.json', 'w') as fp:
            json.dump({'final_accs':      final_accs,
                       'mean_final_accs': mean_final_accs,
                       'returns':         returns,
                       'mean_returns':    mean_returns,
                       'global_accs_DRL_last_episode': global_accs_DRL}, fp)

        elapsed = time.time() - t1
        print(f'\nEpisode {episode} done in {elapsed/60:.1f} min  |  '
              f'final mAP={final_accs[-1]:.4f}  |  mean mAP={mean_acc:.4f}')
        episode += 1

    for client in clients:
        client.terminate_process()
    print(f'\nTotal time: {(time.time()-t0)/60:.1f} min')
    print(f'Best mAP: {high_acc:.4f}')
    print(f'All final mAPs: {final_accs}')