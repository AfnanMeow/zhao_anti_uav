# =============================================================================
# FL + DP + DRL for Drone Detection -- LIGHTWEIGHT CNN VARIANT
# =============================================================================
# Why this file exists (see conversation / CLAUDE.md for the DAMO-YOLO
# history): fl_damoyolo_final.py's DRL agent needs thousands of training
# steps for its explore_decay=0.9996 schedule to matter, but DAMO-YOLO's
# per-round cost (subprocess launch + ModelScope pipeline rebuild + real
# detector training/eval) is ~35-40s at even a tiny (24-image) smoke scale --
# far too slow to get enough episodes/training-steps in any practical
# timeframe. This file replaces DAMO-YOLO with a small, from-scratch,
# in-process (no subprocess, no ModelScope) single-scale grid detector,
# trained directly on single_test_coco, so a full local-training round costs
# a fraction of a second instead of tens of seconds -- enabling hundreds of
# real episodes per day instead of per month.
#
# Reused UNCHANGED from fl_damoyolo_final.py: Actor/Critic/Agent (the DRL
# machinery is architecture-agnostic -- it only sees scalar state/action/
# reward) and the whole-model flatten-clip+noise DP mechanism (validated
# live: no collapse across 5 real rounds, see conversation history).
#
# NOT reused: Net/Client/Training/Server/aggregate/evaluate_map (all
# DAMO-YOLO/ModelScope-specific) -- replaced with lightweight equivalents
# below. Because this model is trained from scratch every episode (matching
# the original MARL paper's toy-CNN setup) there is no pretrained-checkpoint
# calibration to protect, so aggregate() is plain FedAvg over ALL parameters
# (including any head-like layers) -- this needs empirical validation the
# same way seed_perturb and the noise formula were validated, not assumed.
# =============================================================================

import os, sys, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fl_damoyolo_final import Actor, Critic, Agent  # noqa: E402  (DRL machinery, unchanged)

# =============================================================================
# ── MODEL: tiny single-scale grid detector ──────────────────────────────────
# =============================================================================
IMG_W, IMG_H = 320, 256
GRID_W, GRID_H = 20, 16
STRIDE = 16  # IMG_W/GRID_W == IMG_H/GRID_H == 16
ORIG_W, ORIG_H = 640, 512
SCALE_X, SCALE_Y = IMG_W / ORIG_W, IMG_H / ORIG_H


class TinyDetector(nn.Module):
    """Minimal YOLOv1-style single-scale, single-class, single-box-per-image
    detector. 4 stride-2 conv blocks (320x256 -> 20x16 grid), then a 1x1 head
    predicting (objectness, tx, ty, tw, th) per cell. Trains in a fraction of
    a second per local round on a ~150-200 image client shard on this GPU."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, 2, 1); self.b1 = nn.BatchNorm2d(16)
        self.c2 = nn.Conv2d(16, 32, 3, 2, 1); self.b2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 64, 3, 2, 1); self.b3 = nn.BatchNorm2d(64)
        self.c4 = nn.Conv2d(64, 128, 3, 2, 1); self.b4 = nn.BatchNorm2d(128)
        self.head = nn.Conv2d(128, 5, 1)

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.c2(x)))
        x = F.relu(self.b3(self.c3(x)))
        x = F.relu(self.b4(self.c4(x)))
        return self.head(x)  # (B, 5, GRID_H, GRID_W)


# =============================================================================
# ── DATASET: preload + resize once, cache as GPU tensors ───────────────────
# =============================================================================
class ShardDataset:
    """Loads a COCO-style annotation shard, resizes every image once up
    front, and encodes grid targets -- all cached as GPU tensors so repeated
    epochs over the same shard involve zero disk I/O or CPU resizing."""

    def __init__(self, ann_path, device='cuda:0'):
        self.device = device
        coco = COCO(ann_path)
        self.coco = coco
        img_ids = coco.getImgIds()
        imgs, obj_targets, box_targets, self.img_ids = [], [], [], []
        for img_id in img_ids:
            info = coco.loadImgs(img_id)[0]
            path = info['file_name']
            im = cv2.imread(path)
            if im is None:
                continue
            im = cv2.resize(im, (IMG_W, IMG_H))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            imgs.append(torch.from_numpy(im).permute(2, 0, 1))

            obj_t = torch.zeros(GRID_H, GRID_W)
            box_t = torch.zeros(4, GRID_H, GRID_W)
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            if anns:
                a = anns[0]  # dataset has at most ~1 box/image; use the first
                x, y, w, h = a['bbox']
                cx = (x + w / 2) * SCALE_X
                cy = (y + h / 2) * SCALE_Y
                rw = max(w * SCALE_X, 1e-3)
                rh = max(h * SCALE_Y, 1e-3)
                gi = min(int(cx // STRIDE), GRID_W - 1)
                gj = min(int(cy // STRIDE), GRID_H - 1)
                tx = (cx - gi * STRIDE) / STRIDE
                ty = (cy - gj * STRIDE) / STRIDE
                tw = float(np.log(rw / STRIDE))
                th = float(np.log(rh / STRIDE))
                obj_t[gj, gi] = 1.0
                box_t[:, gj, gi] = torch.tensor([tx, ty, tw, th])
            obj_targets.append(obj_t)
            box_targets.append(box_t)
            self.img_ids.append(img_id)

        self.images = torch.stack(imgs).to(device)
        self.obj_targets = torch.stack(obj_targets).to(device)
        self.box_targets = torch.stack(box_targets).to(device)
        self.n = len(self.images)

    def batches(self, batch_size, shuffle=True):
        idx = list(range(self.n))
        if shuffle:
            random.shuffle(idx)
        for i in range(0, self.n, batch_size):
            b = idx[i:i + batch_size]
            yield self.images[b], self.obj_targets[b], self.box_targets[b]


def compute_loss(pred, obj_t, box_t):
    obj_logit = pred[:, 0]
    box_pred = pred[:, 1:5]
    obj_loss = F.binary_cross_entropy_with_logits(obj_logit, obj_t)
    pos_mask = obj_t.unsqueeze(1)
    n_pos = pos_mask.sum().clamp(min=1.0)
    box_loss = (F.smooth_l1_loss(box_pred, box_t, reduction='none') * pos_mask).sum() / n_pos
    return obj_loss + 5.0 * box_loss, obj_loss.item(), box_loss.item()


def train_local(model, dataset, epochs, batch_size, lr, device='cuda:0',
                optimizer='adam', momentum=0.9):
    """optimizer='sgd' matters for DP research, not just speed: Adam divides by
    its running second-moment estimate, so its step size is ~scale-invariant to
    gradient magnitude and the client update norm stays flat (~2.1) even after
    the model has fully converged (measured live over 25 rounds). That erases
    the source paper's core premise -- "gradients shrink during fine-tuning, so
    late-round noise is more damaging" -- because the update magnitude, and
    hence the post-clip signal-to-noise ratio, stops depending on how converged
    the model is. Under plain SGD the update norm does decay with convergence
    (60.5 -> 0.27 over 25 rounds), restoring that premise."""
    model.train()
    if optimizer == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    last = (0.0, 0.0, 0.0)
    for _ in range(epochs):
        for imgs, obj_t, box_t in dataset.batches(batch_size):
            pred = model(imgs)
            loss, ol, bl = compute_loss(pred, obj_t, box_t)
            opt.zero_grad(); loss.backward(); opt.step()
            last = (loss.item(), ol, bl)
    return last


@torch.no_grad()
def evaluate_map(model, dataset, device='cuda:0'):
    """Top-1-box-per-image decode (matches the dataset's <=1 drone/image
    structure), fed into the same pycocotools COCOeval used throughout this
    project."""
    model.eval()
    results = []
    for i in range(dataset.n):
        img = dataset.images[i:i + 1]
        pred = model(img)[0]
        obj = torch.sigmoid(pred[0])
        gj, gi = divmod(int(obj.argmax().item()), GRID_W)
        score = float(obj[gj, gi].item())
        tx, ty, tw, th = pred[1:5, gj, gi].tolist()
        cx = (gi + torch.sigmoid(torch.tensor(tx)).item()) * STRIDE
        cy = (gj + torch.sigmoid(torch.tensor(ty)).item()) * STRIDE
        w = float(np.exp(tw)) * STRIDE
        h = float(np.exp(th)) * STRIDE
        x1 = (cx - w / 2) / SCALE_X
        y1 = (cy - h / 2) / SCALE_Y
        bw = w / SCALE_X
        bh = h / SCALE_Y
        results.append({'image_id': dataset.img_ids[i], 'category_id': 1,
                         'bbox': [x1, y1, bw, bh], 'score': score})
    if not results:
        return 0.0
    tmp = f'_tmp_lightcnn_eval_{os.getpid()}.json'
    with open(tmp, 'w') as f:
        json.dump(results, f)
    coco_dt = dataset.coco.loadRes(tmp)
    ev = COCOeval(dataset.coco, coco_dt, 'bbox')
    ev.params.imgIds = dataset.img_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()
    os.remove(tmp)
    return float(ev.stats[1])


# =============================================================================
# ── DP NOISE (same whole-model flatten-clip+noise mechanism validated for
#    DAMO-YOLO, BN-buffer exclusion included; C_clip is recalibrated for
#    this model's own real delta-norm scale below via measurement) ─────────
# =============================================================================
BN_BUFFER_SUFFIXES = ('.running_mean', '.running_var', '.num_batches_tracked')


def _is_bn_buffer(key):
    return key.endswith(BN_BUFFER_SUFFIXES)


def clip_and_noise(local_sd, global_sd, C_clip, sigma):
    matching_keys = [k for k in local_sd if k in global_sd and not _is_bn_buffer(k)]
    deltas = {k: local_sd[k].float() - global_sd[k].float() for k in matching_keys}
    flat = torch.cat([d.flatten() for d in deltas.values()])
    total_norm = flat.norm(2).item()
    total_numel = flat.numel()
    clip_coef = min(1.0, C_clip / (total_norm + 1e-6))
    noise_std_per_elem = sigma * C_clip / (total_numel ** 0.5)
    noisy_sd = {}
    for k in matching_keys:
        d_clipped = deltas[k] * clip_coef
        noise = torch.randn_like(d_clipped) * noise_std_per_elem
        noisy_sd[k] = global_sd[k].float() + d_clipped + noise
    for k in local_sd:
        if k not in matching_keys:
            noisy_sd[k] = local_sd[k]
    return noisy_sd, total_norm, total_numel, clip_coef


def fedavg(state_dicts):
    """Plain FedAvg over ALL parameters -- no pretrained head to protect
    since this model trains from scratch every episode (matching the
    original paper's toy-CNN setup, unlike DAMO-YOLO's finetuning regime)."""
    avg = {}
    ref = state_dicts[0]
    for k in ref:
        if torch.is_floating_point(ref[k]):
            avg[k] = torch.stack([sd[k].float() for sd in state_dicts]).mean(0)
        else:
            avg[k] = ref[k]
    return avg


class Client:
    def __init__(self, cid, ann_path, C, bs, el, lr, device='cuda:0', optimizer='adam'):
        self.id = cid
        self.dataset = ShardDataset(ann_path, device=device)
        self.C = C
        self.bs = bs
        self.el = el
        self.lr = lr
        self.device = device
        self.optimizer = optimizer
        self.rho_total = 0.0
        self.rho = 0.0

    def reset(self, rho_total):
        self.rho_total = rho_total
        self.rho = rho_total

    def train_one_round(self, global_sd, sigma):
        model = TinyDetector().to(self.device)
        model.load_state_dict(global_sd)
        train_local(model, self.dataset, epochs=self.el, batch_size=self.bs, lr=self.lr,
                    optimizer=self.optimizer)
        local_sd = model.state_dict()
        noisy_sd, total_norm, numel, clip_coef = clip_and_noise(local_sd, global_sd, self.C, sigma)
        return noisy_sd

    def evaluate(self, global_sd):
        model = TinyDetector().to(self.device)
        model.load_state_dict(global_sd)
        return evaluate_map(model, self.dataset)


class Server:
    def __init__(self, state_dict):
        self.state_dict = state_dict

    def aggregate(self, state_dicts):
        self.state_dict = fedavg(state_dicts)


# =============================================================================
# ── MAIN LOOP: full FL + DP + DRL episode loop on single_test_coco ─────────
# =============================================================================
if __name__ == '__main__':
    import time as _time
    import random as _random
    import numpy as np

    N           = 5
    CR_TOTAL    = 8
    EPOCH_LOCAL = 7
    BATCH_SIZE  = 8
    LR          = 1e-3
    C_CLIP      = 1.0
    RHO_MIN, RHO_MAX = 20.0, 40.0
    SIGMA_MIN, SIGMA_MAX = 1.0, 20.0
    RHO_USED_MIN = 2 / SIGMA_MAX**2 * EPOCH_LOCAL
    RHO_USED_MAX = 2 / SIGMA_MIN**2 * EPOCH_LOCAL
    MINIMAL_SIZE = 200
    EPISODES     = 2000
    DETERM_EVERY = 10
    N_TRIALS     = 5

    TRAIN_ANN = 'single_test_coco/annotations/train.json'
    VAL_ANN   = 'single_test_coco/annotations/val.json'
    SHARD_DIR = './fl_workdirs/lightcnn_shards'
    LOG_PATH  = './exp_data/lightcnn_rl_log.json'
    CKPT_PATH = './DRL_Nets/lightcnn_checkpoint.pth'

    def sigma2rho(sigma, el):
        return 2 * el / (sigma ** 2)

    def rho2sigma(rho, el):
        return float(np.sqrt(2 * el / max(rho, 1e-9)))

    def action_mapping(el, sigmas_i, rho_total):
        rho_useds = [sigma2rho(s, el) for s in sigmas_i]
        S = sum(rho_useds)
        T = CR_TOTAL
        if abs(max(sigmas_i) - min(sigmas_i)) < 0.01:
            a = rho_total / max(S, 1e-9); b = 0.0
        else:
            ubs = [rho_total / T]; lbs = [-np.inf]
            for CR in range(T):
                diff = S - T * rho_useds[CR]
                if diff > 1e-7:
                    ubs.append((RHO_USED_MAX * S - rho_total * rho_useds[CR]) / diff)
                    lbs.append((RHO_USED_MIN * S - rho_total * rho_useds[CR]) / diff)
                elif diff < -1e-7:
                    ubs.append((RHO_USED_MIN * S - rho_total * rho_useds[CR]) / diff)
                    lbs.append((RHO_USED_MAX * S - rho_total * rho_useds[CR]) / diff)
            bub = min(ubs); blb = max(lbs)
            dnom = sum((T * rho_useds[CR] - S) ** 2 for CR in range(T))
            b = 0.0 if abs(dnom) < 1e-12 else \
                -sum((rho_useds[CR] * S - rho_total) * (T * rho_useds[CR] - S) for CR in range(T)) / dnom
            b = max(blb, min(bub, b))
            a = (rho_total - b * T) / max(S, 1e-9)
        return [rho2sigma(a * rho_useds[CR] + b, el) for CR in range(CR_TOTAL)]

    def global_eval(global_sd, eval_model, val_ds):
        # Reward/accuracy signal computed ONCE on a held-out set the clients
        # never train on -- not per-client-shard self-eval (verified live:
        # training against self-eval let the agent learn a policy that
        # flattered that proxy without actually generalizing -- it lost to
        # Uniform on held-out data despite "winning" during its own training
        # metric). Also matches the source paper's own reward definition
        # ("global accuracy delta"), and is cheaper (1 eval/round, not N).
        eval_model.load_state_dict(global_sd)
        return evaluate_map(eval_model, val_ds)

    def run_episode(agent, clients, explore, val_ds, init_sd=None):
        # ── Reactive redesign (see conversation history) ────────────────────
        # OLD: the actor picked all CR_TOTAL actions upfront from
        # [round_fraction, budget_fraction] alone, THEN action_mapping()
        # affine-rescaled them to hit the budget exactly. That guaranteed
        # exact rho_total usage (matching the source paper's QP mechanism),
        # but made the agent structurally blind to how THIS episode's
        # training was actually going -- it could never react to real
        # dynamics, which is precisely where a learned policy should have
        # room to beat a fixed closed-form curve. Verified live across
        # several fully-controlled comparisons: it didn't (mixed/negative
        # vs Uniform/Exponential/Arithmetic).
        # NEW: track remaining rho per client round-by-round; state is
        # [round_fraction, remaining_budget_fraction, current_acc,
        # last_round's_acc_delta] -- the agent now sees real performance
        # feedback. Trade-off: no more hard budget-exact guarantee from a
        # QP; instead the last round is forced to spend exactly whatever
        # budget remains (clipped to [sigma_min, sigma_max]), and remaining
        # budget itself is part of the state so overspending early is
        # something the agent has the information to learn to avoid, even
        # though it isn't hard-constrained to.
        global_sd = {k: v.clone() for k, v in init_sd.items()} if init_sd is not None \
            else TinyDetector().cuda().state_dict()
        rhos, remaining = [], []
        for c in clients:
            rho = float(np.random.uniform(RHO_MIN, RHO_MAX))
            rhos.append(rho); remaining.append(rho); c.reset(rho_total=rho)

        eval_model = TinyDetector().cuda()
        last_acc = global_eval(global_sd, eval_model, val_ds)
        baseline = last_acc
        acc_delta = 0.0

        accs = [baseline]
        all_sigmas = [[] for _ in range(N)]
        for CR in range(CR_TOTAL):
            isDone = 1.0 if CR == CR_TOTAL - 1 else 0.0
            states, actions, local_sds = [], [], []
            for i, c in enumerate(clients):
                state = [CR / CR_TOTAL, remaining[i] / RHO_MAX, last_acc, acc_delta]
                if explore:
                    _, an = agent.take_action_with_noise(state=state)
                    a = float(np.clip(an[0], 0, 1))
                else:
                    a = float(np.clip(agent.take_action(state=state)[0], 0, 1))
                sigma = a * (SIGMA_MAX - SIGMA_MIN) + SIGMA_MIN
                if CR == CR_TOTAL - 1:
                    sigma = float(np.clip(rho2sigma(max(remaining[i], 1e-6), c.el), SIGMA_MIN, SIGMA_MAX))
                remaining[i] = max(remaining[i] - sigma2rho(sigma, c.el), 0.0)
                all_sigmas[i].append(sigma)
                local_sds.append(c.train_one_round(global_sd, sigma))
                states.append(state); actions.append((sigma - SIGMA_MIN) / (SIGMA_MAX - SIGMA_MIN))
            global_sd = fedavg(local_sds)
            acc = global_eval(global_sd, eval_model, val_ds)
            accs.append(acc)
            acc_delta = acc - last_acc
            if explore:
                reward = [acc - last_acc]
                next_states = [[(CR + 1) / CR_TOTAL, remaining[i] / RHO_MAX, acc, acc_delta]
                                for i in range(N)]
                agent.train(state=states, action=actions, reward=reward,
                            next_state=next_states, done=[isDone])
            last_acc = acc
        return baseline, accs, all_sigmas, rhos, global_sd

    def run_uniform(clients, rhos, init_sd, val_ds):
        global_sd = {k: v.clone() for k, v in init_sd.items()}
        eval_model = TinyDetector().cuda()
        for i, c in enumerate(clients):
            c.reset(rho_total=rhos[i])
        accs = [global_eval(global_sd, eval_model, val_ds)]
        for CR in range(CR_TOTAL):
            local_sds = []
            for c in clients:
                sigma = float(np.clip((2 * c.el / (c.rho_total / CR_TOTAL)) ** 0.5, SIGMA_MIN, SIGMA_MAX))
                local_sds.append(c.train_one_round(global_sd, sigma))
            global_sd = fedavg(local_sds)
            accs.append(global_eval(global_sd, eval_model, val_ds))
        return accs

    os.makedirs(SHARD_DIR, exist_ok=True)
    os.makedirs('./exp_data', exist_ok=True)
    os.makedirs('./DRL_Nets', exist_ok=True)

    # split annotations into N shards
    d = json.load(open(TRAIN_ANN))
    imgs = d['images']
    _random.shuffle(imgs)
    shard_size = len(imgs) // N
    shard_paths = []
    for i in range(N):
        shard_imgs = imgs[i * shard_size:(i + 1) * shard_size] if i < N - 1 else imgs[i * shard_size:]
        ids = set(im['id'] for im in shard_imgs)
        anns = [a for a in d['annotations'] if a['image_id'] in ids]
        out = {'info': {}, 'licenses': [], 'categories': d['categories'],
               'images': shard_imgs, 'annotations': anns}
        p = os.path.join(SHARD_DIR, f'shard_{i}.json')
        json.dump(out, open(p, 'w'))
        shard_paths.append(p)

    clients = [Client(i, shard_paths[i], C=C_CLIP, bs=BATCH_SIZE, el=EPOCH_LOCAL, lr=LR)
               for i in range(N)]
    print(f'Client shard sizes: {[c.dataset.n for c in clients]}', flush=True)

    val_ds = ShardDataset(VAL_ANN, device='cuda:0')
    print(f'Held-out val set: {val_ds.n} images (never used in any client shard)', flush=True)

    # explore_decay default (0.9996) needs ~11,500 training steps to reach its
    # floor -- at CR_Total=8, 400 episodes only produces ~3,200 steps, so most
    # actions were still near-random exploration even by episode 400 (verified
    # live: deterministic-policy checks showed no improvement trend across the
    # whole run). 0.995 reaches the floor in ~920 steps (~ep 115 of this run),
    # leaving ~94%% of the 2000-episode budget as genuine policy exploitation.
    # dim_state=4: [round_fraction, remaining_budget_fraction, current_acc,
    # last_round's_acc_delta] -- see run_episode()'s reactive-redesign comment.
    agent = Agent(n_agents=N, dim_state=4, dim_action=1, buffer_size=10000,
                  minimal_size=MINIMAL_SIZE, bs=64, explore_decay=0.995)

    log = {'episodes': []}
    print(f'Starting lightcnn RL run: {EPISODES} episodes, N={N}, CR_Total={CR_TOTAL}, '
          f'minimal_size={MINIMAL_SIZE}', flush=True)

    for ep in range(1, EPISODES + 1):
        t0 = _time.time()
        baseline, accs, sigmas, rhos, _ = run_episode(agent, clients, explore=True, val_ds=val_ds)
        buf_len = len(agent.rep_buffer['state'][0])
        entry = {'episode': ep, 'baseline': baseline, 'final_acc': accs[-1],
                  'return': accs[-1] - baseline, 'buffer_len': buf_len,
                  'explore': agent.explore, 'train_step': agent.train_step,
                  'sigma_means': [float(np.mean(s)) for s in sigmas]}
        print(f'[Ep {ep}] baseline={baseline:.4f} final={accs[-1]:.4f} '
              f'return={entry["return"]:+.4f} buf={buf_len} explore={agent.explore:.4f} '
              f'train_step={agent.train_step} ({_time.time()-t0:.1f}s)', flush=True)

        if ep % DETERM_EVERY == 0:
            db, dacc, _, _, _ = run_episode(agent, clients, explore=False, val_ds=val_ds)
            entry['determ_baseline'] = db; entry['determ_final'] = dacc[-1]
            print(f'  [Ep {ep} DETERMINISTIC] baseline={db:.4f} -> final={dacc[-1]:.4f}', flush=True)

        log['episodes'].append(entry)
        with open(LOG_PATH, 'w') as f:
            json.dump(log, f, indent=1)
        if ep % 25 == 0:
            agent.save_checkpoint(episode=ep, training_state={}, path=CKPT_PATH)

    print(f'\n=== Final head-to-head: {N_TRIALS} trials, shared init per trial ===', flush=True)
    trials = []
    for t in range(N_TRIALS):
        _random.seed(2000 + t); np.random.seed(2000 + t)
        trial_rhos = [float(np.random.uniform(RHO_MIN, RHO_MAX)) for _ in range(N)]
        shared_init_sd = TinyDetector().cuda().state_dict()
        for i, c in enumerate(clients):
            c.reset(rho_total=trial_rhos[i])
        drl_baseline, drl_accs, _, _, _ = run_episode(agent, clients, explore=False,
                                                       val_ds=val_ds, init_sd=shared_init_sd)
        uni_accs = run_uniform(clients, trial_rhos, init_sd=shared_init_sd, val_ds=val_ds)
        winner = 'DRL' if drl_accs[-1] > uni_accs[-1] else 'Uniform'
        print(f'[Trial {t+1}] rhos={[f"{r:.2f}" for r in trial_rhos]} '
              f'DRL_final={drl_accs[-1]:.4f} Uniform_final={uni_accs[-1]:.4f} -> {winner}', flush=True)
        trials.append({'rhos': trial_rhos, 'drl_accs': drl_accs, 'uniform_accs': uni_accs,
                        'drl_final': drl_accs[-1], 'uniform_final': uni_accs[-1], 'winner': winner})
        log['final_comparison_trials'] = trials
        with open(LOG_PATH, 'w') as f:
            json.dump(log, f, indent=1)

    drl_finals = [t['drl_final'] for t in trials]
    uni_finals = [t['uniform_final'] for t in trials]
    wins = sum(1 for t in trials if t['winner'] == 'DRL')
    print(f'\n=== Summary over {N_TRIALS} trials ===', flush=True)
    print(f'DRL final mean={np.mean(drl_finals):.4f} (range {min(drl_finals):.4f}-{max(drl_finals):.4f})', flush=True)
    print(f'Uniform final mean={np.mean(uni_finals):.4f} (range {min(uni_finals):.4f}-{max(uni_finals):.4f})', flush=True)
    print(f'DRL won {wins}/{N_TRIALS} trials', flush=True)
