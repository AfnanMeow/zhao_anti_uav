# =============================================================================
# FL + DP + DRL -- DP-SGD VARIANT (matches the source authors' actual mechanism)
# =============================================================================
# Why this file exists: comparing Author/Code/train_CIFAR10.py against the
# thesis port revealed the port silently replaced the paper's DP mechanism.
#
#   AUTHORS (works):  opacus.PrivacyEngine.make_private(..., noise_multiplier=sigma,
#                     max_grad_norm=C) -> DP-SGD. Per-sample gradients are clipped
#                     to C and Gaussian noise is added at EVERY optimizer step,
#                     inside local training. `sigma` is set per round.
#   THESIS PORT:      `import opacus` on line 18 of fl_damoyolo_final.py and never
#                     called. Local training is completely clean; one clip+noise is
#                     applied to the whole model delta once per round (DP-FedAvg).
#
# That difference is why uniform allocation was unbeatable in the port and why
# the paper's premise ("small gradients late are fragile") was inert there:
#   * DP-FedAvg gives the model `el` CLEAN local epochs every round, which simply
#     re-optimize away the previous round's perturbation -- noise is recoverable,
#     so WHEN it is spent barely matters (630+ paired runs: uniform optimal).
#   * DP-SGD corrupts every one of hundreds of gradient steps. Damage compounds and
#     cannot be healed within the round, and the per-step comparison is noise vs the
#     TRUE gradient magnitude -- which really does shrink late in training. That is
#     the asymmetry the paper's adaptive allocation exploits.
#   * C=6 is Opacus's max_grad_norm, a PER-SAMPLE GRADIENT bound (typical 0.1-10).
#     The port applied it as a whole-model DELTA bound (DAMO-YOLO delta norm ~2.8M,
#     i.e. off by ~470,000x) -- the pathology recorded in CLAUDE.md.
#
# Opacus forbids BatchNorm (it mixes information across samples, breaking per-sample
# privacy), so ModuleValidator.fix() swaps BN -> GroupNorm. Side benefit: GroupNorm
# carries no running buffers, so the BN-buffer noise contamination found earlier
# cannot occur here at all.
#
# Privacy accounting is unchanged from the authors: rho_used = 2 * el / sigma**2.
# =============================================================================
import os, sys, json, random, io, contextlib
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fl_lightcnn_final import TinyDetector, ShardDataset, evaluate_map, compute_loss

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


def make_dp_model(device='cuda:0'):
    """Canonical DP-compatible architecture: TinyDetector with BatchNorm swapped
    for GroupNorm. Used for server, clients and eval so state_dicts line up."""
    m = TinyDetector()
    m = ModuleValidator.fix(m)
    return m.to(device)


def _loader_for(dataset, batch_size):
    """Opacus hooks the DataLoader (it needs Poisson sampling), so the custom
    ShardDataset.batches() generator will not do -- wrap the cached tensors."""
    ds = torch.utils.data.TensorDataset(dataset.images, dataset.obj_targets, dataset.box_targets)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)


def train_local_dpsgd(state_dict, dataset, epochs, batch_size, lr, sigma, C,
                       device='cuda:0', momentum=0.0, model_fn=None):
    """One client's local round under DP-SGD. Returns the resulting state_dict.
    noise_multiplier=sigma is exactly the authors' per-round control knob."""
    model = (model_fn or make_dp_model)(device)
    model.load_state_dict(state_dict)
    model.train()
    loader = _loader_for(dataset, batch_size)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    pe = PrivacyEngine()
    model, opt, loader = pe.make_private(
        module=model, optimizer=opt, data_loader=loader,
        noise_multiplier=sigma, max_grad_norm=C,
    )
    for _ in range(epochs):
        for imgs, obj_t, box_t in loader:
            opt.zero_grad()
            loss, _, _ = compute_loss(model(imgs), obj_t, box_t)
            loss.backward()
            opt.step()
    # unwrap GradSampleModule -> plain module state_dict
    inner = model._module if hasattr(model, '_module') else model
    return {k: v.detach().clone() for k, v in inner.state_dict().items()}


def fedavg(state_dicts):
    avg, ref = {}, state_dicts[0]
    for k in ref:
        if torch.is_floating_point(ref[k]):
            avg[k] = torch.stack([sd[k].float() for sd in state_dicts]).mean(0)
        else:
            avg[k] = ref[k]
    return avg


def quiet_eval(state_dict, val_ds, device='cuda:0', model_fn=None):
    m = (model_fn or make_dp_model)(device)
    m.load_state_dict(state_dict)
    with contextlib.redirect_stdout(io.StringIO()):
        return evaluate_map(m, val_ds)


class DPClient:
    def __init__(self, cid, ann_path, C, bs, el, lr, device='cuda:0', momentum=0.0,
                 model_fn=None):
        self.id = cid
        self.dataset = ShardDataset(ann_path, device=device)
        self.C, self.bs, self.el, self.lr = C, bs, el, lr
        self.device, self.momentum = device, momentum
        self.model_fn = model_fn
        self.rho_total = 0.0

    def reset(self, rho_total):
        self.rho_total = rho_total

    def train_one_round(self, global_sd, sigma):
        return train_local_dpsgd(global_sd, self.dataset, self.el, self.bs,
                                  self.lr, sigma, self.C, self.device, self.momentum,
                                  model_fn=self.model_fn)


# =============================================================================
# ── Smaller detector sized for the DP-SGD noise regime ─────────────────────
# =============================================================================
# DP-SGD's per-step signal-to-noise is SNR = B / (sigma * sqrt(D)) -- the clip
# norm C cancels out entirely. The authors sit at B=200, D~59K (LeNet) => SNR
# ~0.82 at sigma=1, with ~100 gradient steps per round (5000 imgs/client).
# TinyDetector at D~98K with B=8 gives SNR ~0.026, which is why it flatlined.
# Halving every channel width brings D to ~25K (sqrt(D)~158), so a full-shard
# batch (B=158, all we have per client) lands SNR ~1.0 -- the authors' regime.
class SmallDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, 2, 1);    self.b1 = nn.BatchNorm2d(8)
        self.c2 = nn.Conv2d(8, 16, 3, 2, 1);   self.b2 = nn.BatchNorm2d(16)
        self.c3 = nn.Conv2d(16, 32, 3, 2, 1);  self.b3 = nn.BatchNorm2d(32)
        self.c4 = nn.Conv2d(32, 64, 3, 2, 1);  self.b4 = nn.BatchNorm2d(64)
        self.head = nn.Conv2d(64, 5, 1)

    def forward(self, x):
        import torch.nn.functional as F
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.c2(x)))
        x = F.relu(self.b3(self.c3(x)))
        x = F.relu(self.b4(self.c4(x)))
        return self.head(x)


def make_small_dp_model(device='cuda:0'):
    return ModuleValidator.fix(SmallDetector()).to(device)


# =============================================================================
# ── Higher-capacity detector for the honest (sequence-disjoint) task ───────
# =============================================================================
# On a leak-free sequence-disjoint split the 25K SmallDetector reaches only
# ~0.09 det-rate -- cross-sequence drone detection is a genuinely harder task
# than the single-sequence split made it look (that one read 0.9+ purely
# because consecutive video frames are near-duplicates on both sides).
# This widens the backbone and adds a stride-8 branch so small drones land on a
# finer grid. GroupNorm throughout: Opacus rejects BatchNorm outright (it mixes
# information across samples, breaking per-sample privacy accounting), and the
# isolation test showed GroupNorm itself costs almost nothing (0.2092 -> 0.2075).
class MidDetector(nn.Module):
    def __init__(self, gn=8):
        super().__init__()
        import torch.nn.functional as F

        def blk(i, o, s=2):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(gn, o), nn.ReLU(inplace=False))

        self.s1 = blk(3, 32)      # 160x128
        self.s2 = blk(32, 64)     # 80x64
        self.s3 = blk(64, 128)    # 40x32
        self.s4 = blk(128, 192)   # 20x16
        self.r3 = nn.Sequential(nn.Conv2d(128, 192, 3, 2, 1), nn.GroupNorm(gn, 192), nn.ReLU(inplace=False))
        self.fuse = nn.Sequential(nn.Conv2d(384, 192, 3, 1, 1), nn.GroupNorm(gn, 192), nn.ReLU(inplace=False))
        self.head = nn.Conv2d(192, 5, 1)

    def forward(self, x):
        x = self.s1(x); x = self.s2(x)
        c3 = self.s3(x)
        c4 = self.s4(c3)
        x = torch.cat([c4, self.r3(c3)], dim=1)
        return self.head(self.fuse(x))


def make_mid_model(device='cuda:0'):
    return ModuleValidator.fix(MidDetector()).to(device)
