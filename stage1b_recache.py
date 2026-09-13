# =============================================================================
# STAGE 1b: re-cache frozen features at an EARLIER cut of the public trunk.
#
# Gate 1 (see FINDINGS_gate1.md) showed that DP-training only the 12.7K head
# makes noise irrelevant across eps=inf .. 0.1 -- there is no accuracy gradient
# for any allocator to exploit. Effective SNR ~ sqrt(B)/(sigma*sqrt(D)), so the
# two live knobs are D (how much of the trunk is DP-trained) and B.
#
# Cutting after s3 instead of after k2 moves the DP-trained parameter count
#   12,677  ->  566,789   (sqrt(D): 113 -> 753, a 6.7x SNR reduction)
# while still caching, so runs stay cheap. s1,s2,s3 stay frozen and public.
#
#   frozen (public, cached): s1 -> s2 -> s3        [64 -> 128ch, stride 8]
#   DP-trained from scratch: k1 -> k2 -> head      [566,789 params]
# =============================================================================
import os, sys, time, warnings
warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage1_public_pretrain import Trunk, Head, reencode_stride8, BIGDATA, OUT

CUT_OUT = os.path.join(OUT, 'cut_s3')
DEV = 'cuda:0'


class DPTail(nn.Module):
    """The DP-trained part when the trunk is cut after s3: k1, k2 and the head.
    Initialised from scratch -- that is what gives the federated phase real
    dynamic range (see FINDINGS_gate1.md section 1)."""

    def __init__(self, gn=8):
        super().__init__()
        self.k1 = nn.Sequential(nn.Conv2d(128, 192, 3, 1, 1), nn.GroupNorm(gn, 192), nn.ReLU(inplace=False))
        self.k2 = nn.Sequential(nn.Conv2d(192, 192, 3, 1, 1), nn.GroupNorm(gn, 192), nn.ReLU(inplace=False))
        self.head = Head()

    def forward(self, f):
        return self.head(self.k2(self.k1(f)))


if __name__ == '__main__':
    os.makedirs(CUT_OUT, exist_ok=True)
    pre = torch.load(os.path.join(OUT, 'public_pretrained.pt'), map_location=DEV)
    trunk = Trunk().to(DEV)
    trunk.load_state_dict(pre['trunk'])
    trunk.eval()
    for p in trunk.parameters():
        p.requires_grad_(False)

    D = sum(p.numel() for p in DPTail().parameters())
    print(f'DP-trained tail: D={D}  sqrt(D)={D**0.5:.0f}   (head-only was 12677 / 113)', flush=True)

    @torch.no_grad()
    def cut_s3(x):
        return trunk.s3(trunk.s2(trunk.s1(x)))

    @torch.no_grad()
    def cache(imgs_u8, tag):
        fs = []
        for s in range(0, imgs_u8.shape[0], 256):
            fs.append(cut_s3(imgs_u8[s:s + 256].to(DEV).float() / 255.0).half().cpu())
        f = torch.cat(fs)
        torch.save(f, os.path.join(CUT_OUT, f'feat_{tag}.pt'))
        return tuple(f.shape)

    t0 = time.time()
    for i in range(5):
        im, o, b = reencode_stride8(os.path.join(BIGDATA, f'client_{i}.pt'))
        print(f'  client {i}: features {cache(im, f"client_{i}")}  ({time.time()-t0:.0f}s)', flush=True)
        del im
    vim, vo, vb = reencode_stride8(os.path.join(BIGDATA, 'val.pt'))
    print(f'  val: features {cache(vim, "val")}', flush=True)
    print('STAGE 1b DONE', flush=True)
