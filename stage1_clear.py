# =============================================================================
# STAGE 1 (clear-sky subset): session-disjoint split, public pretrain, s3 cache.
#
# Two changes from stage1_public_pretrain.py + stage1b_recache.py:
#
# 1. DATA. Restricted to the 29 hand-picked clear-sky sequences. On the cluttered
#    sequences the detector sits at 0.0-0.10 mAP (dpsgd_bigdata_test.json), and
#    you cannot measure DP degradation against a floor -- every schedule scores
#    the same. Removing that floor effect is what makes the allocation question
#    measurable. Results from this split describe the clear-sky subset, not
#    anti-UAV detection in general.
#
# 2. SPLIT BY RECORDING SESSION, never by folder. Clips sharing a timestamp
#    prefix (20190925_111757_1_1 / _1_2 / _1_8) are the same recording seconds
#    apart. The old manifest split by folder and put 5 of 11 val sessions into
#    train as well; the old public-pretrain pick also shared 4 sessions with
#    these 29, so the trunk is re-pretrained here on a session-disjoint pool.
# =============================================================================
import os, sys, json, glob, random, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage1_public_pretrain import (FullNet, loss_fn, load_folder, encode,
                                    eval_full, TRAIN_ROOT)

OUT = './fl_workdirs/clear'
N_CLIENTS, N_VAL_SESSIONS = 5, 4
PUBLIC_FOLDERS, PUBLIC_EPOCHS = 20, 8
VAL_TARGET = 1500
SEED = 4242
DEV = 'cuda:0'

CLEAR = """20190925_101846_1_3 20190925_111757_1_1 20190925_111757_1_2 20190925_111757_1_8
20190925_124612_1_1 20190925_124612_1_2 20190925_130434_1_3 20190925_130434_1_4
20190925_131530_1_7 20190925_134301_1_2 20190925_134301_1_7 20190925_140917_1_1
20190925_140917_1_8 20190925_141417_1_7 20190925_143900_1_1 20190925_143900_1_6
20190925_183946_1_2 20190925_183946_1_8 20190925_193610_1_1 20190925_193610_1_2
20190925_193610_1_3 20190925_205804_1_5 20190925_205804_1_6 20190926_095902_1_1
20190926_102042_1_6 20190926_111509_1_7 20190926_143632_1_3 20190926_143632_1_7
20190926_193515_1_8""".split()


def sess_of(f):
    return '_'.join(f.split('_')[:2])


def build_split():
    by_sess = {}
    for f in CLEAR:
        by_sess.setdefault(sess_of(f), []).append(f)
    counts = {s: sum(len(glob.glob(os.path.join(TRAIN_ROOT, f, '*.jpg'))) for f in fs)
              for s, fs in by_sess.items()}
    sessions = sorted(by_sess, key=lambda s: -counts[s])
    rng = random.Random(SEED)
    val_sess = sorted(rng.sample(sessions, N_VAL_SESSIONS))
    tr_sess = [s for s in sessions if s not in val_sess]
    # greedy balance: biggest remaining session to the currently-smallest client
    clients = [[] for _ in range(N_CLIENTS)]
    load = [0] * N_CLIENTS
    for s in tr_sess:
        i = int(np.argmin(load))
        clients[i].append(s)
        load[i] += counts[s]
    return by_sess, counts, val_sess, clients, load


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    by_sess, counts, val_sess, clients, load = build_split()
    clear_sess = set(by_sess)

    print('clear subset: %d folders / %d sessions / %d frames'
          % (len(CLEAR), len(clear_sess), sum(counts.values())), flush=True)
    print('VAL sessions (%d): %s' % (len(val_sess),
          ', '.join('%s[%d]' % (s, counts[s]) for s in val_sess)), flush=True)
    for i, ss in enumerate(clients):
        print('  client %d: %d sessions, %d frames  (%s)'
              % (i, len(ss), load[i], ', '.join(ss)), flush=True)

    # -- public pretrain pool: session-disjoint from every clear session ------
    allf = sorted(d for d in os.listdir(TRAIN_ROOT)
                  if os.path.isdir(os.path.join(TRAIN_ROOT, d)))
    pool = [f for f in allf if sess_of(f) not in clear_sess]
    rng = random.Random(SEED)
    rng.shuffle(pool)
    pub = pool[:PUBLIC_FOLDERS]
    assert not (set(map(sess_of, pub)) & clear_sess), 'public pool leaks a clear session'
    print('\npublic pretrain: %d folders from a %d-folder session-disjoint pool'
          % (len(pub), len(pool)), flush=True)

    items = []
    for f in pub:
        items.extend(load_folder(os.path.join(TRAIN_ROOT, f)))
    rng.shuffle(items)
    t0 = time.time()
    pim, pobj, pbox = encode(items)
    print('  %d public frames encoded in %.0fs' % (len(items), time.time() - t0), flush=True)
    pim, pobj, pbox = pim.to(DEV), pobj.to(DEV), pbox.to(DEV)

    # -- val, encoded once and subsampled ------------------------------------
    vitems = []
    for s in val_sess:
        for f in by_sess[s]:
            vitems.extend(load_folder(os.path.join(TRAIN_ROOT, f)))
    step = max(1, len(vitems) // VAL_TARGET)
    vitems = vitems[::step][:VAL_TARGET]
    vim, vobj, vbox = encode(vitems)
    print('  val: %d frames (every %dth from %d sessions)'
          % (vim.shape[0], step, len(val_sess)), flush=True)

    net = FullNet().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    B = 128
    t0 = time.time()
    for ep in range(PUBLIC_EPOCHS):
        net.train()
        idx = torch.randperm(pim.shape[0], device=DEV)
        for i in range(0, len(idx), B):
            j = idx[i:i + B]
            opt.zero_grad()
            loss_fn(net(pim[j].float() / 255.0), pobj[j], pbox[j]).backward()
            opt.step()
        c, i25, i50 = eval_full(net.trunk, net.head, vim, vobj, vbox)
        print('  public epoch %d/%d: val cell=%.3f IoU.25=%.3f IoU.50=%.3f (%.0fs)'
              % (ep + 1, PUBLIC_EPOCHS, c, i25, i50, time.time() - t0), flush=True)
    torch.save({'trunk': net.trunk.state_dict(), 'head': net.head.state_dict()},
               os.path.join(OUT, 'public_pretrained.pt'))
    del pim, pobj, pbox
    torch.cuda.empty_cache()

    # -- cache frozen s1->s3 features (the cut_s3 point) for clients + val ----
    net.trunk.eval()
    for p in net.trunk.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def cache(imgs_u8, tag):
        fs = []
        for s0 in range(0, imgs_u8.shape[0], 256):
            x = imgs_u8[s0:s0 + 256].to(DEV).float() / 255.0
            fs.append(net.trunk.s3(net.trunk.s2(net.trunk.s1(x))).half().cpu())
        f = torch.cat(fs)
        torch.save(f, os.path.join(OUT, 'feat_%s.pt' % tag))
        return tuple(f.shape)

    enc, sizes = {}, []
    for i, ss in enumerate(clients):
        its = []
        for s in ss:
            for f in by_sess[s]:
                its.extend(load_folder(os.path.join(TRAIN_ROOT, f)))
        enc[i] = encode(its)
        sizes.append(enc[i][0].shape[0])
    n_eq = min(sizes)
    print('\nequalising clients to %d frames each (raw %s)' % (n_eq, sizes), flush=True)
    for i in range(N_CLIENTS):
        im, o, b = enc[i]
        g = torch.Generator().manual_seed(SEED + i)
        sel = torch.randperm(im.shape[0], generator=g)[:n_eq]
        shp = cache(im[sel], 'client_%d' % i)
        torch.save({'obj': o[sel], 'box': b[sel]}, os.path.join(OUT, 'tgt_client_%d.pt' % i))
        print('  cached client %d: %s' % (i, shp), flush=True)
        del enc[i]
    shp = cache(vim, 'val')
    torch.save({'obj': vobj, 'box': vbox}, os.path.join(OUT, 'tgt_val.pt'))
    print('  cached val: %s' % (shp,), flush=True)

    json.dump({'val_sessions': val_sess, 'client_sessions': clients,
               'public_folders': pub, 'n_per_client': int(n_eq),
               'n_val': int(vim.shape[0])},
              open(os.path.join(OUT, 'manifest.json'), 'w'), indent=1)
    print('\nSTAGE1-CLEAR DONE  (%d x %d train, %d val)'
          % (N_CLIENTS, n_eq, vim.shape[0]), flush=True)
