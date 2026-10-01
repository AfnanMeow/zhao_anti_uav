# =============================================================================
# Defence figures for P3-H, from results/p3h_core.json (+ ladder/armG if present).
# Draws whatever configs exist, so it can be re-run as data arrives.
#
#   figD1_defence_summary   damage / TPR / FPR per defence, sigma setting, attack
#   figD2_acc_over_rounds   IoU.25 per round per defence (vs clean baseline)
#   figH1_norm_band         honest vs malicious norm deviation, raw and /sigma
#   figH2_norm_vs_sigma     per-client norm against sigma, raw and /sigma
#   (figH1, figH2 also as *_colored: blue honest / orange malicious)
#   defence_summary.csv     every number behind figD1 (table view)
#
# Colour = defence (fixed slots of the reference categorical palette, never
# cycled); sigma setting and attack are panels. Honest/malicious use neutral
# grey / violet so they never collide with a defence colour. PDF + real PNG.
# =============================================================================
import os, csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)

CORE = json.load(open(os.path.join(RES, 'p3h_core.json')))['results']

MODES = ['uniform', 'hetero', 'adaptive']
MODE_LBL = {'uniform': 'Uniform σ\n(all clients 1.83)',
            'hetero': 'Heterogeneous σ\n(per-client budgets)',
            'adaptive': 'Adaptive σ\n(per-client, per-round)'}
ATTACKS = ['rmd', 'mpelm', 'mpelm_scale']
ATT_LBL = {'rmd': 'RMD (random update)', 'mpelm': 'α-MPELM (direction attack)',
           'mpelm_scale': 'Band-exploiting α-MPELM'}
DEFS = ['none', 'norm', 'norm_sn', 'mix', 'mix_sn']
DEF_LBL = {'none': 'No defence', 'norm': 'Norm', 'norm_sn': 'Norm-σ (normalised)',
           'mix': 'Mix', 'mix_sn': 'Mix-σ (normalised)'}
# reference palette slots 1-5 (light mode); adjacent pairs validated in the skill
DEF_COL = {'none': '#2a78d6', 'norm': '#eb6834', 'norm_sn': '#1baf7a',
           'mix': '#eda100', 'mix_sn': '#e87ba4'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
HONEST, MALICIOUS = '#b5b3ad', '#2b2a27'      # neutral light / dark: never a defence colour

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
    'ytick.color': INK2, 'text.color': INK, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'axes.axisbelow': True, 'legend.frameon': False,
    'figure.dpi': 110, 'savefig.dpi': 200, 'pdf.fonttype': 42})


def get(mode, att, dfn):
    m = 0 if att == 'none' else 3
    for k, v in CORE.items():
        if k.startswith('%s|%s|%s|m%d|lie1|cap0' % (mode, att, dfn, m)) and '|on' not in k \
                and '|track' not in k:
            return v
    return None


def fin(r):
    return float(np.mean([x['acc'] for x in r['rounds'][-3:]]))


def rates(recs):
    tp = fp = pos = neg = 0
    for r in recs:
        for x in r['rounds']:
            for i, mal in enumerate(x['malicious']):
                rj = x['rejected'][i]
                pos += mal; neg += not mal; tp += mal and rj; fp += (not mal) and rj
    return tp / max(pos, 1), fp / max(neg, 1)


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, '%s.%s' % (name, ext)), bbox_inches='tight',
                    facecolor='white')
    plt.close(fig)
    print('wrote figures/%s.{pdf,png}' % name)


# ── table ───────────────────────────────────────────────────────────────────
rows = []
for att in ATTACKS:
    for mode in MODES:
        clean = get(mode, 'none', 'none')
        for d in DEFS + ['cos', 'cos_sn', 'krum', 'median', 'trimmed']:
            recs = get(mode, att, d)
            if not recs or not clean:
                continue
            n = min(len(recs), len(clean))
            dmg = [fin(recs[t]) - fin(clean[t]) for t in range(n)]
            tpr, fpr = rates(recs)
            rows.append(dict(attack=att, sigma=mode, defence=d, n=n,
                             iou=np.mean([fin(r) for r in recs[:n]]),
                             clean_iou=np.mean([fin(r) for r in clean[:n]]),
                             damage=np.mean(dmg), damage_min=min(dmg), damage_max=max(dmg),
                             tpr=tpr if d != 'none' else np.nan,
                             fpr=fpr if d != 'none' else np.nan))
with open(os.path.join(FIG, 'defence_summary.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ['empty'])
    w.writeheader()
    for r in rows:
        w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
print('wrote figures/defence_summary.csv (%d rows)' % len(rows))
R = {(r['attack'], r['sigma'], r['defence']): r for r in rows}

# ── figD1: defence summary ──────────────────────────────────────────────────
atts = [a for a in ATTACKS if any(k[0] == a for k in R)]
if atts:
    metrics = [('damage', 'IoU.25 change vs clean\n(0 = no harm)'),
               ('tpr', 'Attackers caught (TPR)'), ('fpr', 'Honest clients rejected (FPR)')]
    fig, axes = plt.subplots(3, len(atts), figsize=(3.6 * len(atts), 7.6), squeeze=False)
    w = 0.16
    for c, att in enumerate(atts):
        for rr, (met, lbl) in enumerate(metrics):
            ax = axes[rr, c]
            for j, d in enumerate(DEFS):
                xs, ys, lo, hi = [], [], [], []
                for mi, mode in enumerate(MODES):
                    r = R.get((att, mode, d))
                    if r is None or np.isnan(r[met]):
                        continue
                    xs.append(mi + (j - 2) * w); ys.append(r[met])
                    if met == 'damage':
                        lo.append(r[met] - r['damage_min']); hi.append(r['damage_max'] - r[met])
                ax.bar(xs, ys, w * 0.88, color=DEF_COL[d], label=DEF_LBL[d],
                       edgecolor='white', linewidth=0.6)
                if met == 'damage' and xs:
                    ax.errorbar(xs, ys, yerr=[lo, hi], fmt='none', ecolor=INK2,
                                elinewidth=0.8, capsize=1.5)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_xticks(range(3))
            ax.set_xticklabels([MODE_LBL[m].split('\n')[0] for m in MODES] if rr == 2 else [])
            if met != 'damage':
                ax.set_ylim(0, 1.05)
            if c == 0:
                ax.set_ylabel(lbl)
            if rr == 0:
                ax.set_title(ATT_LBL[att], loc='left', fontweight='bold')
    from matplotlib.patches import Patch
    fig.legend([Patch(color=DEF_COL[d]) for d in DEFS], [DEF_LBL[d] for d in DEFS],
               loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('How well each defence works: 30% malicious clients, 3 trials '
                 '(bars = mean, whiskers = trial range)', x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save(fig, 'figD1_defence_summary')

# ── figD2: accuracy over rounds ─────────────────────────────────────────────
if atts:
    fig, axes = plt.subplots(len(atts), 3, figsize=(10.5, 2.9 * len(atts)), squeeze=False,
                             sharey=True)
    for rr, att in enumerate(atts):
        for c, mode in enumerate(MODES):
            ax = axes[rr, c]
            clean = get(mode, 'none', 'none')
            if clean:
                y = np.mean([[x['acc'] for x in r['rounds']] for r in clean], 0)
                ax.plot(range(1, len(y) + 1), y, color=INK2, lw=1.4, ls='--', label='Clean (no attack)')
            for d in DEFS:
                recs = get(mode, att, d)
                if recs:
                    y = np.mean([[x['acc'] for x in r['rounds']] for r in recs], 0)
                    ax.plot(range(1, len(y) + 1), y, color=DEF_COL[d], lw=2, label=DEF_LBL[d])
            if rr == 0:
                ax.set_title(MODE_LBL[mode].replace('\n', ' '), loc='left')
            if c == 0:
                ax.set_ylabel('%s\nIoU.25' % ATT_LBL[att])
            if rr == len(atts) - 1:
                ax.set_xlabel('Communication round')
    h, l = [], []
    for ax in axes.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, 'figD2_acc_over_rounds')

# ── figH1: the norm band ────────────────────────────────────────────────────
src = {m: get(m, 'rmd', 'none') for m in MODES}
# Two versions: neutral (honest/malicious never collide with a defence colour)
# and _colored (palette slots 1-2, blue/orange, CVD-validated pair) on request.
for HONEST, MALICIOUS, SUFFIX in (('#b5b3ad', '#2b2a27', ''),
                                  ('#2a78d6', '#eb6834', '_colored')):
    if any(src.values()):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
        rng = np.random.default_rng(0)
        for ax, norm_by_sigma, ttl, thr in (
                (axes[0], False, 'Raw update norm (what the Norm detector sees)', 'thr_norm'),
                (axes[1], True, 'Norm ÷ declared σ (what Norm-σ sees)', 'thr_norm_sn')):
            for yi, mode in enumerate(MODES):
                recs = src[mode]
                if not recs:
                    continue
                dev_h, dev_m, half = [], [], []
                for r in recs:
                    for x in r['rounds']:
                        v = np.array(x['norm']) / (np.array(x['sigma_declared']) if norm_by_sigma else 1)
                        med = np.median(v)
                        t = x[thr]
                        half.append((t['hi'] - t['lo']) / 2 / t['median'])
                        for vi, mal in zip(v, x['malicious']):
                            (dev_m if mal else dev_h).append(100 * (vi / med - 1))
                hw = 100 * float(np.median(half))
                ax.fill_betweenx([yi - 0.4, yi + 0.4], -hw, hw, color=GRID, lw=0, zorder=0)
                for dev, col, lab in ((dev_h, HONEST, 'Honest client'), (dev_m, MALICIOUS, 'Malicious (RMD)')):
                    ax.scatter(dev, yi + rng.uniform(-0.28, 0.28, len(dev)), s=9, color=col,
                               alpha=0.55, lw=0, label=lab if yi == 0 else None)
                ax.text(hw, yi + 0.42, ' accepted: ±%.2f%%' % hw, ha='left', va='bottom',
                        fontsize=7.5, color=INK2)
            ax.set_ylim(-0.6, 2.75)
            ax.set_yticks(range(3)); ax.set_yticklabels([MODE_LBL[m].split('\n')[0] for m in MODES])
            ax.set_xlabel('Deviation from the round median (%)')
            ax.set_title(ttl, loc='left'); ax.grid(axis='y', visible=False)
        fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3,
                   bbox_to_anchor=(0.5, -0.04))
        fig.suptitle('Heterogeneous budgets widen the honest band so attackers sit inside it; '
                     'dividing by σ closes it again (grey = detector accept band)',
                     x=0.01, ha='left', fontsize=11)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        save(fig, 'figH1_norm_band' + SUFFIX)

    # ── figH2: norm vs sigma ────────────────────────────────────────────────────
    hs = [m for m in ('hetero', 'adaptive') if src.get(m)]
    if hs:
        fig, axes = plt.subplots(len(hs), 2, figsize=(9, 3.2 * len(hs)), squeeze=False)
        for rr, mode in enumerate(hs):
            for c, norm_by_sigma in enumerate((False, True)):
                ax = axes[rr, c]
                for mal, col, lab in ((False, HONEST, 'Honest'), (True, MALICIOUS, 'Malicious (RMD)')):
                    xs, ys = [], []
                    for r in src[mode]:
                        for x in r['rounds'][2:]:            # skip the first-round transient
                            for n, s, m in zip(x['norm'], x['sigma_true'], x['malicious']):
                                if m == mal:
                                    xs.append(s); ys.append(n / s if norm_by_sigma else n)
                    ax.scatter(xs, ys, s=10, color=col, alpha=0.6, lw=0, label=lab)
                ax.set_xlabel('Client σ this round')
                ax.set_ylabel('Norm ÷ σ' if norm_by_sigma else 'Update norm')
                ax.set_title('%s — %s' % (MODE_LBL[mode].split('\n')[0],
                                          'normalised' if norm_by_sigma else 'raw'), loc='left')
        axes[0, 0].legend(loc='upper left')
        fig.tight_layout()
        save(fig, 'figH2_norm_vs_sigma' + SUFFIX)


# ── figH3: the defence's reach and limits (ladder, Arm G, sigma lying) ───────
def _load(tag):
    p = os.path.join(RES, 'p3h_%s.json' % tag)
    return json.load(open(p))['results'] if os.path.exists(p) else {}


def shadow_rates(recs, field, start=0):
    tp = fp = pos = neg = 0
    for r in recs:
        for x in r['rounds'][start:]:
            rej = set(x[field])
            for i, mal in enumerate(x['malicious']):
                pos += mal; neg += not mal; tp += mal and i in rej; fp += (not mal) and i in rej
    return tp / max(pos, 1), fp / max(neg, 1)


def arm_g(recs, onset=5, k=3.0):          # same rule as analyse_round2.py / PREREG
    tp = fp = pos = neg = 0
    for r in recs:
        W = np.array([x['norm'] for x in r['rounds'][:onset]])
        base, rsd = np.median(W, 0), float(np.median(W.std(0) / W.mean(0)))
        for x in r['rounds'][onset:]:
            flag = np.abs(np.array(x['norm']) / base - 1) > k * rsd
            for i, mal in enumerate(x['malicious']):
                pos += mal; neg += not mal
                tp += bool(mal and flag[i]); fp += bool((not mal) and flag[i])
    return tp / max(pos, 1), fp / max(neg, 1)


LAD, ARMG, ARMF = _load('ladder'), _load('armG'), _load('armF')
HIST_COL = '#008300'                       # palette slot 6: the per-client history detector
if LAD and ARMG and ARMF:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    # (a) ladder
    ax = axes[0, 0]
    pts = [(0.0, get('uniform', 'rmd', 'norm'))] + sorted(
        (float(k.split('|')[0].split(':')[1]), v) for k, v in LAD.items() if '|rmd|norm|' in k)
    xs = [100 * np.mean([np.std(r['rounds'][0]['sigma_true']) / np.mean(r['rounds'][0]['sigma_true'])
                         for r in v]) for _, v in pts]
    for field, d in (('shadow_norm_rej', 'norm'), ('shadow_norm_sn_rej', 'norm_sn')):
        ys = [shadow_rates(v, field)[0] for _, v in pts]
        ax.plot(xs, ys, color=DEF_COL[d], lw=2, marker='o', ms=6, label=DEF_LBL[d])
    ax.set_xlabel('Cross-client σ spread (relative sd, %)'); ax.set_ylabel('Attackers caught (TPR)')
    ax.set_ylim(-0.03, 1.05)
    ax.set_title('(a) Controlled σ ladder, RMD', loc='left')
    ax.legend(loc='center right')
    # (b, c) Arm G vs cross-client detectors
    w = 0.26
    for ax, idx, ttl, yl in ((axes[0, 1], 0, '(b) Per-client history vs cross-client: caught',
                              'Attackers caught (TPR)'),
                             (axes[1, 0], 1, '(c) … and honest clients wrongly rejected',
                              'Honest rejected (FPR)')):
        for j, (lbl, col, fn) in enumerate((
                ('Norm', DEF_COL['norm'], lambda v: shadow_rates(v, 'shadow_norm_rej', 5)),
                ('Norm-σ (normalised)', DEF_COL['norm_sn'], lambda v: shadow_rates(v, 'shadow_norm_sn_rej', 5)),
                ('Per-client history (Arm G)', HIST_COL, arm_g))):
            ys = []
            for m in MODES:
                recs = [v for k, v in ARMG.items() if k.startswith(m + '|')]
                ys.append(fn(recs[0])[idx] if recs else np.nan)
            ax.bar(np.arange(3) + (j - 1) * w, ys, w * 0.88, color=col, label=lbl,
                   edgecolor='white', linewidth=0.6)
        ax.set_xticks(range(3)); ax.set_xticklabels([MODE_LBL[m].split('\n')[0] for m in MODES])
        ax.set_ylim(0, 1.05); ax.set_ylabel(yl); ax.set_title(ttl, loc='left')
    axes[1, 0].legend(*axes[0, 1].get_legend_handles_labels(), loc='upper left', fontsize=8)
    # (d) sigma lying
    ax = axes[1, 1]
    base = get('adaptive', 'rmd', 'norm_sn')
    for cap, ls, mk, lbl in ((0, '-', 'o', 'declared σ taken as-is'),
                             (1, '--', 's', 'declared σ capped to honest range')):
        pts = [(1.0, base)] + [(float(l), ARMF.get('adaptive|rmd|norm_sn|m3|lie%s|cap%d' % (l, cap)))
                               for l in ('0.8', '0.9', '1.1')]
        pts = sorted(p for p in pts if p[1])
        ax.plot([p[0] for p in pts], [rates(p[1])[0] for p in pts], color=DEF_COL['norm_sn'],
                lw=2, ls=ls, marker=mk, ms=6, label=lbl)
    ax.axvline(0.90, color=INK2, lw=0.8, ls=':')
    ax.text(0.905, 0.55, "RMD's true norm ratio\n(the matched lie)", fontsize=7.5, color=INK2)
    ax.set_xlabel('Attacker\'s declared σ ÷ true σ'); ax.set_ylabel('Norm-σ: attackers caught (TPR)')
    ax.set_ylim(-0.03, 1.05); ax.set_title('(d) σ-lying attacker, adaptive σ (capped = uncapped: lines overlap)', loc='left')
    ax.legend(loc='lower right', fontsize=8)
    fig.suptitle('Where each norm defence holds and where it breaks (30% malicious, 3 trials)',
                 x=0.01, ha='left', fontsize=11)
    fig.tight_layout()
    save(fig, 'figH3_defence_limits')


# ── figD3: direction-aware defences and robust aggregators (round 3) ─────────
EXTRA = ['none', 'norm_sn', 'cos', 'cos_sn', 'krum', 'median', 'trimmed']
EXTRA_LBL = {'none': 'No defence', 'norm_sn': 'Norm-σ', 'cos': 'Cosine',
             'cos_sn': 'Cosine + Norm-σ', 'krum': 'Krum', 'median': 'Median',
             'trimmed': 'Trimmed mean'}
DIR_COL, AGG_COL = '#4a3aa7', '#e34948'          # palette slots 7 and 8
EXTRA_COL = {'none': DEF_COL['none'], 'norm_sn': DEF_COL['norm_sn'], 'cos': DIR_COL,
             'cos_sn': DIR_COL, 'krum': AGG_COL, 'median': AGG_COL, 'trimmed': AGG_COL}
if any(get(m, a, 'cos') for m in MODES for a in ATTACKS):
    rows_att = ['none'] + ATTACKS
    row_lbl = {'none': 'Cost with NO attack', **ATT_LBL}
    fig, axes = plt.subplots(4, 3, figsize=(12, 11), sharey=True)
    for rr, att in enumerate(rows_att):
        for c, mode in enumerate(MODES):
            ax = axes[rr, c]
            clean = get(mode, 'none', 'none')
            ys, vals, cols, notes = [], [], [], []
            for yi, d in enumerate(EXTRA):
                if att == 'none' and d == 'none':
                    continue
                recs = get(mode, att, d) if not (att == 'none') else get(mode, 'none', d)
                if not recs or not clean:
                    continue
                n = min(len(recs), len(clean))
                dm = np.mean([fin(recs[t]) - fin(clean[t]) for t in range(n)])
                tpr, fpr = rates(recs)
                ys.append(yi); vals.append(dm); cols.append(EXTRA_COL[d])
                notes.append('' if d in ('none', 'median', 'trimmed') else
                             ('FPR %.2f' % fpr if att == 'none' else 'TPR %.2f  FPR %.2f' % (tpr, fpr)))
            ax.barh(ys, vals, 0.7, color=cols, edgecolor='white', linewidth=0.6)
            ax.axvline(0, color=INK2, lw=0.8)
            lo = min(vals + [0]); hi = max(vals + [0])
            for y, v, t in zip(ys, vals, notes):
                if t:
                    ax.text(max(v, 0) + (hi - lo) * 0.03, y, t, va='center', fontsize=7, color=INK2)
            ax.set_xlim(lo - (hi - lo) * 0.05, hi + (hi - lo) * 0.55)
            ax.set_yticks(range(len(EXTRA))); ax.set_yticklabels([EXTRA_LBL[d] for d in EXTRA])
            ax.invert_yaxis(); ax.grid(axis='y', visible=False)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            if rr == 0:
                ax.set_title(MODE_LBL[mode].replace('\n', ' '), loc='left')
            if c == 0:
                ax.set_ylabel(row_lbl[att], fontweight='bold')
            if rr == 3:
                ax.set_xlabel('IoU.25 change vs clean (0 = no harm)')
    from matplotlib.patches import Patch
    fig.legend([Patch(color=DEF_COL['none']), Patch(color=DEF_COL['norm_sn']), Patch(color=DIR_COL),
                Patch(color=AGG_COL)],
               ['No defence', 'Norm-σ (best norm defence)', 'Direction detectors',
                'Robust aggregators'], loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Direction-aware defences and robust aggregators (30% malicious, 3 trials)',
                 x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    save(fig, 'figD3_direction_defences')

# ── figH5: damage vs sigma spread (round 3 ladder) ───────────────────────────
if LAD and any(k.startswith('ladder') and '|mpelm_scale|' in k for k in LAD):
    def lget(level, att, dfn):
        return LAD.get('%s|%s|%s|m%d|lie1|cap0' % (level, att, dfn, 0 if att == 'none' else 3))
    levels = sorted({k.split('|')[0] for k in LAD}, key=lambda s: float(s.split(':')[1]))
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for d in ('none', 'norm', 'norm_sn'):
        xs, ys, lo, hi = [], [], [], []
        pairs = [(get('uniform', 'mpelm_scale', d), get('uniform', 'none', 'none'))] +                 [(lget(l, 'mpelm_scale', d), lget(l, 'none', 'none')) for l in levels]
        spread = [0.0] + [100 * np.mean([np.std(r['rounds'][0]['sigma_true']) /
                                         np.mean(r['rounds'][0]['sigma_true'])
                                         for r in (lget(l, 'none', 'none') or [])] or [np.nan])
                          for l in levels]
        for sp, (a, c) in zip(spread, pairs):
            if a and c:
                dm = [fin(a[t]) - fin(c[t]) for t in range(min(len(a), len(c)))]
                xs.append(sp); ys.append(np.mean(dm)); lo.append(np.mean(dm) - min(dm))
                hi.append(max(dm) - np.mean(dm))
        ax.errorbar(xs, ys, yerr=[lo, hi], color=DEF_COL[d], lw=2, marker='o', ms=6,
                    capsize=2, elinewidth=0.8, label=DEF_LBL[d])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel('Cross-client σ spread (relative sd, %)')
    ax.set_ylabel('IoU.25 change vs clean')
    ax.set_title('Band-exploiting attacker: damage vs σ spread', loc='left')
    ax.legend(loc='center right')
    fig.tight_layout()
    save(fig, 'figH5_damage_vs_spread')


# ── figD4: how the cosine detector stops alpha-MPELM (deterrence), and its limit ─
if get('hetero', 'mpelm', 'cos'):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axes[0]
    for d, col, lbl in (('none', DEF_COL['none'], 'No defence'),
                        ('norm_sn', DEF_COL['norm_sn'], 'Norm-σ'),
                        ('cos', DIR_COL, 'Cosine')):
        recs = get('hetero', 'mpelm', d)
        g = np.mean([[x['gamma'] for x in r['rounds']] for r in recs], 0)
        ax.plot(range(1, len(g) + 1), g, color=col, lw=2, marker='o', ms=4, label=lbl)
    ax.axhline(0.5, color=INK2, lw=0.8, ls=':')
    ax.text(15, 0.52, 'γ > 0.5: update is flipped (harmful)', ha='right', fontsize=7.5, color=INK2)
    ax.set_xlabel('Communication round'); ax.set_ylabel('Attacker poisoning degree γ')
    ax.set_ylim(0, 1.05)
    ax.set_title('(a) Getting caught makes α-MPELM back off', loc='left')
    ax.legend(loc='upper left')
    ax = axes[1]
    recs = get('hetero', 'mpelm', 'cos')
    R = len(recs[0]['rounds'])
    for mal, col, lbl in ((False, HONEST, 'Honest clients'), (True, MALICIOUS, 'α-MPELM attackers')):
        y = [np.mean([c for r in recs for c, b in zip(r['rounds'][i]['cos'], r['rounds'][i]['malicious'])
                      if b == mal]) for i in range(R)]
        ax.plot(range(1, R + 1), y, color=col, lw=2, marker='o', ms=4, label=lbl)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_yscale('symlog', linthresh=1e-3)
    ax.set_xlabel('Communication round'); ax.set_ylabel('Cosine with the other clients\' mean')
    ax.set_title('(b) The direction signal fades as training converges', loc='left')
    ax.legend(loc='upper right')
    fig.suptitle('Heterogeneous σ: the cosine detector stops α-MPELM by deterrence in round 1, '
                 'not by sustained detection', x=0.01, ha='left', fontsize=11)
    fig.tight_layout()
    save(fig, 'figD4_cosine_deterrence')


# ── figD5: does the cosine detector catch, or only deter? (round 4) ──────────
R4P = os.path.join(RES, 'p3h_round4.json')
if os.path.exists(R4P):
    R4 = json.load(open(R4P))['results']
    VARS = [('mpelm_fixed', 0, 'Never backs off, attacks from round 1', '-', 'o'),
            ('mpelm', 5, 'α-MPELM (backs off), attacks from round 6', '--', 's'),
            ('mpelm_fixed', 5, 'Never backs off, attacks from round 6', ':', '^')]
    k4 = lambda a, d, o: 'hetero|%s|%s|m3|lie1|cap0%s' % (a, d, '|on%d' % o if o else '')
    clean = get('hetero', 'none', 'none')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for a, o, lbl, ls, mk in VARS:
        r = R4[k4(a, 'cos', o)]
        y = np.mean([[x['tp'] for x in t['rounds']] for t in r], 0)
        ax.plot(range(1, len(y) + 1), y, color=DIR_COL, ls=ls, marker=mk, ms=5, lw=2, label=lbl)
    ax.set_xlabel('Communication round'); ax.set_ylabel('Attackers caught by Cosine (of 3)')
    ax.set_ylim(-0.1, 3.2)
    ax.set_title('(a) Cosine catches direction attacks only early', loc='left')
    ax.legend(loc='upper right', fontsize=8)
    ax = axes[1]
    ds = [('none', DEF_COL['none'], None), ('norm_sn', DEF_COL['norm_sn'], None),
          ('cos', DIR_COL, None), ('cos_sn', DIR_COL, '///')]
    w = 0.2
    for j, (d, col, hatch) in enumerate(ds):
        ys = []
        for a, o, *_ in VARS:
            r = R4[k4(a, d, o)]
            ys.append(np.mean([fin(r[t]) - fin(clean[t]) for t in range(min(len(r), len(clean)))]))
        ax.bar(np.arange(3) + (j - 1.5) * w, ys, w * 0.88, color=col, hatch=hatch,
               edgecolor='white', linewidth=0.6,
               label={'none': 'No defence', 'norm_sn': 'Norm-σ', 'cos': 'Cosine',
                      'cos_sn': 'Cosine + Norm-σ'}[d])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['No back-off\nfrom round 1', 'α-MPELM\nfrom round 6', 'No back-off\nfrom round 6'])
    ax.set_ylabel('IoU.25 change vs clean')
    ax.set_title('(b) Damage: only the backing-off attacker is largely stopped', loc='left')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=8)
    fig.suptitle('Heterogeneous σ: harder direction attackers (round 4, 3 trials)',
                 x=0.01, ha='left', fontsize=11)
    fig.tight_layout()
    save(fig, 'figD5_direction_attack_limits')
