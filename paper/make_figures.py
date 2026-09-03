"""
Generate paper figures from the run CSVs + checkpoints.
Palette: Okabe-Ito (CVD-safe). One axis per panel; thin recessive marks; legends
present for >=2 series; direct labels where it helps. Saves PDF (vector) + PNG.
"""
import os, csv, sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import lambertw
import torch

RUN = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

OK = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", verm="#D55E00", purple="#CC79A7", grey="#999999")

mpl.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "lines.linewidth": 1.8, "font.family": "sans-serif",
})

from matplotlib.ticker import FuncFormatter, MaxNLocator
def kfmt(ax):  # step axis in thousands: 0, 10k, 20k...
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: (f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}")))

def rd(path):
    return list(csv.DictReader(open(path)))

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    plt.close(fig); print(f"  -> {name}")

# ---- the NMI law -----------------------------------------------------------
def law(t, A, lam, gam, t0, K):
    x = np.clip(gam*t + t0, 1e-6, None); return A*np.exp(-lam*x)*np.log(x) + K
def fit_law(t, y, lam_max=0.1):
    p, _ = curve_fit(law, t, y, p0=[y.max(), min(1e-4, lam_max/2), 1e-3, 1.0, y.min()],
                     bounds=([0,0,1e-6,1e-3,-1e3],[1e5,lam_max,1,1e4,1e3]), maxfev=200000)
    yh = law(t, *p); r2 = 1 - np.sum((y-yh)**2)/np.sum((y-y.mean())**2)
    return p, r2

def gamma_traj(run, layer, ch, iters):
    g = []
    for it in iters:
        ck = torch.load(f"{run}/ckpt_{it}.pt", map_location="cpu")
        g.append(float(ck["model"][f"transformer.h.{layer}.ln_1.weight"][ch].abs()))
    return np.array(g)

def dissoc(run, layer):
    r = rd(f"{run}/channel_traj_L{layer}.csv"); its=[int(x['iter']) for x in r]; C=len(r[0])-1
    M = np.array([[float(x[f'ch{c}']) for c in range(C)] for x in r])
    peaks = M.max(0); ma = int(np.argmax(peaks))
    def dc(v): ip=int(np.argmax(v)); return (v[ip]-v[-1])/(v[ip]+1e-9)*100
    G0 = torch.load(f"{run}/ckpt_{its[0]}.pt",map_location="cpu")["model"][f"transformer.h.{layer}.ln_1.weight"].abs().numpy()
    G1 = torch.load(f"{run}/ckpt_{its[-1]}.pt",map_location="cpu")["model"][f"transformer.h.{layer}.ln_1.weight"].abs().numpy()
    big = [c for c in range(C) if peaks[c] > peaks[ma]*0.15 and c != ma]
    ma_d = (G0[ma]-G1[ma])/G0[ma]*100 - dc(M[:,ma])
    typ = np.median([ (G0[c]-G1[c])/G0[c]*100 - dc(M[:,c]) for c in big ])
    return ma, ma_d, typ

# ===========================================================================
def fig1_emergence():
    r = rd(f"{RUN}/len_51m_40k/trajectory.csv")
    t = np.array([float(x['iter']) for x in r]); y = np.array([float(x['ratio_L5']) for x in r])
    p, r2 = fit_law(t, y)
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.plot(t, y, ".", ms=3, color=OK['sky'], alpha=0.5, label="measured")
    tt = np.linspace(t.min(), t.max(), 400)
    ax.plot(tt, law(tt, *p), "-", color=OK['blue'], lw=2, label=f"fitted law  ($R^2$={r2:.3f})")
    ax.set_xlabel("training step"); ax.set_ylabel("MA ratio  (max / median), layer 5")
    ax.set_title("Massive activations emerge and follow a 5-parameter law")
    ax.legend(frameon=False, loc="upper left"); kfmt(ax)
    save(fig, "fig1_emergence")

def fig2_global():
    # layer 2: the MA declines WITH the other channels (clean "global, not MA-special-in-shape" message;
    # the L5 protection is the dissociation story, Fig 5).
    r = rd(f"{RUN}/len_51m_40k/channel_traj_L2.csv"); t=np.array([int(x['iter']) for x in r]); C=len(r[0])-1
    M = np.array([[float(x[f'ch{c}']) for c in range(C)] for x in r]); peaks=M.max(0); ma=int(np.argmax(peaks))
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0))
    # A: many channels rise-then-decline, MA is the tail
    order = np.argsort(-peaks)
    for c in order[1:40]:
        if peaks[c] > 3: ax[0].plot(t, M[:,c], color=OK['grey'], lw=0.6, alpha=0.35)
    ax[0].plot(t, M[:,ma], color=OK['verm'], lw=2.2, label=f"MA channel (layer 2)")
    ax[0].plot([], [], color=OK['grey'], lw=1, label="other channels")
    ax[0].set_yscale("log"); ax[0].set_xlabel("training step"); ax[0].set_ylabel("channel magnitude at sink (|x|)")
    ax[0].set_title("The rise-and-decline is global"); ax[0].legend(frameon=False, loc="lower center"); kfmt(ax[0])
    # B: peak-timing vs amplitude, POOLED across layers 0-5 (global family; wide magnitude range)
    allmag=[]; allit=[]; ma_pt=None
    for L in range(6):
        rr=rd(f"{RUN}/len_51m_40k/channel_traj_L{L}.csv"); tt=np.array([int(x['iter']) for x in rr]); CC=len(rr[0])-1
        MM=np.array([[float(x[f'ch{c}']) for c in range(CC)] for x in rr]); pk=MM.max(0)
        for c in range(CC):
            if pk[c]>3:
                allmag.append(pk[c]); allit.append(tt[int(np.argmax(MM[:,c]))])
        if L==5:
            mc=int(np.argmax(pk)); ma_pt=(pk[mc], tt[int(np.argmax(MM[:,mc]))])
    ax[1].scatter(allmag, allit, s=9, color=OK['sky'], alpha=0.45, edgecolor='none')
    ax[1].scatter([ma_pt[0]],[ma_pt[1]], s=55, color=OK['verm'], zorder=5, label="dominant MA (layer 5)")
    # binned median trend
    am=np.array(allmag); ai=np.array(allit); order=np.argsort(am)
    bins=np.logspace(np.log10(am.min()),np.log10(am.max()),8)
    bx=[]; by=[]
    for i in range(len(bins)-1):
        m=(am>=bins[i])&(am<bins[i+1])
        if m.sum()>3: bx.append(np.sqrt(bins[i]*bins[i+1])); by.append(np.median(ai[m]))
    ax[1].plot(bx, by, "-", color=OK['black'], lw=1.4, label="binned median")
    ax[1].set_xscale("log"); ax[1].set_xlabel("channel peak magnitude (all layers)"); ax[1].set_ylabel("peak location (step)")
    ax[1].set_title("Peak timing scales with amplitude"); ax[1].legend(frameon=False, loc="upper left")
    fig.tight_layout(); save(fig, "fig2_global_dynamic")

def fig3_causal():
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0))
    # A: causal arms -- GLOBAL scale (median |activation|), the quantity the global-turnover claim is about.
    # (Single MA channel ch207 is noisier: C is ~flat on it while it declines on the median -- noted in text.)
    arms=[("A_wdON_sched","WD on + schedule",OK['blue'],"-"),
          ("B_wdOFF_sched","WD off",OK['verm'],"-"),
          ("C_wdON_constLR","WD on, LR constant",OK['green'],"--")]
    for tag,lab,col,ls in arms:
        d=rd(f"{RUN}/len_51m_40k/causal{tag}.csv"); it=[int(x['iter']) for x in d]; v=[float(x['median']) for x in d]
        ax[0].plot(it, v, ls, color=col, label=lab)
    ax[0].set_xlabel("training step"); ax[0].set_ylabel("median |activation|  (global scale)")
    ax[0].set_title("Weight decay drives the global decline"); ax[0].legend(frameon=False, loc="center left"); kfmt(ax[0])
    # B: dose-response peak location vs lambda
    lams=[0.05,0.1,0.2,0.4]; tp=[]
    for l in lams:
        d=rd(f"{RUN}/len_51m_40k/lam{l}.csv"); it=np.array([int(x['iter']) for x in d]); v=np.array([float(x['ch207']) for x in d])
        vs=np.convolve(v,np.ones(3)/3,mode='same'); tp.append(it[int(np.argmax(vs))])
    tp=np.array(tp); lams=np.array(lams)
    ax[1].plot(lams, tp, "o", ms=7, color=OK['orange'])
    b,a=np.polyfit(np.log(lams), tp, 1)
    xx=np.linspace(lams.min(),lams.max(),50); ax[1].plot(xx, a+b*np.log(xx), "-", color=OK['black'], lw=1.2,
        label=f"$t_{{peak}}\\approx a-b\\ln\\lambda$\n(corr={np.corrcoef(np.log(lams),tp)[0,1]:.3f})")
    ax[1].set_xscale("log"); ax[1].set_xticks([0.05,0.1,0.2,0.4]); ax[1].set_xticklabels(["0.05","0.1","0.2","0.4"]); ax[1].minorticks_off(); ax[1].set_xlabel("weight-decay coefficient  $\\lambda$")
    ax[1].set_ylabel("MA peak location (step)")
    ax[1].set_title("Dose-response: $\\lambda$ sets the peak"); ax[1].legend(frameon=False, loc="upper right")
    fig.tight_layout(); save(fig, "fig3_weight_decay")

def fig4_regulation():
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0))
    # A: declining-regime kick-up recovery, decay ON vs OFF (near-identical -> decay-independent)
    for wd,col,lab in [("0.1",OK['blue'],"decay ON"),("0",OK['orange'],"decay OFF")]:
        d=rd(f"{RUN}/len_51m_40k/decay_test/ceiling_30000_L2_a2.0_lr1e-04_wd{wd}.csv")
        s=[int(x['step']) for x in d]; ma=[float(x['ma_mag']) for x in d]; ctrl=[float(x['ctrl_mag']) for x in d]
        ax[0].plot(s, ma, "-", color=col, label=f"kicked MA ({lab})")
    ax[0].plot(s, ctrl, ":", color=OK['grey'], label="control (setpoint)")
    ax[0].set_xlabel("resume step"); ax[0].set_ylabel("MA magnitude")
    ax[0].set_title("Declining-regime ceiling is decay-independent"); ax[0].legend(frameon=False, loc="upper right")
    # B: specificity bars — gap closed, MA vs random, decay on/off
    cats=["MA\ndecay ON","MA\ndecay OFF","random\ndecay ON","random\ndecay OFF"]
    vals=[62,63,20,21]; cols=[OK['blue'],OK['orange'],OK['sky'],OK['yellow']]
    ax[1].bar(range(4), vals, color=cols, width=0.7)
    for i,v in enumerate(vals): ax[1].text(i, v+1.5, f"{v}%", ha='center', fontsize=8)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(cats, fontsize=7.5)
    ax[1].set_ylabel("gap-to-control closed (%)"); ax[1].set_ylim(0,75)
    ax[1].set_title("MA-specific and decay-independent")
    fig.tight_layout(); save(fig, "fig4_regulation")

def fig5_dissociation():
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0))
    # A: LN L5 MA — activation preserved vs gamma collapses (normalized to peak, one axis)
    run=f"{RUN}/len_51m_40k"; r=rd(f"{run}/channel_traj_L5.csv"); its=[int(x['iter']) for x in r]
    act=np.array([float(x['ch207']) for x in r]); gam=gamma_traj(run,5,207,its)
    t=np.array(its)
    ax[0].plot(t, act/act.max(), "-", color=OK['verm'], label="activation |x|  (backward role, rms)")
    ax[0].plot(t, gam/gam.max(), "-", color=OK['blue'], label="LayerNorm gain $\\gamma$  (forward role)")
    ax[0].set_xlabel("training step"); ax[0].set_ylabel("fraction of peak"); ax[0].set_ylim(0,1.08)
    ax[0].set_title("Raw / functional dissociation (MA, layer 5)"); ax[0].legend(frameon=False, loc="lower left"); kfmt(ax[0])
    # B: dissociation across conditions
    conds=[("len_51m_40k","LayerNorm\nseed 1337"),("rms_51m_40k","RMSNorm\nseed 1337"),("seed2024_51m_40k","LayerNorm\nseed 2024")]
    ma_ds=[]; typ_ds=[]
    for run_name,_ in conds:
        _,md,td=dissoc(f"{RUN}/{run_name}",5); ma_ds.append(md); typ_ds.append(td)
    x=np.arange(3); w=0.36
    ax[1].bar(x-w/2, ma_ds, w, color=OK['verm'], label="MA channel")
    ax[1].bar(x+w/2, typ_ds, w, color=OK['grey'], label="typical big channel")
    ax[1].axhline(0, color='k', lw=0.6)
    for i,v in enumerate(ma_ds): ax[1].text(i-w/2, v+2, f"+{v:.0f}", ha='center', fontsize=8)
    ax[1].set_xticks(x); ax[1].set_xticklabels([c[1] for c in conds], fontsize=7.5)
    ax[1].set_ylabel("dissociation  ($\\gamma$-decline $-$ act-decline, pp)")
    ax[1].set_ylim(-58, 66); ax[1].set_title("MA-specific, across architectures & seeds"); ax[1].legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0,0.82))
    fig.tight_layout(); save(fig, "fig5_dissociation")

def fig4_controls():
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0))
    # (a) g / (1/rms prediction) per condition -- MA on the line with other large channels
    d = rd(f"{RUN}/len_51m_40k/floor_test.csv")
    over = [float(x['g_over_pred']) for x in d]; names = [x['cond'] for x in d]
    cols = [OK['blue'] if n == 'control' else (OK['verm'] if n == 'ablate_MA' else OK['grey']) for n in names]
    lab = ['control' if n == 'control' else ('MA' if n == 'ablate_MA' else n.replace('ablate_ch', 'ch')) for n in names]
    ax[0].bar(range(len(d)), over, color=cols, width=0.72)
    ax[0].axhline(1.0, color='k', lw=0.8, ls=':')
    ax[0].set_xticks(range(len(d))); ax[0].set_xticklabels(lab, rotation=45, ha='right', fontsize=7)
    ax[0].set_ylabel(r"$g\ /\ (1/\mathrm{rms})$ prediction"); ax[0].set_ylim(0, 1.15)
    ax[0].plot([], [], 's', color=OK['verm'], label="ablate MA"); ax[0].plot([], [], 's', color=OK['grey'], label="ablate other large ch")
    ax[0].legend(frameon=False, loc="lower center", fontsize=7.5)
    ax[0].set_title(r"Gradient obeys $g\propto1/\mathrm{rms}$; MA on the line")
    # (b) direction test: matched-rms kicks pulled back equally regardless of x_hat distortion
    e = rd(f"{RUN}/len_51m_40k/direction_test.csv")
    s = [int(x['step']) for x in e]
    ax[1].plot(s, [float(x['rmsA']) for x in e], "-", color=OK['verm'], label=r"kick MA (distorts $\hat x$ $11\times$)")
    ax[1].plot(s, [float(x['rmsB']) for x in e], "-", color=OK['blue'], label=r"kick top-16 (preserves $\hat x$)")
    ax[1].plot(s, [float(x['rms_ctrl']) for x in e], ":", color=OK['grey'], label="control")
    ax[1].set_xlabel("resume step"); ax[1].set_ylabel("sink rms")
    ax[1].set_title("Matched-rms kicks: equal pullback")
    ax[1].legend(frameon=False, loc="upper right", fontsize=7.5)
    fig.tight_layout(); save(fig, "fig4_controls")


def fig5_scale():
    """Scale grows MA separation; training tokens grow the bimodal isolation (cliff).
    Our from-scratch models sit in the early-training continuum regime."""
    import collections
    prof = collections.defaultdict(dict); npar = {}
    for r in rd(f"{RUN}/scale_profile/profiles.csv"):
        prof[r["model"]][int(r["rank"])] = float(r["mag_over_median"]); npar[r["model"]] = int(r["n_params"])
    summ = {r["model"]: r for r in rd(f"{RUN}/scale_profile/summary.csv")}
    ours = sorted([m for m in npar if m.startswith("ours")], key=lambda m: npar[m])
    pyth = sorted([m for m in npar if m.startswith("pythia")], key=lambda m: npar[m])
    def cliff_of(pdict):                                             # biggest consecutive drop, top-15
        s = [pdict[k] for k in range(1, 16) if k in pdict]
        return max(s[i] / s[i + 1] for i in range(len(s) - 1))
    ours_cliffs = [cliff_of(prof[m]) for m in ours]
    tr410 = rd(f"{RUN}/scale_profile/pythia410m_traj.csv")
    tr14 = rd(f"{RUN}/scale_profile/pythia1.4b_traj.csv")

    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.15))
    # (a) MA separation (max/median) grows with scale -- both families
    for fam, c, name in [(pyth, OK["blue"], "Pythia (final)"), (ours, OK["verm"], "ours (from scratch)")]:
        px = [npar[m] for m in fam]; py = [float(summ[m]["max_over_median"]) for m in fam]
        ax[0].plot(px, py, "-o", color=c, lw=1.6, ms=5, label=name)
    for m in ours + pyth:
        dy = (4, 5) if m.startswith("ours") else (4, -9)            # ours above, Pythia below
        ax[0].annotate(m.replace("pythia-", "").replace("ours-", ""),
                       (npar[m], float(summ[m]["max_over_median"])),
                       fontsize=6.2, color=(OK["verm"] if m.startswith("ours") else "#444"),
                       xytext=dy, textcoords="offset points")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("parameters"); ax[0].set_ylabel("MA separation  (max / median)")
    ax[0].set_title("Scale grows the MA (both families)")
    ax[0].legend(frameon=False, fontsize=7.2, loc="lower right")
    # (b) bimodal isolation (cliff) grows with TRAINING TOKENS; ours = early regime
    for tr, c, name in [(tr410, OK["blue"], "Pythia-410m"), (tr14, OK["purple"], "Pythia-1.4b")]:
        st = [int(x["step"]) for x in tr]; cl = [float(x["cliff"]) for x in tr]
        ax[1].plot(st, cl, "-o", color=c, lw=1.6, ms=4, label=name)
    lo, hi = min(ours_cliffs), max(ours_cliffs)
    ax[1].axhspan(lo, hi, color=OK["verm"], alpha=0.18)
    ax[1].text(1.05e3, hi + 0.5, "our from-scratch models\n(10–40k steps)", fontsize=6.6, color=OK["verm"])
    ax[1].set_xscale("log")
    ax[1].set_xlabel("training step"); ax[1].set_ylabel("bimodal isolation  (cliff, ×)")
    ax[1].set_title("Tokens grow the isolation; ours = early regime")
    ax[1].legend(frameon=False, fontsize=7.2, loc="upper left")
    # (c) decoupling in Pythia-410m: MA magnitude saturates while isolation keeps rising
    st = np.array([int(x["step"]) for x in tr410])
    mm = np.array([float(x["max_over_median"]) for x in tr410]); cl = np.array([float(x["cliff"]) for x in tr410])
    ax[2].plot(st, mm / mm.max(), "-o", color=OK["blue"], lw=1.7, ms=4, label="MA separation (max/med)")
    ax[2].plot(st, cl / cl.max(), "-s", color=OK["verm"], lw=1.7, ms=4, label="bimodal isolation (cliff)")
    ax[2].set_xscale("log"); ax[2].set_ylim(0, 1.08)
    ax[2].set_xlabel("training step (Pythia-410m)"); ax[2].set_ylabel("fraction of final value")
    ax[2].set_title("Magnitude saturates; isolation keeps rising")
    ax[2].legend(frameon=False, fontsize=7.0, loc="lower right")
    fig.tight_layout(); save(fig, "fig5_scale")


def fig6_dropout():
    """pythia-410m 16k->143k: gradual staggered erosion of the middle while winners
    peak-and-decline; no abrupt onset."""
    d = rd(f"{RUN}/scale_profile/pythia410m_dropout.csv")
    st = np.array([int(x["step"]) for x in d])
    fig, ax = plt.subplots(1, 2, figsize=(7.8, 3.1))
    # (a) retention trajectories: winners vs middle ranks
    for k in range(4, 13):                                            # middle ranks, thin grey
        y = [float(x[f"r{k}"]) for x in d]
        ax[0].plot(st, y, "-", color=OK["grey"], lw=0.9, alpha=0.7)
    ax[0].plot(st, [float(x["mid_mean"]) for x in d], "-o", color=OK["blue"], lw=2.2, ms=4,
               label="ranks 4–12 (mean)", zorder=5)
    ax[0].plot(st, [float(x["top3_mean"]) for x in d], "-o", color=OK["verm"], lw=2.2, ms=4,
               label="top-3 (winners)", zorder=5)
    ax[0].axhline(1.0, color="k", lw=0.7, ls=":")
    ax[0].set_yscale("log"); kfmt(ax[0])
    ax[0].set_xlabel("training step"); ax[0].set_ylabel("retention  (mag / mag@16k)")
    ax[0].set_title("Middle erodes gradually; winners peak-and-decline")
    ax[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    # (b) cliff (bimodal isolation) builds continuously, no single onset
    ax[1].plot(st, [float(x["cliff"]) for x in d], "-o", color=OK["purple"], lw=1.8, ms=4)
    ax[1].set_xlabel("training step"); ax[1].set_ylabel("bimodal isolation (cliff, ×)")
    ax[1].set_title("Isolation builds continuously over 16k–143k"); kfmt(ax[1])
    fig.tight_layout(); save(fig, "fig6_dropout")


def fig7_adam():
    """gp_adam (Adam-preconditioned maintaining force) for the sink cohort vs random,
    across 3 checkpoints, with the weight-decay threshold lambda=0.01."""
    d = rd(f"{RUN}/scale_profile/grad_hard.csv")
    steps = [16000, 32000, 48000]; LAM = 0.01
    def group(role): return [r for r in d if r["role"] == role]
    cohort = group("winner") + group("collapse"); rnd = group("random")
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    xs = np.arange(len(steps))
    for r in cohort:
        ax.plot(xs + np.random.uniform(-.08, .08, len(steps)), [float(r[f"gp_{s}"]) for s in steps],
                "o", color=OK["blue"], ms=3, alpha=0.35)
    for r in rnd:
        ax.plot(xs + np.random.uniform(-.08, .08, len(steps)), [float(r[f"gp_{s}"]) for s in steps],
                "o", color=OK["grey"], ms=3, alpha=0.35)
    cm = [np.mean([float(r[f"gp_{s}"]) for r in cohort]) for s in steps]
    rm = [np.mean([float(r[f"gp_{s}"]) for r in rnd]) for s in steps]
    ax.plot(xs, cm, "-D", color=OK["blue"], lw=2, ms=7, label="sink cohort (n=15)")
    ax.plot(xs, rm, "-s", color=OK["grey"], lw=2, ms=6, label="random (n=10)")
    ax.axhline(LAM, color=OK["verm"], ls="--", lw=1.4)
    ax.text(2.05, LAM, r"$\lambda=0.01$ (decay)", color=OK["verm"], fontsize=7.5, va="bottom", ha="right")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xticks(xs); ax.set_xticklabels([f"{s//1000}k" for s in steps])
    ax.set_xlabel("training step"); ax.set_ylabel(r"maintaining force $g^{\mathrm{p}}_{\mathrm{adam}}$")
    ax.set_title("Adam preconditioning sustains the sink cohort\nabove decay; random channels sit at $\\sim$0")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(); save(fig, "fig7_adam")


def fig8_balance():
    """Balance theory: ||w|| ~ sqrt(t) pre-peak, and peak magnitude ~ lambda^{-1/2}."""
    fig, ax = plt.subplots(1, 2, figsize=(7.8, 3.1))
    # (a) ||w|| vs sqrt(t) for lambda=0.1, pre-peak
    w = rd(f"{RUN}/len_51m_40k/wnorm_lam0.1.csv")
    it = np.array([float(x["iter"]) for x in w]); wn = np.array([float(x["w_norm_ma"]) for x in w])
    pre = it <= 20000
    ax[0].plot(np.sqrt(it), wn, "o", color=OK["blue"], ms=2.5, alpha=0.5)
    A = np.polyfit(np.sqrt(it[pre]), wn[pre], 1)
    xs = np.linspace(np.sqrt(it.min()), np.sqrt(20000), 50)
    ax[0].plot(xs, np.polyval(A, xs), "-", color=OK["verm"], lw=1.8, label=r"linear in $\sqrt{t}$ (pre-peak)")
    ax[0].set_xlabel(r"$\sqrt{\mathrm{step}}$"); ax[0].set_ylabel(r"MA writer norm $\|w\|$")
    ax[0].set_title(r"$\|w\| \sim \sqrt{t}$ (pre-peak, $\lambda=0.1$)")
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    # (b) peak MA magnitude vs lambda, log-log, slope -1/2 reference
    lams = [0.05, 0.1, 0.2, 0.4]; peaks = []
    for lam in lams:
        r = rd(f"{RUN}/len_51m_40k/lam{lam}.csv")
        peaks.append(max(float(x["ch207"]) for x in r))
    lams = np.array(lams); peaks = np.array(peaks)
    sl = np.polyfit(np.log(lams), np.log(peaks), 1)[0]
    ax[1].plot(lams, peaks, "o", color=OK["blue"], ms=8, label=f"measured (slope {sl:.2f})")
    ref = peaks[1] * (lams / lams[1]) ** (-0.5)
    ax[1].plot(lams, ref, "--", color=OK["verm"], lw=1.6, label=r"$\lambda^{-1/2}$ reference")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel(r"weight decay $\lambda$"); ax[1].set_ylabel("peak MA magnitude")
    ax[1].set_title(r"peak $\propto \lambda^{-1/2}$")
    ax[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(); save(fig, "fig8_balance")


def fig10_redundancy():
    """Sink is a redundant collective: single-channel ablations weak, trio superadditive."""
    # measured by ablate_sink.py (pythia-410m step16k): drop in sink attention on ablation
    labels = ["ch357", "ch130", "ch966", "all\nthree", "random"]
    vals = [0.0155, 0.0042, 0.0047, 0.0795, 0.0006]
    cols = [OK["blue"], OK["blue"], OK["blue"], OK["verm"], OK["grey"]]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.bar(range(len(vals)), vals, color=cols, width=0.7)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("drop in sink attention when ablated")
    ax.set_title("The sink is a redundant collective\n(trio $\\sim$3$\\times$ superadditive over singles)")
    ax.annotate("", xy=(3, 0.0795), xytext=(1, 0.024),
                arrowprops=dict(arrowstyle="->", color="#555", lw=0.8))
    fig.tight_layout(); save(fig, "fig10_redundancy")


def fig9_lockin():
    """Early lock-in: magnitude rank vs step. Winners flat at 1-3; the rank-4
    channel collapses; a climber rises. Left: Pythia-410m; right: our 51M (with the
    early argmax migration ch221->ch308->ch207)."""
    import json
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.2))
    # (a) Pythia-410m
    d = rd(f"{RUN}/scale_profile/pythia410m_ranktraj.csv")
    st = np.array([int(x["step"]) for x in d])
    series = [("357", OK["verm"], "winner (r1)"), ("130", OK["orange"], "winner (r2)"),
              ("966", OK["yellow"], "winner (r3)"), ("752", OK["blue"], "ch752: stuck at r4, later fades"),
              ("125", OK["green"], "climber")]
    for ch, c, lab in series:
        ax[0].plot(st, [int(x[f"rank_{ch}"]) for x in d], "-o", color=c, lw=1.6, ms=3, label=lab)
    ax[0].set_yscale("log"); ax[0].invert_yaxis(); ax[0].set_xscale("log")
    ax[0].set_xlabel("training step"); ax[0].set_ylabel("magnitude rank (1 = largest)")
    ax[0].set_title("Pythia-410m: winners lock by $\\sim$4k")
    ax[0].legend(frameon=False, fontsize=6.8, loc="lower right")
    # (b) our 51M from saved trajectory
    M = np.load(f"{RUN}/scale_profile/ours51m_traj.npy")
    meta = json.load(open(f"{RUN}/scale_profile/ours51m_meta.json"))
    it = np.array(meta["iters"]); cols = meta["cols"]
    def rank_of(name):
        c = cols.index(name); return [int((M[j] > M[j, c]).sum()) + 1 for j in range(len(it))]
    for name, c, lab in [("ch207", OK["verm"], "ch207 (final MA)"),
                          ("ch399", OK["orange"], "ch399"), ("ch308", OK["yellow"], "ch308"),
                          ("ch221", OK["blue"], "ch221 (early argmax)")]:
        ax[1].plot(it, rank_of(name), "-o", color=c, lw=1.6, ms=3, label=lab)
    ax[1].set_yscale("log"); ax[1].invert_yaxis(); ax[1].set_xscale("log")
    ax[1].set_xlabel("training step"); ax[1].set_ylabel("magnitude rank")
    ax[1].set_title("Our 51M: argmax migrates, then locks by $\\sim$5k")
    ax[1].legend(frameon=False, fontsize=6.8, loc="lower right")
    fig.tight_layout(); save(fig, "fig9_lockin")


def fig11_gain():
    """Gain dissociation is a large-channel effect, not MA-specific. (a) trajectory:
    MA magnitude grows while its LayerNorm gain falls. (b) at the MA's layer, gain vs
    magnitude: all large channels are suppressed and the MA sits among its peers."""
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.2))
    # (a) trajectory
    d = rd(f"{RUN}/scale_profile/gain_traj.csv")
    it = np.array([int(x["iter"]) for x in d])
    mag = np.array([float(x["ma_mag"]) for x in d]); g = np.array([float(x["ma_gain"]) for x in d])
    ok = ~np.isnan(mag)
    axb = ax[0].twinx()
    l1, = ax[0].plot(it[ok], mag[ok], "-", color=OK["verm"], lw=2, label="MA magnitude")
    l2, = axb.plot(it, g, "-", color=OK["blue"], lw=2, label=r"MA LayerNorm gain $\gamma$")
    ax[0].set_xlabel("training step"); ax[0].set_ylabel("MA magnitude", color=OK["verm"])
    axb.set_ylabel(r"MA gain $\gamma$", color=OK["blue"]); kfmt(ax[0])
    ax[0].tick_params(axis="y", labelcolor=OK["verm"]); axb.tick_params(axis="y", labelcolor=OK["blue"])
    ax[0].set_title("Dissociation: magnitude grows, gain falls")
    ax[0].legend(handles=[l1, l2], frameon=False, fontsize=7.5, loc="center right")
    # (b) gain vs magnitude at the MA layer (block-6 pairing: h5 magnitude, its normalising gain)
    s = [r for r in rd(f"{RUN}/scale_profile/gain_scatter.csv") if int(r["layer"]) == 6]
    m = np.array([float(r["mag"]) for r in s]); gg = np.array([float(r["gain"]) for r in s])
    isma = np.array([int(r["is_ma"]) for r in s])
    order = np.argsort(-m); big = order[:8]
    mask = np.ones(len(m), bool); mask[big] = False; mask[isma == 1] = False
    ax[1].scatter(m[mask], gg[mask], s=6, color=OK["grey"], alpha=0.4, label="bulk channels")
    ax[1].scatter(m[big], gg[big], s=26, color=OK["blue"], label="top-8 (equal-magnitude peers)")
    mac = np.where(isma == 1)[0][0]
    ax[1].scatter([m[mac]], [gg[mac]], s=130, marker="*", color=OK["verm"], zorder=5, label="MA")
    ax[1].axhline(np.median(gg[order[50:]]), color="#888", ls=":", lw=1)
    ax[1].text(ax[1].get_xlim()[1], np.median(gg[order[50:]]), " bulk median", fontsize=7, color="#666", va="bottom", ha="right")
    ax[1].set_xscale("log"); ax[1].set_xlabel("channel magnitude at sink"); ax[1].set_ylabel(r"LayerNorm gain $\gamma$")
    ax[1].set_title("Large channels are suppressed; MA among peers")
    ax[1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout(); save(fig, "fig11_gain")


if __name__ == "__main__":
    print("building figures...")
    fig1_emergence(); fig2_global(); fig3_causal(); fig4_controls(); fig5_scale(); fig6_dropout()
    fig7_adam(); fig8_balance(); fig9_lockin(); fig10_redundancy(); fig11_gain()
    print("done ->", OUT)
