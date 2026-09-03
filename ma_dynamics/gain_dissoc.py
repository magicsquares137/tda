"""
Reconstruct the gain-dissociation analysis for a figure.
Claim: the MA preserves magnitude while its LayerNorm gain gamma collapses; gain is
NOT magnitude-organized (per-layer slopes sign-changing); and the MA is NOT an outlier
below its magnitude trend once scored on the BULK (z~+0.3), whereas naive leave-one-out
manufactures an outlier (z~+1.5) via mutual contamination among the top channels.

gamma_l[c] = ln_1.weight of block l, applied to norm(h_{l-1}); pair it with the sink
magnitude at the INPUT to block l. Per layer, regress gamma on magnitude, score the MA
(argmax magnitude) out-of-sample against (a) the bulk trend (exclude top-8) and
(b) a leave-one-out fit (all channels but the MA).
"""
import os, csv, json
import numpy as np, torch
from common import build_model, Data

DEV = "cuda:0"


def per_layer(ckpt):
    ck = torch.load(ckpt, map_location=DEV); ma = ck["model_args"]
    model, cfg = build_model(ma["n_layer"], ma["n_head"], ma["n_embd"], ma["block_size"],
                             vocab_size=ma["vocab_size"], bias=ma["bias"], device=DEV, norm=ma.get("norm", "layer"))
    model.load_state_dict(ck["model"]); model.eval()
    data = Data(ma["block_size"], 8, DEV); X, _ = data.fixed_probe_batch()
    ins = {}
    hs = [model.transformer.h[l].register_forward_pre_hook(
            (lambda l: lambda m, i: ins.__setitem__(l, i[0].detach()))(l)) for l in range(cfg.n_layer)]
    with torch.no_grad(): model(X)
    for h in hs: h.remove()
    out = {}
    for l in range(cfg.n_layer):
        m = ins[l].float()[:, 0, :].abs().mean(0).cpu().numpy()      # sink magnitude into block l
        g = ck["model"][f"transformer.h.{l}.ln_1.weight"].float().cpu().numpy()  # gain (signed)
        out[l] = (m, g)
    del model; torch.cuda.empty_cache()
    return out, cfg.n_layer


def zscore(m, g, exclude):
    """fit g ~ a*m + b on channels NOT in exclude; return slope and residual-z of each excluded ch."""
    keep = np.ones_like(m, bool); keep[list(exclude)] = False
    A = np.polyfit(m[keep], g[keep], 1); pred = np.polyval(A, m)
    sd = (g - pred)[keep].std()
    return A[0], {c: (g[c] - pred[c]) / (sd + 1e-12) for c in exclude}


ckpt = "runs/len_51m_40k/ckpt_40000.pt"
data, nL = per_layer(ckpt)
print(f"{'layer':>5} {'MA ch':>6} {'MA mag':>8} {'MA gain':>8} {'slope':>8} {'z_bulk':>8} {'z_loo':>8}")
rows = []
zb_all, zl_all, slopes = [], [], []
for l in range(nL):
    m, g = data[l]; ma_c = int(m.argmax())
    top8 = list(np.argsort(-m)[:8])
    slope, zb = zscore(m, g, top8)                 # bulk fit (exclude top-8), score MA
    _, zl = zscore(m, g, [ma_c])                    # leave-one-out (exclude MA only)
    zb_ma, zl_ma = zb[ma_c], zl[ma_c]
    slopes.append(slope); zb_all.append(zb_ma); zl_all.append(zl_ma)
    rows.append(dict(layer=l, ma_ch=ma_c, ma_mag=float(m[ma_c]), ma_gain=float(g[ma_c]),
                     slope=float(slope), z_bulk=float(zb_ma), z_loo=float(zl_ma)))
    print(f"{l:>5} {ma_c:>6} {m[ma_c]:>8.1f} {g[ma_c]:>8.3f} {slope:>8.2f} {zb_ma:>+8.2f} {zl_ma:>+8.2f}")
print(f"\nslopes range [{min(slopes):.1f}, {max(slopes):.1f}] (sign-changing: {min(slopes)<0<max(slopes)})")
print(f"z_bulk mean {np.mean(zb_all):+.2f} ({sum(z>0 for z in zb_all)}/{nL} positive)")
print(f"z_loo  mean {np.mean(zl_all):+.2f}")

# save per-layer + the L5 trajectory for the figure
with open("runs/scale_profile/gain_dissoc.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows: w.writerow(r)
# trajectory: MA magnitude (channel_traj_L5) + MA gain over training at L5
import glob
LT = 5
its = sorted(int(''.join(filter(str.isdigit, p.split('ckpt_')[1]))) for p in glob.glob("runs/len_51m_40k/ckpt_*.pt"))
tr = []
mag_by_iter = {int(r["iter"]): float(r["ch207"]) if "ch207" in r else None
               for r in csv.DictReader(open("runs/len_51m_40k/channel_traj_L5.csv"))} if False else None
# read magnitude straight from channel_traj_L5 (ch207 column)
cm = {int(r["iter"]): float(r.get("ch207", "nan")) for r in csv.DictReader(open("runs/len_51m_40k/channel_traj_L5.csv"))}
for it in its:
    p = f"runs/len_51m_40k/ckpt_{it}.pt"
    g = torch.load(p, map_location="cpu")["model"][f"transformer.h.{LT}.ln_1.weight"][207].item()
    tr.append(dict(iter=it, ma_gain_L5=g, ma_mag_L5=cm.get(it, float("nan"))))
with open("runs/scale_profile/gain_traj.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["iter", "ma_gain_L5", "ma_mag_L5"]); w.writeheader()
    for r in tr: w.writerow(r)
print("[gain_dissoc] -> gain_dissoc.csv, gain_traj.csv")
