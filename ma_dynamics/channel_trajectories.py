"""
Is the MA channel's rise-then-decline shape SPECIFIC to it, or do other channels
do the same (just smaller)? For each checkpoint, record the sink-token (pos 0)
magnitude of EVERY residual channel at a layer, then classify each channel's
trajectory shape. If only the MA channel (+a few) peak-and-decline while the rest
are flat/monotone -> decline is MA-specific. If most channels peak-and-decline ->
it's a global activation-scale effect and the MA is special only in magnitude.
"""

import os
import csv
import argparse
import numpy as np
import torch

from common import build_model, make_ctx, Data
from ma_probe import MAProbe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", default="runs/len_51m_40k")
    p.add_argument("--ckpts", type=int, nargs="+",
                   default=[2000, 6000, 10000, 14000, 18000, 22000, 26000, 30000, 34000, 40000])
    p.add_argument("--layer", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    dev = args.device
    ctx = make_ctx(dev)
    ck0 = torch.load(os.path.join(args.run_dir, f"ckpt_{args.ckpts[0]}.pt"), map_location=dev)
    bs = ck0["model_args"]["block_size"]
    data = Data(bs, 24, dev); X = data.fixed_probe_batch()[0]

    mags = []      # [n_ckpt, C] mean |residual[pos0, c]|
    ma_chans = []
    for it in args.ckpts:
        ck = torch.load(os.path.join(args.run_dir, f"ckpt_{it}.pt"), map_location=dev)
        ma = ck["model_args"]
        model, cfg = build_model(ma["n_layer"], ma["n_head"], ma["n_embd"], ma["block_size"],
                                 vocab_size=ma["vocab_size"], bias=ma["bias"], device=dev,
                                 norm=ma.get("norm", "layer"))
        model.load_state_dict(ck["model"])
        cap = {}
        h = model.transformer.h[args.layer].register_forward_hook(
            lambda m, i, o: cap.__setitem__("r", o.detach()))
        with torch.no_grad(), ctx:
            model(X)
        r = cap["r"].float()                      # (B,T,C)
        mags.append(r[:, 0, :].abs().mean(dim=0).cpu().numpy())
        pr = MAProbe(model).attach(); ma_chans.append(pr.measure(X, ctx=ctx)["ma_channel"][args.layer]); pr.detach()
        h.remove(); del model; torch.cuda.empty_cache()

    M = np.stack(mags)          # [n_ckpt, C]
    it = np.array(args.ckpts)
    C = M.shape[1]
    ma_chan = max(set(ma_chans), key=ma_chans.count)

    # per-channel: peak location + decline fraction from peak to end
    def shape(v):
        ip = int(np.argmax(v)); pk = v[ip]; fin = v[-1]
        decl = (pk - fin) / (pk + 1e-9)
        # "peak-then-decline": peak not at the very end AND declines > 15%
        peaks = (it[ip] < 0.8 * it[-1]) and (decl > 0.15)
        return ip, pk, decl, peaks

    peak_decline = 0; results = []
    for c in range(C):
        ip, pk, decl, peaks = shape(M[:, c])
        results.append((c, pk, it[ip], decl, peaks))
        if peaks and pk > 2.0:   # ignore tiny channels
            peak_decline += 1
    big = [r for r in results if r[1] > 2.0]
    print(f"layer {args.layer} | MA channel = {ma_chan} | {C} channels, {len(big)} with peak-mag>2")
    print(f"channels that PEAK-then-DECLINE (>15%, peak before 80% of run): {peak_decline} of {len(big)} sizeable")
    print(f"\nMA channel {ma_chan}: peak-mag {M[:,ma_chan].max():.1f} @ iter {it[np.argmax(M[:,ma_chan])]}, "
          f"decline {shape(M[:,ma_chan])[2]*100:.0f}%")
    # top-8 channels by peak magnitude and their shapes
    top = sorted(big, key=lambda r: -r[1])[:8]
    print(f"\n{'chan':>5} {'peakmag':>8} {'@iter':>6} {'decline%':>9} {'peak-decline?':>13}")
    for c, pk, ipk, decl, peaks in top:
        tag = "YES" if peaks else "no"
        star = " <-MA" if c == ma_chan else ""
        print(f"{c:>5} {pk:>8.1f} {ipk:>6} {decl*100:>8.0f}% {tag:>13}{star}")
    # median-channel trajectory for contrast
    med = np.median(M, axis=1)
    print(f"\nMEDIAN channel magnitude trajectory: {' '.join(f'{v:.2f}' for v in med)}")
    print(f"  (iters: {' '.join(str(i) for i in it)})")
    # save full matrix
    out = os.path.join(args.run_dir, f"channel_traj_L{args.layer}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iter"] + [f"ch{c}" for c in range(C)])
        for i in range(len(it)):
            w.writerow([it[i]] + [f"{M[i,c]:.4f}" for c in range(C)])
    print(f"[channel_traj] -> {out}")


if __name__ == "__main__":
    main()
