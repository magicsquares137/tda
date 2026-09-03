"""
Causal test for the GLOBAL rise-then-decline: is weight decay what drives channels
(and the MA) down after their peak? Resume a near-peak checkpoint and continue
UNPERTURBED with weight decay ON vs OFF (same batch order), tracking the MA channel
+ other big channels + the median. If decay-OFF stops the decline -> decay is the
cause. If channels still decline with decay off -> it's loss/convergence-driven.
"""

import os
import csv
import math
import argparse
import torch

from common import make_ctx, Data
from ma_probe import MAProbe
from perturb_recover import load_model_opt, scheduled_lr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--constant_lr", type=float, default=-1.0, help=">0 = hold LR fixed (isolates LR-decay); -1 = follow schedule")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--layer", type=int, default=2)
    p.add_argument("--track_channels", type=int, nargs="+", default=[207, 221, 468])
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--probe_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    dev = args.device
    ctx = make_ctx(dev, args.dtype)

    ck = torch.load(args.ckpt, map_location=dev)
    it0 = ck.get("iter_num", 0)
    tcfg = ck["config"]
    model, opt, cfg = load_model_opt(ck, dev)
    for g in opt.param_groups:                 # override decay (Adam moments preserved)
        if g.get("weight_decay", 0) > 0 or args.weight_decay == 0:
            g["weight_decay"] = args.weight_decay
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    bs = ck["model_args"]["block_size"]
    data = Data(bs, args.batch_size, dev)
    probeX = data.fixed_probe_batch()[0]
    probe = MAProbe(model).attach()

    L = args.layer
    ma_c = args.track_channels[0]               # the MA channel (207)
    cap = {}
    h = model.transformer.h[L].register_forward_hook(lambda m, i, o: cap.__setitem__("r", o.detach()))

    @torch.no_grad()
    def ma_writer_norm():
        # ||w_MA||: aggregate L2 norm of the c_proj rows feeding the MA channel across all layers
        s = 0.0
        for block in model.transformer.h:
            s += float((block.mlp.c_proj.weight[ma_c] ** 2).sum())
            s += float((block.attn.c_proj.weight[ma_c] ** 2).sum())
        return s ** 0.5

    def snap():
        with torch.no_grad(), ctx:
            model(probeX)
        r = cap["r"].float()[:, 0, :]           # sink-token residual (B,C)
        med = r.abs().median().item()
        chans = {c: r[:, c].abs().mean().item() for c in args.track_channels}
        meas = probe.measure(probeX, ctx=ctx)
        return med, chans, meas["ratio"][L], ma_writer_norm()

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "wd", "ratio_L", "median", "w_norm_ma"] + [f"ch{c}" for c in args.track_channels])

    gen = torch.Generator().manual_seed(args.seed)
    print(f"resume ckpt {it0} | wd={args.weight_decay} | layer {L} | tracking {args.track_channels}")
    X, Y = data.get_batch("train", generator=gen)
    for step in range(args.steps + 1):
        it = it0 + step
        if step % args.probe_every == 0:
            med, chans, ratio, wnorm = snap()
            with open(args.out, "a", newline="") as f:
                csv.writer(f).writerow([it, args.weight_decay, f"{ratio:.3f}", f"{med:.4f}", f"{wnorm:.5f}"]
                                       + [f"{chans[c]:.4f}" for c in args.track_channels])
            if step % (args.probe_every * 10) == 0:
                print(f"  it {it:6d} ratio {ratio:6.1f} med {med:.3f} | "
                      + " ".join(f"ch{c}:{chans[c]:.1f}" for c in args.track_channels))
        if step == args.steps:
            break
        lr = args.constant_lr if args.constant_lr > 0 else scheduled_lr(it, tcfg)
        for g in opt.param_groups:
            g["lr"] = lr
        for _ in range(args.grad_accum):
            with ctx:
                _, loss = model(X, Y)
                loss = loss / args.grad_accum
            X, Y = data.get_batch("train", generator=gen)
            scaler.scale(loss).backward()
        if args.grad_clip:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    print(f"[causal_decay] wd={args.weight_decay} -> {args.out}")


if __name__ == "__main__":
    main()
