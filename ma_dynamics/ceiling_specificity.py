"""
Is the saturated-regime "ceiling" (a doubled MA relaxing back down) MA-SPECIFIC
regulation, or generic relaxation of any weight perturbation near convergence?

For a checkpoint, double (alpha=2) EITHER the MA channel OR a matched-norm random
channel, resume at a given LR with fixed batches, and track *that channel's own*
magnitude (its max|activation| and its writer-row L2 norm) over `steps`.

  - MA relaxes but random does NOT  -> MA-specific downward regulation (real ceiling)
  - both relax about equally         -> generic near-minimum weight relaxation (artifact)

Also supports an LR override so we can disentangle "regime" from "learning rate":
run ckpt 8000 (saturated) at high LR and ckpt 2000 (rising) at low LR.
"""

import os
import csv
import argparse
import torch

from common import make_ctx, Data
from ma_probe import MAProbe, scale_channel_all_layers, channel_writer_norms
from perturb_recover import scheduled_lr, load_model_opt


def channel_writer_norm_one(model, channel, targets=("mlp", "attn")):
    s = 0.0
    for block in model.transformer.h:
        for tgt in targets:
            lin = block.mlp.c_proj if tgt == "mlp" else block.attn.c_proj
            s += float((lin.weight.data[channel] ** 2).sum())
    return s ** 0.5


def run(ckpt, device, ctx, data, probe_X, layer, channel, alpha, lr, steps,
        grad_accum, grad_clip, dtype, batch_seed=777, weight_decay=-1.0):
    model, opt, cfg = load_model_opt(ckpt, device)
    if weight_decay >= 0:                       # override AdamW decoupled weight decay (Adam moments stay valid)
        for g in opt.param_groups:
            if g.get("weight_decay", 0) > 0:    # only the decayed group
                g["weight_decay"] = weight_decay
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    if alpha != 1.0:
        scale_channel_all_layers(model, channel, alpha)
    probe = MAProbe(model).attach()
    gen = torch.Generator().manual_seed(batch_seed)

    def snap():
        cm = probe.channel_magnitude(probe_X, layer, channel, ctx=ctx)
        return cm["max_abs"], channel_writer_norm_one(model, channel)
    rec = [snap()]
    for _ in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr
        for micro in range(grad_accum):
            X, Y = data.get_batch("train", generator=gen)
            with ctx:
                _, loss = model(X, Y)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
        if grad_clip != 0.0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        rec.append(snap())
    probe.detach(); del model, opt, scaler; torch.cuda.empty_cache()
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=-1.0, help="override frozen LR; -1 = native scheduled")
    p.add_argument("--layer", type=int, default=-1, help="target layer; -1 = auto (max ratio)")
    p.add_argument("--weight_decay", type=float, default=-1.0, help="override AdamW weight decay; -1 = ckpt's, 0 = off")
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--out_dir", default="runs/pilot_v2/ceiling")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ctx = make_ctx(args.device, args.dtype)

    ckpt = torch.load(args.ckpt, map_location=args.device)
    it = ckpt.get("iter_num", -1)
    bs = ckpt["model_args"]["block_size"]
    data = Data(bs, args.batch_size, args.device)
    probe_X = data.fixed_probe_batch()[0]
    lr = args.lr if args.lr >= 0 else scheduled_lr(it, ckpt["config"])

    m0, _o, cfg = load_model_opt(ckpt, args.device)
    probe = MAProbe(m0).attach()
    meas = probe.measure(probe_X, ctx=ctx)
    layer = args.layer if args.layer >= 0 else int(max(range(cfg.n_layer), key=lambda l: meas["ratio"][l]))
    ma_channel = meas["ma_channel"][layer]
    norms = channel_writer_norms(m0); tn = norms[ma_channel].clone(); norms[ma_channel] = float("inf")
    rand_channel = int((norms - tn).abs().argmin().item())
    probe.detach(); del m0, _o; torch.cuda.empty_cache()

    print(f"ckpt {it} | L{layer} | MA ch {ma_channel} | rand ch {rand_channel} | "
          f"alpha {args.alpha} | lr {lr:.2e} | wd {args.weight_decay} | steps {args.steps}")

    base_kw = dict(ckpt=ckpt, device=args.device, ctx=ctx, data=data, probe_X=probe_X,
                   layer=layer, lr=lr, steps=args.steps,
                   grad_accum=args.grad_accum, grad_clip=args.grad_clip, dtype=args.dtype,
                   weight_decay=args.weight_decay)
    ctrl = run(channel=ma_channel, alpha=1.0, **base_kw)   # unperturbed moving baseline for the MA channel
    ma = run(channel=ma_channel, alpha=args.alpha, **base_kw)
    rd = run(channel=rand_channel, alpha=args.alpha, **base_kw)

    wd_tag = "wdCK" if args.weight_decay < 0 else f"wd{args.weight_decay:g}"
    out = os.path.join(args.out_dir, f"ceiling_{it}_L{layer}_a{args.alpha}_lr{lr:.0e}_{wd_tag}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "ctrl_mag", "ma_mag", "ma_wnorm", "rand_mag", "rand_wnorm"])
        for i in range(len(ma)):
            w.writerow([i, f"{ctrl[i][0]:.4f}", f"{ma[i][0]:.4f}", f"{ma[i][1]:.4f}",
                        f"{rd[i][0]:.4f}", f"{rd[i][1]:.4f}"])
    # ceiling test: does the kicked MA channel return toward the control trajectory?
    gap0 = ma[0][0] - ctrl[0][0]; gapE = ma[-1][0] - ctrl[-1][0]
    print(f"  MA vs CONTROL: kick gap {gap0:+.1f} -> {gapE:+.1f}  "
          f"(ctrl {ctrl[0][0]:.0f}->{ctrl[-1][0]:.0f}, kicked {ma[0][0]:.0f}->{ma[-1][0]:.0f}) "
          f"| {'CEILING: returns toward ctrl' if gapE < 0.7*gap0 else 'NO ceiling: gap holds/widens'}")

    def frac(rec):  # fraction of the alpha-induced excess magnitude that relaxed away
        m0v, mEnd = rec[0][0], rec[-1][0]
        # baseline (pre-double) magnitude ~ m0/alpha; excess = m0 - baseline
        base = m0v / args.alpha
        excess0 = m0v - base
        return (m0v - mEnd) / (excess0 + 1e-9)
    print(f"  MA channel : mag {ma[0][0]:.2f}->{ma[-1][0]:.2f}  wnorm {ma[0][1]:.3f}->{ma[-1][1]:.3f}  "
          f"| relaxed {frac(ma)*100:.0f}% of excess")
    print(f"  RAND channel: mag {rd[0][0]:.2f}->{rd[-1][0]:.2f}  wnorm {rd[0][1]:.3f}->{rd[-1][1]:.3f}  "
          f"| relaxed {frac(rd)*100:.0f}% of excess")
    print(f"  => {'MA-SPECIFIC (MA relaxes more)' if frac(ma) > frac(rd)+0.1 else 'GENERIC (both relax similarly)'}")
    print(f"[ceiling] -> {out}")


if __name__ == "__main__":
    main()
