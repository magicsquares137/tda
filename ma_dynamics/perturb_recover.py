"""
Stage 1 + 2 -- causal perturb-and-recover test for massive activations.

For each checkpoint we fork three continuations, each resumed for `resume_steps`
with the *same batch order* and a *frozen learning rate* (removing the LR-decay
confound):

  1. control_unpert : resume unchanged  -> gives the moving baseline trajectory
  2. perturb_ma     : scale the c_proj row that writes the MA channel by alpha
  3. control_rand   : scale a matched-norm *random* channel's row by alpha

We track the MA-channel ratio each step and measure the gap of (2) and (3) to
(1). If the MA channel returns to the baseline -> MAs are a regulated variable
with a setpoint (Stage-1 premise). Fitting gap(t) = gap0 * exp(-t/tau) gives the
relaxation time tau; sweeping checkpoints across emergence onset gives tau(iter)
and tests for critical slowing down (Stage 2).

Usage:
  # Stage 1 (one settled checkpoint, does it recover at all?)
  python perturb_recover.py --ckpts runs/pilot/ckpt_3000.pt --alpha 0 --device cuda:0
  # Stage 2 (sweep the onset bracket)
  python perturb_recover.py --ckpts runs/pilot/ckpt_500.pt runs/pilot/ckpt_800.pt ... --alpha 0
"""

import os
import csv
import math
import argparse

import torch

from common import build_model, make_ctx, Data
from ma_probe import (MAProbe, scale_channel_all_layers, channel_writer_norms)


def scheduled_lr(it, cfg):
    lr, warmup = cfg["learning_rate"], cfg["warmup_iters"]
    decay_iters, min_lr = cfg["lr_decay_iters"], cfg["min_lr"]
    if it < warmup:
        return lr * (it + 1) / (warmup + 1)
    if it > decay_iters:
        return min_lr
    r = (it - warmup) / (decay_iters - warmup)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * r)) * (lr - min_lr)


def load_model_opt(ckpt, device):
    ma = ckpt["model_args"]
    model, cfg = build_model(ma["n_layer"], ma["n_head"], ma["n_embd"],
                             ma["block_size"], vocab_size=ma["vocab_size"],
                             bias=ma["bias"], dropout=ma.get("dropout", 0.0),
                             device=device, norm=ma.get("norm", "layer"))
    model.load_state_dict(ckpt["model"])
    tcfg = ckpt["config"]
    opt = model.configure_optimizers(tcfg["weight_decay"], tcfg["learning_rate"],
                                     (tcfg["beta1"], tcfg["beta2"]), "cuda")
    opt.load_state_dict(ckpt["optimizer"])
    return model, opt, cfg


def resume_fork(ckpt, device, ctx, data, probe_X, layer, channel,
                resume_steps, frozen_lr, grad_accum, grad_clip, dtype,
                perturb=None, batch_seed=777):
    """Load a fresh copy, optionally perturb, resume with fixed batches + frozen
    LR, and return the per-step MA-channel ratio trajectory."""
    model, opt, cfg = load_model_opt(ckpt, device)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    if perturb is not None:
        scale_channel_all_layers(model, perturb["channel"], perturb["alpha"],
                                 targets=perturb["targets"])
    probe = MAProbe(model).attach()

    # deterministic, identical batch stream across all forks
    gen = torch.Generator().manual_seed(batch_seed)

    ratios = []
    # t=0 measurement (immediately after perturb, before any resume step)
    ratios.append(probe.channel_magnitude(probe_X, layer, channel, ctx=ctx)["ratio"])
    for step in range(resume_steps):
        for g in opt.param_groups:
            g["lr"] = frozen_lr
        for micro in range(grad_accum):
            X, Y = data.get_batch("train", generator=gen)
            with ctx:
                _, loss = model(X, Y)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
        if grad_clip != 0.0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        ratios.append(probe.channel_magnitude(probe_X, layer, channel, ctx=ctx)["ratio"])
    probe.detach()
    del model, opt, scaler
    torch.cuda.empty_cache()
    return ratios


def fit_tau(gap):
    """Fit the recovery of |gap(t)| to an *offset* exponential
        |gap(t)| = c + Delta * exp(-t/tau)
    which handles incomplete recovery (a plateau) -- a plain single exponential
    mis-fits the fast-rebound-then-stall shape. tau is found by a 1-D search over
    log-spaced candidates, solving linearly for (c, Delta) at each.

    Returns (tau, r2, recovered_fraction) where recovered_fraction = 1 - c/|gap0|
    is the asymptotic fraction of the displacement that is restored."""
    import numpy as np
    g = np.abs(np.asarray(gap, dtype=float))
    t = np.arange(len(g), dtype=float)
    if g[0] < 1e-6 or len(g) < 5:
        return float("nan"), float("nan"), float("nan")
    best_r2, best = -1e18, (float("inf"), 0.0, 0.0)  # tau, c, Delta
    ss_tot = ((g - g.mean()) ** 2).sum() + 1e-12
    for tau in np.logspace(0, np.log10(max(10.0, 4.0 * len(g))), 60):
        basis = np.vstack([np.ones_like(t), np.exp(-t / tau)]).T
        coef, *_ = np.linalg.lstsq(basis, g, rcond=None)
        r2 = 1.0 - ((g - basis @ coef) ** 2).sum() / ss_tot
        if r2 > best_r2:
            best_r2 = r2
            best = (float(tau), float(coef[0]), float(coef[1]))
    tau, c, delta = best
    r2 = best_r2
    # if the plateau c already exceeds gap0 (gap grew, no recovery) tau is meaningless
    if c >= g[0]:
        tau = float("inf")
    recovered = 1.0 - c / (g[0] + 1e-9)
    # if Delta ~ 0 the gap never moved coherently -> tau meaningless
    if abs(delta) < 1e-3 * g[0]:
        tau = float("inf")
    return float(tau), float(r2), float(recovered)


def t_half(gap):
    """Model-free half-recovery time: steps to close half of the gap that is
    ultimately closed. Robust alternative to the exponential tau. Returns nan if
    the channel never recovers (final gap >= initial)."""
    import numpy as np
    g = np.abs(np.asarray(gap, dtype=float))
    total_closed = g[0] - g[-1]
    if total_closed <= 0:
        return float("nan")
    closed = g[0] - g
    idx = np.argmax(closed >= 0.5 * total_closed)
    return float(idx)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--alpha", type=float, default=0.0, help="perturb scale (0=zero write, 2=double)")
    p.add_argument("--target", type=str, default="both", choices=["mlp", "attn", "both"])
    p.add_argument("--layer", type=int, default=-1, help="MA birth (max-ratio) layer for measurement; -1 = auto")
    p.add_argument("--ma_channel", type=int, default=-1, help="pin MA residual channel across ckpts; -1 = auto per ckpt")
    p.add_argument("--min_displacement", type=float, default=0.25,
                   help="require |perturbed-base|/base at t=0 above this to trust tau")
    p.add_argument("--n_reps", type=int, default=1, help="resume-seed replicates per ckpt (error bars)")
    p.add_argument("--seed", type=int, default=777, help="base resume batch seed")
    p.add_argument("--resume_steps", type=int, default=200)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--freeze_lr", action="store_true", default=True)
    p.add_argument("--out_dir", type=str, default="runs/pilot/perturb")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bfloat16")
    return p.parse_args()


def main():
    args = parse()
    os.makedirs(args.out_dir, exist_ok=True)
    ctx = make_ctx(args.device, args.dtype)

    summary_path = os.path.join(args.out_dir, "tau_summary.csv")
    with open(summary_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["ckpt_iter", "layer", "ma_channel", "rand_channel", "alpha", "base0",
             "n_reps", "disp_ma",
             "tau_mean", "tau_std", "thalf_mean", "thalf_std",
             "recovered_mean", "recovered_std", "disp_rand", "recovered_rand"])

    for ckpt_path in args.ckpts:
        ckpt = torch.load(ckpt_path, map_location=args.device)
        it = ckpt.get("iter_num", -1)
        cfg_train = ckpt["config"]
        block_size = ckpt["model_args"]["block_size"]
        data = Data(block_size, args.batch_size, args.device)
        probe_X = data.fixed_probe_batch()[0]

        # identify MA channel + birth layer on the loaded (unperturbed) model
        m0, _opt0, cfg = load_model_opt(ckpt, args.device)
        probe0 = MAProbe(m0).attach()
        meas = probe0.measure(probe_X, ctx=ctx)
        probe0.detach()
        layer = args.layer if args.layer >= 0 else int(
            max(range(cfg.n_layer), key=lambda l: meas["ratio"][l]))
        # the massive activation IS the argmax channel (verified: zeroing it collapses
        # the ratio; persist_chan/highest-mean is a different, non-massive channel).
        # auto-detect per checkpoint so we track the MA through its onset migration.
        ma_channel = args.ma_channel if args.ma_channel >= 0 else meas["ma_channel"][layer]
        targets = ("mlp", "attn") if args.target == "both" else (args.target,)
        norms = channel_writer_norms(m0, targets=targets)
        tn = norms[ma_channel].clone()
        norms[ma_channel] = float("inf")
        rand_channel = int((norms - tn).abs().argmin().item())
        del m0, _opt0
        torch.cuda.empty_cache()

        frozen_lr = scheduled_lr(it, cfg_train) if args.freeze_lr else cfg_train["learning_rate"]
        print(f"\n=== ckpt {it} | birth L{layer} | MA chan {ma_channel} | "
              f"rand chan {rand_channel} | frozen lr {frozen_lr:.2e} | alpha {args.alpha} ===")

        # replicates: same weight perturbation, different resume batch stream ->
        # spread over seeds gives an error bar on tau / recovery
        import numpy as np
        tau_reps, rec_reps, th_reps, disp_reps = [], [], [], []
        for rep in range(args.n_reps):
            seed = args.seed + 1000 * rep
            kw = dict(device=args.device, ctx=ctx, data=data, probe_X=probe_X,
                      layer=layer, channel=ma_channel, resume_steps=args.resume_steps,
                      frozen_lr=frozen_lr, grad_accum=args.grad_accum,
                      grad_clip=args.grad_clip, dtype=args.dtype, batch_seed=seed)
            base = resume_fork(ckpt, perturb=None, **kw)
            ma = resume_fork(ckpt, perturb=dict(channel=ma_channel, alpha=args.alpha, targets=targets), **kw)
            gap_ma = [ma[i] - base[i] for i in range(len(base))]
            disp = abs(gap_ma[0]) / (base[0] + 1e-9)
            tau, r2, rec = fit_tau(gap_ma)
            th = t_half(gap_ma)
            if disp >= args.min_displacement:
                tau_reps.append(tau); th_reps.append(th)
            rec_reps.append(rec); disp_reps.append(disp)
            # rep 0 also runs the random-channel control (specificity check)
            if rep == 0:
                rand = resume_fork(ckpt, perturb=dict(channel=rand_channel, alpha=args.alpha, targets=targets), **kw)
                gap_rand = [rand[i] - base[i] for i in range(len(base))]
                disp_rand = abs(gap_rand[0]) / (base[0] + 1e-9)
                _, _, rec_rand = fit_tau(gap_rand)
                base0 = base[0]
                with open(os.path.join(args.out_dir, f"recover_{it}.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["step", "base_ratio", "ma_ratio", "rand_ratio", "gap_ma", "gap_rand"])
                    for i in range(len(base)):
                        w.writerow([i, f"{base[i]:.4f}", f"{ma[i]:.4f}", f"{rand[i]:.4f}",
                                    f"{gap_ma[i]:.4f}", f"{gap_rand[i]:.4f}"])

        def ms(xs):
            xs = [x for x in xs if np.isfinite(x)]
            return (float(np.mean(xs)), float(np.std(xs))) if xs else (float("nan"), float("nan"))
        tau_m, tau_s = ms(tau_reps)
        rec_m, rec_s = ms(rec_reps)
        th_m, th_s = ms(th_reps)
        disp_m = float(np.mean(disp_reps))

        with open(summary_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [it, layer, ma_channel, rand_channel, args.alpha, f"{base0:.3f}",
                 args.n_reps, f"{disp_m:.3f}",
                 f"{tau_m:.3f}", f"{tau_s:.3f}", f"{th_m:.3f}", f"{th_s:.3f}",
                 f"{rec_m:.3f}", f"{rec_s:.3f}", f"{disp_rand:.3f}", f"{rec_rand:.3f}"])

        print(f"  MA   : disp={disp_m:.2f} tau={tau_m:6.1f}±{tau_s:.1f} "
              f"t_half={th_m:5.1f}±{th_s:.1f} recovered={rec_m:+.2f}±{rec_s:.2f}  (base ratio {base0:.1f})")
        print(f"  RAND : disp={disp_rand:.2f} recovered={rec_rand:+.2f} (specificity control)")

    print(f"\n[perturb] summary -> {summary_path}")


if __name__ == "__main__":
    main()
