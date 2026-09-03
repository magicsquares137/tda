"""
Stage 0 -- grow massive activations in a mini SLM and record their trajectory.

Trains a small GPT-2-style model from scratch on OpenWebText/Dolma, measures the
per-layer max/median activation ratio on a *fixed* probe batch every
`probe_interval` iters (cheap -> dense trajectory), and saves full checkpoints
(weights + optimizer, needed for perturb/resume) every `ckpt_interval` iters.

Outputs:
  <out_dir>/trajectory.csv   one row per probe: iter, lr, loss, and per-layer ratio
  <out_dir>/ckpt_<iter>.pt   full checkpoints for the perturb/recover stage

Usage (single GPU):
  python train_dense.py --device=cuda:0 --out_dir=runs/pilot --max_iters=4000
Overrides use the same `--key=value` convention as luvai/train.py.
"""

import os
import csv
import math
import time
import argparse

import torch

from common import build_model, make_ctx, Data
from ma_probe import MAProbe


def get_lr(it, lr, warmup, decay_iters, min_lr, decay=True):
    if not decay:
        return lr
    if it < warmup:
        return lr * (it + 1) / (warmup + 1)
    if it > decay_iters:
        return min_lr
    r = (it - warmup) / (decay_iters - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * r))
    return min_lr + coeff * (lr - min_lr)


def parse():
    p = argparse.ArgumentParser()
    # model (~30M default: 8 layers, 512 wide)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=512)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--bias", action="store_true", default=False)
    p.add_argument("--norm", type=str, default="layer", choices=["layer", "rms"])
    # optim
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=6e-4)
    p.add_argument("--min_lr", type=float, default=6e-5)
    p.add_argument("--warmup_iters", type=int, default=200)
    p.add_argument("--max_iters", type=int, default=4000)
    p.add_argument("--lr_decay_iters", type=int, default=4000)
    p.add_argument("--weight_decay", type=float, default=1e-1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    # probing / checkpointing
    p.add_argument("--probe_interval", type=int, default=25)
    p.add_argument("--ckpt_interval", type=int, default=100)
    p.add_argument("--ckpt_start", type=int, default=0, help="start saving full ckpts at this iter")
    # io / system
    p.add_argument("--out_dir", type=str, default="runs/pilot")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    args = parse()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, cfg = build_model(args.n_layer, args.n_head, args.n_embd,
                             args.block_size, bias=args.bias,
                             dropout=args.dropout, device=args.device, norm=args.norm)
    ctx = make_ctx(args.device, args.dtype)
    data = Data(args.block_size, args.batch_size, args.device)
    probe_batch = data.fixed_probe_batch()  # (X, Y); we only need X
    probe_X = probe_batch[0]

    opt = model.configure_optimizers(args.weight_decay, args.learning_rate,
                                     (args.beta1, args.beta2), "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    probe = MAProbe(model).attach()

    traj_path = os.path.join(args.out_dir, "trajectory.csv")
    header = ["iter", "lr", "loss"] + [f"ratio_L{l}" for l in range(cfg.n_layer)] \
        + ["birth_layer", "birth_ratio", "birth_channel", "birth_position"]
    with open(traj_path, "w", newline="") as f:
        csv.writer(f).writerow(header)

    model_args = dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
                      block_size=cfg.block_size, bias=cfg.bias,
                      vocab_size=cfg.vocab_size, dropout=cfg.dropout, norm=args.norm)

    def save_ckpt(it):
        ck = {"model": model.state_dict(), "optimizer": opt.state_dict(),
              "model_args": model_args, "iter_num": it, "config": vars(args)}
        torch.save(ck, os.path.join(args.out_dir, f"ckpt_{it}.pt"))

    print(f"[stage0] {model.get_num_params()/1e6:.1f}M params | out={args.out_dir} | device={args.device}")
    X, Y = data.get_batch("train")
    t0 = time.time()
    last_loss = float("nan")
    for it in range(args.max_iters + 1):
        lr = get_lr(it, args.learning_rate, args.warmup_iters,
                    args.lr_decay_iters, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        # -- probe (dense, cheap) --
        if it % args.probe_interval == 0:
            m = probe.measure(probe_X, ctx=ctx)
            bl = int(max(range(cfg.n_layer), key=lambda l: m["ratio"][l]))
            row = [it, lr, last_loss] + [f"{m['ratio'][l]:.4f}" for l in range(cfg.n_layer)] \
                + [bl, f"{m['ratio'][bl]:.4f}", m["ma_channel"][bl], m["ma_position"][bl]]
            with open(traj_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            print(f"iter {it:5d} | loss {last_loss:.3f} | lr {lr:.2e} | "
                  f"birth L{bl} ratio {m['ratio'][bl]:7.1f} chan {m['ma_channel'][bl]} pos {m['ma_position'][bl]} | "
                  f"{(time.time()-t0):.0f}s")

        # -- full checkpoint (for resume) --
        if it % args.ckpt_interval == 0 and it >= args.ckpt_start and it > 0:
            save_ckpt(it)

        if it == args.max_iters:
            break

        # -- train step (grad accum) --
        for micro in range(args.grad_accum):
            with ctx:
                _, loss = model(X, Y)
                loss = loss / args.grad_accum
            X, Y = data.get_batch("train")
            scaler.scale(loss).backward()
        if args.grad_clip != 0.0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        last_loss = loss.item() * args.grad_accum

    probe.detach()
    print(f"[stage0] done. trajectory -> {traj_path}")


if __name__ == "__main__":
    main()
