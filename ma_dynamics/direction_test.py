"""
Is the declining-regime ceiling a REPRESENTATIONAL cost (distortion of the sink
direction x_hat) or an rms cost? Two matched-rms displacements:

  Kick A: scale the MA channel only            -> same Delta-rms, LARGE x_hat distortion
  Kick B: scale the top-K channels proportionally -> same Delta-rms, x_hat PRESERVED

Resume both (frozen LR, fixed batches) alongside an unperturbed control, and track,
per step, rms(x_sink) and the x_hat distortion (1 - cos to the control's x_hat).

Prediction (representational): the x_hat distortion of Kick A is corrected
(pulled back) while Kick B's rms excess persists -> the ceiling is about direction,
not rms. If instead both rms returns equally, the ceiling is about rms.
"""
import os, csv, argparse, math
import numpy as np
import torch
from common import make_ctx, Data
from ma_probe import MAProbe, scale_channel_all_layers
from perturb_recover import load_model_opt, scheduled_lr


def sink_vec(model, L, probeX, ctx):
    cap = {}
    h = model.transformer.h[L].register_forward_hook(lambda m, i, o: cap.__setitem__("r", o.detach()))
    with torch.no_grad(), ctx:
        model(probeX)
    h.remove()
    x = cap["r"].float()[:, 0, :].mean(0)          # mean sink residual (C,)
    return x


def resume_track(ckpt, dev, ctx, data, probeX, L, kick, steps, lr, grad_accum, grad_clip, dtype, seed=999):
    model, opt, cfg = load_model_opt(ckpt, dev)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    if kick is not None:
        for c, f in kick:                          # list of (channel, factor)
            scale_channel_all_layers(model, c, f)
    gen = torch.Generator().manual_seed(seed)
    xs = [sink_vec(model, L, probeX, ctx).cpu()]
    for _ in range(steps):
        for g in opt.param_groups: g["lr"] = lr
        for _ in range(grad_accum):
            X, Y = data.get_batch("train", generator=gen)
            with ctx:
                _, loss = model(X, Y); loss = loss / grad_accum
            scaler.scale(loss).backward()
        if grad_clip: scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        xs.append(sink_vec(model, L, probeX, ctx).cpu())
    del model, opt, scaler; torch.cuda.empty_cache()
    return torch.stack(xs).numpy()                 # (steps+1, C)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/len_51m_40k/ckpt_30000.pt")
    p.add_argument("--layer", type=int, default=2)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", default="runs/len_51m_40k/direction_test.csv")
    args = p.parse_args()
    dev = args.device; ctx = make_ctx(dev, args.dtype)
    ck = torch.load(args.ckpt, map_location=dev); it0 = ck.get("iter_num", 0)
    L = args.layer; bs = ck["model_args"]["block_size"]
    data = Data(bs, args.batch_size, dev); probeX = data.fixed_probe_batch()[0]
    lr = scheduled_lr(it0, ck["config"])
    C = ck["model_args"]["n_embd"]; d = C

    # detect MA channel + top-K by sink magnitude on the loaded model
    m0, _o, cfg = load_model_opt(ck, dev)
    x0 = sink_vec(m0, L, probeX, ctx).cpu().numpy()
    ma = int(np.argmax(np.abs(x0)))
    topk = list(np.argsort(-np.abs(x0))[:args.topk])
    if ma not in topk: topk = [ma] + topk[:-1]
    rms0 = float(np.sqrt((x0**2).mean()))

    # Kick A: MA channel x alpha. Measure resulting rms to set Kick B's beta (match rms).
    mA, _o2, _c = load_model_opt(ck, dev); scale_channel_all_layers(mA, ma, args.alpha)
    rmsA = float(np.sqrt((sink_vec(mA, L, probeX, ctx).cpu().numpy()**2).mean())); del mA
    beta = rmsA / rms0                                   # first guess (top-K dominate rms)
    for _ in range(3):                                    # refine beta to match rmsA
        mB, _o3, _c = load_model_opt(ck, dev)
        for c in topk: scale_channel_all_layers(mB, int(c), beta)
        rmsB = float(np.sqrt((sink_vec(mB, L, probeX, ctx).cpu().numpy()**2).mean())); del mB
        beta *= rmsA / rmsB
        if abs(rmsB - rmsA) / rmsA < 0.03: break
    del m0; torch.cuda.empty_cache()
    print(f"ckpt {it0} L{L} | MA ch {ma} | rms0 {rms0:.2f} -> kickA(x{args.alpha}) {rmsA:.2f} ; "
          f"kickB(top{args.topk} x{beta:.2f}) {rmsB:.2f} (matched) | lr {lr:.2e}")

    kw = dict(ckpt=ck, dev=dev, ctx=ctx, data=data, probeX=probeX, L=L, steps=args.steps,
              lr=lr, grad_accum=args.grad_accum, grad_clip=args.grad_clip, dtype=args.dtype)
    ctrlX = resume_track(kick=None, **kw)
    aX = resume_track(kick=[(ma, args.alpha)], **kw)
    bX = resume_track(kick=[(int(c), beta) for c in topk], **kw)

    def rms(v): return np.sqrt((v**2).mean(-1))
    def xhat(v): return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    ch = xhat(ctrlX)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "rmsA", "rmsB", "rms_ctrl", "distortA", "distortB"])
        for s in range(len(ctrlX)):
            dA = 1 - float(np.dot(xhat(aX[s]), ch[s])); dB = 1 - float(np.dot(xhat(bX[s]), ch[s]))
            w.writerow([s, f"{rms(aX[s]):.4f}", f"{rms(bX[s]):.4f}", f"{rms(ctrlX[s]):.4f}", f"{dA:.5f}", f"{dB:.5f}"])
    # summary: how much of the initial excess (rms and distortion) is removed by the end?
    def frac(v, ctrl): return (v[0]-v[-1])/((v[0]-ctrl[-1]) + 1e-9)
    rA, rB, rc = rms(aX), rms(bX), rms(ctrlX)
    dA0, dAe = 1-np.sum(xhat(aX)*ch,1)[0], 1-np.sum(xhat(aX)*ch,1)[-1]
    dB0, dBe = 1-np.sum(xhat(bX)*ch,1)[0], 1-np.sum(xhat(bX)*ch,1)[-1]
    print(f"  rms excess removed:   A {100*frac(rA,rc):.0f}%   B {100*frac(rB,rc):.0f}%")
    print(f"  x_hat distortion:     A {dA0:.4f}->{dAe:.4f} ({100*(dA0-dAe)/(dA0+1e-9):.0f}% corrected)   "
          f"B {dB0:.4f}->{dBe:.4f} ({100*(dB0-dBe)/(dB0+1e-9):.0f}% corrected)")
    print(f"[direction_test] -> {args.out}")


if __name__ == "__main__":
    main()
