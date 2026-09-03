"""
Is the gradient-attenuation FLOOR (Chen et al.) MA-specific or magnitude/rms-specific?
No training: one forward+backward per condition.

Their mechanism: the norm Jacobian attenuates the sink gradient in proportion to
1/rms(x). So the downstream sink gradient g should obey g * rms ~ const if
attenuation is purely a function of total rms. We ablate (a) the MA channel and
(b) several other large channels individually, measuring rms(x_sink) and the
downstream gradient g at the sink. If every point lies on one g ~ 1/rms curve
(g*rms constant), the floor is on TOTAL rms and the MA merely dominates it
(magnitude-specific). If ablating the MA raises g far above the 1/rms prediction,
the MA does extra, specific attenuation (MA-specific).
"""
import os, csv, argparse
import numpy as np
import torch
from common import build_model, Data
from ma_probe import MAProbe, scale_channel_all_layers


def measure(ck, dev, L, X, Y, ablate=None):
    ma = ck["model_args"]
    model, cfg = build_model(ma["n_layer"], ma["n_head"], ma["n_embd"], ma["block_size"],
                             vocab_size=ma["vocab_size"], bias=ma["bias"], device=dev,
                             norm=ma.get("norm", "layer"))
    model.load_state_dict(ck["model"]); model.train()
    if ablate is not None:
        for c in ablate: scale_channel_all_layers(model, int(c), 0.0)
    cap = {}
    h = model.transformer.h[L].register_forward_hook(
        lambda m, i, o: (o.retain_grad(), cap.__setitem__("r", o)) and None)
    model.zero_grad(set_to_none=True)
    _, loss = model(X, Y); loss.backward()
    r = cap["r"]; x = r[:, 0, :].float(); g = r.grad[:, 0, :].float()
    C = x.shape[-1]
    rms = (x.pow(2).mean(-1).sqrt()).mean().item()
    gnorm = g.norm(dim=-1).mean().item()
    h.remove(); del model; torch.cuda.empty_cache()
    return rms, gnorm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/len_51m_40k/ckpt_16000.pt")
    p.add_argument("--layer", type=int, default=-1, help="-1 = auto (max-ratio MA layer)")
    p.add_argument("--n_other", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="runs/len_51m_40k/floor_test.csv")
    args = p.parse_args()
    dev = args.device
    torch.backends.cuda.matmul.allow_tf32 = True
    ck = torch.load(args.ckpt, map_location=dev)
    bs = ck["model_args"]["block_size"]
    data = Data(bs, args.batch_size, dev); X, Y = data.fixed_probe_batch()

    # detect MA layer + channel + other large channels (by sink magnitude)
    ma = ck["model_args"]
    m0, cfg = build_model(ma["n_layer"], ma["n_head"], ma["n_embd"], ma["block_size"],
                          vocab_size=ma["vocab_size"], bias=ma["bias"], device=dev, norm=ma.get("norm","layer"))
    m0.load_state_dict(ck["model"])
    pr = MAProbe(m0).attach(); meas = pr.measure(X); pr.detach()
    L = args.layer if args.layer >= 0 else int(max(range(cfg.n_layer), key=lambda l: meas["ratio"][l]))
    # sink vector at L
    cap = {}
    hh = m0.transformer.h[L].register_forward_hook(lambda m,i,o: cap.__setitem__("r", o.detach()))
    with torch.no_grad(): m0(X)
    hh.remove()
    xs = cap["r"][:,0,:].float().abs().mean(0).cpu().numpy()
    ma_c = int(np.argmax(xs))
    others = [int(c) for c in np.argsort(-xs) if int(c) != ma_c][:args.n_other]
    del m0; torch.cuda.empty_cache()

    rows = []
    rms0, g0 = measure(ck, dev, L, X, Y, ablate=None)
    rows.append(("control", None, rms0, g0))
    r, g = measure(ck, dev, L, X, Y, ablate=[ma_c]); rows.append(("ablate_MA", ma_c, r, g))
    for c in others:
        r, g = measure(ck, dev, L, X, Y, ablate=[c]); rows.append((f"ablate_ch{c}", c, r, g))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cond","channel","rms","g","g_x_rms","g_pred_1overrms","g_over_pred"])
        print(f"floor test @ {ck.get('iter_num','?')} L{L} | MA ch {ma_c} | control rms {rms0:.3f} g {g0:.5f} (g*rms={g0*rms0:.4f})")
        print(f"{'condition':>12} {'rms':>7} {'g':>9} {'g*rms':>7} {'pred g(1/rms)':>13} {'g/pred':>7}")
        for name, c, r, g in rows:
            pred = g0 * rms0 / r                      # if g ~ 1/rms
            over = g / pred
            w.writerow([name, c, f"{r:.4f}", f"{g:.6f}", f"{g*r:.5f}", f"{pred:.6f}", f"{over:.3f}"])
            tag = " <-MA" if name == "ablate_MA" else ""
            print(f"{name:>12} {r:>7.3f} {g:>9.5f} {g*r:>7.4f} {pred:>13.5f} {over:>7.3f}{tag}")
    print("\ng/pred ~ 1.0 for all => attenuation is pure 1/rms (magnitude-specific).")
    print("ablate_MA g/pred >> others => MA does specific extra attenuation (MA-specific floor).")


if __name__ == "__main__":
    main()
