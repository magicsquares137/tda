"""
Is the "massive activation" a DISTINCT CLASS (a rank-1/rank-2 gap that appears and
widens with scale) or the TOP OF A CONTINUUM (smooth, power-law-ish tail)?

For each model we find the (layer, token) whose hidden vector has the largest
max/median ratio -- "where the MA lives" -- and report that vector's sorted
per-channel magnitude profile. The two numbers that actually answer the question:

  rank1/rank2  : is the top channel separated from its NEIGHBOURS (not from the
                 median, which is tiny and inflates max/median)?
  r1_energy    : fraction of the vector's rms^2 carried by the top channel
                 (the analog of the ablation test: if ~1.0 the MA stands alone;
                  if small, many channels contribute comparably).

Run across scale (my 51M + Pythia 70M..6.9B) and see whether the gap grows.
"""
import os, csv, argparse
import numpy as np
import torch

PASSAGE = (
    "The history of science is the study of the development of science, including "
    "both the natural and social sciences. Science is a body of empirical, theoretical, "
    "and practical knowledge about the natural world, produced by researchers making use "
    "of observation and explanation. In many parts of the world, the earliest recorded "
    "evidence of scientific reasoning comes from written records. The nature of the "
    "physical world was investigated by philosophers who sought general principles that "
    "could account for the phenomena they observed, and their explanations laid the "
    "foundation for later developments. Over centuries, methods of inquiry were refined, "
    "instruments improved, and mathematics increasingly applied, so that predictions could "
    "be tested against careful measurement and the results shared among a growing community."
) * 8


def profile_from_absmean(A):
    """A: (T, C) mean-over-batch abs activations. Returns dict for the max-ratio token."""
    med = A.median(dim=-1).values                         # (T,)
    mx = A.max(dim=-1).values                              # (T,)
    ratio = mx / (med + 1e-12)
    t = int(ratio.argmax().item())
    v = A[t]                                               # (C,) the MA vector
    sv, _ = torch.sort(v, descending=True)
    energy = (sv ** 2)
    return {
        "pos": t,
        "sorted": sv.cpu().numpy(),
        "median": float(v.median().item()),
        "ratio": float(ratio[t].item()),
        "r1_energy_frac": float((energy[0] / energy.sum()).item()),
    }


def summarize(name, nparams, layer, prof, K=12):
    s = prof["sorted"]
    r1, r2 = float(s[0]), float(s[1])
    med = prof["median"]
    print(f"\n== {name} ({nparams/1e6:.0f}M) | MA layer {layer} pos {prof['pos']} ==")
    print(f"   top{K}: " + ", ".join(f"{x:.1f}" for x in s[:K]))
    print(f"   median {med:.3f} | rank1/rank2 {r1/r2:.2f} | max/median {r1/(med+1e-12):.0f} "
          f"| rank1 energy frac {prof['r1_energy_frac']:.3f}")
    return dict(model=name, n_params=int(nparams), layer=int(layer), pos=int(prof["pos"]),
                median=med, rank1=r1, rank2=r2, r1_over_r2=r1/r2,
                max_over_median=r1/(med+1e-12), r1_energy_frac=prof["r1_energy_frac"])


# ---------- my 51M model ----------
def run_mine(ckpt, dev, batch_size=8):
    from common import build_model, Data
    ck = torch.load(ckpt, map_location=dev)
    ma = ck["model_args"]
    model, cfg = build_model(ma["n_layer"], ma["n_head"], ma["n_embd"], ma["block_size"],
                             vocab_size=ma["vocab_size"], bias=ma["bias"], device=dev,
                             norm=ma.get("norm", "layer"))
    model.load_state_dict(ck["model"]); model.eval()
    data = Data(ma["block_size"], batch_size, dev)
    X, _ = data.fixed_probe_batch()
    caps = {}
    hs = [model.transformer.h[l].register_forward_hook(
            (lambda l: lambda m, i, o: caps.__setitem__(l, o.detach()))(l))
          for l in range(cfg.n_layer)]
    with torch.no_grad():
        model(X)
    for h in hs: h.remove()
    # per layer: mean over batch of |h| -> (T,C); pick global max-ratio layer
    best = None
    for l in range(cfg.n_layer):
        A = caps[l].float().abs().mean(0)                 # (T,C)
        prof = profile_from_absmean(A)
        if best is None or prof["ratio"] > best[1]["ratio"]:
            best = (l, prof)
    nparams = sum(p.numel() for p in model.parameters())
    del model; torch.cuda.empty_cache()
    return best[0], best[1], nparams


# ---------- Pythia via HF ----------
def run_pythia(size, dev, batch_size=4, seqlen=512, **_):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = f"EleutherAI/pythia-{size}"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(dev).eval()
    ids = tok(PASSAGE, return_tensors="pt").input_ids[0]
    need = batch_size * seqlen
    if ids.numel() < need:
        ids = ids.repeat((need // ids.numel()) + 1)
    ids = ids[:need].view(batch_size, seqlen).to(dev)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = out.hidden_states                                # tuple len n_layer+1
    best = None
    for l in range(1, len(hs)):
        A = hs[l].float().abs().mean(0)                   # (T,C)
        prof = profile_from_absmean(A)
        if best is None or prof["ratio"] > best[1]["ratio"]:
            best = (l, prof)
    nparams = sum(p.numel() for p in model.parameters())
    del model; torch.cuda.empty_cache()
    return best[0], best[1], nparams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ckpt", default="runs/len_51m_40k/ckpt_40000.pt")
    p.add_argument("--pythia", default="70m,160m,410m,1.4b,2.8b,6.9b")
    p.add_argument("--pbatch", type=int, default=4)
    p.add_argument("--pseq", type=int, default=512)
    p.add_argument("--skip_mine", action="store_true")
    p.add_argument("--append", action="store_true", help="append to existing summary/profiles")
    p.add_argument("--outdir", default="runs/scale_profile")
    args = p.parse_args()
    dev = args.device
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.outdir, exist_ok=True)

    summary, profiles = [], []

    if not args.skip_mine:
        l, prof, npar = run_mine(args.ckpt, dev)
        summary.append(summarize("ours-51M", npar, l, prof))
        for r, m in enumerate(prof["sorted"][:64], 1):
            profiles.append(("ours-51M", npar, r, float(m), float(m)/(prof["median"]+1e-12)))

    for size in [s for s in args.pythia.split(",") if s]:
        try:
            l, prof, npar = run_pythia(size, dev, batch_size=args.pbatch, seqlen=args.pseq)
            summary.append(summarize(f"pythia-{size}", npar, l, prof))
            for r, m in enumerate(prof["sorted"][:64], 1):
                profiles.append((f"pythia-{size}", npar, r, float(m), float(m)/(prof["median"]+1e-12)))
        except Exception as e:
            print(f"[skip pythia-{size}] {type(e).__name__}: {e}")

    smode = "a" if args.append else "w"
    spath = os.path.join(args.outdir, "summary.csv")
    write_hdr = (not args.append) or (not os.path.exists(spath))
    with open(spath, smode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        if write_hdr: w.writeheader()
        for row in summary: w.writerow(row)
    ppath = os.path.join(args.outdir, "profiles.csv")
    with open(ppath, smode, newline="") as f:
        w = csv.writer(f)
        if write_hdr: w.writerow(["model", "n_params", "rank", "mag", "mag_over_median"])
        for row in profiles: w.writerow(row)
    print(f"\n[scale_profile] -> {args.outdir}/summary.csv, profiles.csv")


if __name__ == "__main__":
    main()
