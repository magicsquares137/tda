"""
Is the early sink lock-in STRUCTURAL (same channel indices win across seeds) or
STOCHASTIC (different seeds pick different channels)? Same architecture + same data,
different RNG seed. Compare the top sink-channel INDICES at the mature MA locus.
  same indices  -> structural: the architecture/data funnel the sink to fixed coords
  diff indices  -> stochastic early symmetry-breaking: whoever leads by ~4k wins
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np, torch


def top_ours(ckpt, dev, k=8):
    from common import build_model, Data
    ck = torch.load(ckpt, map_location=dev); ma = ck["model_args"]
    model, cfg = build_model(ma["n_layer"], ma["n_head"], ma["n_embd"], ma["block_size"],
                             vocab_size=ma["vocab_size"], bias=ma["bias"], device=dev, norm=ma.get("norm", "layer"))
    model.load_state_dict(ck["model"]); model.eval()
    data = Data(ma["block_size"], 8, dev); X, _ = data.fixed_probe_batch()
    caps = {}
    hs = [model.transformer.h[l].register_forward_hook((lambda l: lambda m, i, o: caps.__setitem__(l, o.detach()))(l))
          for l in range(cfg.n_layer)]
    with torch.no_grad(): model(X)
    for h in hs: h.remove()
    best = None
    for l in range(cfg.n_layer):
        A = caps[l].float().abs().mean(0); r = A.max(-1).values / (A.median(-1).values + 1e-9); t = int(r.argmax())
        if best is None or r[t] > best[2]: best = (l, t, float(r[t]), A[t])
    v = best[3].cpu().numpy(); idx = np.argsort(-v)[:k]
    del model; torch.cuda.empty_cache()
    return best[0], best[1], [(int(c), float(v[c])) for c in idx]


def top_pythia(name, dev, k=8):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scale_profile import PASSAGE
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(dev).eval()
    ids = tok(PASSAGE, return_tensors="pt").input_ids[0]
    need = 4 * 512
    if ids.numel() < need: ids = ids.repeat(need // ids.numel() + 1)
    ids = ids[:need].view(4, 512).to(dev)
    with torch.no_grad(): hs = model(ids, output_hidden_states=True).hidden_states
    best = None
    for l in range(1, len(hs)):
        A = hs[l].float().abs().mean(0); r = A.max(-1).values / (A.median(-1).values + 1e-9); t = int(r.argmax())
        if best is None or r[t] > best[2]: best = (l, t, float(r[t]), A[t])
    v = best[3].cpu().numpy(); idx = np.argsort(-v)[:k]
    del model; torch.cuda.empty_cache()
    return best[0], best[1], [(int(c), float(v[c])) for c in idx]


def report(name, layer, pos, tops):
    chans = [c for c, _ in tops]
    print(f"  {name:22s} L{layer:>2} pos{pos:<3} top: " + " ".join(f"{c}({m:.0f})" for c, m in tops))
    return chans


def overlap(a, b, k=3):
    sa, sb = set(a[:k]), set(b[:k])
    return len(sa & sb), sorted(sa & sb)


dev = "cuda:0"
torch.backends.cuda.matmul.allow_tf32 = True

print("== OUR 51M (seed 1337 vs 2024; same arch+data) ==")
oa = report("seed1337", *top_ours("runs/len_51m_40k/ckpt_40000.pt", dev))
ob = report("seed2024", *top_ours("runs/seed2024_51m_40k/ckpt_40000.pt", dev))
n, sh = overlap(oa, ob, 3); print(f"  -> top-3 shared: {n}/3  {sh}")

print("\n== PYTHIA-410m (default vs seed1 vs seed2; same data) ==")
runs = {}
for nm in ["EleutherAI/pythia-410m", "EleutherAI/pythia-410m-seed1", "EleutherAI/pythia-410m-seed2"]:
    try:
        runs[nm] = report(nm.split("/")[-1], *top_pythia(nm, dev))
    except Exception as e:
        print(f"  [skip {nm}] {type(e).__name__}: {e}")
ks = list(runs)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        n, sh = overlap(runs[ks[i]], runs[ks[j]], 3)
        print(f"  -> {ks[i].split('/')[-1]} vs {ks[j].split('/')[-1]}: top-3 shared {n}/3  {sh}")
