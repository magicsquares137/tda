"""
Harden the two claims before writing them down:
  (1) the sink channels are PASSAGE-ROBUST (same channels regardless of input text)
  (2) the early LOCK-IN (winners settled by ~step 4k) is SEED-GENERAL and holds in
      a second seed of both Pythia-410m and our 51M.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np, torch

P1 = ("The history of science is the study of the development of science. Science is a body of "
      "empirical, theoretical, and practical knowledge about the natural world, produced by "
      "researchers making use of observation and explanation of the phenomena they encounter. ") * 12
P2 = ("A compiler translates source code written in a high-level programming language into machine "
      "instructions. Modern optimizing compilers perform register allocation, loop unrolling, and "
      "dead-code elimination to make the resulting binaries run faster on the target hardware. ") * 12
P3 = ("The river wound slowly through the valley, past meadows where cattle grazed in the long "
      "afternoon light. By evening the mist would gather over the water and the herons would return "
      "to the shallows to fish among the reeds before darkness settled on the hills. ") * 12
DEV = "cuda:0"


def ids_of(tok, passage, need=4 * 512):
    x = tok(passage, return_tensors="pt").input_ids[0]
    if x.numel() < need: x = x.repeat(need // x.numel() + 1)
    return x[:need].view(4, 512).to(DEV)


def pythia_locus_vec(name, revision, layer=None, pos=None, passage=P1):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, torch_dtype=torch.float16).to(DEV).eval()
    with torch.no_grad():
        hs = model(ids_of(tok, passage), output_hidden_states=True).hidden_states
    if layer is None:
        best = None
        for l in range(1, len(hs)):
            A = hs[l].float().abs().mean(0); r = A.max(-1).values / (A.median(-1).values + 1e-9); t = int(r.argmax())
            if best is None or r[t] > best[2]: best = (l, t, float(r[t]))
        layer, pos = best[0], best[1]
    v = hs[layer].float().abs().mean(0)[pos].cpu().numpy()
    del model; torch.cuda.empty_cache()
    return v, layer, pos


def ours_locus_vec(ckpt, layer=None):
    from common import build_model, Data
    ck = torch.load(ckpt, map_location=DEV); ma = ck["model_args"]
    model, cfg = build_model(ma["n_layer"], ma["n_head"], ma["n_embd"], ma["block_size"],
                             vocab_size=ma["vocab_size"], bias=ma["bias"], device=DEV, norm=ma.get("norm", "layer"))
    model.load_state_dict(ck["model"]); model.eval()
    data = Data(ma["block_size"], 8, DEV); X, _ = data.fixed_probe_batch()
    caps = {}
    hs = [model.transformer.h[l].register_forward_hook((lambda l: lambda m, i, o: caps.__setitem__(l, o.detach()))(l))
          for l in range(cfg.n_layer)]
    with torch.no_grad(): model(X)
    for h in hs: h.remove()
    if layer is None:
        best = None
        for l in range(cfg.n_layer):
            A = caps[l].float().abs().mean(0); r = A.max(-1).values / (A.median(-1).values + 1e-9); t = int(r.argmax())
            if best is None or r[t] > best[2]: best = (l, t, float(r[t]))
        layer = best[0]
    v = caps[layer].float().abs().mean(0)[0].cpu().numpy()
    del model; torch.cuda.empty_cache()
    return v, layer


def rank(v, c): return int((v > v[c]).sum()) + 1
def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


print("== (1) PASSAGE ROBUSTNESS: pythia-410m (final), 3 different texts ==")
tops = {}
for nm, p in [("science", P1), ("compiler", P2), ("river", P3)]:
    v, L, POS = pythia_locus_vec("EleutherAI/pythia-410m", "step143000", passage=p)
    t = [int(c) for c in np.argsort(-v)[:3]]
    tops[nm] = t; print(f"  {nm:9s} L{L} pos{POS} top-3: {t}")
allshared = set(tops["science"]) & set(tops["compiler"]) & set(tops["river"])
print(f"  -> channels shared across all 3 passages: {sorted(allshared)} ({len(allshared)}/3)")

print("\n== (2) SECOND-SEED EARLY LOCK-IN: pythia-410m-seed1 ==")
vfin, L1, P1pos = pythia_locus_vec("EleutherAI/pythia-410m-seed1", "step143000")
win1 = [int(c) for c in np.argsort(-vfin)[:4]]
print(f"  seed1 mature locus L{L1} pos{P1pos}; its winners: {win1}")
steps = [4000, 8000, 16000, 32000, 64000, 143000]
vv = {}
for st in steps:
    vv[st], _, _ = pythia_locus_vec("EleutherAI/pythia-410m-seed1", f"step{st}", layer=L1, pos=P1pos)
print(f"  {'chan':>5} " + " ".join(f"r@{s//1000}k" for s in steps))
for c in win1:
    print(f"  {c:>5} " + " ".join(f"{rank(vv[s], c):>5}" for s in steps))
sp = spearman(vv[8000], vv[143000])
print(f"  -> Spearman(rank@8k, rank@143k) all channels: {sp:+.2f}")

print("\n== (3) SECOND-SEED EARLY LOCK-IN: our 51M seed2024 ==")
vend, L2 = ours_locus_vec("runs/seed2024_51m_40k/ckpt_40000.pt")
win2 = [int(c) for c in np.argsort(-vend)[:4]]
print(f"  seed2024 locus L{L2}; its winners: {win2}")
osteps = [2000, 4000, 8000, 16000, 40000]
ov = {}
for st in osteps:
    ov[st], _ = ours_locus_vec(f"runs/seed2024_51m_40k/ckpt_{st}.pt", layer=L2)
print(f"  {'chan':>5} " + " ".join(f"r@{s//1000}k" for s in osteps))
for c in win2:
    print(f"  {c:>5} " + " ".join(f"{rank(ov[s], c):>5}" for s in osteps))
print(f"  -> Spearman(rank@4k, rank@40k) all channels: {spearman(ov[4000], ov[40000]):+.2f}")
