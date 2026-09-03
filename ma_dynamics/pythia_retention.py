"""
Does the large isolation in pythia-410m (cliff 1.7x@16k -> 16.8x@143k) arise from
MAGNITUDE-GRADED protection, or an MA-SPECIFIC jump?

Fix the MA layer/token (detected at the LATE checkpoint), measure per-channel
|activation| at step16000 (still a continuum) and step143000 (bimodal). Retention =
mag143/mag16. Rank channels by their step-16k magnitude and report the RANK-WISE z
profile against the bulk retention-vs-magnitude trend. A smooth z ladder over the
top ranks = graded (no MA-specific jump). A rank-1 spike far above rank-2 = a real
MA-specific force surviving.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scale_profile import PASSAGE

MODEL = "EleutherAI/pythia-410m"; DEV = "cuda:0"


def hidden_at(revision, layer, batch=4, seq=512):
    tok = AutoTokenizer.from_pretrained(MODEL, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=revision, torch_dtype=torch.float16).to(DEV).eval()
    ids = tok(PASSAGE, return_tensors="pt").input_ids[0]
    need = batch * seq
    if ids.numel() < need: ids = ids.repeat(need // ids.numel() + 1)
    ids = ids[:need].view(batch, seq).to(DEV)
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    if layer is None:                                   # detect max-ratio layer/pos
        best = None
        for l in range(1, len(hs)):
            A = hs[l].float().abs().mean(0); r = A.max(-1).values / (A.median(-1).values + 1e-9)
            t = int(r.argmax())
            if best is None or r[t] > best[2]: best = (l, t, float(r[t]))
        layer = best[0]; pos = best[1]
    else:
        A = hs[layer].float().abs().mean(0); pos = 0
    v = hs[layer].float().abs().mean(0)[pos].cpu().numpy()
    del model; torch.cuda.empty_cache()
    return v, layer, pos


# detect the mature MA layer at 143k, then measure both checkpoints there
v143, L, pos = hidden_at("step143000", None)
v16, _, _ = hidden_at("step16000", L)
print(f"fixed locus: layer {L}, token pos {pos}")

ret = v143 / (v16 + 1e-9)
order16 = np.argsort(-v16)                               # rank by EARLY magnitude
lm = np.log10(v16 + 1e-9)
mask = np.ones_like(v16, bool); mask[order16[:16]] = False
A = np.polyfit(lm[mask], ret[mask], 1); pred = np.polyval(A, lm)
sd = (ret - pred)[mask].std()
z = (ret - pred) / sd

ma16 = int(order16[0]); giants143 = np.argsort(-v143)[:5]
print(f"corr(retention, log mag16) all-ch: {np.corrcoef(ret, lm)[0,1]:+.3f} | "
      f"bulk slope {A[0]:+.3f} (positive => bigger retained more)")
rank_of = {int(c): r for r, c in enumerate(order16, 1)}
print("late giants (rank at step16k): " + ", ".join(f"ch{int(c)}=rank{rank_of[int(c)]}" for c in giants143))
print(f"\n{'rank@16k':>8} {'chan':>5} {'mag16':>8} {'mag143':>9} {'retain':>7} {'z':>6}")
for r, c in enumerate(order16[:20], 1):
    tag = " <-MA@16k" if c == ma16 else ("  (late giant)" if c in giants143 else "")
    print(f"{r:>8} {c:>5} {v16[c]:>8.2f} {v143[c]:>9.1f} {ret[c]:>7.3f} {z[c]:>+6.2f}{tag}")
print("\nread: smooth z ladder over top ranks => graded/magnitude; rank-1 spike >> rank-2 => MA-specific.")
