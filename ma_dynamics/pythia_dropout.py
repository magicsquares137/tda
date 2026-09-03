"""
Is the late large-channel drop-out GRADUAL (continuous with the measured decline)
or an ABRUPT onset (a separate transition)?

Fix the mature MA locus (L,pos detected at step143000). Rank channels by magnitude
at step16000. Measure each channel's |activation| across intermediate checkpoints
16k..143k and track retention = mag(step)/mag(16k). If ranks 4-12 fade smoothly
from ~32k onward -> plausibly the same per-weight-efficiency decline, unification
holds. If they hold flat then collapse abruptly at some step -> separate transition.
"""
import os, csv
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scale_profile import PASSAGE

MODEL = "EleutherAI/pythia-410m"; DEV = "cuda:0"
STEPS = [16000, 24000, 32000, 48000, 64000, 96000, 128000, 143000]


def vec_at(revision, layer, pos, batch=4, seq=512):
    tok = AutoTokenizer.from_pretrained(MODEL, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=revision, torch_dtype=torch.float16).to(DEV).eval()
    ids = tok(PASSAGE, return_tensors="pt").input_ids[0]
    need = batch * seq
    if ids.numel() < need: ids = ids.repeat(need // ids.numel() + 1)
    ids = ids[:need].view(batch, seq).to(DEV)
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    if layer is None:
        best = None
        for l in range(1, len(hs)):
            A = hs[l].float().abs().mean(0); r = A.max(-1).values / (A.median(-1).values + 1e-9); t = int(r.argmax())
            if best is None or r[t] > best[2]: best = (l, t, float(r[t]))
        layer, pos = best[0], best[1]
    v = hs[layer].float().abs().mean(0)[pos].cpu().numpy()
    del model; torch.cuda.empty_cache()
    return v, layer, pos


_, L, POS = vec_at("step143000", None, None)
print(f"fixed locus: layer {L} pos {POS}")
vecs = {}
for st in STEPS:
    v, _, _ = vec_at(f"step{st}", L, POS)
    vecs[st] = v
    print(f"step{st} measured")

v16 = vecs[16000]
order = np.argsort(-v16)                      # rank by 16k magnitude
top = order[:12]

def cliff(v):
    s = np.sort(v)[::-1][:15]
    return float(max(s[i] / (s[i+1] + 1e-9) for i in range(len(s)-1)))

rows = []
for st in STEPS:
    v = vecs[st]
    ret = {f"r{r}": float(v[c] / (v16[c] + 1e-9)) for r, c in enumerate(top, 1)}
    row = dict(step=st, cliff=cliff(v),
               top3_mean=float(np.mean([v[c]/(v16[c]+1e-9) for c in order[:3]])),
               mid_mean=float(np.mean([v[c]/(v16[c]+1e-9) for c in order[3:12]])),
               **ret)
    rows.append(row)

with open("runs/scale_profile/pythia410m_dropout.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows: w.writerow(r)

print(f"\n{'step':>7} {'cliff':>6} {'top3':>6} {'mid(4-12)':>9} | per-rank retention (ranks 4-12)")
for r in rows:
    mids = " ".join(f"{r[f'r{k}']:.2f}" for k in range(4, 13))
    print(f"{r['step']:>7} {r['cliff']:>5.1f}x {r['top3_mean']:>6.2f} {r['mid_mean']:>9.2f} | {mids}")
print("[pythia_dropout] -> runs/scale_profile/pythia410m_dropout.csv")
