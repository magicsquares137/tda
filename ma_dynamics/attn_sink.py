"""
Softmax-competition hypothesis: the sink channels compete to implement an attention
bias; once enough supply it, the softmax saturates and further channels get zero
gradient -> weight decay erodes them (competitive exclusion + fixed small count).

Timing prediction: the attention mass on the sink token should SATURATE right as the
mid-rank channels collapse (~24-32k in pythia-410m). Measure mean attention to pos0
across query positions/heads/layers over training and compare to the channel collapse
(mid-rank retention, from pythia410m_dropout.csv).
"""
import os, csv
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scale_profile import PASSAGE

MODEL = "EleutherAI/pythia-410m"; DEV = "cuda:0"
STEPS = [8000, 16000, 24000, 32000, 48000, 64000, 96000, 143000]


def sink_attention(revision, seq=256):
    tok = AutoTokenizer.from_pretrained(MODEL, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=revision, torch_dtype=torch.float32, attn_implementation="eager").to(DEV).eval()
    ids = tok(PASSAGE, return_tensors="pt").input_ids[0]
    if ids.numel() < seq: ids = ids.repeat(seq // ids.numel() + 1)
    ids = ids[:seq].view(1, seq).to(DEV)
    with torch.no_grad():
        att = model(ids, output_attentions=True).attentions      # tuple[L] of (1, H, T, T)
    per_layer = []
    for a in att:
        a = a[0].float()                                          # (H, T, T)
        # attention TO pos0 from query positions >=1, mean over heads & queries
        per_layer.append(a[:, 1:, 0].mean().item())
    del model; torch.cuda.empty_cache()
    return float(np.mean(per_layer)), float(np.max(per_layer)), per_layer


rows = []
for st in STEPS:
    m, mx, pl = sink_attention(f"step{st}")
    rows.append(dict(step=st, sink_attn_mean=m, sink_attn_maxlayer=mx))
    print(f"step {st:>7}: sink-attn mean {m:.3f}  max-layer {mx:.3f}")

with open("runs/scale_profile/pythia410m_sinkattn.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows: w.writerow(r)

# overlay with the channel collapse (mid-rank retention)
try:
    dd = {int(x["step"]): float(x["mid_mean"]) for x in csv.DictReader(open("runs/scale_profile/pythia410m_dropout.csv"))}
    print("\nstep      sink-attn(mean)   mid-rank retention (collapse)")
    for r in rows:
        mid = dd.get(r["step"], float("nan"))
        print(f"{r['step']:>7}      {r['sink_attn_mean']:.3f}            {mid:.2f}")
except Exception as e:
    print("overlay skipped:", e)
print("[attn_sink] -> runs/scale_profile/pythia410m_sinkattn.csv")
