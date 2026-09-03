"""
Count hypothesis: if the attention sink saturates at some total bias, only as many
channels as needed to supply it are maintained -> the survivor COUNT should track the
sink-attention capacity. Controlled test: seed reruns at fixed scale (70m x5, 160m x5,
410m x3) vary the count; does total sink attention track the count across seeds?
"""
import os, csv
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scale_profile import PASSAGE

DEV = "cuda:0"
FAMILIES = {
    "70m":  ["pythia-70m", "pythia-70m-seed1", "pythia-70m-seed2", "pythia-70m-seed3", "pythia-70m-seed4"],
    "160m": ["pythia-160m", "pythia-160m-seed1", "pythia-160m-seed2", "pythia-160m-seed3", "pythia-160m-seed4"],
    "410m": ["pythia-410m", "pythia-410m-seed1", "pythia-410m-seed2"],
}


def measure(name, seq=256):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float32, attn_implementation="eager").to(DEV).eval()
    ids = tok(PASSAGE, return_tensors="pt").input_ids[0]
    if ids.numel() < seq: ids = ids.repeat(seq // ids.numel() + 1)
    ids = ids[:seq].view(1, seq).to(DEV)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, output_attentions=True)
    # sink locus = max-ratio layer/pos
    best = None
    for l in range(1, len(out.hidden_states)):
        A = out.hidden_states[l][0].float().abs()                 # (T,C)
        r = A.max(-1).values / (A.median(-1).values + 1e-9); t = int(r.argmax())
        if best is None or r[t] > best[2]: best = (l, t, float(r[t]), A[t])
    v = best[3].cpu().numpy(); s = np.sort(v)[::-1]
    # count = cluster size (rank at biggest consecutive drop in top-12)
    drops = [(s[i] / (s[i + 1] + 1e-9), i + 1) for i in range(min(12, len(s) - 1))]
    cliff, count = max(drops)
    n_above = int((v > 0.15 * v.max()).sum())                     # alt count
    # sink attention: mean over layers of mean(attn to pos0 from q>=1, heads)
    pl = [a[0].float()[:, 1:, 0].mean().item() for a in out.attentions]
    sink_attn = float(np.mean(pl)); sink_attn_max = float(np.max(pl))
    del model; torch.cuda.empty_cache()
    return dict(count=count, cliff=round(cliff, 1), n_above=n_above,
                sink_attn=round(sink_attn, 3), sink_attn_max=round(sink_attn_max, 3))


rows = []
for fam, names in FAMILIES.items():
    print(f"\n== {fam} ==")
    for nm in names:
        try:
            r = measure(f"EleutherAI/{nm}")
            r = dict(family=fam, model=nm.replace("pythia-", ""), **r)
            rows.append(r)
            print(f"  {r['model']:16s} count {r['count']} (cliff {r['cliff']}x, n>15%max {r['n_above']:2d}) | "
                  f"sink_attn {r['sink_attn']:.3f} (max {r['sink_attn_max']:.3f})")
        except Exception as e:
            print(f"  [skip {nm}] {type(e).__name__}: {e}")

with open("runs/scale_profile/count_capacity.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows: w.writerow(r)

print("\n== within-family correlation: count vs sink attention ==")
for fam in FAMILIES:
    fr = [r for r in rows if r["family"] == fam]
    if len(fr) < 3: continue
    cnt = np.array([r["count"] for r in fr]); sa = np.array([r["sink_attn"] for r in fr])
    na = np.array([r["n_above"] for r in fr])
    def corr(a, b): return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    print(f"  {fam}: counts {list(cnt)} sink_attn {list(sa)} | corr(count,sink_attn)={corr(cnt,sa):+.2f} "
          f"corr(n_above,sink_attn)={corr(na,sa):+.2f}")
print("[count_capacity] -> runs/scale_profile/count_capacity.csv")
