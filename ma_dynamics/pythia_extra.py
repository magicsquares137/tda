"""Two data-strengthening runs for the writeup:
 (1) passage robustness x10 (does the sink live on the same channels across 10 texts?)
 (2) rank trajectory: rank of winners / rank-4 collapser / a climber across training,
     at the mature locus (L17 pos0) -> the lock-in figure.
"""
import os, csv
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "EleutherAI/pythia-410m"; DEV = "cuda:0"; L, POS = 17, 0

PASSAGES = [
 "The history of science is the study of the development of science over many centuries.",
 "A compiler translates high-level source code into machine instructions for a processor.",
 "The river wound slowly through the valley past meadows where cattle grazed at dusk.",
 "Photosynthesis converts carbon dioxide and water into glucose using light energy.",
 "In contract law an agreement requires offer, acceptance, consideration, and intent.",
 "The orchestra tuned to the oboe before the conductor raised the baton for the symphony.",
 "Tectonic plates grind past one another along faults, building stress that releases as quakes.",
 "She simmered the stock for hours, skimming the surface until the broth ran clear and golden.",
 "The spacecraft executed a gravity assist around Jupiter to gain speed for the outer planets.",
 "Supply and demand set prices in a competitive market absent externalities or collusion.",
]


def sink_vec(model, tok, text):
    ids = tok(text * 20, return_tensors="pt").input_ids[0][:256].view(1, 256).to(DEV)
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    # locus for passage robustness = global max-ratio; for trajectory we pass fixed L,POS
    return hs


def top3_auto(model, tok, text):
    hs = sink_vec(model, tok, text)
    best = None
    for l in range(1, len(hs)):
        A = hs[l][0].float().abs(); r = A.max(-1).values / (A.median(-1).values + 1e-9); t = int(r.argmax())
        if best is None or r[t] > best[2]: best = (l, t, float(r[t]), A[t])
    v = best[3].cpu().numpy()
    return [int(c) for c in np.argsort(-v)[:3]]


def vec_at(model, tok, text, l, pos):
    hs = sink_vec(model, tok, text)
    return hs[l][0, pos].float().abs().cpu().numpy()


# (1) passage robustness x10 on the final model
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()
print("== passage robustness (pythia-410m final), top-3 per passage ==")
tops = []
for i, p in enumerate(PASSAGES):
    t3 = top3_auto(m, tok, p); tops.append(t3); print(f"  p{i}: {t3}")
from collections import Counter
allc = Counter(c for t in tops for c in t)
common = [c for c, n in allc.items() if n == len(PASSAGES)]
print(f"  channels in ALL 10 passages' top-3: {sorted(common)} ({len(common)}/3)")
del m; torch.cuda.empty_cache()

# (2) rank trajectory at fixed locus L17 pos0
TRACK = [357, 130, 966, 752, 125, 550]
STEPS = [2000, 4000, 8000, 16000, 24000, 32000, 48000, 64000, 96000, 143000]
rows = []
for st in STEPS:
    tok = AutoTokenizer.from_pretrained(MODEL, revision=f"step{st}")
    mm = AutoModelForCausalLM.from_pretrained(MODEL, revision=f"step{st}", torch_dtype=torch.float16).to(DEV).eval()
    v = vec_at(mm, tok, PASSAGES[0], L, POS)
    ranks = {c: int((v > v[c]).sum()) + 1 for c in TRACK}
    rows.append(dict(step=st, **{f"rank_{c}": ranks[c] for c in TRACK}))
    print(f"  step {st:>6}: " + " ".join(f"ch{c}=r{ranks[c]}" for c in TRACK))
    del mm; torch.cuda.empty_cache()
with open("runs/scale_profile/pythia410m_ranktraj.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows: w.writerow(r)
print("[pythia_extra] done -> pythia410m_ranktraj.csv")
