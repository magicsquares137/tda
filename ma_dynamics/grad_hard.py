"""
Harden the Adam-preconditioning result: ~10 winners vs ~10 collapsers vs 10 random
controls, three checkpoints, steady-state AdamW preconditioning.

For each residual channel c, over a window of N WikiText batches, accumulate writer-row
gradient mean m and second moment v (steady-state Adam EMAs -> full-window m, v). The
preconditioned maintaining force is gp_adam = -<w, m/(sqrt(v)+eps)> / ||w||^2 with the
AdamW eps; ||w|| grows iff gp_adam > lambda (=0.01 for Pythia). Prediction: winners
gp_adam >> lambda, collapsers ~lambda, random ~0.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "EleutherAI/pythia-410m"; DEV = "cuda:0"; L = 17
SEQ = 128; NBATCH = 200; EPS = 1e-8; LAM = 0.01
STEPS = [16000, 32000, 48000]


def load(rev):
    tok = AutoTokenizer.from_pretrained(MODEL, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=rev, torch_dtype=torch.float32).to(DEV).eval()
    return tok, model


def sink_vec(model, tok):
    x = tok("The history of science is the study of the development of science. " * 20,
            return_tensors="pt").input_ids[0][:256].view(1, 256).to(DEV)
    with torch.no_grad():
        v = model(x, output_hidden_states=True).hidden_states[L][0, 0].float().abs()
    return v.cpu().numpy()


def wmats(model):
    return sum([[l.attention.dense.weight, l.mlp.dense_4h_to_h.weight] for l in model.gpt_neox.layers], [])


# ---- define groups from fate ----
tok, m143 = load("step143000"); v143 = sink_vec(m143, tok); del m143; torch.cuda.empty_cache()
tok, m16 = load("step16000"); v16 = sink_vec(m16, tok)
ret = v143 / (v16 + 1e-9)
winners = [int(c) for c in np.argsort(-v143)[:10]]
big16 = list(np.argsort(-v16)[:40])
collapsers = [int(c) for c in big16 if ret[c] < 0.2 and c not in winners][:10]
rng = np.random.default_rng(0)
pool = [c for c in range(len(v16)) if c not in winners + collapsers]
randoms = [int(c) for c in rng.choice(pool, 10, replace=False)]
GROUPS = {"winner": winners, "collapse": collapsers, "random": randoms}
ALL = winners + collapsers + randoms
print("winners:", winners, "\ncollapsers:", collapsers, "\nrandoms:", randoms)
del m16; torch.cuda.empty_cache()

# ---- corpus ----
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
text = "\n".join(t for t in ds["text"] if len(t) > 40)
ids_all = tok(text, return_tensors="pt").input_ids[0]

results = {}
for st in STEPS:
    tok, model = load(f"step{st}")
    n = min(NBATCH, ids_all.numel() // SEQ)
    chunks = ids_all[:n * SEQ].view(n, SEQ).to(DEV)
    wvec = {c: torch.cat([w.detach()[c].flatten() for w in wmats(model)]) for c in ALL}
    sg = {c: 0.0 for c in ALL}; sg2 = {c: 0.0 for c in ALL}
    for b in range(n):
        out = model(chunks[b:b+1], labels=chunks[b:b+1])
        model.zero_grad(set_to_none=True); out.loss.backward()
        W = wmats(model)
        for c in ALL:
            g = torch.cat([w.grad[c].flatten() for w in W]).detach()
            sg[c] = sg[c] + g; sg2[c] = sg2[c] + g * g
    gp = {}
    for c in ALL:
        m = sg[c] / n; v = sg2[c] / n; ghat = m / (v.sqrt() + EPS)
        wv = wvec[c]; gp[c] = float(-(wv * ghat).sum() / (wv * wv).sum())
    results[st] = gp
    del model; torch.cuda.empty_cache()
    print(f"\nstep {st} (N={n}):")
    for grp, chans in GROUPS.items():
        vals = np.array([gp[c] for c in chans])
        print(f"  {grp:9s} gp_adam mean {vals.mean():+.3f}  median {np.median(vals):+.3f}  "
              f"[{vals.min():+.3f},{vals.max():+.3f}]  frac>lambda {np.mean(vals>LAM):.2f}")

print(f"\n== per-channel gp_adam (grows iff > lambda={LAM}) ==")
print(f"{'chan':>5} {'role':>9} " + " ".join(f"{'@'+str(s//1000)+'k':>8}" for s in STEPS))
for grp, chans in GROUPS.items():
    for c in chans:
        print(f"{c:>5} {grp:>9} " + " ".join(f"{results[s][c]:>+8.3f}" for s in STEPS))
import csv
with open("runs/scale_profile/grad_hard.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["chan", "role"] + [f"gp_{s}" for s in STEPS])
    for grp, chans in GROUPS.items():
        for c in chans: w.writerow([c, grp] + [results[s][c] for s in STEPS])
print("[grad_hard] -> runs/scale_profile/grad_hard.csv")
