"""
Are the eventual winners predictable at step 8k (still a continuum), BEFORE they win?
Magnitude is the wrong predictor (shown). Test whether OUTPUT-DECOUPLING predicts
survival: a channel the unembedding does not read from / the final LN suppresses can
carry magnitude without hurting logits -> "parked" -> survives as a sink; a channel
the output reads from gets regulated back down -> collapses.

Per residual channel c, at step 8k, measure:
  unembed_read  = ||embed_out.weight[:, c]||         (how much logits read channel c)
  lnf_gain      = |final_layer_norm.weight[c]|        (final-LN suppression)
  out_coupling  = lnf_gain * unembed_read             (effective output coupling)
  embed_write   = ||embed_in.weight[:, c]||           (how much tokens write to c)
Winners/collapsers are DEFINED by their 143k fate (measured at the mature locus).
If out_coupling separates winners (low) from equal-era collapsers (high), that's a
candidate mechanism + a prediction testable on other models.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scale_profile import PASSAGE

MODEL = "EleutherAI/pythia-410m"; DEV = "cuda:0"


def sink_vec(revision, layer, pos, batch=4, seq=512):
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


def features(revision):
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=revision, torch_dtype=torch.float32).to("cpu").eval()
    sd = dict(model.named_parameters())
    Wout = sd["embed_out.weight"].detach()                       # (vocab, d)
    gain = sd["gpt_neox.final_layer_norm.weight"].detach()       # (d,)
    Win = sd["gpt_neox.embed_in.weight"].detach()                # (vocab, d)
    unembed_read = Wout.norm(dim=0).numpy()                      # per-channel column norm
    lnf_gain = gain.abs().numpy()
    embed_write = Win.norm(dim=0).numpy()
    del model
    return unembed_read, lnf_gain, embed_write


# locus + fate
v16, L, POS = sink_vec("step16000", None, None)
v143, _, _ = sink_vec("step143000", L, POS)
order16 = np.argsort(-v16)
ret = v143 / (v16 + 1e-9)
winners = [int(c) for c in np.argsort(-v143)[:4]]                # late giants
early_big = [int(c) for c in order16[:12]]
collapsers = [c for c in early_big if ret[c] < 0.2]             # big@16k, gone@143k
print(f"locus L{L} pos{POS}")
print(f"winners (top-4 @143k): {winners}")
print(f"collapsers (big@16k, ret<0.2): {collapsers}")

ur, gn, ew = features("step8000")
C = len(ur)
def z(x): return (x - np.median(x)) / (np.percentile(x, 84) - np.percentile(x, 16) + 1e-9)
coupling = gn * ur
feats = {"unembed_read": ur, "lnf_gain": gn, "out_coupling": coupling, "embed_write": ew}

def pct(x, c): return int((x < x[c]).mean() * 100)               # percentile rank of channel c
print(f"\nper-channel features at step8k (value | field-percentile):")
hdr = "chan  role      " + "".join(f"{k:>16}" for k in feats)
print(hdr)
for role, chans in [("WINNER", winners), ("COLLAPSE", collapsers)]:
    for c in chans:
        cells = "".join(f"{feats[k][c]:>8.3f}({pct(feats[k], c):>3d}%)" for k in feats)
        print(f"{c:>4}  {role:8s}  {cells}")
print(f"{'MED':>4}  {'field':8s}  " + "".join(f"{np.median(feats[k]):>8.3f}({'50':>3}%)" for k in feats))

print("\n== winner vs collapser medians (and separation) ==")
for k, x in feats.items():
    w = np.median([x[c] for c in winners]); co = np.median([x[c] for c in collapsers])
    print(f"{k:>14}: winners {w:.3f}  collapsers {co:.3f}  ratio {w/(co+1e-9):.2f}")
