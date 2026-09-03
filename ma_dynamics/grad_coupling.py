"""
Measure the SPINE of the mechanism instead of inferring it. Claim: winners' growth
raises sink rms -> g proportional to 1/rms suppresses the maintaining gradient on the
FIELD channels -> weight decay erodes them. Directly measure the gradient reaching a
collapsing channel's writer rows across training, vs sink rms.

Writer rows for residual channel c = row c of every layer's attention.dense.weight and
mlp.dense_4h_to_h.weight (the projections that write into the residual). g_writer(c) =
L2 norm of the loss-gradient on those rows, summed over layers.

Prediction: for a collapsing channel, g_writer * rms is ROUGHLY CONSTANT (gradient
falls as 1/rms) while the channel is still large. If g_writer is flat while rms rises,
weight decay alone erodes it and the 1/rms link is decorative.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scale_profile import PASSAGE

MODEL = "EleutherAI/pythia-410m"; DEV = "cuda:0"; L = 17
STEPS = [8000, 16000, 32000, 48000]
CHANS = {752: "collapse", 550: "collapse", 357: "winner", 130: "winner"}


def measure(rev):
    tok = AutoTokenizer.from_pretrained(MODEL, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=rev, torch_dtype=torch.float32).to(DEV).eval()
    ids = tok(PASSAGE, return_tensors="pt").input_ids[0]
    need = 4 * 256
    if ids.numel() < need: ids = ids.repeat(need // ids.numel() + 1)
    ids = ids[:need].view(4, 256).to(DEV)
    out = model(ids, labels=ids, output_hidden_states=True)
    va = out.hidden_states[L].float().abs().mean(0)[0]            # sink |activation| per channel
    sink_rms = float(out.hidden_states[L][:, 0, :].float().pow(2).mean(-1).sqrt().mean())
    model.zero_grad(set_to_none=True)
    out.loss.backward()
    res = {}
    for c in CHANS:
        gw2 = wn2 = 0.0
        for layer in model.gpt_neox.layers:
            for W in (layer.attention.dense.weight, layer.mlp.dense_4h_to_h.weight):
                gw2 += float(W.grad[c].pow(2).sum()); wn2 += float(W[c].pow(2).sum())
        res[c] = (gw2 ** 0.5, wn2 ** 0.5, float(va[c]))
    del model; torch.cuda.empty_cache()
    return sink_rms, float(out.loss), res


data = {}
for st in STEPS:
    rms, loss, res = measure(f"step{st}")
    data[st] = (rms, res)
    print(f"step {st:>6}: sink_rms {rms:6.2f} loss {loss:.3f}")

print(f"\n{'chan':>5} {'role':>9} " + " ".join(f"{'g@'+str(s//1000)+'k':>9}" for s in STEPS))
print("  -- writer-row gradient g_writer --")
for c, role in CHANS.items():
    print(f"{c:>5} {role:>9} " + " ".join(f"{data[s][1][c][0]:>9.4f}" for s in STEPS))
print("  -- g_writer * sink_rms  (flat => gradient falls as 1/rms) --")
for c, role in CHANS.items():
    print(f"{c:>5} {role:>9} " + " ".join(f"{data[s][1][c][0]*data[s][0]:>9.3f}" for s in STEPS))
print("  -- channel |activation| (is it still large while gradient falls?) --")
for c, role in CHANS.items():
    print(f"{c:>5} {role:>9} " + " ".join(f"{data[s][1][c][2]:>9.1f}" for s in STEPS))
print("\nread: collapser g_writer falls while its |activation| is still large, and")
print("g_writer*rms ~ flat => the 1/rms suppression removes the maintaining gradient (coupling MEASURED).")
