"""
Decisive redundancy test for competitive exclusion. At step16000, the collapser
ch752 is still large (rank4, mag ~242). Softmax-competition predicts it is already
REDUNDANT to the attention sink -> ablating it barely changes sink attention, while
the survivors (357,130,966) are load-bearing -> ablating them drops it. If a
large-but-redundant channel is the one that later collapses, that's the mechanism
(redundant -> ~zero gradient -> weight decay erodes it), independent of magnitude.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scale_profile import PASSAGE

MODEL = "EleutherAI/pythia-410m"; DEV = "cuda:0"; REV = "step16000"


def load():
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REV, torch_dtype=torch.float32, attn_implementation="eager").to(DEV).eval()
    ids = tok(PASSAGE, return_tensors="pt").input_ids[0]
    if ids.numel() < 256: ids = ids.repeat(256 // ids.numel() + 1)
    return model, ids[:256].view(1, 256).to(DEV)


def sink_attn(model, ids, ablate=None):
    hooks = []
    if ablate is not None:
        def zero_hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out           # layer: tuple; embed: tensor
            for c in ablate: t[:, :, c] = 0.0
            return out
        hooks.append(model.gpt_neox.embed_in.register_forward_hook(zero_hook))
        for layer in model.gpt_neox.layers:
            hooks.append(layer.register_forward_hook(zero_hook))
    with torch.no_grad():
        att = model(ids, output_attentions=True).attentions
    for h in hooks: h.remove()
    pl = [a[0].float()[:, 1:, 0].mean().item() for a in att]
    return float(np.mean(pl))


model, ids = load()
base = sink_attn(model, ids)
print(f"baseline sink-attn @step16k: {base:.4f}\n")

# magnitudes at 16k for context (from earlier): 357~495 130~458 966~373 752~242 550~142
survivors = [357, 130, 966]
collapsers = [752, 550, 509, 421]
rng = np.random.default_rng(0)
randch = [int(c) for c in rng.choice([c for c in range(1024) if c not in survivors + collapsers], 3, replace=False)]

print(f"{'ablate':>16} {'sink-attn':>10} {'drop':>8}")
def run(name, chans):
    s = sink_attn(model, ids, ablate=chans)
    print(f"{name:>16} {s:>10.4f} {base - s:>+8.4f}")
    return s
for c in survivors: run(f"survivor {c}", [c])
for c in collapsers: run(f"collapser {c}", [c])
for c in randch: run(f"random {c}", [c])
run("ALL 3 survivors", survivors)
run("survivors+752", survivors + [752])
print("\nread: survivors drop sink-attn, 752 (large but redundant) does not => "
      "redundancy (not magnitude) predicts collapse.")
