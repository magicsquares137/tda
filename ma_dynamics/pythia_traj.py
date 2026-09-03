"""
Does the continuum -> bimodal (cliff) structure track TRAINING TOKENS or is it
present from the start? Load one Pythia model at successive training revisions and
measure, at the max-ratio (layer, token): MA separation (max/median), the cliff
(biggest consecutive channel drop in the top-15 = how bimodal the profile is), and
rank1/rank2. If early checkpoints look like a continuum (small cliff, like our
from-scratch models) and the cliff grows with tokens, the ours-vs-Pythia difference
is a training-maturity effect, not parameter count.
"""
import os, csv, argparse
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np
import torch
from scale_profile import PASSAGE, profile_from_absmean


def measure_rev(model_name, revision, dev, batch_size=4, seqlen=512):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, torch_dtype=torch.float16).to(dev).eval()
    ids = tok(PASSAGE, return_tensors="pt").input_ids[0]
    need = batch_size * seqlen
    if ids.numel() < need:
        ids = ids.repeat((need // ids.numel()) + 1)
    ids = ids[:need].view(batch_size, seqlen).to(dev)
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    best = None
    for l in range(1, len(hs)):
        A = hs[l].float().abs().mean(0)
        prof = profile_from_absmean(A)
        if best is None or prof["ratio"] > best[1]["ratio"]:
            best = (l, prof)
    del model
    torch.cuda.empty_cache()
    l, prof = best
    s = prof["sorted"]
    cliff = max(s[i] / s[i + 1] for i in range(min(14, len(s) - 1)))
    return dict(layer=l, pos=prof["pos"], median=prof["median"],
                max_over_median=prof["ratio"], r1_over_r2=float(s[0] / s[1]),
                cliff=float(cliff), r1_energy_frac=prof["r1_energy_frac"],
                top8="|".join(f"{x:.1f}" for x in s[:8]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model", default="EleutherAI/pythia-410m")
    p.add_argument("--steps", default="1000,2000,4000,8000,16000,32000,64000,143000")
    p.add_argument("--out", default="runs/scale_profile/pythia410m_traj.csv")
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    steps = [int(s) for s in args.steps.split(",") if s]
    rows = []
    for st in steps:
        try:
            r = measure_rev(args.model, f"step{st}", args.device)
            r = dict(step=st, **r)
            rows.append(r)
            print(f"step {st:>7}: L{r['layer']} pos{r['pos']} | max/med {r['max_over_median']:7.0f} "
                  f"| cliff {r['cliff']:5.1f}x | r1/r2 {r['r1_over_r2']:.2f} | top8 {r['top8']}")
        except Exception as e:
            print(f"[skip step{st}] {type(e).__name__}: {e}")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[pythia_traj] -> {args.out}")


if __name__ == "__main__":
    main()
