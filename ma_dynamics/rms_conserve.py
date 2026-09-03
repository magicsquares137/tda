"""
Conservation test: the gradient floor is on the sink token's COLLECTIVE rms
(established: g proportional to 1/rms). If the loss enforces an rms floor and the
field (ranks 4+) collapses, the survivors must GROW to hold total rms -> growth and
erosion are ONE process (redistribution under a constraint), not two.

Prediction: total sink rms stays ~flat across 16k->143k while its composition
concentrates (winner energy rises as field energy falls, gain ~ loss).
Measure at the mature sink locus (L17 pos0).
"""
import os, csv
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scale_profile import PASSAGE

MODEL = "EleutherAI/pythia-410m"; DEV = "cuda:0"; L, POS = 17, 0
STEPS = [8000, 16000, 24000, 32000, 48000, 64000, 96000, 143000]


def sink_vec(revision):
    tok = AutoTokenizer.from_pretrained(MODEL, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=revision, torch_dtype=torch.float16).to(DEV).eval()
    ids = tok(PASSAGE, return_tensors="pt").input_ids[0]
    need = 4 * 512
    if ids.numel() < need: ids = ids.repeat(need // ids.numel() + 1)
    ids = ids[:need].view(4, 512).to(DEV)
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    v = hs[L].float()[:, POS, :].mean(0).cpu().numpy()          # signed mean; use abs for energy
    va = hs[L].float().abs().mean(0)[POS].cpu().numpy()
    del model; torch.cuda.empty_cache()
    return va


# winners = top-3 at final
vfin = sink_vec("step143000")
winners = list(np.argsort(-vfin)[:3])
print(f"locus L{L} pos{POS}; winners (top-3 @143k): {[int(c) for c in winners]}\n")

rows = []
for st in STEPS:
    v = sink_vec(f"step{st}")
    E_total = float((v ** 2).sum())
    E_win = float((v[winners] ** 2).sum())
    E_field = E_total - E_win
    total_rms = float(np.sqrt((v ** 2).mean()))
    rows.append(dict(step=st, total_rms=total_rms, E_total=E_total, E_win=E_win, E_field=E_field,
                     win_frac=E_win / E_total))
    print(f"step {st:>7}: total_rms {total_rms:6.2f} | E_total {E_total:9.0f} | "
          f"E_win {E_win:9.0f} | E_field {E_field:8.0f} | win_frac {E_win/E_total:.2f}")

with open("runs/scale_profile/pythia410m_rms.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows: w.writerow(r)

# is winner gain ~ field loss (redistribution), or does total change?
r16 = next(r for r in rows if r["step"] == 16000); r143 = rows[-1]
print(f"\n16k -> 143k:")
print(f"  total_rms:  {r16['total_rms']:.2f} -> {r143['total_rms']:.2f}  ({100*(r143['total_rms']/r16['total_rms']-1):+.0f}%)")
print(f"  E_field:    {r16['E_field']:.0f} -> {r143['E_field']:.0f}  (lost {r16['E_field']-r143['E_field']:.0f})")
print(f"  E_win:      {r16['E_win']:.0f} -> {r143['E_win']:.0f}  (gained {r143['E_win']-r16['E_win']:.0f})")
print(f"  winner gain / field loss = {(r143['E_win']-r16['E_win'])/(r16['E_field']-r143['E_field']+1e-9):.2f}")
print("  ~1.0 => redistribution (conservation);  >>1 => winners grow beyond what the field sheds")
print("[rms_conserve] -> runs/scale_profile/pythia410m_rms.csv")
