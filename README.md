# The Training Dynamics of Massive Activations

Code and figure data for *The Training Dynamics of Massive Activations Are
Magnitude-Organized and Weight-Decay-Driven* (sequel to *Hidden Dynamics of Massive
Activations*, arXiv:2508.03616).

A massive activation (MA) is traced over its full life cycle: a sink born by stochastic
early symmetry-breaking (its channel identity seed-dependent), rising and peaking under
a two-sided balance of loss benefit and weight decay, declining as a per-weight
efficiency reorganization, and consolidating onto a redundant few. Experiments use
transformers trained from scratch (51M/203M/380M) plus the released Pythia suite
(70M–6.9B, including seed re-runs) analyzed by inference only.

## Layout

```
ma_dynamics/     harness + one script per experiment (see table below)
  model.py         vendored nanoGPT-style GPT (definition used by all runs)
  common.py        model builder + memmap data loader
  ma_probe.py      the per-layer sink probe
  train_dense.py   from-scratch training with dense checkpointing
  prepare_data.py  tokenize a corpus -> train.bin / val.bin (GPT-2 BPE)
paper/           main.tex, make_figures.py, figures/
results/         the small CSVs behind every figure (checked in)
```

## Reproduce the figures

Every figure regenerates from the checked-in `results/` CSVs — no GPU or checkpoints
needed:

```bash
pip install -r requirements.txt
cd paper && python make_figures.py      # writes figures/*.pdf,*.png
```

## Rerun the experiments

Scripts read/write under `runs/` by default (device defaults to `cuda:0`; override with
`--device`). Pythia analyses download weights from the Hub automatically.

| Figure / Table | Script |
|---|---|
| Fig 1–2 emergence / global dynamic | `train_dense.py`, `channel_trajectories.py` |
| Fig 3 + Fig 8 weight-decay cause & balance law | `causal_decay_run.py` (λ-sweep) |
| Fig 4 floor / direction controls | `floor_test.py`, `direction_test.py` |
| §controls perturb-and-restore | `perturb_recover.py`, `ceiling_specificity.py` |
| Fig 5 scale | `scale_profile.py`, `pythia_traj.py` |
| Fig 6 consolidation | `pythia_dropout.py` |
| Fig 7 Adam maintenance | `grad_hard.py` |
| Fig 9 early lock-in | `pythia_extra.py` |
| Fig 10 redundant collective | `ablate_sink.py` |
| Fig 11 gain dissociation | `gain_dissoc.py` |
| Table 1 seed identity | `seed_test.py`, `harden_lockin.py` |
| Table 2 refuted mechanisms | `rms_conserve.py`, `count_capacity.py`, `grad_coupling.py` |
| supporting | `attn_sink.py`, `pythia_winners.py`, `pythia_retention.py` |

From-scratch training (primary 51M model):

```bash
python ma_dynamics/prepare_data.py --input /path/to/dolma --out_dir data/dolma
TDA_DATA_DIR=data/dolma python ma_dynamics/train_dense.py \
  --n_layer 8 --n_embd 512 --n_head 8 --block_size 512 \
  --weight_decay 0.1 --learning_rate 6e-4 --max_iters 40000 --seed 1337
```

## Data & checkpoints

- **Data.** The Dolma corpus (AI2, ODC-BY) is not redistributed here; `prepare_data.py`
  tokenizes it into the `train.bin`/`val.bin` format the loader expects.
- **Checkpoints.** The full dense checkpoint history is large and hosted separately
  (Hugging Face — link TBD). The released `results/` CSVs are sufficient to reproduce
  all figures; checkpoints are only needed to re-probe activations or rerun the
  perturb-and-restore protocol.

## License & attribution

Code released under Apache-2.0 (see `LICENSE`). `model.py` is derived from nanoGPT.
Pythia (Apache-2.0) and Dolma (ODC-BY) are used under their respective licenses.
