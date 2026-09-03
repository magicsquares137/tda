"""Shared helpers: the GPT model (vendored in model.py), a small config builder, a
memmap data loader, and an autocast context. Kept deliberately tiny.

Data location is configurable via the TDA_DATA_DIR environment variable (default
./data/dolma), which must contain train.bin / val.bin (uint16 GPT-2 BPE tokens; see
prepare_data.py)."""

import os
import sys  # noqa: F401  (kept for downstream scripts)
import contextlib
import numpy as np
import torch

from model import GPTConfig, GPT  # vendored nanoGPT-style model definition

DATA_DIR = os.environ.get("TDA_DATA_DIR", os.path.join("data", "dolma"))


def build_model(n_layer, n_head, n_embd, block_size, vocab_size=50304, bias=False,
                dropout=0.0, device="cuda:0", norm="layer"):
    if norm == "rms":
        import model_rms as M
    else:
        import model as M
    cfg = M.GPTConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                      block_size=block_size, vocab_size=vocab_size, bias=bias,
                      dropout=dropout)
    model = M.GPT(cfg)
    model.to(device)
    return model, cfg


def make_ctx(device, dtype="bfloat16"):
    if "cuda" not in device:
        return contextlib.nullcontext()
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
               "float16": torch.float16}[dtype]
    return torch.amp.autocast(device_type="cuda", dtype=ptdtype)


class Data:
    """memmap batch sampler over train.bin / val.bin (uint16 GPT-2 BPE)."""

    def __init__(self, block_size, batch_size, device, data_dir=DATA_DIR):
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.data_dir = data_dir

    def _mm(self, split):
        fn = "train.bin" if split == "train" else "val.bin"
        return np.memmap(os.path.join(self.data_dir, fn), dtype=np.uint16, mode="r")

    def get_batch(self, split, generator=None):
        data = self._mm(split)
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,),
                           generator=generator)
        x = torch.stack([torch.from_numpy(data[i:i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + self.block_size].astype(np.int64)) for i in ix])
        if "cuda" in self.device:
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    def fixed_probe_batch(self, split="val", seed=1234):
        """A single fixed batch used to measure the MA trajectory reproducibly
        across training (no sampling noise between checkpoints)."""
        g = torch.Generator().manual_seed(seed)
        return self.get_batch(split, generator=g)
