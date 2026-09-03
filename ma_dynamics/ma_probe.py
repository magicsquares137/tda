"""
Massive-activation probe for a nanoGPT-style GPT model (luvai/model.py).

Measures, per layer, the ratio of the largest activation magnitude to the
median activation magnitude in the residual stream -- the same r_{l,t} = h^max /
h^median quantity fit in "Hidden Dynamics of Massive Activations" (2508.03616).

Also identifies:
  - the MA *channel* d* (the residual-stream dimension that holds the massive
    value) at each layer,
  - the *birth layer* (layer with the largest ratio),
  - the position at which the max occurs (to check the position-0 attention-sink
    story).

And provides weight-surgery helpers to scale the c_proj rows that *write into* a
given channel -- this is the causal perturbation used by the perturb/recover
experiment (Option B).

The probe hooks the output of each transformer Block, i.e. the residual stream
`x` after `x = x + mlp(ln_2(x))`. In luvai/model.py that is `model.transformer.h[l]`.
"""

import torch


class MAProbe:
    """Hook-based reader of residual-stream statistics. Works on the *raw*
    (uncompiled, un-DDP) GPT model. Call `.attach()` once, `.measure(...)` many
    times, `.detach()` when done."""

    def __init__(self, raw_model):
        self.model = raw_model
        self.blocks = list(raw_model.transformer.h)
        self.n_layer = len(self.blocks)
        self._captures = {}
        self._handles = []

    # -- hook lifecycle --------------------------------------------------------
    def attach(self):
        self.detach()
        for li, block in enumerate(self.blocks):
            def hook(module, inp, out, li=li):
                # out is the residual stream after this block: (B, T, C)
                self._captures[li] = out.detach()
            self._handles.append(block.register_forward_hook(hook))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._captures = {}

    # -- measurement -----------------------------------------------------------
    @torch.no_grad()
    def measure(self, idx, ctx=None):
        """Run a forward pass on token ids `idx` (B, T) and return per-layer
        statistics computed in fp32.

        Returns a dict with numpy-friendly python lists:
          ratio[l]        = max|h| / median|h|  over all (position, channel)
          ma_channel[l]   = channel index d* holding the global max at layer l
          ma_position[l]  = token position holding the global max
          persist_chan[l] = channel with the largest mean|h| (persistent MA dim)
          max_abs[l]      = the raw max|h|
          median_abs[l]   = the raw median|h|
        """
        was_training = self.model.training
        self.model.eval()
        self._captures = {}
        import contextlib
        cm = ctx if ctx is not None else contextlib.nullcontext()
        with cm:
            self.model(idx)  # targets=None -> just populates hooks
        if was_training:
            self.model.train()

        ratio, ma_channel, ma_position = [], [], []
        persist_chan, max_abs, median_abs = [], [], []
        for li in range(self.n_layer):
            h = self._captures[li].float()          # (B, T, C)
            B, T, C = h.shape
            a = h.abs()
            # global max over B,T,C and its location
            flat = a.reshape(-1, C)                  # (B*T, C)
            max_val, max_c = flat.max(dim=1)         # per-token max over channels
            tok_argmax = int(max_val.argmax().item())
            d_star = int(max_c[tok_argmax].item())
            pos = tok_argmax % T
            m_max = float(a.max().item())
            m_med = float(a.median().item())
            ratio.append(m_max / (m_med + 1e-12))
            ma_channel.append(d_star)
            ma_position.append(pos)
            # persistent channel: largest mean magnitude across all tokens
            persist_chan.append(int(a.mean(dim=(0, 1)).argmax().item()))
            max_abs.append(m_max)
            median_abs.append(m_med)

        self._captures = {}
        return {
            "ratio": ratio,
            "ma_channel": ma_channel,
            "ma_position": ma_position,
            "persist_chan": persist_chan,
            "max_abs": max_abs,
            "median_abs": median_abs,
        }

    @torch.no_grad()
    def channel_magnitude(self, idx, layer, channel, ctx=None):
        """Mean and max |activation| of one specific (layer, channel) on `idx`.
        Used to track a fixed MA channel during perturb/recover."""
        self._captures = {}
        import contextlib
        cm = ctx if ctx is not None else contextlib.nullcontext()
        with cm:
            self.model(idx)
        h = self._captures[layer].float()           # (B, T, C)
        col = h[:, :, channel].abs()
        med = h.abs().median().item()
        self._captures = {}
        return {
            "mean_abs": float(col.mean().item()),
            "max_abs": float(col.max().item()),
            "ratio": float(col.max().item() / (med + 1e-12)),
        }


# -- weight surgery -----------------------------------------------------------
def _writer_linear(raw_model, layer, target):
    """Return the nn.Linear whose output rows write into the residual stream at
    `layer`. target='mlp' -> block.mlp.c_proj ; target='attn' -> block.attn.c_proj.
    Both map (..)->n_embd, so output-feature index == residual channel index."""
    block = raw_model.transformer.h[layer]
    if target == "mlp":
        return block.mlp.c_proj
    elif target == "attn":
        return block.attn.c_proj
    raise ValueError(f"unknown target {target!r}")


@torch.no_grad()
def scale_writer_row(raw_model, layer, channel, alpha, target="mlp"):
    """Multiply the output row of the writer linear that feeds `channel` by
    `alpha` (in place). Returns the pre-perturbation norm of that row so a
    matched-magnitude control can be constructed. alpha=0 zeroes the write;
    alpha=2 doubles it."""
    lin = _writer_linear(raw_model, layer, target)
    row = lin.weight.data[channel]                  # (in_features,)
    norm = float(row.norm().item())
    lin.weight.data[channel] = row * alpha
    if lin.bias is not None:
        lin.bias.data[channel] = lin.bias.data[channel] * alpha
    return norm


@torch.no_grad()
def scale_channel_all_layers(raw_model, channel, alpha, targets=("mlp", "attn"), layers=None):
    """Scale the writer rows feeding `channel` across *every* layer (attn.c_proj
    and mlp.c_proj by default). Because the residual stream has a fixed width, the
    channel index is the same dimension at every layer, so this removes/scales all
    block contributions to that residual dimension -- a strong displacement of an
    established massive activation (a single-layer row is too weak once the MA is
    redundantly written). Returns the aggregate L2 norm of all displaced rows so a
    matched-magnitude control can be built."""
    n = len(raw_model.transformer.h)
    layers = range(n) if layers is None else layers
    total_sq = 0.0
    for l in layers:
        for tgt in targets:
            lin = _writer_linear(raw_model, l, tgt)
            row = lin.weight.data[channel]
            total_sq += float((row ** 2).sum().item())
            lin.weight.data[channel] = row * alpha
            if lin.bias is not None:
                lin.bias.data[channel] = lin.bias.data[channel] * alpha
    return total_sq ** 0.5


@torch.no_grad()
def channel_writer_norms(raw_model, targets=("mlp", "attn")):
    """Aggregate writer-row L2 norm per residual channel, summed across all layers
    and targets. Used to pick a matched-magnitude control channel."""
    n_embd = raw_model.transformer.h[0].mlp.c_proj.weight.shape[0]
    sq = torch.zeros(n_embd)
    for block in raw_model.transformer.h:
        for tgt in targets:
            lin = block.mlp.c_proj if tgt == "mlp" else block.attn.c_proj
            sq += (lin.weight.data ** 2).sum(dim=1).cpu()
    return sq.sqrt()


@torch.no_grad()
def pick_matched_control_channel(raw_model, layer, ma_channel, target="mlp", seed=0):
    """Pick a non-MA channel whose writer-row norm is closest to the MA channel's,
    so a perturbation of equal magnitude can be applied as a control."""
    lin = _writer_linear(raw_model, layer, target)
    norms = lin.weight.data.norm(dim=1)             # (out_features,)
    target_norm = norms[ma_channel].clone()
    norms[ma_channel] = float("inf")                # exclude the MA channel itself
    # closest norm to the MA channel's row norm
    control = int((norms - target_norm).abs().argmin().item())
    return control
