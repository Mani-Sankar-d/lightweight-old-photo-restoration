"""
DirectMapping: the final z_A -> z_B mapping, a plain residual conv network in the
frozen VAE latent space (same mechanism as Microsoft's mapping network, ~3x smaller).

It is rf_model.VelocityNet with the time input removed. With t fixed to 0:
  - the time embedding is a constant vector, so each FiLM layer is a constant
    per-channel scale/shift right after norm2, which norm2's own affine absorbs;
  - the input concat(z_t, z_A) is concat(z_A, z_A), so the first conv's two weight
    halves merge into one conv on z_A.
So the function class is exactly that of the t = 0 model, and fold_rf() turns a
trained t = 0 RFMapping into a DirectMapping with identical outputs.

    z_B = scale * (a + net(a)),   a = z_A / scale

Residual, with a zero-initialized output conv, so training starts at the identity.
forward(z_A, mask=None) matches OldPhotoModel.mapping's call signature: obj.mapping = net.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """FiLMResBlock without FiLM: GroupNorm -> SiLU -> dilated conv -> GroupNorm -> SiLU -> dropout -> conv, + skip."""

    def __init__(self, channels, dilation=1, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.dropout(F.silu(self.norm2(h)))
        return x + self.conv2(h)


class DirectMapping(nn.Module):
    def __init__(self, latent_ch, width=256, n_blocks=9, dropout=0.1):
        super().__init__()
        self.latent_ch, self.width, self.n_blocks = latent_ch, width, n_blocks
        self.in_conv = nn.Conv2d(latent_ch, width, 3, padding=1)
        dilations = (1, 2, 4)
        self.blocks = nn.ModuleList([ResBlock(width, dilations[i % len(dilations)], dropout)
                                     for i in range(n_blocks)])
        self.out_conv = nn.Conv2d(width, latent_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        self.register_buffer("scale", torch.ones(()))  # latent std, set from training data

    def forward(self, z_A, mask=None):
        a = z_A / self.scale
        h = self.in_conv(a)
        for block in self.blocks:
            h = block(h)
        return (a + self.out_conv(F.silu(h))) * self.scale

    def arch(self):
        return {"type": "direct", "latent_ch": self.latent_ch, "width": self.width, "n_blocks": self.n_blocks}


@torch.no_grad()
def fold_rf(rf):
    """t = 0 RFMapping -> DirectMapping with identical outputs (both in eval mode)."""
    v, c = rf.v, rf.latent_ch
    d = DirectMapping(c, rf.width, rf.n_blocks).to(rf.scale.device)
    d.scale.copy_(rf.scale)
    w = v.in_conv.weight                                  # (width, 2c, 3, 3): halves for z_t and z_A
    d.in_conv.weight.copy_(w[:, :c] + w[:, c:])
    d.in_conv.bias.copy_(v.in_conv.bias)
    t_emb = v.time_emb(torch.zeros(1, device=w.device))   # constant at t = 0
    for bd, bv in zip(d.blocks, v.blocks):
        scale, shift = bv.film(t_emb)[0].chunk(2)
        for name in ("norm1", "conv1", "conv2"):
            getattr(bd, name).load_state_dict(getattr(bv, name).state_dict())
        # norm2(h) * (1 + scale) + shift = gn(h) * w (1 + scale) + (b (1 + scale) + shift)
        bd.norm2.weight.copy_(bv.norm2.weight * (1 + scale))
        bd.norm2.bias.copy_(bv.norm2.bias * (1 + scale) + shift)
    d.out_conv.load_state_dict(v.out_conv.state_dict())
    return d.eval()


def load_direct(path, device, use_ema=True):
    """DirectMapping from a train_direct checkpoint (or the exported weights file). Returns (net, step)."""
    ck = torch.load(path, map_location=device)
    arch = {k: v for k, v in ck["arch"].items() if k != "type"}
    net = DirectMapping(**arch).to(device)
    net.load_state_dict(ck["ema"] if use_ema else ck["net"])
    net.eval()
    print(f"loaded {path} | step {ck['step']} | {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params")
    return net, ck["step"]
