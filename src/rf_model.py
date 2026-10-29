"""
Rectified flow model mapping z_A -> z_B in the frozen VAE latent space.

VelocityNet: v_theta(z_t, t | z_A), fully convolutional (works at any latent
resolution). FiLM-conditioned residual blocks, dilated convs for receptive
field, zero-initialized output so the flow starts as the identity.

RFMapping: wraps VelocityNet with the scalar latent normalization and the
Euler ODE integrator used both at train time (for validation samples) and at
inference time (obj.mapping replacement).

Bridge matching (sigma > 0): the straight path gets Brownian-bridge noise,
    z_t = (1 - t) z_A + t z_B + sigma * sqrt(t (1 - t)) * eps,
zero at both ends. The network output v defines the predicted endpoint
z_B_hat = z_t + (1 - t) v, trained against the true z_B, and sampling uses the
exact bridge posterior step
    z <- z + dt / (1 - t) * (z_B_hat - z) + sigma * sqrt(dt (1 - t') / (1 - t)) * eps.
A deterministic flow (sigma = 0) regresses to the AVERAGE of all clean images
consistent with z_A (soft texture, washed-out colour); with sigma > 0 the
sampler draws ONE plausible z_B instead. At sigma = 0 the sampler is exactly
the old Euler RF sampler, so RF checkpoints warm-start a bridge model directly.

direct_loss() (t = 0 ablation): only the point the 1-step sampler uses, z_t = z_A,
t = 0 -- plain direct regression z_A -> z_B with the same network, i.e. the
Microsoft-style mapping objective. No bridge noise, no t > 0.

traj_loss() trains the velocity at points the sampler actually reaches
(the model's own Euler trajectory), not only on the ideal straight line, so
the model learns to correct its own drift instead of accumulating it.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):  # t: (B,) in [0,1]
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = (t[:, None] * 1000.0) * freqs[None]
        return self.mlp(torch.cat([args.sin(), args.cos()], dim=1))


class FiLMResBlock(nn.Module):
    def __init__(self, channels, tdim, dilation=1, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film = nn.Linear(tdim, 2 * channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(t_emb)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.dropout(F.silu(h))
        return x + self.conv2(h)


class VelocityNet(nn.Module):
    """v(z_t, t | z_A). Input is concat(z_t, z_A) along channels."""

    def __init__(self, latent_ch, width=256, n_blocks=9, tdim=256, dropout=0.1):
        super().__init__()
        self.time_emb = TimeEmb(tdim)
        self.in_conv = nn.Conv2d(2 * latent_ch, width, 3, padding=1)
        dilations = (1, 2, 4)
        self.blocks = nn.ModuleList([
            FiLMResBlock(width, tdim, dilation=dilations[i % len(dilations)], dropout=dropout)
            for i in range(n_blocks)
        ])
        self.out_conv = nn.Conv2d(width, latent_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, z_t, t, z_A):
        t_emb = self.time_emb(t)
        h = self.in_conv(torch.cat([z_t, z_A], dim=1))
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out_conv(F.silu(h))


class RFMapping(nn.Module):
    """Rectified-flow replacement for the repo's mapping net.

    forward(z_A) integrates dz/dt = v(z, t | z_A) from t=0 (z_A) to t=1 (z_B)
    with `steps` Euler steps, matching the OldPhotoModel.mapping call signature
    closely enough to drop in via rf_mapping.install() / obj.mapping = rf.
    """

    def __init__(self, latent_ch, width=256, n_blocks=9, steps=8, sigma=0.0):
        super().__init__()
        self.latent_ch = latent_ch
        self.width = width
        self.n_blocks = n_blocks
        self.v = VelocityNet(latent_ch, width, n_blocks)
        self.steps = steps
        self.sigma = float(sigma)     # bridge noise (0 = deterministic RF); stored in checkpoint["arch"]
        self.sample_sigma = None      # inference override: None = use self.sigma, 0 = deterministic
        self.register_buffer("scale", torch.ones(()))  # latent std, set from training data

    def forward(self, z_A, mask=None, steps=None):
        return self.sample(z_A, steps)

    @staticmethod
    def _noise(z, generator=None):
        if generator is None:
            return torch.randn_like(z)
        return torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype)

    @torch.no_grad()
    def sample(self, z_A, steps=None, sigma=None, generator=None):
        """n bridge-posterior steps from z_A. sigma: noise level (default:
        sample_sigma if set, else the trained sigma). sigma=0 is the old Euler
        RF sampler exactly. generator: optional torch.Generator for repeatable noise."""
        n = steps or self.steps
        if sigma is None:
            sigma = self.sigma if self.sample_sigma is None else self.sample_sigma
        a = z_A / self.scale
        z = a.clone()
        for i in range(n):
            t, t2 = i / n, (i + 1) / n
            z = z + self.v(z, torch.full((z.shape[0],), t, device=z.device), a) / n   # = z + dt/(1-t) (z_B_hat - z)
            if sigma > 0 and i < n - 1:
                z = z + sigma * math.sqrt((t2 - t) * (1 - t2) / (1 - t)) * self._noise(z, generator)
        return z * self.scale

    def rf_loss(self, z_A, z_B, time_sampler="logit_normal", return_endpoint=False):
        """Rectified-flow training loss: MSE between v_theta and the straight-line
        velocity (z_B - z_A), evaluated at a random point on the line.

        return_endpoint=True also returns z_B_hat = z_t + (1 - t) * v, the
        model's one-jump guess of z_B from that point (unscaled, differentiable),
        for image-space losses on decode_B(z_B_hat)."""
        a = z_A / self.scale
        b = z_B / self.scale
        bsz = a.shape[0]
        if time_sampler == "logit_normal":
            t = torch.sigmoid(torch.randn(bsz, device=a.device))
        else:
            t = torch.rand(bsz, device=a.device)
        tt = t[:, None, None, None]
        z_t = (1 - tt) * a + tt * b
        if self.sigma > 0:
            # bridge: noise on the path; target = drift towards z_B from the noisy point.
            # (b - z_t) / (1 - t) has noise / sqrt(1 - t), so the error is weighted by
            # (1 - t)^2 (= endpoint-space error), normalized to mean 1, as in traj_loss.
            t = t.clamp(max=0.999)
            tt = t[:, None, None, None]
            z_t = z_t + self.sigma * torch.sqrt(tt * (1 - tt)) * torch.randn_like(z_t)
            target = (b - z_t) / (1 - tt)
            pred = self.v(z_t, t, a)
            w = (1 - t) ** 2
            w = w / w.mean()
            loss = (w * ((pred - target) ** 2).flatten(1).mean(1)).mean()
        else:
            target = b - a
            pred = self.v(z_t, t, a)
            loss = F.mse_loss(pred, target)
        if not return_endpoint:
            return loss
        z_B_hat = (z_t + (1 - tt) * pred) * self.scale
        return loss, z_B_hat

    def direct_loss(self, z_A, z_B, return_endpoint=False):
        """t = 0 only: v(z_A, 0 | z_A) regresses z_B - z_A, so z_B_hat = z_A + v is
        exactly what sample(steps=1) returns. Same MSE (scaled latent) as rf_loss,
        which at t = 0 has the same target and weight."""
        a = z_A / self.scale
        b = z_B / self.scale
        t = torch.zeros(a.shape[0], device=a.device)
        pred = self.v(a, t, a)
        loss = F.mse_loss(pred, b - a)
        if not return_endpoint:
            return loss
        return loss, (a + pred) * self.scale

    @torch.no_grad()
    def rollout(self, a, k, n, sigma=0.0):
        """Per-sample partial sampling in scaled space: sample i takes the first
        k[i] of n[i] steps from a[i] (with bridge noise if sigma > 0, exactly as
        sample() does). a: (B,C,H,W) = z_A / scale; k, n: (B,) float tensors
        with k <= n - 1. Returns the reached points z."""
        z = a.clone()
        for i in range(int(k.max().item())):
            active = (i < k)[:, None, None, None]
            # finished samples (i >= k) are still in the batch: clamp their time below 1 so nothing
            # divides by zero, and select with where() -- 0 * nan would still be nan
            t = torch.minimum(torch.full_like(n, float(i)), n - 1) / n
            t2 = t + 1 / n
            step = self.v(z, t, a) / n[:, None, None, None]
            if sigma > 0:
                std = sigma * torch.sqrt((t2 - t) * (1 - t2) / (1 - t))
                step = step + std[:, None, None, None] * torch.randn_like(z)
            z = torch.where(active, z + step, z)
        return z

    def traj_loss(self, z_A, z_B, roller, step_choices=(4, 8, 16), return_endpoint=False, n_mode="sample"):
        """Like rf_loss, but at a point on the model's OWN trajectory: `roller`
        (normally the EMA copy, i.e. the weights used at inference) runs k of n
        Euler steps from z_A, with n drawn from step_choices and k uniform in
        [1, n-1] per sample. The target is the velocity that reaches the true
        z_B from there in the remaining time: (z_B - z) / (1 - t).

        The per-sample error is weighted by (1 - t)^2, i.e. measured where the
        jump lands ((z + (1-t) v) vs z_B) rather than in velocity, and the
        weights are normalized to mean 1. Without it, 1 / (1 - t) up to n
        amplifies drift late in the path (x256 in MSE at n=16) and those few
        samples dominate the loss.

        n_mode="batch" draws ONE sampler length n for the whole batch instead of
        one per sample. The rollout runs until the largest k in the batch, so
        with long samplers (e.g. 64) in the choices, per-sample n makes almost
        every batch pay ~n forward passes; per-batch n costs ~2.5x less on
        average and covers the same (n, k) distribution over training."""
        a = z_A / self.scale
        b = z_B / self.scale
        bsz = a.shape[0]
        choices = torch.tensor(step_choices, dtype=torch.float32, device=a.device)
        if n_mode == "batch":
            n = choices[torch.randint(len(step_choices), (1,), device=a.device)].expand(bsz)
        else:
            n = choices[torch.randint(len(step_choices), (bsz,), device=a.device)]
        k = torch.floor(torch.rand(bsz, device=a.device) * (n - 1)) + 1
        z = roller.rollout(a, k, n, self.sigma)
        t = k / n
        tt = t[:, None, None, None]
        target = (b - z) / (1 - tt)
        pred = self.v(z, t, a)
        w = (1 - t) ** 2
        w = w / w.mean()
        loss = (w * ((pred - target) ** 2).flatten(1).mean(1)).mean()
        if not return_endpoint:
            return loss
        z_B_hat = (z + (1 - tt) * pred) * self.scale
        return loss, z_B_hat

    def clone_for_ema(self):
        ema = RFMapping(self.latent_ch, self.width, self.n_blocks, self.steps, self.sigma)
        ema.load_state_dict(self.state_dict())
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema

    @torch.no_grad()
    def ema_update(self, source, decay=0.999):
        for p_ema, p in zip(self.v.parameters(), source.v.parameters()):
            p_ema.mul_(decay).add_(p, alpha=1 - decay)
        self.scale.copy_(source.scale)  # scalar: just mirror, no EMA needed
