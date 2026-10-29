"""
Trains RFMapping (rf_model.py) to map z_A -> z_B, using (A, B) pixel pairs
from online_dataset.py and the frozen VAEs from OldPhotoModel.

- The VAEs are never trained; only rf.v's parameters get gradients.
- Loss = latent rectified-flow MSE + (optional) IMAGE-SPACE losses on the
  decoded endpoint prediction. Microsoft's own mapping net is trained with
  image-space losses (VGG / feature matching / GAN on decode_B output); a
  latent-only MSE leaves small latent errors that vae_B decodes into colour
  blotches and noise. The image terms are:
    pixel_weight      L1(decode_B(z_B_hat), B)
    chroma_weight     L1 on chroma (pixel minus its RGB mean) -- colour
                      fidelity; for gray targets it forces the output gray
    luma_weight       L1 on brightness (Y) -- tone + grain, gray and colour
    grad_weight       L1 on brightness gradients -- edges/detail; penalizes
                      both missing edges (blur) and extra ones (grain)
    perceptual_weight VGG16 feature L1 (can add grain if set too high)
    gan_weight /      LSGAN patch discriminator + its feature matching, the
    fm_weight         sharpness terms Microsoft's mapping training also uses
    gray_weight       multiplier on the per-sample terms for gray targets
                      (1.0 = gray and colour count equally)
    lowfreq_weight    L1 on 8x8 average-pooled images -- global tone /
                      brightness, which drifted in multi-step sampling
- Bridge matching (sigma > 0): see rf_model.py. sigma = 0 reproduces the
  deterministic RF exactly (same loss, same sampler). Validation samples use
  a fixed noise seed, so validation numbers are comparable across evals.
- Achievable target (target="vae"): the image losses compare the prediction
  with decode_B(encode_B(B)) -- the best image the frozen VAE B can produce
  -- instead of B itself, which even a perfect latent cannot reproduce (the
  VAE ceiling: dE00 ~5.3 vs ground truth). For gray targets the VAE
  reconstruction is replaced by its luminance, so gray stays exactly gray.
  The GAN / feature matching still use the real B as "real".
- Trajectory training (traj_prob > 0): on that fraction of steps the loss is
  taken at a point on the EMA model's own Euler trajectory instead of the
  ideal straight line (RFMapping.traj_loss), so the model learns to correct
  the drift that multi-step sampling accumulates (seen as a brightness bias
  that grew with the step count).
  z_B_hat is the one-jump endpoint estimate from a random point t on the
  straight path, so every t the sampler visits is supervised in image space,
  not only t=0.
- Validation encodes the held-out set once, then measures ODE-sampled latent
  MSE at a few step counts, plus (when image losses are on) decoded-image
  L1 / chroma / gradient error, reported separately for gray and colour
  targets. The best checkpoint is then picked by the average of the gray and
  colour scores, so neither can be sacrificed for the other.
- Checkpoints: latest.pt is saved every `save_every` steps, ATOMICALLY (written to
  a temp file, then renamed), with optimizer, scheduler, EMA and discriminator
  state, so a Colab disconnect never corrupts it; re-running train() with the
  same out_dir resumes from it. Optional snapshots every `snapshot_every` steps.
  With select_best=False / early_stop=False (fixed schedule, like Microsoft's
  training), no metric-based best.pt is written and the final checkpoint is used.
- t_mode="direct": every step uses RFMapping.direct_loss, i.e.
  only t = 0 and z_t = z_A -- the one point 1-step sampling evaluates. No bridge
  noise, no trajectory training; validation is 1-step only. t_mode is stored in
  checkpoint["arch"] and checked on resume.
- disc_type="multiscale": pix2pixHD-style discriminator as used by Microsoft's
  mapping training (Pix2PixHDModel_Mapping): num_D scales of an n_layers
  PatchGAN with instance norm, LSGAN, optionally conditional on the degraded
  input, feature matching weighted 4/(n_layers+1) * 1/num_D per layer;
  perceptual_type="vgg19": pix2pixHD VGGLoss (relu1_1..relu5_1, weights
  1/32..1). Microsoft weights both feature matching and VGG with lambda_feat=10.
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from rf_model import RFMapping


class VGGPerceptualLoss(nn.Module):
    """Feature-space loss: penalizes perceptual/structural difference, not just
    per-pixel color, so the network is pushed toward confident, realistic
    predictions instead of the pixelwise-MSE-optimal blurry average."""

    def __init__(self, device, layers=(3, 8, 15, 22)):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg
        self.layers = set(layers)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def forward(self, pred, target):
        # pred/target are in [-1,1] (old_photo_model convention) -> back to [0,1] -> ImageNet norm
        pred_n = ((pred + 1) / 2 - self.mean) / self.std
        with torch.no_grad():  # target never needs gradients -- don't build/store its graph at all
            target_n = ((target + 1) / 2 - self.mean) / self.std
            y_feats = []
            y = target_n
            for i, layer in enumerate(self.vgg):
                y = layer(y)
                if i in self.layers:
                    y_feats.append(y)
                if i >= max(self.layers):
                    break

        loss = 0.0
        x = pred_n
        fi = 0
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                loss = loss + F.l1_loss(x, y_feats[fi])
                fi += 1
            if i >= max(self.layers):
                break
        return loss


def chroma(x):
    """Per-pixel deviation from the RGB mean: exactly 0 for a gray image."""
    return x - x.mean(dim=1, keepdim=True)


def luma(x):
    """Brightness (BT.601 Y), (N,1,H,W): where grain and fine detail live."""
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def per_sample_l1(x, y):
    """(N,) mean absolute error per sample."""
    return (x - y).abs().flatten(1).mean(1)


def grad_l1(x, y):
    """(N,) L1 between horizontal/vertical gradients of x and y: penalizes
    missing edges (blur) and extra edges (grain) alike."""
    dx = lambda t: t[..., :, 1:] - t[..., :, :-1]
    dy = lambda t: t[..., 1:, :] - t[..., :-1, :]
    return per_sample_l1(dx(x), dx(y)) + per_sample_l1(dy(x), dy(y))


def is_gray(x, tol=1e-3):
    """(N,) bool: the image has no colour."""
    return chroma(x).abs().flatten(1).amax(1) < tol


def lowfreq_l1(x, y, k=8):
    """(N,) L1 between k x k average-pooled images: only global tone / brightness."""
    return per_sample_l1(F.avg_pool2d(x, k), F.avg_pool2d(y, k))


def wmean(values, weights):
    return (values * weights).sum() / weights.sum()


class PatchDiscriminator(nn.Module):
    """Small PatchGAN with spectral norm. forward() -> (patch logits, features)."""

    def __init__(self, ch=64, n_layers=3):
        super().__init__()
        sn = nn.utils.spectral_norm
        layers, c_in = [], 3
        for i in range(n_layers):
            c_out = ch * 2 ** i
            layers.append(nn.Sequential(sn(nn.Conv2d(c_in, c_out, 4, 2, 1)), nn.LeakyReLU(0.2)))
            c_in = c_out
        layers.append(nn.Sequential(sn(nn.Conv2d(c_in, c_in, 4, 1, 1)), nn.LeakyReLU(0.2)))
        self.body = nn.ModuleList(layers)
        self.head = sn(nn.Conv2d(c_in, 1, 4, 1, 1))

    def forward(self, x):
        feats = []
        for layer in self.body:
            x = layer(x)
            feats.append(x)
        return self.head(x), feats


class NLayerDiscriminator(nn.Module):
    """pix2pixHD NLayerDiscriminator with intermediate features:
    forward() -> [feat_1, ..., feat_{n_layers+1}, logits]."""

    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm="instance"):
        super().__init__()
        kw, padw = 4, 2
        norm_layer = (lambda c: nn.InstanceNorm2d(c, affine=False)) if norm == "instance" else (lambda c: nn.Identity())
        seq = [[nn.Conv2d(input_nc, ndf, kw, 2, padw), nn.LeakyReLU(0.2)]]
        nf = ndf
        for _ in range(1, n_layers):
            nf_prev, nf = nf, min(nf * 2, 512)
            seq.append([nn.Conv2d(nf_prev, nf, kw, 2, padw), norm_layer(nf), nn.LeakyReLU(0.2)])
        nf_prev, nf = nf, min(nf * 2, 512)
        seq.append([nn.Conv2d(nf_prev, nf, kw, 1, padw), norm_layer(nf), nn.LeakyReLU(0.2)])
        seq.append([nn.Conv2d(nf, 1, kw, 1, padw)])
        self.layers = nn.ModuleList(nn.Sequential(*m) for m in seq)

    def forward(self, x):
        out = []
        for layer in self.layers:
            x = layer(x)
            out.append(x)
        return out


class MultiscaleDiscriminator(nn.Module):
    """pix2pixHD MultiscaleDiscriminator: num_D NLayerDiscriminators on the
    image downsampled 1x, 2x, ... forward() -> list (one per scale) of feature lists."""

    def __init__(self, input_nc=3, ndf=64, n_layers=3, num_D=2, norm="instance"):
        super().__init__()
        self.num_D, self.n_layers = num_D, n_layers
        self.ds = nn.ModuleList(NLayerDiscriminator(input_nc, ndf, n_layers, norm) for _ in range(num_D))
        self.down = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        res = []
        for i, d in enumerate(self.ds):
            res.append(d(x))
            if i != self.num_D - 1:
                x = self.down(x)
        return res


def ms_gan_g_loss(preds_fake):
    """LSGAN generator loss, summed over scales (pix2pixHD GANLoss)."""
    return sum(((p[-1] - 1) ** 2).mean() for p in preds_fake)


def ms_gan_d_loss(preds_real, preds_fake):
    return 0.5 * (sum(((p[-1] - 1) ** 2).mean() for p in preds_real) + sum((p[-1] ** 2).mean() for p in preds_fake))


def ms_fm_loss(preds_fake, preds_real, n_layers, num_D):
    """pix2pixHD discriminator feature matching (all intermediate features, not the logits)."""
    feat_w, d_w = 4.0 / (n_layers + 1), 1.0 / num_D
    return sum(d_w * feat_w * F.l1_loss(pf[j], pr[j].detach())
               for pf, pr in zip(preds_fake, preds_real) for j in range(len(pf) - 1))


class VGG19Loss(nn.Module):
    """pix2pixHD VGGLoss: L1 between VGG19 relu1_1, relu2_1, relu3_1, relu4_1,
    relu5_1 features, weighted 1/32, 1/16, 1/8, 1/4, 1. Inputs in [-1,1] are
    mapped to ImageNet normalization first."""

    def __init__(self, device):
        super().__init__()
        vgg = tvm.vgg19(weights=tvm.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        for q in vgg.parameters():
            q.requires_grad_(False)
        cuts = (2, 7, 12, 21, 30)
        self.slices = nn.ModuleList(vgg[a:b] for a, b in zip((0,) + cuts[:-1], cuts))
        self.weights = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1.0)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def forward(self, pred, target):
        x = ((pred + 1) / 2 - self.mean) / self.std
        with torch.no_grad():
            y = ((target + 1) / 2 - self.mean) / self.std
            ys = []
            for sl in self.slices:
                y = sl(y)
                ys.append(y)
        loss = 0.0
        for sl, w, yf in zip(self.slices, self.weights, ys):
            x = sl(x)
            loss = loss + w * F.l1_loss(x, yf)
        return loss


# --------------------------------------------------------------------------- #
# Validation: precompute encoded val latents once, then ODE-sample at eval time
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_loader(obj, loader, max_batches=None, keep_images=64):
    """Returns (zA, zB, B_images); B_images keeps the first `keep_images`
    targets (CPU) for the image-space validation metrics."""
    zA_all, zB_all, B_keep = [], [], []
    n_kept = 0
    for i, (A, B) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        if n_kept < keep_images:
            B_keep.append(B[:keep_images - n_kept])
            n_kept += B_keep[-1].shape[0]
        A, B = A.to(obj.device), B.to(obj.device)
        zA_all.append(obj.encode_A(A).cpu())
        zB_all.append(obj.encode_B(B).cpu())
    B_img = torch.cat(B_keep) if B_keep else None
    return torch.cat(zA_all), torch.cat(zB_all), B_img


@torch.no_grad()
def validate(rf, zA_val, zB_val, device, step_counts=(1, 4, 8), batch_size=16,
             obj=None, B_img=None, pix_steps=8):
    rf.eval()
    n = zA_val.shape[0]
    sums = {s: 0.0 for s in step_counts}
    gen = torch.Generator(device=device).manual_seed(1234)   # same bridge noise every eval
    for i in range(0, n, batch_size):
        a = zA_val[i:i + batch_size].to(device)
        b = zB_val[i:i + batch_size].to(device)
        for s in step_counts:
            pred = rf.sample(a, steps=s, generator=gen)
            sums[s] += F.mse_loss(pred, b, reduction="sum").item()
    numel = zA_val.numel()
    out = {s: v / numel for s, v in sums.items()}

    if obj is not None and B_img is not None:
        per = {"l1": [], "chroma": [], "grad": [], "bias": [], "gray": []}
        m = B_img.shape[0]
        for i in range(0, m, batch_size):
            j = min(i + batch_size, m)
            a = zA_val[i:j].to(device)
            tgt = B_img[i:j].to(device)
            img = obj.decode_B(rf.sample(a, steps=pix_steps, generator=gen))
            per["l1"].append(per_sample_l1(img, tgt).cpu())
            per["chroma"].append(per_sample_l1(chroma(img), chroma(tgt)).cpu())
            per["grad"].append(grad_l1(luma(img), luma(tgt)).cpu())
            per["bias"].append(((img - tgt) / 2).flatten(1).mean(1).cpu())  # brightness bias on a [0,1] scale
            per["gray"].append(is_gray(tgt).cpu())
        per = {k: torch.cat(v) for k, v in per.items()}
        scores = []
        for name, mask in (("gray", per["gray"]), ("color", ~per["gray"])):
            if mask.any():
                for k in ("l1", "chroma", "grad", "bias"):
                    out[f"{name}_{k}"] = per[k][mask].mean().item()
                scores.append(out[f"{name}_l1"] + out[f"{name}_chroma"] + out[f"{name}_grad"])
        out["img"] = sum(scores) / len(scores)  # gray and colour weighted equally
    return out


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(path, rf, ema, opt, sched, step, best_val, patience_ct, extra=None):
    """Atomic: write to path.tmp, then rename, so an interrupted save (e.g. a Colab
    disconnect) never leaves a corrupt checkpoint behind."""
    tmp = path + ".tmp"
    torch.save({
        **(extra or {}),
        "rf": rf.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "step": step,
        "best_val": best_val,
        "patience_ct": patience_ct,
        "arch": {"latent_ch": rf.latent_ch, "width": rf.width, "n_blocks": rf.n_blocks, "sigma": rf.sigma,
                 "t_mode": getattr(rf, "t_mode", "bridge")},
    }, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(
    obj,                      # OldPhotoModel instance (frozen VAEs)
    train_loader,
    val_loader,
    out_dir,
    width=256,
    n_blocks=9,
    lr=2e-4,
    weight_decay=1e-4,
    max_steps=20000,
    eval_every=500,
    ema_decay=0.999,
    patience=8,                # eval rounds without improvement before stopping
    val_max_batches=None,      # None = use the whole val_loader for encoding
    resume=True,
    init_from=None,            # path to a checkpoint whose EMA weights warm-start rf/ema;
                                # optimizer/scheduler/step/best_val/patience all start fresh.
                                # Ignored if `resume` finds an existing latest_path in out_dir.
    pixel_weight=0.0,          # L1 on decode_B(z_B_hat) vs B
    chroma_weight=0.0,         # L1 on chroma -- colour fidelity / anti-blotching
    luma_weight=0.0,           # L1 on brightness -- tone + grain
    grad_weight=0.0,           # L1 on brightness gradients -- detail without grain
    perceptual_weight=0.0,     # VGG16 feature L1 on decode_B(z_B_hat) vs B
    gan_weight=0.0,            # LSGAN patch-discriminator loss -- sharpness
    fm_weight=0.0,             # discriminator feature matching (needs gan_weight > 0)
    gan_start=0,               # step at which the GAN terms switch on
    d_lr=2e-4,                 # discriminator learning rate
    gray_weight=1.0,           # loss multiplier for gray targets (1.0 = equal to colour)
    lowfreq_weight=0.0,        # L1 on 8x8-pooled images -- global tone / brightness
    traj_prob=0.0,             # fraction of steps trained on the EMA model's own trajectory
    traj_steps=(4, 8, 16),     # sampler step counts the trajectory points are drawn from
    traj_n_mode="sample",      # "batch": one sampler length per batch (much cheaper with long samplers)
    target="clean",            # image-loss target: "clean" = B, "vae" = decode_B(encode_B(B)) (achievable)
    sigma=0.0,                 # bridge noise level (0 = deterministic RF)
    t_mode="bridge",           # "bridge": t ~ U[0,1] (+ trajectory); "direct": t = 0 only (ablation)
    disc_type="patch",         # "patch" (small PatchGAN) or "multiscale" (pix2pixHD, as Microsoft)
    num_D=2,                   # multiscale: number of scales
    n_layers_D=3,              # multiscale: layers per discriminator
    d_conditional=False,       # multiscale: discriminator also sees the degraded input (pix2pixHD cGAN)
    d_start=None,              # step at which the discriminator starts training (None = gan_start);
                                # lets a fresh discriminator warm up before it drives the generator
    perceptual_type="vgg16",   # "vgg16" or "vgg19" (pix2pixHD VGGLoss, as Microsoft)
    save_every=200,            # write latest.pt every N steps (crash-safe resume)
    snapshot_every=0,          # also keep step_XXXXXX.pt every N steps (0 = off)
    early_stop=True,           # False: fixed schedule, never stop early
    select_best=True,          # False: no metric-based best.pt; use the final checkpoint
    val_pix_steps=8,           # sampling steps for the image-space validation metrics
    pixel_every=1,             # only add the image-space terms every N steps, to control compute cost
    val_pix_images=128,        # val images kept for the image-space val metrics
):
    os.makedirs(out_dir, exist_ok=True)
    latest_path = os.path.join(out_dir, "latest.pt")
    best_path = os.path.join(out_dir, "best.pt")
    device = obj.device
    assert target in ("clean", "vae"), target
    assert traj_n_mode in ("sample", "batch"), traj_n_mode
    assert t_mode in ("bridge", "direct"), t_mode
    assert t_mode == "bridge" or traj_prob == 0, "t_mode='direct' trains only t = 0: set traj_prob=0"
    image_losses = any(w > 0 for w in (pixel_weight, chroma_weight, luma_weight, grad_weight,
                                       perceptual_weight, gan_weight, lowfreq_weight))

    # infer latent channel count from the loaded VAEs
    with torch.no_grad():
        latent_ch = obj.encode_A(torch.zeros(1, 3, 64, 64, device=device)).shape[1]

    rf = RFMapping(latent_ch, width, n_blocks, sigma=sigma).to(device)
    rf.t_mode = t_mode
    print(f"t_mode: {t_mode}" + (" (t = 0 only: direct regression z_A -> z_B)" if t_mode == "direct" else ""))

    if init_from and not (resume and os.path.exists(latest_path)):
        ck = load_checkpoint(init_from, device)
        rf.load_state_dict(ck["ema"])  # warm-start from the better (EMA) weights
        if ck["arch"].get("sigma", 0.0) != sigma:
            print(f"NOTE: init_from was trained with sigma={ck['arch'].get('sigma', 0.0)}, this run uses sigma={sigma}")
        print(f"warm-started weights from {init_from} (its step {ck['step']}, best_val {ck['best_val']:.5f}); "
              f"optimizer/scheduler/step/patience all reset")

    # encode the validation set once (VAEs frozen, so this doesn't change during training)
    print("encoding validation set...")
    zA_val, zB_val, B_val = encode_loader(obj, val_loader, val_max_batches,
                                          keep_images=val_pix_images if image_losses else 0)
    rf.scale.fill_(float(zB_val.float().std()))
    print(f"val set: {zA_val.shape[0]} samples | latent shape {tuple(zA_val.shape[1:])} | scale {float(rf.scale):.4f}")
    path_rms = float(((zB_val - zA_val).float() / rf.scale.cpu()).pow(2).mean().sqrt())
    print(f"bridge sigma {sigma} | RMS |z_B - z_A| (scaled) {path_rms:.3f} | "
          f"max path noise std sigma/2 = {sigma / 2:.3f} ({sigma / 2 / max(path_rms, 1e-6):.0%} of the path length)")

    ema = rf.clone_for_ema().to(device).eval()   # eval: no dropout in trajectory rollouts
    assert disc_type in ("patch", "multiscale") and perceptual_type in ("vgg16", "vgg19")
    d_start = gan_start if d_start is None else d_start
    perceptual = None
    if perceptual_weight > 0:
        perceptual = VGG19Loss(device) if perceptual_type == "vgg19" else VGGPerceptualLoss(device)
    opt = torch.optim.AdamW(rf.v.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_steps)

    disc = opt_d = None
    if gan_weight > 0:
        if disc_type == "multiscale":
            disc = MultiscaleDiscriminator(6 if d_conditional else 3, 64, n_layers_D, num_D).to(device)
        else:
            disc = PatchDiscriminator().to(device)
        opt_d = torch.optim.Adam(disc.parameters(), lr=d_lr, betas=(0.5, 0.999))
        print(f"discriminator: {disc_type}" + (f" (num_D={num_D}, n_layers={n_layers_D}, "
              f"conditional={d_conditional})" if disc_type == "multiscale" else "")
              + f" | {sum(q.numel() for q in disc.parameters()) / 1e6:.2f}M params")

    def d_in(img, cond):
        return torch.cat([cond, img], 1) if (disc_type == "multiscale" and d_conditional) else img

    def extra_state():
        return ({"disc": disc.state_dict(), "opt_d": opt_d.state_dict(), "disc_type": disc_type,
                 "d_conditional": d_conditional} if disc is not None else None)

    start_step, best_val, patience_ct = 0, float("inf"), 0
    if resume and os.path.exists(latest_path):
        ck = load_checkpoint(latest_path, device)
        rf.load_state_dict(ck["rf"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_step, best_val, patience_ct = ck["step"], ck["best_val"], ck["patience_ct"]
        if ck["arch"].get("sigma", 0.0) != sigma:
            raise ValueError(f"{latest_path} was trained with sigma={ck['arch'].get('sigma', 0.0)} but sigma={sigma} "
                             f"was passed -- use the same sigma, or a new out_dir")
        if ck["arch"].get("t_mode", "bridge") != t_mode:
            raise ValueError(f"{latest_path} was trained with t_mode={ck['arch'].get('t_mode', 'bridge')} but "
                             f"t_mode={t_mode} was passed -- use a new out_dir")
        if disc is not None and "disc" in ck:
            if ck.get("disc_type", "patch") == disc_type and ck.get("d_conditional", False) == d_conditional:
                disc.load_state_dict(ck["disc"]); opt_d.load_state_dict(ck["opt_d"])
            else:
                print("NOTE: checkpoint has a different discriminator type -- discriminator starts fresh")
        print(f"RESUMED from {latest_path} at step {start_step}/{max_steps}")

    rf.train()
    terms = ("pix", "chroma", "luma", "grad", "low", "vgg")
    run = dict.fromkeys(("loss", "rf", *terms, "gan", "fm", "d"), 0.0)
    n_steps, n_img_steps, n_gan_steps, n_traj = 0, 0, 0, 0
    t0 = time.time()
    step = start_step
    train_iter = iter(train_loader)

    while step < max_steps:
        try:
            A, B = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            A, B = next(train_iter)

        A, B = A.to(device), B.to(device)
        with torch.no_grad():                     # VAEs stay frozen
            z_A = obj.encode_A(A)
            z_B = obj.encode_B(B)

        gan_on = False
        use_traj = traj_prob > 0 and torch.rand(()).item() < traj_prob
        n_traj += int(use_traj)
        if image_losses and step % pixel_every == 0:
            if use_traj:
                loss_rf, z_B_hat = rf.traj_loss(z_A, z_B, ema, traj_steps, return_endpoint=True,
                                                n_mode=traj_n_mode)
            elif t_mode == "direct":
                loss_rf, z_B_hat = rf.direct_loss(z_A, z_B, return_endpoint=True)
            else:
                # uniform t so the endpoint guess is supervised near t=0 too, where
                # few-step sampling starts (logit-normal rarely samples there)
                loss_rf, z_B_hat = rf.rf_loss(z_A, z_B, time_sampler="uniform", return_endpoint=True)
            # obj.decode_B is @torch.no_grad-wrapped for inference; call the VAE directly
            # so gradients flow from the image-space losses back into rf.v.
            pred_img = obj.vae_B.forward(z_B_hat, flow="dec")
            gray_b = is_gray(B)
            w = 1.0 + (gray_weight - 1.0) * gray_b.float()   # per-sample emphasis
            if target == "vae":
                with torch.no_grad():   # the best image VAE B can make from the true clean latent
                    tgt = obj.vae_B.forward(z_B, flow="dec")
                    if gray_b.any():    # gray targets stay exactly gray
                        tgt[gray_b] = luma(tgt[gray_b]).expand(-1, 3, -1, -1)
            else:
                tgt = B
            y_pred, y_tgt = luma(pred_img), luma(tgt)
            l = {
                "pix": wmean(per_sample_l1(pred_img, tgt), w),
                "chroma": wmean(per_sample_l1(chroma(pred_img), chroma(tgt)), w),
                "luma": wmean(per_sample_l1(y_pred, y_tgt), w),
                "grad": wmean(grad_l1(y_pred, y_tgt), w),
                "low": wmean(lowfreq_l1(pred_img, tgt), w),
                "vgg": perceptual(pred_img, tgt) if perceptual is not None else torch.zeros((), device=device),
            }
            loss = (loss_rf + pixel_weight * l["pix"] + chroma_weight * l["chroma"]
                    + luma_weight * l["luma"] + grad_weight * l["grad"] + lowfreq_weight * l["low"]
                    + perceptual_weight * l["vgg"])

            gan_on = disc is not None and step >= d_start          # discriminator trains this step
            g_adv = disc is not None and step >= gan_start         # adversarial terms drive the generator
            if g_adv:
                if disc_type == "multiscale":
                    p_fake = disc(d_in(pred_img, A))
                    with torch.no_grad():
                        p_real = disc(d_in(B, A))
                    l["gan"] = ms_gan_g_loss(p_fake)
                    l["fm"] = ms_fm_loss(p_fake, p_real, n_layers_D, num_D)
                else:
                    logits, f_fake = disc(pred_img)
                    with torch.no_grad():
                        _, f_real = disc(B)
                    l["gan"] = ((logits - 1) ** 2).mean()
                    l["fm"] = sum(F.l1_loss(a, b) for a, b in zip(f_fake, f_real)) / len(f_real)
                loss = loss + gan_weight * l["gan"] + fm_weight * l["fm"]
                n_gan_steps += 1

            for k, v in l.items():
                run[k] += v.item()
            n_img_steps += 1
        else:
            loss_rf = (rf.traj_loss(z_A, z_B, ema, traj_steps, n_mode=traj_n_mode) if use_traj
                       else rf.direct_loss(z_A, z_B) if t_mode == "direct" else rf.rf_loss(z_A, z_B))
            loss = loss_rf

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rf.v.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.ema_update(rf, decay=ema_decay)

        if gan_on:  # discriminator step: real B vs the (detached) prediction
            if disc_type == "multiscale":
                l_d = ms_gan_d_loss(disc(d_in(B, A)), disc(d_in(pred_img.detach(), A)))
            else:
                d_real, _ = disc(B)
                d_fake, _ = disc(pred_img.detach())
                l_d = 0.5 * (((d_real - 1) ** 2).mean() + (d_fake ** 2).mean())
            opt_d.zero_grad(set_to_none=True)
            l_d.backward()
            opt_d.step()
            run["d"] += l_d.item()

        run["loss"] += loss.item()
        run["rf"] += loss_rf.item()
        n_steps += 1
        step += 1

        if save_every and step % save_every == 0 and step % eval_every != 0:
            save_checkpoint(latest_path, rf, ema, opt, sched, step, best_val, patience_ct, extra_state())
        if snapshot_every and step % snapshot_every == 0:
            save_checkpoint(os.path.join(out_dir, f"step_{step:06d}.pt"), rf, ema, opt, sched, step,
                            best_val, patience_ct, extra_state())

        if step % eval_every == 0 or step == max_steps:
            msg = f"train_loss {run['loss'] / n_steps:.4f} (rf {run['rf'] / n_steps:.4f}"
            if n_img_steps:
                msg += "".join(f", {k} {run[k] / n_img_steps:.4f}" for k in terms)
            if n_gan_steps:
                msg += "".join(f", {k} {run[k] / n_gan_steps:.4f}" for k in ("gan", "fm", "d"))
            if traj_prob > 0:
                msg += f", traj {n_traj / n_steps:.0%} of steps"
            msg += ")"
            run = dict.fromkeys(run, 0.0)
            n_steps, n_img_steps, n_gan_steps, n_traj = 0, 0, 0, 0

            val = validate(ema, zA_val, zB_val, device, obj=obj, B_img=B_val, pix_steps=val_pix_steps,
                           step_counts=(1,) if t_mode == "direct" else (1, 4, 8))
            elapsed = time.time() - t0
            print(f"step {step}/{max_steps} | {msg}\n"
                  "    val_mse " + "  ".join(f"{s}-step {val[s]:.5f}" for s in (1, 4, 8) if s in val) +
                  f" | {elapsed:.0f}s | lr {sched.get_last_lr()[0]:.2e}")
            for name in ("gray", "color"):
                if f"{name}_l1" in val:
                    print(f"    val {name:5s}  l1 {val[name + '_l1']:.4f}  chroma {val[name + '_chroma']:.4f}"
                          f"  grad {val[name + '_grad']:.4f}  brightness bias {val[name + '_bias']:+.4f}"
                          f"  ({val_pix_steps}-step)")

            # best-checkpoint / early-stopping metric
            val_metric = val["img"] if "img" in val else val[8]

            if val_metric < best_val - 1e-6:
                best_val = val_metric
                patience_ct = 0
                if select_best:
                    save_checkpoint(best_path, rf, ema, opt, sched, step, best_val, patience_ct, extra_state())
                    print(f"  new best (val={best_val:.5f}), saved {best_path}")
            else:
                patience_ct += 1
                if select_best or early_stop:
                    print(f"  no improvement ({patience_ct}/{patience})")
            # saved after the best/patience update so a resume sees current values
            save_checkpoint(latest_path, rf, ema, opt, sched, step, best_val, patience_ct, extra_state())
            print(f"  saved {latest_path} (step {step})")
            if early_stop and patience_ct >= patience:
                print(f"early stopping at step {step} (best val={best_val:.5f})")
                break

            rf.train()
            t0 = time.time()

    save_checkpoint(latest_path, rf, ema, opt, sched, step, best_val, patience_ct, extra_state())
    if not select_best:
        final_path = os.path.join(out_dir, "final.pt")
        save_checkpoint(final_path, rf, ema, opt, sched, step, best_val, patience_ct, extra_state())
        print(f"training finished at step {step} -> {final_path} (fixed schedule, final checkpoint)")
    else:
        print(f"training finished at step {step}. best val={best_val:.5f} -> {best_path}")
    return rf, ema
