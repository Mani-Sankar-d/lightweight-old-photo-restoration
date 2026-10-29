"""
Trains the final DirectMapping (direct_model.py): z_A -> z_B with the frozen VAEs, using
Microsoft's loss types only:

    loss = MSE(z_B_hat / scale, z_B / scale)                 latent loss
         + vgg_weight * VGG19(decode_B(z_B_hat), B)          pix2pixHD VGGLoss
         + gan_weight * LSGAN(D(A, decode_B(z_B_hat)))       conditional multi-scale D   } from
         + fm_weight  * feature matching(D)                                              } gan_start

AdamW + cosine LR, gradient clipping 1.0, EMA 0.999 (the EMA weights are the model),
discriminator Adam(2e-4, betas 0.5/0.999) trained from d_start. Fixed schedule, no
metric-based checkpoint selection: final.pt is the model.

Crash-safe: latest.pt (all states) is written atomically at every evaluation (and every
save_every steps if > 0); re-running with the same out_dir resumes. history.csv keeps the
validation curve. At the end, final.pt and a small weights-only mapping_direct.pt (EMA).
"""
import os
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from direct_model import DirectMapping


def chroma(x):
    """Per-pixel deviation from the RGB mean: exactly 0 for a gray image."""
    return x - x.mean(dim=1, keepdim=True)


def luma(x):
    """Brightness (BT.601 Y), (N,1,H,W)."""
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def per_sample_l1(x, y):
    return (x - y).abs().flatten(1).mean(1)


def grad_l1(x, y):
    dx = lambda t: t[..., :, 1:] - t[..., :, :-1]
    dy = lambda t: t[..., 1:, :] - t[..., :-1, :]
    return per_sample_l1(dx(x), dx(y)) + per_sample_l1(dy(x), dy(y))


def is_gray(x, tol=1e-3):
    return chroma(x).abs().flatten(1).amax(1) < tol


# --- pix2pixHD discriminator, LSGAN, feature matching, VGG19 loss: same code as train_rf.py ---
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


@torch.no_grad()
def encode_loader(obj, loader, keep_images=128):
    """Encodes the (fixed) validation set once: (zA, zB, first keep_images clean targets)."""
    zA, zB, keep, n_kept = [], [], [], 0
    for A, B in loader:
        if n_kept < keep_images:
            keep.append(B[:keep_images - n_kept])
            n_kept += keep[-1].shape[0]
        zA.append(obj.encode_A(A.to(obj.device)).cpu())
        zB.append(obj.encode_B(B.to(obj.device)).cpu())
    return torch.cat(zA), torch.cat(zB), torch.cat(keep)


@torch.no_grad()
def validate(net, obj, zA_val, zB_val, B_img, batch_size=16):
    """Latent MSE over the whole val set + image-space L1 / chroma / gradient / brightness bias,
    separately for gray and colour targets ("img" = their average, gray and colour equally weighted)."""
    device = obj.device
    net.eval()
    sse = 0.0
    for i in range(0, len(zA_val), batch_size):
        sse += F.mse_loss(net(zA_val[i:i + batch_size].to(device)), zB_val[i:i + batch_size].to(device),
                          reduction="sum").item()
    out = {"latent_mse": sse / zA_val.numel()}
    per = {"l1": [], "chroma": [], "grad": [], "bias": [], "gray": []}
    for i in range(0, len(B_img), batch_size):
        j = min(i + batch_size, len(B_img))
        img = obj.decode_B(net(zA_val[i:j].to(device)))
        tgt = B_img[i:j].to(device)
        per["l1"].append(per_sample_l1(img, tgt).cpu())
        per["chroma"].append(per_sample_l1(chroma(img), chroma(tgt)).cpu())
        per["grad"].append(grad_l1(luma(img), luma(tgt)).cpu())
        per["bias"].append(((img - tgt) / 2).flatten(1).mean(1).cpu())
        per["gray"].append(is_gray(tgt).cpu())
    per = {k: torch.cat(v) for k, v in per.items()}
    scores = []
    for name, mask in (("gray", per["gray"]), ("color", ~per["gray"])):
        if mask.any():
            for k in ("l1", "chroma", "grad", "bias"):
                out[f"{name}_{k}"] = per[k][mask].mean().item()
            scores.append(out[f"{name}_l1"] + out[f"{name}_chroma"] + out[f"{name}_grad"])
    out["img"] = sum(scores) / len(scores)
    return out


def _atomic_save(obj_, path):
    tmp = path + ".tmp"
    torch.save(obj_, tmp)
    os.replace(tmp, path)


def train_direct(
    obj, train_loader, val_loader, out_dir,
    width=256, n_blocks=9, dropout=0.1,
    lr=2e-4, weight_decay=1e-4, max_steps=20000, eval_every=500, ema_decay=0.999,
    vgg_weight=10.0, gan_weight=1.0, fm_weight=10.0,
    gan_start=2000, d_start=0, d_lr=2e-4,
    num_D=2, n_layers_D=3, d_conditional=True,
    val_pix_images=128, save_every=0, resume=True,
):
    os.makedirs(out_dir, exist_ok=True)
    latest_path, hist_path = f"{out_dir}/latest.pt", f"{out_dir}/history.csv"
    device = obj.device
    with torch.no_grad():
        latent_ch = obj.encode_A(torch.zeros(1, 3, 64, 64, device=device)).shape[1]
    net = DirectMapping(latent_ch, width, n_blocks, dropout).to(device)
    print(f"DirectMapping: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params")

    print("encoding validation set...")
    zA_val, zB_val, B_val = encode_loader(obj, val_loader, val_pix_images)
    net.scale.fill_(float(zB_val.float().std()))
    print(f"val set: {zA_val.shape[0]} samples | latent {tuple(zA_val.shape[1:])} | scale {float(net.scale):.4f}")

    ema = DirectMapping(latent_ch, width, n_blocks, dropout).to(device).eval()
    ema.load_state_dict(net.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    vgg = VGG19Loss(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_steps)
    disc = MultiscaleDiscriminator(6 if d_conditional else 3, 64, n_layers_D, num_D).to(device)
    opt_d = torch.optim.Adam(disc.parameters(), lr=d_lr, betas=(0.5, 0.999))
    d_in = (lambda img, cond: torch.cat([cond, img], 1)) if d_conditional else (lambda img, cond: img)
    print(f"discriminator: multiscale (num_D={num_D}, n_layers={n_layers_D}, conditional={d_conditional}) | "
          f"{sum(q.numel() for q in disc.parameters()) / 1e6:.2f}M params")

    def save(path, step):
        _atomic_save({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                      "sched": sched.state_dict(), "disc": disc.state_dict(), "opt_d": opt_d.state_dict(),
                      "step": step, "arch": net.arch()}, path)

    step, history = 0, []
    if resume and os.path.exists(latest_path):
        ck = torch.load(latest_path, map_location=device)
        if ck["arch"] != net.arch():
            raise ValueError(f"{latest_path} has arch {ck['arch']}, this run builds {net.arch()}: use a new out_dir")
        net.load_state_dict(ck["net"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        disc.load_state_dict(ck["disc"]); opt_d.load_state_dict(ck["opt_d"])
        step = ck["step"]
        if os.path.exists(hist_path):
            history = [r for r in pd.read_csv(hist_path).to_dict("records") if r["step"] <= step]
        print(f"RESUMED from {latest_path} at step {step}/{max_steps}")

    net.train()
    run = dict.fromkeys(("loss", "latent", "vgg", "gan", "fm", "d"), 0.0)
    n_steps, n_gan = 0, 0
    t0 = time.time()
    train_iter = iter(train_loader)
    while step < max_steps:
        try:
            A, B = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            A, B = next(train_iter)
        A, B = A.to(device), B.to(device)
        with torch.no_grad():                       # VAEs stay frozen
            z_A, z_B = obj.encode_A(A), obj.encode_B(B)

        z_hat = net(z_A)
        l_lat = F.mse_loss(z_hat / net.scale, z_B / net.scale)
        pred_img = obj.vae_B.forward(z_hat, flow="dec")   # not obj.decode_B: that one is no_grad
        l_vgg = vgg(pred_img, B)
        loss = l_lat + vgg_weight * l_vgg
        if step >= gan_start:
            p_fake = disc(d_in(pred_img, A))
            with torch.no_grad():
                p_real = disc(d_in(B, A))
            l_gan, l_fm = ms_gan_g_loss(p_fake), ms_fm_loss(p_fake, p_real, n_layers_D, num_D)
            loss = loss + gan_weight * l_gan + fm_weight * l_fm
            run["gan"] += l_gan.item()
            run["fm"] += l_fm.item()
            n_gan += 1

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for p_e, p in zip(ema.parameters(), net.parameters()):
                p_e.mul_(ema_decay).add_(p, alpha=1 - ema_decay)
            ema.scale.copy_(net.scale)

        if step >= d_start:                         # discriminator: real B vs the detached prediction
            l_d = ms_gan_d_loss(disc(d_in(B, A)), disc(d_in(pred_img.detach(), A)))
            opt_d.zero_grad(set_to_none=True)
            l_d.backward()
            opt_d.step()
            run["d"] += l_d.item()

        run["loss"] += loss.item()
        run["latent"] += l_lat.item()
        run["vgg"] += l_vgg.item()
        n_steps += 1
        step += 1

        if save_every and step % save_every == 0 and step % eval_every != 0:
            save(latest_path, step)
        if step % eval_every == 0 or step == max_steps:
            tr = {k: run[k] / (n_gan if k in ("gan", "fm") and n_gan else n_steps) for k in run}
            val = validate(ema, obj, zA_val, zB_val, B_val)
            net.train()
            history.append({"step": step, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in val.items()}})
            hist_tmp = hist_path + ".tmp"
            pd.DataFrame(history).to_csv(hist_tmp, index=False)
            os.replace(hist_tmp, hist_path)
            save(latest_path, step)
            print(f"step {step}/{max_steps} | loss {tr['loss']:.4f} (latent {tr['latent']:.4f}, vgg {tr['vgg']:.4f}, "
                  f"gan {tr['gan']:.4f}, fm {tr['fm']:.4f}, d {tr['d']:.4f}) | {time.time() - t0:.0f}s | "
                  f"lr {sched.get_last_lr()[0]:.2e}\n"
                  f"    val latent_mse {val['latent_mse']:.5f} | img {val['img']:.4f}"
                  + "".join(f" | {n} l1 {val[n + '_l1']:.4f} bias {val[n + '_bias']:+.4f}"
                            for n in ("gray", "color") if f"{n}_l1" in val)
                  + f"\n    saved {latest_path} (step {step})")
            run = dict.fromkeys(run, 0.0)
            n_steps, n_gan = 0, 0
            t0 = time.time()

    save(latest_path, step)
    save(f"{out_dir}/final.pt", step)
    _atomic_save({"ema": ema.state_dict(), "arch": ema.arch(), "step": step}, f"{out_dir}/mapping_direct.pt")
    print(f"training finished at step {step} -> {out_dir}/final.pt (+ weights-only mapping_direct.pt)")
    return net, ema
