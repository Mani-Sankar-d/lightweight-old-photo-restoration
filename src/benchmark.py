"""
Large, independent, frozen benchmark for the old-photo mapping.

build_benchmark(): takes clean photos the models never saw (e.g. COCO val2017 or
DIV2K valid), makes degraded/clean 256x256 pairs with the SAME synthetic
degradation as training (make_online_pair: online_add_degradation_v2, crop,
50% gray), each pair seeded by its index, and saves them as PNGs + meta.csv.
Built once (resumable), so every model is scored on exactly the same pairs.

evaluate_benchmark(): streams the pairs from disk and reports, per method,
PSNR, SSIM, LPIPS, FID, KID (unbiased at any n; mean +- std over subsets),
CIEDE2000 colour error, a gray / colour split, and mapping cost. Reference rows:
degraded input, Microsoft mapping, and the VAE B ceiling (clean -> VAE B -> clean,
the best any mapping can do).

FID / KID are computed between each method's outputs and the PAIRED clean targets
(same scenes). FID values are therefore only comparable between methods on the
same benchmark, never with other papers or other n; KID is unbiased, so its
differences can be judged against its std. (A clean-vs-clean comparison of two
disjoint halves is NOT a valid floor here: it measures scene differences too,
which is why it can exceed the ceiling's FID.)
"""
import csv
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity

from online_dataset import list_clean_images, make_online_pair


# --------------------------------------------------------------------------- #
# Building the benchmark
# --------------------------------------------------------------------------- #
def _to_uint8(t):
    """(3,H,W) in [-1,1] -> (H,W,3) uint8."""
    return ((t.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()


def build_benchmark(src_dir, out_dir, n=5000, crops_per_image=1, seed=0, grayscale_prob=0.5):
    """Saves n degraded/clean pairs: out_dir/A/000000.png, out_dir/B/000000.png, meta.csv.
    Resumable: pairs already listed in meta.csv are skipped. Pair i uses source image
    i // crops_per_image (of a seeded shuffle) and its own fixed random seed."""
    os.makedirs(f"{out_dir}/A", exist_ok=True)
    os.makedirs(f"{out_dir}/B", exist_ok=True)
    meta_path = f"{out_dir}/meta.csv"
    files = list_clean_images(src_dir)
    random.Random(seed).shuffle(files)
    total = min(n, len(files) * crops_per_image)
    done = set()
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            done = {int(r["idx"]) for r in csv.DictReader(f)}
    print(f"benchmark: {total} pairs from {len(files)} source images ({len(done)} already built)")
    new = not os.path.exists(meta_path)
    with open(meta_path, "a", newline="") as mf:
        w = csv.writer(mf)
        if new:
            w.writerow(["idx", "source", "gray"])
        t0, made = time.time(), 0
        for i in range(total):
            if i in done:
                continue
            src = files[i // crops_per_image]
            s = seed * 1_000_003 + i
            random.seed(s)
            np.random.seed(s % 2 ** 32)
            try:
                a, b = make_online_pair(src, grayscale_prob)
            except Exception as e:
                print(f"  skipping {src}: {e}")
                continue
            gray = bool((b - b.mean(0, keepdim=True)).abs().max() < 1e-3)
            Image.fromarray(_to_uint8(a)).save(f"{out_dir}/A/{i:06d}.png")
            Image.fromarray(_to_uint8(b)).save(f"{out_dir}/B/{i:06d}.png")
            w.writerow([i, os.path.basename(src), int(gray)])
            made += 1
            if made % 250 == 0:
                mf.flush()
                print(f"  {len(done) + made}/{total} pairs ({made / (time.time() - t0):.1f}/s)")
    print(f"benchmark ready: {out_dir}")


class BenchPairs(torch.utils.data.Dataset):
    """(A, B, gray) from a built benchmark; A, B in [-1,1]."""

    def __init__(self, bench_dir, max_images=None):
        self.dir = bench_dir
        with open(f"{bench_dir}/meta.csv") as f:
            rows = sorted(csv.DictReader(f), key=lambda r: int(r["idx"]))
        self.rows = rows[:max_images] if max_images else rows

    def __len__(self):
        return len(self.rows)

    def _load(self, sub, idx):
        a = np.asarray(Image.open(f"{self.dir}/{sub}/{idx:06d}.png").convert("RGB"), dtype=np.float32)
        return torch.from_numpy(a / 127.5 - 1.0).permute(2, 0, 1)

    def __getitem__(self, k):
        idx = int(self.rows[k]["idx"])
        return self._load("A", idx), self._load("B", idx), int(self.rows[k]["gray"])


# --------------------------------------------------------------------------- #
# Colour science (same code as color_analysis.py, checked against scikit-image)
# --------------------------------------------------------------------------- #
_M_RGB2XYZ = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                           [0.2126729, 0.7151522, 0.0721750],
                           [0.0193339, 0.1191920, 0.9503041]])
_WHITE_D65 = torch.tensor([0.95047, 1.0, 1.08883])


def srgb_to_lab(rgb):
    """(N,3,H,W) sRGB in [0,1] -> (N,3,H,W) CIELAB (D65, 2 deg), like skimage.color.rgb2lab."""
    rgb = rgb.clamp(0, 1)
    lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = torch.einsum("ij,njhw->nihw", _M_RGB2XYZ.to(rgb), lin) / _WHITE_D65.to(rgb)[None, :, None, None]
    eps = (6 / 29) ** 3
    f = torch.where(xyz > eps, xyz.clamp_min(1e-12) ** (1 / 3), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return torch.stack([L, a, b], 1)


def ciede2000(lab1, lab2):
    """Per-pixel CIEDE2000 between (N,3,H,W) Lab tensors -> (N,H,W). (Sharma et al. 2005)"""
    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]
    C1 = torch.sqrt(a1 ** 2 + b1 ** 2)
    C2 = torch.sqrt(a2 ** 2 + b2 ** 2)
    Cb7 = ((C1 + C2) / 2) ** 7
    G = 0.5 * (1 - torch.sqrt(Cb7 / (Cb7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p = torch.sqrt(a1p ** 2 + b1 ** 2)
    C2p = torch.sqrt(a2p ** 2 + b2 ** 2)
    h1p = torch.rad2deg(torch.atan2(b1, a1p)) % 360
    h2p = torch.rad2deg(torch.atan2(b2, a2p)) % 360
    zero = (C1p * C2p) == 0

    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dh = torch.where(dh > 180, dh - 360, torch.where(dh < -180, dh + 360, dh))
    dh = torch.where(zero, torch.zeros_like(dh), dh)
    dHp = 2 * torch.sqrt(C1p * C2p) * torch.sin(torch.deg2rad(dh / 2))

    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    hsum = h1p + h2p
    hbp = torch.where((h1p - h2p).abs() <= 180, hsum / 2,
                      torch.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2))
    hbp = torch.where(zero, hsum, hbp)
    T = (1 - 0.17 * torch.cos(torch.deg2rad(hbp - 30)) + 0.24 * torch.cos(torch.deg2rad(2 * hbp))
         + 0.32 * torch.cos(torch.deg2rad(3 * hbp + 6)) - 0.20 * torch.cos(torch.deg2rad(4 * hbp - 63)))
    dtheta = 30 * torch.exp(-(((hbp - 275) / 25) ** 2))
    Cbp7 = Cbp ** 7
    RC = 2 * torch.sqrt(Cbp7 / (Cbp7 + 25.0 ** 7))
    SL = 1 + 0.015 * (Lbp - 50) ** 2 / torch.sqrt(20 + (Lbp - 50) ** 2)
    SC = 1 + 0.045 * Cbp
    SH = 1 + 0.015 * Cbp * T
    RT = -torch.sin(torch.deg2rad(2 * dtheta)) * RC
    return torch.sqrt(((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2
                       + RT * (dCp / SC) * (dHp / SH)).clamp_min(0))


# --------------------------------------------------------------------------- #
# Distribution metrics
# --------------------------------------------------------------------------- #
def fid_from_feats(f1, f2):
    from pytorch_fid.fid_score import calculate_frechet_distance
    f1, f2 = f1.astype(np.float64), f2.astype(np.float64)
    return float(calculate_frechet_distance(f1.mean(0), np.cov(f1, rowvar=False),
                                            f2.mean(0), np.cov(f2, rowvar=False)))


def kid_from_feats(f1, f2, n_subsets=100, subset_size=1000, seed=0):
    """Kernel Inception Distance (Binkowski et al. 2018): unbiased MMD^2 with the
    cubic polynomial kernel k(x,y) = (x.y/d + 1)^3, averaged over random subsets.
    Returns (mean, std). Unlike FID it is unbiased, so it is comparable across n."""
    rng = np.random.default_rng(seed)
    d = f1.shape[1]
    m = min(subset_size, len(f1), len(f2))
    vals = []
    for _ in range(n_subsets):
        x = f1[rng.choice(len(f1), m, replace=False)].astype(np.float64)
        y = f2[rng.choice(len(f2), m, replace=False)].astype(np.float64)
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        mmd = ((kxx.sum() - np.trace(kxx)) + (kyy.sum() - np.trace(kyy))) / (m * (m - 1)) - 2 * kxy.mean()
        vals.append(mmd)
    return float(np.mean(vals)), float(np.std(vals))


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
class _Metrics:
    def __init__(self, device):
        import lpips
        from pytorch_fid.inception import InceptionV3
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()

    @torch.no_grad()
    def per_image(self, y, t):
        """y, t: (N,3,H,W) in [0,1] -> dict of per-image metrics (CPU)."""
        mse = ((y - t) ** 2).flatten(1).mean(1)
        p = y.permute(0, 2, 3, 1).cpu().numpy()
        q = t.permute(0, 2, 3, 1).cpu().numpy()
        ssim = [structural_similarity(q[k], p[k], channel_axis=2, data_range=1.0, gaussian_weights=True,
                                      sigma=1.5, use_sample_covariance=False) for k in range(len(p))]
        return {"psnr": (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).cpu(),
                "ssim": torch.tensor(ssim, dtype=torch.float32),
                "lpips": self.lpips(y * 2 - 1, t * 2 - 1).flatten().cpu(),
                "de00": ciede2000(srgb_to_lab(t), srgb_to_lab(y)).flatten(1).mean(1).cpu()}

    @torch.no_grad()
    def feats(self, y):
        return self.inception(y)[0].flatten(1).cpu().numpy()


@torch.no_grad()
def evaluate_benchmark(obj, rfs, bench_dir, rf_steps=(1,), batch_size=16, max_images=None,
                       include_input=True, include_ceiling=True, out_csv=None, num_workers=2,
                       include_microsoft=True):
    """obj: OldPhotoModel (quality); rfs: {"name": RFMapping}. Bridge noise is seeded per method."""
    device = obj.device
    metrics = _Metrics(device)
    ds = BenchPairs(bench_dir, max_images)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"evaluating on {len(ds)} pairs from {bench_dir}")

    methods = []
    if include_input:
        methods.append(("degraded input", "input", None, None))
    if include_microsoft:
        methods.append(("Microsoft mapping", "net", obj._core.mapping_net, None))
    for name, rf in rfs.items():
        for s in rf_steps:            # s=None: a direct mapping (no sampler), labelled by its name only
            methods.append((name if s is None else f"{name} ({s}-step)", "net", rf, s))
    if include_ceiling:
        methods.append(("VAE B ceiling", "ceiling", None, None))

    to01 = lambda x: ((x.float() + 1) / 2).clamp(0, 1)
    real_feats, gray_all, rows = None, None, []
    for name, kind, net, steps in methods:
        per = {k: [] for k in ("psnr", "ssim", "lpips", "de00")}
        feats, real, grays, t_map = [], [], [], 0.0
        torch.manual_seed(0)
        for A, B, g in loader:
            A, B = A.to(device), B.to(device)
            if kind == "input":
                y = to01(A)
            elif kind == "ceiling":
                y = to01(obj.decode_B(obj.encode_B(B)))
            else:
                if steps is not None:
                    net.steps = steps
                z_A = obj.encode_A(A)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                z_B = net(z_A)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_map += time.time() - t0
                y = to01(obj.decode_B(z_B))
            t = to01(B)
            for k, v in metrics.per_image(y, t).items():
                per[k].append(v)
            feats.append(metrics.feats(y))
            if real_feats is None:
                real.append(metrics.feats(t))
                grays.append(g.bool())
        if real_feats is None:
            real_feats, gray_all = np.concatenate(real), torch.cat(grays)
        per = {k: torch.cat(v) for k, v in per.items()}
        f = np.concatenate(feats)
        kid_m, kid_s = kid_from_feats(f, real_feats)
        row = {"method": name, "PSNR↑": per["psnr"].mean().item(), "SSIM↑": per["ssim"].mean().item(),
               "LPIPS↓": per["lpips"].mean().item(), "FID↓": fid_from_feats(f, real_feats),
               "KID×1000↓": 1000 * kid_m, "KID std×1000": 1000 * kid_s, "ΔE00↓": per["de00"].mean().item()}
        for tag, m in (("gray", gray_all), ("colour", ~gray_all)):
            if m.any():
                for k, col in (("psnr", "PSNR"), ("ssim", "SSIM"), ("lpips", "LPIPS"), ("de00", "ΔE00")):
                    row[f"{col} {tag}"] = per[k][m].mean().item()
        if kind == "net":
            row["mapping params (M)"] = sum(p.numel() for p in net.parameters()) / 1e6
            row["mapping ms/img"] = 1000 * t_map / len(ds)
        rows.append(row)
        print(f"{name:34s} PSNR {row['PSNR↑']:.2f}  SSIM {row['SSIM↑']:.4f}  LPIPS {row['LPIPS↓']:.4f}  "
              f"FID {row['FID↓']:.2f}  KID×1000 {row['KID×1000↓']:.2f}±{row['KID std×1000']:.2f}  ΔE00 {row['ΔE00↓']:.2f}")

    df = pd.DataFrame(rows).set_index("method")
    if out_csv:
        df.to_csv(out_csv)
        print(f"saved {out_csv}")
    return df.round(4)
