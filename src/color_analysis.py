"""
Colour-drift analysis: Microsoft mapping vs RF vs ground truth.

All comparisons are done in CIELAB (D65), where directions are readable:
    L*  lightness         0 (black) .. 100 (white)
    a*  green (-) .. red (+)
    b*  blue  (-) .. yellow (+)
    C*  chroma  = sqrt(a*^2 + b*^2)   (saturation)
    h   hue angle = atan2(b*, a*) in degrees (0 = pinkish red, 90 = yellow,
        180 = cyan/green-blue, 270 = blue)

What is measured (output vs the clean ground truth, same pixels):
  1. Overall error: CIEDE2000 dE (median / mean / p95), lightness shift dL*,
     and the colour change split into its two parts:
       - cast = shift of NEUTRAL pixels (gray/white in the ground truth), the
         white-balance definition, which desaturation cannot fake;
       - fit  output_ab = s * true_ab + c  over all pixels: s = saturation
         scale, c = cast. (A plain mean da*/db* mixes both: desaturating mostly
         magenta images looks like a green/yellow "cast".)
     plus chroma ratio (saturation gain/loss),
     colourfulness (Hasler-Suesstrunk), local lightness-contrast ratio
     (fine detail such as stripes collapsing).
  2. Per hue (colour images): for ground-truth pixels of each hue, the mean
     signed hue shift, chroma ratio, lightness shift, dE, and where they end up.
  3. Hue transition matrix: ground-truth colour class -> output colour class
     ("neutral" + 12 hues). Off-diagonal mass = colours turning into others;
     the "neutral" row/column = tinting gray / desaturating colour.
  4. Per tone (shadows / midtones / highlights): da*, db*, dL*, dC*: casts that
     differ by brightness (e.g. blue shadows, yellow highlights).
  5. Gray images: false colour added to gray content (mean a*, b*, C*, and
     share of visibly tinted pixels).

The test set is built exactly like evaluate.py's (held-out training split or a
separate folder, fixed seed), so numbers line up with the metric tables.
"""
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from online_dataset import list_clean_images, make_online_pair

HUE_NAMES = ["pink", "red", "orange", "yellow", "lime", "green",
             "teal", "cyan", "sky", "blue", "violet", "magenta"]   # 30-degree bins of Lab hue, approximate
CLASSES = ["neutral"] + HUE_NAMES
TONES = [("shadows", 0, 33), ("midtones", 33, 66), ("highlights", 66, 101)]


# --------------------------------------------------------------------------- #
# Test set (same construction as evaluate.py)
# --------------------------------------------------------------------------- #
def build_test_set(clean_dir=None, test_clean_dir=None, n=500, grayscale_prob=0.5, seed=0, val_fraction=0.05):
    """A, B (N,3,256,256) in [-1,1] and gray (N,) bool. seed / val_fraction must
    match training's make_online_dataloaders call for a held-out split."""
    if test_clean_dir:
        files = list_clean_images(test_clean_dir)
    else:
        shuffled = list_clean_images(clean_dir)
        random.Random(seed).shuffle(shuffled)
        files = shuffled[:max(1, int(len(shuffled) * val_fraction))]
    state = random.getstate(), np.random.get_state()
    random.seed(seed + 123)
    np.random.seed(seed + 123)
    A, B, i = [], [], 0
    while len(A) < n and i < 20 * n:
        f = files[i % len(files)]
        i += 1
        try:
            a, b = make_online_pair(f, grayscale_prob)
        except Exception as e:
            print(f"[test set] skipping unreadable {f}: {e}")
            continue
        A.append(a)
        B.append(b)
    random.setstate(state[0])
    np.random.set_state(state[1])
    A, B = torch.stack(A), torch.stack(B)
    gray = (B - B.mean(1, keepdim=True)).abs().flatten(1).amax(1) < 1e-3
    print(f"test set: {len(A)} pairs from {len(files)} source images "
          f"({int(gray.sum())} gray, {int((~gray).sum())} colour)")
    return A, B, gray


# --------------------------------------------------------------------------- #
# Colour science (torch, on GPU)
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


def colourfulness(rgb):
    """(N,3,H,W) in [0,1] -> (N,) Hasler & Suesstrunk colourfulness (0-255 scale)."""
    x = rgb * 255
    rg = x[:, 0] - x[:, 1]
    yb = 0.5 * (x[:, 0] + x[:, 1]) - x[:, 2]
    std = torch.sqrt(rg.flatten(1).var(1) + yb.flatten(1).var(1))
    mean = torch.sqrt(rg.flatten(1).mean(1) ** 2 + yb.flatten(1).mean(1) ** 2)
    return std + 0.3 * mean


def local_contrast(L, k=8):
    """(N,H,W) lightness -> (N,) mean std of L* inside k x k patches (fine-detail contrast)."""
    L = L[:, None]
    m = F.avg_pool2d(L, k)
    v = (F.avg_pool2d(L * L, k) - m * m).clamp_min(0)
    return v.sqrt().flatten(1).mean(1)


def color_class(lab, chroma_thresh):
    """(N,3,H,W) Lab -> (N,H,W) long: 0 = neutral (C* < thresh), 1..12 = 30-degree hue bins."""
    C = torch.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2)
    h = torch.rad2deg(torch.atan2(lab[:, 2], lab[:, 1])) % 360
    hb = (h // 30).long().clamp(0, 11) + 1
    return torch.where(C < chroma_thresh, torch.zeros_like(hb), hb)


# --------------------------------------------------------------------------- #
# Statistics accumulation
# --------------------------------------------------------------------------- #
_DE_BINS = torch.linspace(0, 60, 601)   # dE histogram for median / p95


class _Acc:
    """Accumulates colour statistics for one method across batches."""

    def __init__(self, device):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)
        self.de_hist = {"colour": z(600), "gray": z(600)}
        self.img = {"colour": {k: [] for k in ("dE", "dL", "da", "db", "chroma_ratio", "cf_out", "cf_gt", "contrast_ratio")},
                    "gray": {k: [] for k in ("dE", "dL", "a", "b", "C", "tinted", "contrast_ratio")}}
        self.hue = {k: z(12) for k in ("n", "n_both", "dh", "dh_abs", "C_gt", "C_out", "dL", "dE",
                                       "a_gt", "b_gt", "a_out", "b_out", "stay")}
        self.trans = z(13, 13)
        self.gray_trans = z(13)
        self.tone = {k: z(3) for k in ("n", "dL", "dC", "n_neu", "da_neu", "db_neu")}
        self.fit = z(7)       # N, sum a_g, sum b_g, sum a_o, sum b_o, sum |g|^2, sum o.g  (colour images)
        self.neutral = z(3)   # n, sum da, sum db over ground-truth-neutral pixels (colour images)
        self.hist = {k: z(100) for k in ("L_gt", "L_out", "a_gt", "a_out", "b_gt", "b_out")}

    def add(self, lab_o, lab_g, rgb_o, rgb_g, is_gray, chroma_thresh, tint_thresh):
        dE = ciede2000(lab_g, lab_o)
        d = lab_o - lab_g
        C_o = torch.sqrt(lab_o[:, 1] ** 2 + lab_o[:, 2] ** 2)
        C_g = torch.sqrt(lab_g[:, 1] ** 2 + lab_g[:, 2] ** 2)
        contrast = local_contrast(lab_o[:, 0]) / local_contrast(lab_g[:, 0]).clamp_min(1e-3)

        for tag, m in (("colour", ~is_gray), ("gray", is_gray)):
            if not m.any():
                continue
            self.de_hist[tag] += torch.histc(dE[m].double(), 600, 0, 60)
            rec = self.img[tag]
            rec["dE"].append(dE[m].flatten(1).mean(1).cpu())
            rec["dL"].append(d[m, 0].flatten(1).mean(1).cpu())
            rec["contrast_ratio"].append(contrast[m].cpu())
            if tag == "colour":
                rec["da"].append(d[m, 1].flatten(1).mean(1).cpu())
                rec["db"].append(d[m, 2].flatten(1).mean(1).cpu())
                rec["chroma_ratio"].append((C_o[m].flatten(1).mean(1) / C_g[m].flatten(1).mean(1).clamp_min(1e-3)).cpu())
                rec["cf_out"].append(colourfulness(rgb_o[m]).cpu())
                rec["cf_gt"].append(colourfulness(rgb_g[m]).cpu())
            else:
                rec["a"].append(lab_o[m, 1].flatten(1).mean(1).cpu())
                rec["b"].append(lab_o[m, 2].flatten(1).mean(1).cpu())
                rec["C"].append(C_o[m].flatten(1).mean(1).cpu())
                rec["tinted"].append((C_o[m] > tint_thresh).float().flatten(1).mean(1).cpu())
                cls_o = color_class(lab_o[m], chroma_thresh)
                self.gray_trans += torch.bincount(cls_o.flatten(), minlength=13).double()

        if (~is_gray).any():
            m = ~is_gray
            lo, lg = lab_o[m], lab_g[m]
            cg, co = color_class(lg, chroma_thresh), color_class(lo, chroma_thresh)
            self.trans += torch.bincount((cg * 13 + co).flatten(), minlength=169).double().view(13, 13)

            Cg, Co = C_g[m].flatten(), C_o[m].flatten()
            hg = torch.rad2deg(torch.atan2(lg[:, 2], lg[:, 1])).flatten() % 360
            ho = torch.rad2deg(torch.atan2(lo[:, 2], lo[:, 1])).flatten() % 360
            dh = (ho - hg + 180) % 360 - 180          # signed, + = counter-clockwise (red -> yellow -> green -> blue)
            bin_g = (hg // 30).long().clamp(0, 11)
            chrom = Cg >= chroma_thresh                # hue only meaningful for chromatic ground truth
            both = chrom & (Co >= chroma_thresh)
            flat = lambda x: x.flatten()
            dEm, dLm = flat(dE[m]), flat(d[m, 0])
            vals = {"n": chrom.double(), "n_both": both.double(),
                    "dh": torch.where(both, dh, torch.zeros_like(dh)).double(),
                    "dh_abs": torch.where(both, dh.abs(), torch.zeros_like(dh)).double(),
                    "C_gt": torch.where(chrom, Cg, torch.zeros_like(Cg)).double(),
                    "C_out": torch.where(chrom, Co, torch.zeros_like(Co)).double(),
                    "dL": torch.where(chrom, dLm, torch.zeros_like(dLm)).double(),
                    "dE": torch.where(chrom, dEm, torch.zeros_like(dEm)).double(),
                    "a_gt": torch.where(chrom, flat(lg[:, 1]), torch.zeros_like(Cg)).double(),
                    "b_gt": torch.where(chrom, flat(lg[:, 2]), torch.zeros_like(Cg)).double(),
                    "a_out": torch.where(chrom, flat(lo[:, 1]), torch.zeros_like(Cg)).double(),
                    "b_out": torch.where(chrom, flat(lo[:, 2]), torch.zeros_like(Cg)).double(),
                    "stay": (chrom & (flat(co) == bin_g + 1)).double()}
            for k, v in vals.items():
                self.hue[k] += torch.bincount(bin_g, weights=v, minlength=12)

            ag, bg, ao, bo = (x.double().flatten() for x in (lg[:, 1], lg[:, 2], lo[:, 1], lo[:, 2]))
            self.fit += torch.stack([torch.tensor(float(ag.numel()), device=ag.device, dtype=torch.float64),
                                     ag.sum(), bg.sum(), ao.sum(), bo.sum(),
                                     (ag * ag + bg * bg).sum(), (ao * ag + bo * bg).sum()])
            neu = C_g[m] < chroma_thresh
            if neu.any():
                self.neutral += torch.stack([neu.sum().double(), (lo[:, 1] - lg[:, 1])[neu].double().sum(),
                                             (lo[:, 2] - lg[:, 2])[neu].double().sum()])

            Lg = lg[:, 0]
            for ti, (_, lo_, hi_) in enumerate(TONES):
                tm = (Lg >= lo_) & (Lg < hi_)
                if tm.any():
                    self.tone["n"][ti] += tm.sum()
                    self.tone["dL"][ti] += (lo[:, 0] - lg[:, 0])[tm].double().sum()
                    self.tone["dC"][ti] += (C_o[m] - C_g[m])[tm].double().sum()
                    tn = tm & neu
                    if tn.any():
                        self.tone["n_neu"][ti] += tn.sum()
                        self.tone["da_neu"][ti] += (lo[:, 1] - lg[:, 1])[tn].double().sum()
                        self.tone["db_neu"][ti] += (lo[:, 2] - lg[:, 2])[tn].double().sum()

            for ch, idx, rng in (("L", 0, (0, 100)), ("a", 1, (-80, 80)), ("b", 2, (-80, 80))):
                self.hist[f"{ch}_gt"] += torch.histc(lg[:, idx].double(), 100, *rng)
                self.hist[f"{ch}_out"] += torch.histc(lo[:, idx].double(), 100, *rng)

    def finish(self):
        out = {"de_hist": {k: v.cpu() for k, v in self.de_hist.items()},
               "img": {t: {k: torch.cat(v) if v else torch.empty(0) for k, v in d.items()} for t, d in self.img.items()},
               "hue": {k: v.cpu() for k, v in self.hue.items()},
               "trans": self.trans.cpu(), "gray_trans": self.gray_trans.cpu(),
               "tone": {k: v.cpu() for k, v in self.tone.items()},
               "fit": self.fit.cpu(), "neutral": self.neutral.cpu(),
               "hist": {k: v.cpu() for k, v in self.hist.items()}}
        return out


def _to01(y):
    return ((y.float() + 1) / 2).clamp(0, 1)


@torch.no_grad()
def analyze(obj, rfs, A, B, gray, rf_steps=(8,), batch_size=16, include_input=True,
            chroma_thresh=8.0, tint_thresh=5.0):
    """obj: OldPhotoModel (quality); rfs: {"name": RFMapping}. Returns {method: stats}.
    chroma_thresh: C* below which a pixel counts as neutral (gray).
    tint_thresh:   C* above which a pixel of a gray image counts as visibly tinted."""
    device = obj.device
    methods = {}
    if include_input:
        methods["degraded input"] = (None, None)
    methods["Microsoft mapping"] = (obj._core.mapping_net, None)
    for name, rf in rfs.items():
        for s in rf_steps:            # s=None: a direct mapping (no sampler), labelled by its name only
            methods[name if s is None else f"{name} ({s}-step)"] = (rf, s)

    stats = {}
    n = len(A)
    for name, (net, steps) in methods.items():
        acc = _Acc(device)
        for i in range(0, n, batch_size):
            a = A[i:i + batch_size].to(device)
            rgb_g = _to01(B[i:i + batch_size].to(device))
            if net is None:
                rgb_o = _to01(a)
            else:
                if steps is not None:
                    net.steps = steps
                rgb_o = _to01(obj.decode_B(net(obj.encode_A(a))))
            acc.add(srgb_to_lab(rgb_o), srgb_to_lab(rgb_g), rgb_o, rgb_g,
                    gray[i:i + batch_size].to(device), chroma_thresh, tint_thresh)
        stats[name] = acc.finish()
        print(f"analyzed {name}")
    stats["_config"] = {"chroma_thresh": chroma_thresh, "tint_thresh": tint_thresh}
    return stats


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _methods(stats):
    return [k for k in stats if not k.startswith("_")]


def _hist_quantile(hist, q):
    c = hist.cumsum(0)
    if c[-1] == 0:
        return float("nan")
    idx = int(torch.searchsorted(c, q * c[-1]))
    return float(_DE_BINS[min(idx + 1, 600)])


def _cast_name(da, db, tol=0.5):
    parts = []
    if db < -tol:
        parts.append("blue")
    elif db > tol:
        parts.append("yellow")
    if da > tol:
        parts.append("red/magenta")
    elif da < -tol:
        parts.append("green")
    return "+".join(parts) if parts else "none"


def _fit_sat_cast(fit):
    """Least squares output_ab = s * true_ab + c  ->  (s, c_a, c_b)."""
    N, sag, sbg, sao, sbo, sgg, sog = fit.tolist()
    if N < 1:
        return float("nan"), float("nan"), float("nan")
    s = (sog - (sao * sag + sbo * sbg) / N) / max(sgg - (sag ** 2 + sbg ** 2) / N, 1e-9)
    return s, (sao - s * sag) / N, (sbo - s * sbg) / N


MIN_NEUTRAL_PX = 2000   # fewer ground-truth-neutral pixels than this -> use the fitted cast instead


def _cast(s):
    """(cast_a, cast_b, source) for one method's colour images."""
    n, sa, sb = s["neutral"].tolist()
    if n >= MIN_NEUTRAL_PX:
        return sa / n, sb / n, "neutral px"
    _, ca, cb = _fit_sat_cast(s["fit"])
    return ca, cb, "fit"


def summary_tables(stats):
    """Returns (colour_df, gray_df): one row per method."""
    col_rows, gray_rows = [], []
    for name in _methods(stats):
        s = stats[name]
        c, g = s["img"]["colour"], s["img"]["gray"]
        if len(c["dE"]):
            cast_a, cast_b, src = _cast(s)
            sat, fit_a, fit_b = _fit_sat_cast(s["fit"])
            col_rows.append({
                "method": name,
                "dE00 median": _hist_quantile(s["de_hist"]["colour"], 0.5),
                "dE00 mean": c["dE"].mean().item(),
                "dE00 p95": _hist_quantile(s["de_hist"]["colour"], 0.95),
                "dL* (lightness)": c["dL"].mean().item(),
                "cast a* (+red/-green)": cast_a,
                "cast b* (+yellow/-blue)": cast_b,
                "cast": _cast_name(cast_a, cast_b),
                "cast measured on": src,
                "saturation scale s (fit)": sat,
                "cast a* (fit)": fit_a,
                "cast b* (fit)": fit_b,
                "mean da* (cast+sat mixed)": c["da"].mean().item(),
                "mean db* (cast+sat mixed)": c["db"].mean().item(),
                "chroma ratio (out/gt)": c["chroma_ratio"].mean().item(),
                "colourfulness out": c["cf_out"].mean().item(),
                "colourfulness gt": c["cf_gt"].mean().item(),
                "detail contrast ratio": c["contrast_ratio"].mean().item(),
                "mean |hue shift| deg": (s["hue"]["dh_abs"].sum() / s["hue"]["n_both"].sum().clamp_min(1)).item(),
            })
        if len(g["dE"]):
            a, b = g["a"].mean().item(), g["b"].mean().item()
            gray_rows.append({
                "method": name,
                "dE00 median": _hist_quantile(s["de_hist"]["gray"], 0.5),
                "dE00 mean": g["dE"].mean().item(),
                "dL* (lightness)": g["dL"].mean().item(),
                "false a* (+red/-green)": a,
                "false b* (+yellow/-blue)": b,
                "tint": _cast_name(a, b),
                "false chroma C*": g["C"].mean().item(),
                f"% pixels tinted (C*>{stats['_config']['tint_thresh']:g})": 100 * g["tinted"].mean().item(),
                "detail contrast ratio": g["contrast_ratio"].mean().item(),
            })
    to_df = lambda r: pd.DataFrame(r).set_index("method").round(3) if r else pd.DataFrame()
    return to_df(col_rows), to_df(gray_rows)


def hue_table(stats, method):
    """Per ground-truth hue (colour images): where that colour goes."""
    h = stats[method]["hue"]
    trans = stats[method]["trans"]
    rows = []
    total = h["n"].sum().clamp_min(1)
    for i, hn in enumerate(HUE_NAMES):
        n = h["n"][i]
        if n < 1:
            continue
        nb = h["n_both"][i].clamp_min(1)
        row_t = trans[i + 1]
        dest = row_t.clone()
        dest[i + 1] = 0
        top = int(dest.argmax())
        rows.append({
            "gt hue": f"{hn} ({30 * i}-{30 * i + 30} deg)",
            "% of colour px": (100 * n / total).item(),
            "hue shift deg (+ccw)": (h["dh"][i] / nb).item(),
            "chroma ratio": (h["C_out"][i] / h["C_gt"][i].clamp_min(1e-6)).item(),
            "dL*": (h["dL"][i] / n).item(),
            "dE00": (h["dE"][i] / n).item(),
            "% stay same hue": (100 * h["stay"][i] / n).item(),
            "% -> neutral": (100 * row_t[0] / row_t.sum().clamp_min(1)).item(),
            "most common drift": f"{CLASSES[top]} ({100 * dest[top] / row_t.sum().clamp_min(1):.0f}%)",
        })
    return pd.DataFrame(rows).set_index("gt hue").round(2)


def tone_table(stats):
    """Per lightness band (colour images): lightness and saturation shift over all
    pixels, and the cast measured on ground-truth-neutral pixels of that band."""
    rows = []
    for name in _methods(stats):
        t = stats[name]["tone"]
        for i, (tn, lo, hi) in enumerate(TONES):
            n = t["n"][i].clamp_min(1)
            nn_ = t["n_neu"][i]
            ok = bool(nn_ >= MIN_NEUTRAL_PX / 3)
            da = (t["da_neu"][i] / nn_).item() if ok else float("nan")
            db = (t["db_neu"][i] / nn_).item() if ok else float("nan")
            rows.append({"method": name, "tone": f"{tn} (L* {lo}-{min(hi, 100)})",
                         "dL*": (t["dL"][i] / n).item(), "dC* (saturation)": (t["dC"][i] / n).item(),
                         "cast da*": da, "cast db*": db,
                         "cast": _cast_name(da, db) if ok else "n/a (few neutral px)",
                         "neutral px": int(nn_)})
    return pd.DataFrame(rows).set_index(["method", "tone"]).round(2)


def transition_table(stats, method):
    """Row-normalized % : ground-truth colour class (rows) -> output colour class (cols)."""
    t = stats[method]["trans"]
    pct = 100 * t / t.sum(1, keepdim=True).clamp_min(1)
    df = pd.DataFrame(pct.numpy(), index=[f"gt {c}" for c in CLASSES], columns=[f"-> {c}" for c in CLASSES])
    return df[t.sum(1).numpy() > 0].round(1)


def describe(stats, min_share=3.0, min_shift=8.0, min_chroma_change=0.15):
    """Plain-language findings per method."""
    col, gray = summary_tables(stats)
    lines = []
    for name in _methods(stats):
        lines.append(f"\n=== {name} ===")
        if name in col.index:
            r = col.loc[name]
            lines.append(f"Colour images: median dE00 {r['dE00 median']:.1f} (p95 {r['dE00 p95']:.1f}); "
                         f"global cast: {r['cast']} (a* {r['cast a* (+red/-green)']:+.2f}, "
                         f"b* {r['cast b* (+yellow/-blue)']:+.2f}, on {r['cast measured on']}); "
                         f"lightness {r['dL* (lightness)']:+.2f} L*; saturation x{r['chroma ratio (out/gt)']:.2f} "
                         f"(fit s={r['saturation scale s (fit)']:.2f}); "
                         f"fine-detail contrast x{r['detail contrast ratio']:.2f}.")
            ht = hue_table(stats, name)
            for hue, h in ht.iterrows():
                if h["% of colour px"] < min_share:
                    continue
                notes = []
                shift = h["hue shift deg (+ccw)"]
                if abs(shift) >= min_shift:
                    i = HUE_NAMES.index(hue.split()[0])
                    towards = HUE_NAMES[(i + 1) % 12] if shift > 0 else HUE_NAMES[i - 1]  # + = ccw on the hue wheel
                    notes.append(f"hue {shift:+.0f} deg (towards {towards})")
                if abs(h["chroma ratio"] - 1) >= min_chroma_change:
                    notes.append(("more" if h["chroma ratio"] > 1 else "less") + f" saturated (x{h['chroma ratio']:.2f})")
                if h["% -> neutral"] >= 20:
                    notes.append(f"{h['% -> neutral']:.0f}% washed out to neutral")
                if notes:
                    lines.append(f"  - {hue}: " + ", ".join(notes) + f"; most common drift: {h['most common drift']}")
        if name in gray.index:
            r = gray.loc[name]
            tint_col = [c for c in gray.columns if c.startswith("% pixels tinted")][0]
            lines.append(f"Gray images: false tint {r['tint']} (a* {r['false a* (+red/-green)']:+.2f}, "
                         f"b* {r['false b* (+yellow/-blue)']:+.2f}, C* {r['false chroma C*']:.2f}); "
                         f"{r[tint_col]:.1f}% of pixels visibly tinted; lightness {r['dL* (lightness)']:+.2f} L*.")
    tone = tone_table(stats)
    lines.append("\nTone-dependent casts (colour images):")
    for (name, tn), r in tone.iterrows():
        if r["cast"] != "none" and not r["cast"].startswith("n/a"):
            lines.append(f"  {name} | {tn}: {r['cast']} (da* {r['cast da*']:+.2f}, db* {r['cast db*']:+.2f}), "
                         f"dL* {r['dL*']:+.2f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _lab_to_rgb_tuple(L, a, b):
    """Single Lab colour -> matplotlib RGB tuple (for drawing hue swatches)."""
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    inv = lambda f: f ** 3 if f ** 3 > (6 / 29) ** 3 else 3 * (6 / 29) ** 2 * (f - 4 / 29)
    xyz = np.array([inv(fx), inv(fy), inv(fz)]) * _WHITE_D65.numpy()
    lin = np.linalg.solve(_M_RGB2XYZ.numpy(), xyz)
    rgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.clip(lin, 0, None) ** (1 / 2.4) - 0.055)
    return tuple(np.clip(rgb, 0, 1))


def _savefig(plt, path):
    """Save the current figure, creating its folder first (e.g. a new Drive folder)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=150)


def plot_ab_drift(stats, save=None):
    """a*b* plane: arrow from each ground-truth hue's mean colour to where the method puts it."""
    import matplotlib.pyplot as plt
    names = [n for n in _methods(stats) if n != "degraded input"]
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5), squeeze=False)
    lim = 30.0   # common axis range covering every dot and arrow tip
    for name in names:
        h = stats[name]["hue"]
        n = h["n"].clamp_min(1)
        for k in ("a_gt", "b_gt", "a_out", "b_out"):
            lim = max(lim, float((h[k] / n)[h["n"] >= 1].abs().max()) * 1.15 if (h["n"] >= 1).any() else lim)
    for ax, name in zip(axes[0], names):
        h = stats[name]["hue"]
        for i, hn in enumerate(HUE_NAMES):
            n = h["n"][i]
            if n < 1:
                continue
            ag, bg = (h["a_gt"][i] / n).item(), (h["b_gt"][i] / n).item()
            ao, bo = (h["a_out"][i] / n).item(), (h["b_out"][i] / n).item()
            colour = _lab_to_rgb_tuple(60, ag, bg)
            ax.scatter([ag], [bg], s=60 + 400 * (n / h["n"].sum()).item(), color=colour, edgecolors="k", zorder=3)
            ax.annotate("", xy=(ao, bo), xytext=(ag, bg),
                        arrowprops=dict(arrowstyle="->", color="k", lw=1.5), zorder=4)
            ax.text(ag, bg, f" {hn}", fontsize=8, va="bottom")
        g = stats[name]["img"]["gray"]
        if len(g["a"]):
            ax.scatter([g["a"].mean()], [g["b"].mean()], marker="x", s=120, color="k", zorder=5,
                       label="gray images: mean false colour")
            ax.legend(loc="lower left", fontsize=8)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.axvline(0, color="0.7", lw=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("a*  (green -  /  + red)")
        ax.set_ylabel("b*  (blue -  /  + yellow)")
        ax.set_title(f"{name}\ndot = true mean colour per hue, arrow = output")
        ax.set_aspect("equal")
    plt.tight_layout()
    if save:
        _savefig(plt, save)
    plt.show()


def plot_transitions(stats, save=None):
    """Heatmap of ground-truth colour class -> output colour class (% of each row)."""
    import matplotlib.pyplot as plt
    names = [n for n in _methods(stats) if n != "degraded input"]
    fig, axes = plt.subplots(1, len(names), figsize=(6.5 * len(names), 6), squeeze=False)
    for ax, name in zip(axes[0], names):
        t = stats[name]["trans"]
        keep = (t.sum(1) > 0).numpy()
        pct = (100 * t / t.sum(1, keepdim=True).clamp_min(1)).numpy()[keep]
        im = ax.imshow(pct, cmap="viridis", vmin=0, vmax=100)
        ax.set_xticks(range(13))
        ax.set_xticklabels(CLASSES, rotation=60, fontsize=8)
        ax.set_yticks(range(int(keep.sum())))
        ax.set_yticklabels([c for c, k in zip(CLASSES, keep) if k], fontsize=8)
        for r in range(pct.shape[0]):
            for c in range(13):
                if pct[r, c] >= 5:
                    ax.text(c, r, f"{pct[r, c]:.0f}", ha="center", va="center", fontsize=7,
                            color="w" if pct[r, c] < 60 else "k")
        ax.set_xlabel("output colour")
        ax.set_ylabel("ground-truth colour")
        ax.set_title(f"{name}\n% of each true colour ending up as ...")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    if save:
        _savefig(plt, save)
    plt.show()


def plot_tone_casts(stats, save=None):
    """da*, db*, dL* per lightness band, per method."""
    import matplotlib.pyplot as plt
    tone = tone_table(stats)
    names = _methods(stats)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    width = 0.8 / len(names)
    for ax, col, lab in zip(axes, ("cast da*", "cast db*", "dL*"),
                            ("cast da* on neutral px  (+ red / - green)", "cast db* on neutral px  (+ yellow / - blue)",
                             "dL*  (+ brighter / - darker)")):
        for j, name in enumerate(names):
            vals = [tone.loc[(name, f"{tn} (L* {lo}-{min(hi, 100)})"), col] for tn, lo, hi in TONES]
            ax.bar(np.arange(3) + j * width, vals, width, label=name)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(np.arange(3) + width * (len(names) - 1) / 2)
        ax.set_xticklabels([t[0] for t in TONES])
        ax.set_ylabel(lab)
    axes[0].legend(fontsize=7)
    fig.suptitle("Colour shift by tone (colour images, output - ground truth)")
    plt.tight_layout()
    if save:
        _savefig(plt, save)
    plt.show()


def plot_histograms(stats, save=None):
    """L*, a*, b* distributions of outputs vs ground truth (colour images)."""
    import matplotlib.pyplot as plt
    names = _methods(stats)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, ch, rng, lab in zip(axes, ("L", "a", "b"), ((0, 100), (-80, 80), (-80, 80)),
                                ("L* (lightness)", "a* (- green / + red)", "b* (- blue / + yellow)")):
        x = np.linspace(rng[0], rng[1], 100)
        gt = stats[names[0]]["hist"][f"{ch}_gt"].numpy()
        ax.plot(x, gt / gt.sum(), "k", lw=2.5, label="ground truth")
        for name in names:
            h = stats[name]["hist"][f"{ch}_out"].numpy()
            ax.plot(x, h / h.sum(), lw=1.2, label=name)
        ax.set_xlabel(lab)
        if ch != "L":
            ax.set_yscale("log")
    axes[0].legend(fontsize=7)
    fig.suptitle("Pixel distributions (colour images)")
    plt.tight_layout()
    if save:
        _savefig(plt, save)
    plt.show()


def save_tables(stats, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    col, gray = summary_tables(stats)
    col.to_csv(os.path.join(out_dir, "colour_summary.csv"))
    gray.to_csv(os.path.join(out_dir, "gray_summary.csv"))
    tone_table(stats).to_csv(os.path.join(out_dir, "tone_casts.csv"))
    for name in _methods(stats):
        safe = "".join(ch if ch.isalnum() else "_" for ch in name)
        hue_table(stats, name).to_csv(os.path.join(out_dir, f"hue_{safe}.csv"))
        transition_table(stats, name).to_csv(os.path.join(out_dir, f"transitions_{safe}.csv"))
    with open(os.path.join(out_dir, "findings.txt"), "w") as f:
        f.write(describe(stats))
    print(f"saved tables + findings.txt to {out_dir}")
