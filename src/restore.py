"""
Restore photos with the lightweight direct mapping (and, optionally, BOPBTL's original mapping for comparison).

    python src/restore.py --input photos/ --output restored/
    python src/restore.py --input photo.jpg --output restored/ --compare --weights mapping_direct.pt

Without --weights, the released model is downloaded once to ~/.cache/lightweight_old_photos/.

Uses BOPBTL's frozen global-restoration VAEs (quality mode). The Microsoft repository and its pretrained
checkpoints are found or downloaded by OldPhotoModel (pass --repo to point at an existing clone).

Outputs are the raw model outputs. --guard applies an optional post-processing step to EVERY method in the same
way; never use it when judging which model is better.
"""
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image

from direct_model import load_direct
from old_photo_model import OldPhotoModel

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
WEIGHTS_URL = ("https://github.com/Mani-Sankar-d/lightweight-old-photo-restoration/"
               "releases/download/v1.0/mapping_direct.pt")


def default_weights():
    """The released model, downloaded once to ~/.cache/lightweight_old_photos/."""
    path = os.path.join(os.path.expanduser("~"), ".cache", "lightweight_old_photos", "mapping_direct.pt")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading {WEIGHTS_URL}")
        torch.hub.download_url_to_file(WEIGHTS_URL, path)
    return path


def mean_chroma(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    return float(np.abs(a - a.mean(axis=2, keepdims=True)).mean())


def apply_guard(inp, out, guard):
    """Optional post-processing. "gray": outputs of gray inputs become gray. "near_gray": for low-saturation inputs,
    keep the output's lightness and take the input's smoothed colour."""
    if guard == "gray" and mean_chroma(inp) <= 2.0:
        return out.convert("L").convert("RGB")
    if guard == "near_gray" and mean_chroma(inp) <= 12.0:
        import cv2
        lab_in = cv2.cvtColor(np.asarray(inp.convert("RGB").resize(out.size)), cv2.COLOR_RGB2LAB)
        lab_out = cv2.cvtColor(np.asarray(out.convert("RGB")), cv2.COLOR_RGB2LAB)
        ab = cv2.GaussianBlur(lab_in[..., 1:], (31, 31), 0)
        return Image.fromarray(cv2.cvtColor(np.dstack([lab_out[..., :1], ab]), cv2.COLOR_LAB2RGB))
    return out


def restore(obj, img, net):
    """Raw output of one mapping; identical VAEs and pre/post-processing for every mapping."""
    mapping, uses_mask = obj.mapping, obj.uses_mask
    obj.mapping, obj.uses_mask = net, False
    try:
        with torch.no_grad():
            return obj.restore(img, keep_size=True, normalize=False)
    finally:
        obj.mapping, obj.uses_mask = mapping, uses_mask


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=None,
                   help="mapping_direct.pt or a train_direct final.pt (default: download the released model)")
    p.add_argument("--input", required=True, help="an image file or a folder of images")
    p.add_argument("--output", required=True, help="output folder")
    p.add_argument("--repo", default=None, help="path to a clone of microsoft/Bringing-Old-Photos-Back-to-Life")
    p.add_argument("--compare", action="store_true", help="also restore with BOPBTL's original mapping")
    p.add_argument("--guard", choices=("none", "gray", "near_gray"), default="none")
    p.add_argument("--max-side", type=int, default=1024, help="downscale larger photos (GPU memory)")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    obj = OldPhotoModel(repo_root=args.repo, mode="quality", gpu=-1 if args.cpu else 0)
    nets = {"ours": load_direct(args.weights or default_weights(), obj.device)[0]}
    if args.compare:
        nets["bopbtl"] = obj._core.mapping_net

    paths = [args.input] if os.path.isfile(args.input) else sorted(
        f for f in glob.glob(os.path.join(args.input, "*")) if f.lower().endswith(IMG_EXT))
    if not paths:
        raise SystemExit(f"no images found at {args.input}")
    os.makedirs(args.output, exist_ok=True)
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")
        if args.max_side and max(img.size) > args.max_side:
            s = args.max_side / max(img.size)
            img = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.BICUBIC)
        for tag, net in nets.items():
            out = apply_guard(img, restore(obj, img, net), args.guard)
            out.save(os.path.join(args.output, f"{name}_{tag}.png"))
        print(f"restored {path}")


if __name__ == "__main__":
    main()
