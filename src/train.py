"""
Train the lightweight direct mapping (default) or, for the ablation, the bridge-flow mapping.

    python src/train.py --clean-dir data/voc_clean --out checkpoints/direct
    python src/train.py --clean-dir data/voc_clean --out checkpoints/bridge_flow --model bridge_flow

Training pairs are made on the fly from clean photos with BOPBTL's synthetic degradation; 5% of the source
images are held out for validation. Re-running the same command resumes from <out>/latest.pt.
"""
import argparse
import random

import numpy as np
import torch

from old_photo_model import OldPhotoModel
from online_dataset import make_online_dataloaders

COMMON = dict(width=256, n_blocks=9, lr=2e-4, weight_decay=1e-4, eval_every=500, gan_start=2000, d_start=0,
              d_lr=2e-4, num_D=2, n_layers_D=3, d_conditional=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clean-dir", required=True, help="folder of clean photos (both sides >= 256 px)")
    p.add_argument("--out", required=True, help="checkpoint folder")
    p.add_argument("--model", choices=("direct", "bridge_flow"), default="direct")
    p.add_argument("--repo", default=None, help="path to a clone of microsoft/Bringing-Old-Photos-Back-to-Life")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    obj = OldPhotoModel(repo_root=args.repo, mode="quality", gpu=0)
    train_loader, val_loader = make_online_dataloaders(
        clean_dir=args.clean_dir, batch_size=args.batch_size, epoch_length=20000,
        num_workers=args.workers, val_fraction=0.05, val_samples=400, grayscale_prob=0.5)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.model == "direct":
        from train_direct import train_direct
        train_direct(obj, train_loader, val_loader, args.out, max_steps=args.steps,
                     vgg_weight=10.0, gan_weight=1.0, fm_weight=10.0, **COMMON)
    else:
        from train_rf import train
        train(obj, train_loader, val_loader, args.out, max_steps=args.steps, sigma=0.5, t_mode="bridge",
              traj_prob=0.5, traj_steps=(4, 8, 16), disc_type="multiscale", gan_weight=1.0, fm_weight=10.0,
              perceptual_type="vgg19", perceptual_weight=10.0, val_pix_steps=1, pixel_every=1,
              early_stop=False, select_best=False, save_every=0, snapshot_every=0, **COMMON)


if __name__ == "__main__":
    main()
