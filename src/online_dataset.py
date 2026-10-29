"""
Training pairs generated on the fly: every __getitem__ draws a fresh clean image, a fresh random crop and a
fresh random degradation, as in the original BOPBTL training, so the model never sees the same degraded
instance twice.

Validation uses a fixed, in-memory set built once with a fixed seed from held-out source images (never used
for training), so validation numbers are comparable across runs.
"""
import glob
import math
import os
import random

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from degradation import online_add_degradation_v2

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".ppm")
FINE_SIZE = 256

_to_tensor_norm = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1,1]
])


def list_clean_images(clean_dir, min_side=256):
    files = sorted(f for f in glob.glob(os.path.join(clean_dir, "*")) if f.lower().endswith(IMG_EXT))
    kept, corrupt = [], 0
    for f in files:
        try:
            with Image.open(f) as im:
                im.verify()          # catches truncated/corrupt files
            with Image.open(f) as im:  # verify() invalidates the handle, reopen to read size
                w, h = im.size
        except Exception:
            corrupt += 1
            continue
        if w >= min_side and h >= min_side:
            kept.append(f)
    print(f"{len(kept)}/{len(files)} images pass the >= {min_side}px filter "
          f"({corrupt} unreadable/corrupt files skipped)")
    return kept


def _crop_params(size, fine_size=FINE_SIZE):
    w, h = size
    x = random.randint(0, max(0, w - fine_size))
    y = random.randint(0, max(0, h - fine_size))
    flip = random.random() > 0.5
    return (x, y), flip


def _apply_crop_flip(img, pos, flip, fine_size=FINE_SIZE):
    ow, oh = img.size
    x1, y1 = pos
    if ow > fine_size or oh > fine_size:
        img = img.crop((x1, y1, x1 + fine_size, y1 + fine_size))
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def make_online_pair(clean_path, grayscale_prob=0.1):
    """Fresh (A, B) tensor pair every call: new crop position, new flip, new
    degradation realization -- even for the same source image.

    grayscale_prob: fraction of pairs where BOTH A and B are desaturated
    (matches the repo's own training, which also keeps B grayscale when A is).
    Real target old photos are overwhelmingly B&W, so the default 0.1 (matching
    the original repo) under-represents that case for a mapping net trained
    from scratch without the repo's additional real-old-photo VAE_A domain
    adaptation -- raise this for training runs targeting B&W restoration."""
    B = Image.open(clean_path).convert("RGB")
    A = online_add_degradation_v2(B)
    if A.size != B.size:
        A = A.resize(B.size, Image.BILINEAR)

    if random.uniform(0, 1) < grayscale_prob:
        A = A.convert("L").convert("RGB")
        B = B.convert("L").convert("RGB")

    pos, flip = _crop_params(B.size)
    A = _apply_crop_flip(A, pos, flip)
    B = _apply_crop_flip(B, pos, flip)

    if A.size != (FINE_SIZE, FINE_SIZE):
        A = A.resize((FINE_SIZE, FINE_SIZE), Image.BICUBIC)
        B = B.resize((FINE_SIZE, FINE_SIZE), Image.BICUBIC)

    return _to_tensor_norm(A), _to_tensor_norm(B)


class OnlineQualityDataset(Dataset):
    """len() is nominal (controls how many samples count as "one epoch" for
    DataLoader bookkeeping / shuffling), but __getitem__ always draws a fresh
    random source image + fresh degradation, regardless of idx. This is the
    key difference from a dataset that reads a fixed PNG from disk per idx."""

    def __init__(self, clean_dir, length=20000, min_side=256, grayscale_prob=0.1):
        self.files = list_clean_images(clean_dir, min_side)
        assert self.files, f"no usable images (>= {min_side}px both sides) in {clean_dir}"
        self.length = length
        self.grayscale_prob = grayscale_prob

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        for _ in range(5):  # a handful of retries in case of a corrupt/transient-unreadable file
            f = random.choice(self.files)
            try:
                return make_online_pair(f, self.grayscale_prob)
            except Exception as e:
                print(f"[OnlineQualityDataset] skipping unreadable {f}: {e}")
        raise RuntimeError("too many consecutive unreadable files -- check your image folder")


def make_online_train_loader(clean_dir, batch_size=16, epoch_length=20000, num_workers=4, grayscale_prob=0.1):
    """Online training loader: every sample is freshly degraded. Use a
    multi-worker DataLoader since degradation (PIL/cv2 ops) is CPU-bound and
    would otherwise bottleneck the GPU."""
    ds = OnlineQualityDataset(clean_dir, length=epoch_length, grayscale_prob=grayscale_prob)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )
    print(f"online train loader: {len(ds.files)} source images, "
          f"{epoch_length} samples/epoch, {len(loader)} batches/epoch")
    return loader


# --------------------------------------------------------------------------- #
# Train/val split at the SOURCE-IMAGE level, both drawn from the same folder.
# --------------------------------------------------------------------------- #
class FixedPairDataset(Dataset):
    """A small, pre-generated, in-memory set of (A, B) tensor pairs. Built
    once with a fixed seed so it's identical across runs/evals, but drawn
    from source images that were NEVER used for training."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def make_online_dataloaders(clean_dir, batch_size=16, epoch_length=20000,
                             num_workers=2, val_fraction=0.05, val_samples=400, seed=0,
                             grayscale_prob=0.1):
    """Splits source images into train/val pools (no overlap), builds the
    online training loader from the train pool, and a FIXED val set (built
    once, with a fixed seed, same degradation pipeline) from the held-out
    val pool -- so val is comparable across runs AND actually measures
    generalization on the same visual domain the model is training on."""
    all_files = list_clean_images(clean_dir)
    rng = random.Random(seed)
    shuffled = all_files[:]
    rng.shuffle(shuffled)
    n_val_files = max(1, int(len(shuffled) * val_fraction))
    val_files, train_files = shuffled[:n_val_files], shuffled[n_val_files:]
    print(f"split: {len(train_files)} train source images, {len(val_files)} val source images (no overlap)")

    train_ds = OnlineQualityDataset.__new__(OnlineQualityDataset)  # bypass list_clean_images re-scan
    train_ds.files = train_files
    train_ds.length = epoch_length
    train_ds.grayscale_prob = grayscale_prob
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )

    # build the fixed val set once, with a local seeded RNG so it's reproducible
    state = random.getstate()
    random.seed(seed + 1)
    pairs = []
    i = 0
    while len(pairs) < val_samples:
        f = val_files[i % len(val_files)]
        try:
            pairs.append(make_online_pair(f, grayscale_prob))
        except Exception as e:
            print(f"[val set] skipping unreadable {f}: {e}")
        i += 1
    random.setstate(state)

    val_loader = DataLoader(FixedPairDataset(pairs), batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"online train loader: {len(train_files)} source images, {epoch_length} samples/epoch, "
          f"{len(train_loader)} batches/epoch | fixed val: {len(pairs)} samples from {len(val_files)} held-out images")
    return train_loader, val_loader
