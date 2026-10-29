"""
OldPhotoModel: single-class wrapper around microsoft/Bringing-Old-Photos-Back-to-Life
(Global restoration stage). Finds the repo (Colab: /content/...), downloads the pretrained
weights if missing, builds VAE A, VAE B and the mapping network, loads their weights and
verifies that every checkpoint key matched.

    from old_photo_model import OldPhotoModel
    obj = OldPhotoModel(mode="quality", gpu=0)

    x   = obj.to_tensor(PIL_image)
    z_A = obj.encode_A(x)
    z_B = obj.map(z_A)
    y   = obj.decode_B(z_B)            # or obj.translate(x)
    obj.to_image(y).save("out.png")
    obj.restore(PIL_image).save("out.png")

Modes: "quality", "scratch" (needs mask), "scratch_hr" (needs mask, high-res).
Scratch modes: x, m = obj.prepare(image, mask); obj.map(z_A, m) / obj.translate(x, m).
"""
import contextlib
import glob
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

import numpy as np
import torch
from PIL import Image, ImageFilter

_REPO_ZIP = "https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/archive/refs/heads/master.zip"
_CKPT_ZIP = ("https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/"
             "releases/download/v1.0/global_checkpoints.zip")
_DEFAULT_ROOT = "/content/Bringing-Old-Photos-Back-to-Life"

# (checkpoint sub-folder, network label) required per mode -> files named "<epoch>_net_<label>.pth"
_REQUIRED = {
    "quality":    [("VAE_A_quality", "G"), ("VAE_B_quality", "G"), ("mapping_quality", "mapping_net")],
    "scratch":    [("VAE_A_quality", "G"), ("VAE_B_scratch", "G"), ("mapping_scratch", "mapping_net")],
    "scratch_hr": [("VAE_A_quality", "G"), ("VAE_B_scratch", "G"), ("mapping_Patch_Attention", "mapping_net")],
}


def _download(url, dest):
    print(f"[OldPhotoModel] downloading {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def _find_repo(hint=None):
    """Return repo root: hint, /content/<repo>, or any folder under /content containing Global/models/mapping_model.py."""
    marker = os.path.join("Global", "models", "mapping_model.py")
    cands = [hint, _DEFAULT_ROOT, _DEFAULT_ROOT + "-master"]
    cands += glob.glob("/content/*/") + glob.glob("/content/*/*/")
    for c in cands:
        if c and os.path.isfile(os.path.join(c, marker)):
            return os.path.abspath(c)
    return None


@contextlib.contextmanager
def _cpu_safe_torch_load():
    """The repo calls torch.load(path) without map_location, which fails on CPU-only machines."""
    orig = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return orig(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = orig


class OldPhotoModel:

    def __init__(self, repo_root=None, mode="quality", gpu=0, auto_download=True, verify=True,
                 test_mode="Full"):
        assert mode in _REQUIRED, f"mode must be one of {list(_REQUIRED)}"
        assert test_mode in ("Full", "Scale"), "test_mode: 'Full' (keep size) or 'Scale' (short side -> 256)"
        self.mode = mode
        self.test_mode = test_mode  # only used in quality mode, like --test_mode in Global/test.py

        found = _find_repo(repo_root)
        if found is None:
            if not auto_download:
                raise FileNotFoundError("Repo not found under /content; pass repo_root=...")
            repo_root = os.path.abspath(repo_root or _DEFAULT_ROOT)
            self._ensure_repo(repo_root)  # zip download as fallback
        else:
            repo_root = found
        global_dir = os.path.join(repo_root, "Global")
        print(f"[OldPhotoModel] using repo at {repo_root}")

        if auto_download:
            self._ensure_weights(global_dir, mode)
        else:
            self._check_weights(global_dir, mode, raise_error=True)

        if gpu >= 0 and not torch.cuda.is_available():
            print("[OldPhotoModel] CUDA not available -> running on CPU")
            gpu = -1

        opt = self._build_opt(global_dir, mode, gpu)

        from models.mapping_model import Pix2PixHDModel_Mapping  # repo code

        prev = os.getcwd()
        os.chdir(global_dir)
        try:
            with _cpu_safe_torch_load():
                core = Pix2PixHDModel_Mapping()
                core.initialize(opt)  # builds netG_A, netG_B, mapping_net + loads weights
        finally:
            os.chdir(prev)

        self.vae_A = core.netG_A      # old-photo domain VAE
        self.vae_B = core.netG_B      # clean domain VAE
        self.mapping = core.mapping_net
        for net in (self.vae_A, self.vae_B, self.mapping):
            net.eval()
            for p in net.parameters():
                p.requires_grad = False

        self.uses_mask = bool(getattr(opt, "NL_use_mask", False))
        self._hr = bool(getattr(opt, "inference_optimize", False))
        self._mask_dilation = int(getattr(opt, "mask_dilation", 0) or 0)
        self.device = next(self.mapping.parameters()).device
        self._core = core
        self._opt = opt

        if verify:
            self._verify_weights(opt, mode)

    # ------------------------------------------------------------------ #
    # Core methods
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode_A(self, x):
        return self.vae_A.forward(x, flow="enc")

    @torch.no_grad()
    def encode_B(self, x):
        return self.vae_B.forward(x, flow="enc")

    @torch.no_grad()
    def map(self, z_A, mask=None):
        if not self.uses_mask:
            return self.mapping(z_A)
        if mask is None:
            raise ValueError("scratch mode: pass mask (1,1,H,W) tensor, same H/W as the input image "
                             "(use obj.prepare(image, mask)).")
        if self._hr:
            return self.mapping.inference_forward(z_A, mask)
        return self.mapping(z_A, mask)

    @torch.no_grad()
    def decode_A(self, z_A):
        return self.vae_A.forward(z_A, flow="dec")

    @torch.no_grad()
    def decode_B(self, z_B):
        return self.vae_B.forward(z_B, flow="dec")

    @torch.no_grad()
    def translate(self, x_A, mask=None):
        z_A = self.encode_A(x_A)
        z_B = self.map(z_A, mask)
        return self.decode_B(z_B)

    # ------------------------------------------------------------------ #
    # Helpers: PIL <-> tensor, and one-call restore
    # ------------------------------------------------------------------ #
    def _fit(self, img):
        """Same resizing as Global/test.py (BILINEAR, multiples of 4).
        Scratch modes never rescale in test.py, so only the multiple-of-4 rounding applies there."""
        w, h = img.size
        ow, oh = w, h
        if self.test_mode == "Scale" and not self.uses_mask:
            if w < h:
                ow, oh = 256, h / w * 256
            else:
                oh, ow = 256, w / h * 256
        size = (int(round(ow / 4) * 4), int(round(oh / 4) * 4))
        return img if size == (w, h) else img.resize(size, Image.BILINEAR)

    def to_tensor(self, image):
        """PIL -> (1,3,H,W) tensor in [-1,1]; H,W rounded to multiples of 4."""
        image = self._fit(image.convert("RGB"))
        x = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return ((x - 0.5) / 0.5).unsqueeze(0).to(self.device)

    def mask_to_tensor(self, mask, size):
        """PIL mask (white = scratch) -> (1,1,H,W) tensor, resized to size=(W,H)."""
        m = mask.convert("RGB").resize(size, Image.NEAREST)
        m = torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return m[:1].unsqueeze(0).to(self.device)

    def prepare(self, image, mask=None):
        """PIL (+ PIL mask in scratch modes) -> (x, mask_tensor or None), preprocessed like Global/test.py:
        size handled like test.py, mask dilated (HR mode), scratched pixels painted white."""
        img = self._fit(image.convert("RGB"))
        if not self.uses_mask:
            return self.to_tensor(img), None
        if mask is None:
            raise ValueError("scratch mode needs a mask image (white = scratch).")
        mk = mask.convert("RGB").resize(img.size, Image.NEAREST)
        for _ in range(self._mask_dilation):
            mk = mk.filter(ImageFilter.MaxFilter(3))
        m = np.asarray(mk, dtype=np.float32) / 255.0
        holed = np.asarray(img, dtype=np.float32) * (1.0 - m) + 255.0 * m
        img = Image.fromarray(holed.round().clip(0, 255).astype(np.uint8))
        return self.to_tensor(img), self.mask_to_tensor(mk, img.size)

    @staticmethod
    def to_image(x, normalize=True):
        """(1,3,H,W) tensor in [-1,1] -> PIL. normalize=True min-max stretches like
        save_image(..., normalize=True) in Global/test.py; False = plain clamp."""
        y = (x.detach().cpu().float() + 1.0) / 2.0
        if normalize:
            lo, hi = y.min(), y.max()
            y = (y - lo) / (hi - lo).clamp_min(1e-5)
        y = y.clamp(0, 1)[0]
        return Image.fromarray((y.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))

    def restore(self, image, mask=None, keep_size=True, normalize=True):
        """PIL in -> PIL out (full pipeline)."""
        x, m = self.prepare(image, mask)
        out = self.to_image(self.translate(x, m), normalize)
        if keep_size and out.size != image.size:
            out = out.resize(image.size, Image.BICUBIC)
        return out

    # ------------------------------------------------------------------ #
    # Setup: repo, weights, options
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ensure_repo(repo_root):
        if os.path.isfile(os.path.join(repo_root, "Global", "models", "mapping_model.py")):
            return
        os.makedirs(os.path.dirname(repo_root), exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            zpath = os.path.join(tmp, "repo.zip")
            _download(_REPO_ZIP, zpath)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(tmp)
            extracted = os.path.join(tmp, "Bringing-Old-Photos-Back-to-Life-master")
            if os.path.exists(repo_root):
                shutil.rmtree(repo_root)
            shutil.move(extracted, repo_root)

    def _ensure_weights(self, global_dir, mode):
        """Download the checkpoint zip (layout: checkpoints/restoration/<folder>/*.pth) and copy
        any missing or empty files for the needed folders."""
        if not self._missing(global_dir, mode):
            return
        target = os.path.join(global_dir, "checkpoints", "restoration")
        wanted = {f for f, _ in _REQUIRED[mode]}
        with tempfile.TemporaryDirectory() as tmp:
            zpath = os.path.join(tmp, "global_checkpoints.zip")
            _download(_CKPT_ZIP, zpath)
            extract_dir = os.path.join(tmp, "extracted")
            with zipfile.ZipFile(zpath) as z:
                z.extractall(extract_dir)
            src_root = os.path.join(extract_dir, "checkpoints", "restoration")
            for folder in wanted:
                os.makedirs(os.path.join(target, folder), exist_ok=True)
                for f in os.listdir(os.path.join(src_root, folder)):
                    src, dst = os.path.join(src_root, folder, f), os.path.join(target, folder, f)
                    if not os.path.isfile(dst) or os.path.getsize(dst) != os.path.getsize(src):
                        shutil.copy2(src, dst)
        self._check_weights(global_dir, mode, raise_error=True)

    @staticmethod
    def _missing(global_dir, mode):
        ckpt = os.path.join(global_dir, "checkpoints", "restoration")
        missing = []
        for folder, label in _REQUIRED[mode]:
            if not glob.glob(os.path.join(ckpt, folder, f"*_net_{label}.pth")):
                missing.append(os.path.join(ckpt, folder, f"latest_net_{label}.pth"))
        return missing

    def _check_weights(self, global_dir, mode, raise_error):
        missing = self._missing(global_dir, mode)
        if missing and raise_error:
            raise FileNotFoundError("Missing checkpoints:\n  " + "\n  ".join(missing) +
                                    f"\nDownload {_CKPT_ZIP} and unzip it inside {global_dir}")
        return missing

    @staticmethod
    def _build_opt(global_dir, mode, gpu):
        if global_dir not in sys.path:
            sys.path.insert(0, global_dir)  # repo imports `options`, `models`, `util` from Global/
        from options.test_options import TestOptions

        flag = "--Quality_restore" if mode == "quality" else "--Scratch_and_Quality_restore"
        backup = sys.argv
        sys.argv = ["old_photo_model", flag, "--gpu_ids", str(gpu)]
        if mode == "scratch_hr":
            sys.argv.append("--HR")
        try:
            opt = TestOptions().parse(save=False)
        finally:
            sys.argv = backup

        # mirrors parameter_set() in Global/test.py
        ckpt = os.path.join(global_dir, "checkpoints", "restoration")
        opt.serial_batches = True
        opt.no_flip = True
        opt.label_nc = 0
        opt.n_downsample_global = 3
        opt.mc = 64
        opt.k_size = 4
        opt.start_r = 1
        opt.mapping_n_block = 6
        opt.map_mc = 512
        opt.no_instance = True
        opt.checkpoints_dir = ckpt

        if mode == "quality":
            opt.name = "mapping_quality"
            opt.load_pretrainA = os.path.join(ckpt, "VAE_A_quality")
            opt.load_pretrainB = os.path.join(ckpt, "VAE_B_quality")
        else:
            opt.NL_res = True
            opt.use_SN = True
            opt.correlation_renormalize = True
            opt.NL_use_mask = True
            opt.NL_fusion_method = "combine"
            opt.non_local = "Setting_42"
            opt.name = "mapping_scratch"
            opt.load_pretrainA = os.path.join(ckpt, "VAE_A_quality")
            opt.load_pretrainB = os.path.join(ckpt, "VAE_B_scratch")
            if mode == "scratch_hr":
                opt.HR = True
                opt.mapping_exp = 1              # Mapping_Model_with_mask_2 (patch attention)
                opt.inference_optimize = True    # enables mapping.inference_forward
                opt.mask_dilation = 3
                opt.name = "mapping_Patch_Attention"
        return opt

    def _verify_weights(self, opt, mode):
        """Re-load each checkpoint with strict=True so a silent partial load can't go unnoticed."""
        ckpt = opt.checkpoints_dir
        vae_epoch = getattr(opt, "use_vae_which_epoch", "latest")
        map_epoch = getattr(opt, "which_epoch", "latest")
        jobs = [
            (self.vae_A, os.path.join(opt.load_pretrainA, f"{vae_epoch}_net_G.pth")),
            (self.vae_B, os.path.join(opt.load_pretrainB, f"{vae_epoch}_net_G.pth")),
            (self.mapping, os.path.join(ckpt, opt.name, f"{map_epoch}_net_mapping_net.pth")),
        ]
        for net, path in jobs:
            try:
                net.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"Checkpoint {path} does not match the network built for mode '{mode}': {e}\n"
                    "The option flags in _build_opt() probably differ from Global/test.py."
                ) from e
