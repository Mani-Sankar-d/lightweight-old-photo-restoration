"""
Port of the synthetic degradation pipeline of microsoft/Bringing-Old-Photos-Back-to-Life
(Global/data/online_dataset_for_old_photos.py): blur, Gaussian / speckle / salt-and-pepper noise,
downscaling and JPEG compression, applied in random order.
"""
import random
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def _pil_to_np(img):
    ar = np.array(img)
    ar = ar.transpose(2, 0, 1) if ar.ndim == 3 else ar[None, ...]
    return ar.astype(np.float32) / 255.0


def _np_to_pil(ar):
    ar = np.clip(ar * 255, 0, 255).astype(np.uint8)
    ar = ar[0] if ar.shape[0] == 1 else ar.transpose(1, 2, 0)
    return Image.fromarray(ar)


def synthesize_salt_pepper(image, amount, salt_vs_pepper):
    img = _pil_to_np(image)
    out = img.copy()
    flipped = np.random.choice([True, False], size=img.shape, p=[amount, 1 - amount])
    salted = np.random.choice([True, False], size=img.shape, p=[salt_vs_pepper, 1 - salt_vs_pepper])
    peppered = ~salted
    out[flipped & salted] = 1
    out[flipped & peppered] = 0.0
    return _np_to_pil(np.clip(out, 0, 1).astype(np.float32))


def synthesize_gaussian(image, std_l, std_r):
    img = _pil_to_np(image)
    std = random.uniform(std_l / 255.0, std_r / 255.0)
    noisy = img + np.random.normal(0, std, size=img.shape)
    return _np_to_pil(np.clip(noisy, 0, 1).astype(np.float32))


def synthesize_speckle(image, std_l, std_r):
    img = _pil_to_np(image)
    std = random.uniform(std_l / 255.0, std_r / 255.0)
    noisy = img + np.random.normal(0, std, size=img.shape) * img
    return _np_to_pil(np.clip(noisy, 0, 1).astype(np.float32))


def synthesize_low_resolution(img):
    w, h = img.size
    new_w = random.randint(int(w / 2), w)
    new_h = random.randint(int(h / 2), h)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    method = Image.NEAREST if random.uniform(0, 1) < 0.5 else Image.BILINEAR
    return img.resize((w, h), method)


def convert_to_jpeg(img, quality):
    with BytesIO() as f:
        img.save(f, format="JPEG", quality=quality)
        f.seek(0)
        return Image.open(f).convert("RGB")


def blur_image_v2(img):
    x = np.array(img)
    kernel_size = random.choice([(3, 3), (5, 5), (7, 7)])
    std = random.uniform(1.0, 5.0)
    blur = cv2.GaussianBlur(x, kernel_size, std)
    return Image.fromarray(blur.astype(np.uint8))


def online_add_degradation_v2(img):
    """The repo's exact composite degradation used for training domain A / the mapping net."""
    task_id = np.random.permutation(4)
    for x in task_id:
        if x == 0 and random.uniform(0, 1) < 0.7:
            img = blur_image_v2(img)
        if x == 1 and random.uniform(0, 1) < 0.7:
            flag = random.choice([1, 2, 3])
            if flag == 1:
                img = synthesize_gaussian(img, 5, 50)
            elif flag == 2:
                img = synthesize_speckle(img, 5, 50)
            elif flag == 3:
                img = synthesize_salt_pepper(img, random.uniform(0, 0.01), random.uniform(0.3, 0.8))
        if x == 2 and random.uniform(0, 1) < 0.7:
            img = synthesize_low_resolution(img)
        if x == 3 and random.uniform(0, 1) < 0.7:
            img = convert_to_jpeg(img, random.randint(40, 100))
    return img


def irregular_hole_synthesize(img, mask):
    """Paint the masked (white) region white, exactly as the repo does before encoding domain A in scratch mode."""
    img_np = np.array(img).astype("uint8")
    mask_np = np.array(mask).astype("uint8") / 255
    img_new = img_np * (1 - mask_np) + mask_np * 255
    hole_img = Image.fromarray(img_new.astype("uint8")).convert("RGB")
    return hole_img, mask.convert("L")
