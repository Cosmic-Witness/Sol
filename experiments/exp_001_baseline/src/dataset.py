"""Dataset and cache construction for experiment 001.

Two-stage design
----------------
Conditioning a 2048x2048 frame costs roughly a second: a Hough transform, a
64-annulus radial fit, then CLAHE. Repeating that for 974 records across 60
epochs would spend more wall-clock on preprocessing than on gradient steps, and
free-tier Colab has no wall-clock to spare.

`build_cache` therefore runs the conditioning once and writes 512x512 PNGs.
`FilamentDataset` reads those PNGs. Preprocessing is keyed on the observation,
not on the annotation record, because three annotation batches of one photograph
share one conditioned image.
"""

from __future__ import annotations

import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from shared.preprocessing import preprocess


def build_cache(
    annotation_path: str,
    images_dir: str,
    cache_dir: str,
    image_size: int,
    clahe_clip: float,
    clahe_grid: int,
    blur_sigma: float,
    verbose: bool = True,
) -> None:
    """Condition every observation once and write the training cache.

    Layout produced under `cache_dir`:

        images/<stem>.png   conditioned observation, image_size square
        masks/<image_id>.png  binary union of that batch's filaments

    The function is resumable. An existing output file is left untouched, so a
    Colab disconnect midway through cache construction costs only the remainder.
    """
    coco = COCO(annotation_path)
    img_out = Path(cache_dir) / "images"
    mask_out = Path(cache_dir) / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    records = coco.loadImgs(coco.getImgIds())
    total = len(records)
    for n, record in enumerate(records, start=1):
        stem = Path(record["file_name"]).stem
        image_id = record["id"]
        image_path = img_out / f"{stem}.png"
        mask_path = mask_out / f"{image_id}.png"

        if not image_path.exists():
            raw = cv2.imread(os.path.join(images_dir, record["file_name"]), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                raise FileNotFoundError(os.path.join(images_dir, record["file_name"]))
            conditioned = preprocess(raw, clahe_clip, clahe_grid, blur_sigma)
            resized = cv2.resize(conditioned, (image_size, image_size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(image_path), resized)

        if not mask_path.exists():
            union = np.zeros((record["height"], record["width"]), dtype=np.uint8)
            for ann in coco.loadAnns(coco.getAnnIds(imgIds=[image_id])):
                union |= coco.annToMask(ann)
            # INTER_NEAREST would drop filament threads thinner than the
            # downscale factor. INTER_AREA followed by a threshold keeps any
            # structure that covers part of a destination pixel.
            small = cv2.resize(
                union * 255, (image_size, image_size), interpolation=cv2.INTER_AREA
            )
            cv2.imwrite(str(mask_path), (small > 32).astype(np.uint8) * 255)

        if verbose and n % 100 == 0:
            print(f"cache {n}/{total}", flush=True)


def build_transforms(image_size: int, train: bool) -> A.Compose:
    """Augmentation policy.

    Solar frames carry no canonical orientation, so flips and 90-degree
    rotations are label-preserving. Brightness and contrast jitter stands in for
    the seeing and exposure variation across the six GONG sites. Elastic
    deformation is deliberately mild: filament morphology is the signal, and a
    strong warp would teach shapes the Sun does not make.
    """
    if not train:
        return A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.ElasticTransform(alpha=20, sigma=6, p=0.15),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class FilamentDataset(Dataset):
    """Serves conditioned images and binary filament masks from the cache.

    One item is one annotation record, not one observation. An observation with
    three independent annotations therefore appears three times, each with its
    own mask. That is intentional: the disagreement between annotators is real
    signal about which structures are ambiguous, and averaging it away would
    discard it.
    """

    def __init__(
        self,
        image_ids: list[str],
        id_to_stem: dict[str, str],
        cache_dir: str,
        image_size: int,
        train: bool,
    ):
        self.image_ids = image_ids
        self.id_to_stem = id_to_stem
        self.cache_dir = Path(cache_dir)
        self.transform = build_transforms(image_size, train)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        stem = Path(self.id_to_stem[image_id]).stem

        image = cv2.imread(str(self.cache_dir / "images" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.cache_dir / "masks" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"cache miss for {image_id}; run build_cache first")

        # ImageNet encoder stems expect three channels.
        image = np.repeat(image[:, :, None], 3, axis=2)
        mask = (mask > 127).astype(np.float32)

        augmented = self.transform(image=image, mask=mask)
        image_t = augmented["image"].transpose(2, 0, 1).astype(np.float32)
        mask_t = augmented["mask"][None, :, :].astype(np.float32)
        return image_t, mask_t, image_id
