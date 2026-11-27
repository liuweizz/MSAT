# SPDX-License-Identifier: Apache-2.0
"""Reconstruct admitted layouts and evaluation masks from explicit raw data roots."""
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .io import checked_relative, read_json, sha256


def load_ground_truth(manifest: dict, data_root: str | Path) -> np.ndarray:
    """Load original-frame GT using the pinned grayscale/nearest/>0 convention.

    Args:
        manifest: Portable category manifest with labels, image and mask paths.
        data_root: Explicit root for those relative paths.

    Returns:
        Boolean masks with shape (images, 518, 518), in admitted sample order.

    Raises:
        ValueError: Pixel GT is inapplicable or a positive sample lacks its mask.
        FileNotFoundError: A referenced image or mask is missing.
    """
    if manifest["geometry"] == "classification-only":
        raise ValueError("This manifest has image labels and no pixel ground truth.")
    masks = []
    for sample in manifest["samples"]:
        if sample.get("mask"):
            path = checked_relative(data_root, sample["mask"])
            with Image.open(path) as image:
                mask = image.convert("L").resize((518, 518), Image.Resampling.NEAREST)
        elif sample["label"] == 0:
            mask = Image.new("L", (518, 518), 0)
        else:
            raise ValueError(f"Positive sample {sample['id']} has no admitted mask path.")
        masks.append(np.asarray(mask) > 0)
    return np.stack(masks)


def prepare_data(manifest_path: str | Path, source_root: str | Path,
                 output_root: str | Path) -> dict:
    """Create the admitted portable layout without resplitting source samples.

    Args:
        manifest_path: Manifest containing original source-relative paths.
        source_root: Explicit root for the recorded original source image/mask paths.
        output_root: New root for manifest image, mask, and KSDD band paths.

    Returns:
        Counts of copied images, converted masks and JPEG bands.

    Raises:
        ValueError: A source binding is absent or an existing output differs.
    """
    manifest = read_json(manifest_path)
    images = masks = bands = 0
    for sample in manifest["samples"]:
        source = checked_relative(source_root, sample.get("source_image", sample["image"]))
        destination = checked_relative(output_root, sample["image"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256(source) != sha256(destination):
                raise ValueError(f"Existing prepared image differs for {sample['id']}.")
        else:
            shutil.copyfile(source, destination)
        images += 1
        if sample.get("mask"):
            source_mask = sample.get("source_mask")
            if not source_mask:
                raise ValueError(f"Original mask binding is absent for {sample['id']}.")
            with Image.open(checked_relative(source_root, source_mask)) as image:
                mask = image.convert("L")
            target = checked_relative(output_root, sample["mask"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                with Image.open(target) as existing:
                    if not np.array_equal(np.asarray(existing.convert("L")), np.asarray(mask)):
                        raise ValueError(f"Existing prepared mask differs for {sample['id']}.")
            else:
                mask.save(target)
            masks += 1
        for band in sample.get("bands") or []:
            with Image.open(source) as image:
                crop = image.convert("RGB").crop(band["source_crop_edges"])
            target = checked_relative(output_root, band["image"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                crop.save(target, quality=95)
            expected = band.get("image_sha256")
            if expected and sha256(target) != expected:
                raise ValueError(f"Prepared KSDD band differs for {sample['id']}.")
            bands += 1
    return dict(images=images, masks=masks, bands=bands)
