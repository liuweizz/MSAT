# SPDX-License-Identifier: Apache-2.0
"""Exact pooled pixel evaluation and inherited binary-image metric conventions."""
from pathlib import Path

import numpy as np

from .io import read_json, sha256, write_json


def evaluate(manifest_path: str | Path, prediction_root: str | Path,
             gt_root: str | Path | None, output: str | Path, device: str = "cpu",
             data_root: str | Path | None = None,
             identity_map: str | Path | None = None) -> dict:
    """Evaluate saved predictions in a category's exact admitted sample order.

    Args:
        manifest_path: Category manifest with real labels and applicability flags.
        prediction_root: Directory of 00000.npz and matching completion markers.
        gt_root: Artifact root containing benchmark/<key>/GT.npy, used only here.
        output: Destination JSON file containing eight metrics and applicability.
        device: cpu or a PyTorch CUDA device for the exact pooled sorter.
        data_root: Optional explicit raw mask root, as an alternative to gt_root.
        identity_map: Explicit identity mapping for predictions from the single-image
            API without a bound manifest. Each record binds its NPZ hash, ID, index
            and this manifest's hash.

    Returns:
        Category-level measured image and pixel metrics as fractions in [0, 1].

    Raises:
        ValueError: Completion markers, identities, or GT geometry are invalid.
    """
    from ._vendor.radiology.metrics import compute_image_metrics
    from ._vendor.radiology.pixel_metrics import _trapezoid
    from ._vendor.radiology.exact_metrics import gt_region_weights, pooled_metrics

    manifest = read_json(manifest_path)
    manifest_hash = sha256(manifest_path)
    mapping = read_json(identity_map) if identity_map is not None else {}
    labels = np.array([sample["label"] for sample in manifest["samples"]])
    scores = []
    maps = []
    localization = manifest["geometry"] != "classification-only"
    for index, sample in enumerate(manifest["samples"]):
        path = Path(prediction_root) / f"{index:05d}.npz"
        marker = read_json(path.with_suffix(".complete.json"))
        if marker["status"] != "completed" or sha256(path) != marker["sha256"]:
            raise ValueError(f"Prediction is incomplete or changed: {path.name}")
        expected_identity = dict(sample_id=sample["id"], index=index,
                                 manifest_sha256=manifest_hash)
        identity = marker.get("identity")
        if identity is None:
            explicit = mapping.get(path.name)
            if explicit is None or explicit.get("prediction_sha256") != marker["sha256"]:
                raise ValueError(f"Prediction {path.name} requires an explicit sample identity mapping.")
            identity = {name: explicit.get(name) for name in expected_identity}
        if identity != expected_identity:
            raise ValueError(f"Prediction identity differs from the manifest at index {index}.")
        with np.load(path) as prediction:
            scores.append(float(prediction["J"]))
            if localization:
                maps.append(prediction["Q"])
    metrics = {name: None for name in (
        "image_auroc", "image_ap", "image_f1", "pixel_auroc", "pixel_ap", "pixel_f1",
        "aupro", "pro_at_fpr_0p01")}
    if len(set(labels.tolist())) == 2:
        image = compute_image_metrics(np.asarray(scores), labels, ["test"] * len(labels))["overall"]
        metrics.update(image_auroc=float(image["auroc"]), image_ap=float(image["ap"]),
                       image_f1=float(image["f1"]))
    if localization:
        if data_root is not None:
            from .data import load_ground_truth

            truth = load_ground_truth(manifest, data_root)
        elif gt_root is not None:
            truth = np.load(Path(gt_root) / "benchmark" / manifest["key"] / "GT.npy").astype(bool)
        else:
            raise ValueError("Evaluation requires an explicit artifact root or raw mask data root.")
        predictions = np.stack(maps)
        if truth.shape != predictions.shape:
            raise ValueError("GT and prediction sample/pixel geometry differ.")
        weights, regions = gt_region_weights(truth)
        pixel = pooled_metrics(predictions, truth, weights, regions, device, _trapezoid)
        metrics.update({name: pixel[name] for name in metrics if not name.startswith("image_")})
    result = dict(key=manifest["key"], count=len(labels), metrics=metrics,
                  image_applicable=len(set(labels.tolist())) == 2,
                  pixel_applicable=localization)
    write_json(output, result)
    return result
