# SPDX-License-Identifier: Apache-2.0
"""Two-scale evidence fusion, image scoring, and monotone distribution anchoring."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Prediction:
    """Prediction arrays and scalar scores.

    Attributes:
        image_score: Float64 promoted native top-eight/global mixture.
        baseline_score: Global score from the 448-resolution observation,
            maximized over bands for three-band inputs.
        native: Float32 array with shape (bands, 784).
        unanchored_map: Float32 518-by-518 map before distribution anchoring.
        anomaly_map: Float32 anchored map, or None for classification-only use.
        baseline_map: Float32 anchoring target from the 448-resolution field,
            or None for classification-only use.
    """

    image_score: float
    baseline_score: float
    native: np.ndarray
    unanchored_map: np.ndarray | None
    anomaly_map: np.ndarray | None
    baseline_map: np.ndarray | None


def area_weights() -> np.ndarray:
    """Return exact area-overlap weights from a 42-grid to a 28-grid.

    Returns:
        Float64 array of shape (28, 42). Every row sums to one.
    """
    low = np.arange(28, dtype=np.int64)[:, None]
    high = np.arange(42, dtype=np.int64)[None, :]
    overlap = np.maximum(
        0, np.minimum(3 * (low + 1), 2 * (high + 1)) - np.maximum(3 * low, 2 * high)
    )
    return overlap.astype(np.float64) / 3.0


def fuse(low: np.ndarray, high: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    """Area-align and average two margin fields with a foreground-dependent floor.

    Args:
        low: Finite float32 margins with shape (784,).
        high: Finite float32 margins with shape (1764,).
        foreground: Boolean foreground with shape (784,).

    Returns:
        Float32 native margins with shape (784,).

    Raises:
        ValueError: An input has invalid shape, dtype, or nonfinite values.
    """
    if (low.shape != (784,) or high.shape != (1764,)
            or foreground.shape != (784,) or foreground.dtype != bool
            or low.dtype != np.float32 or high.dtype != np.float32
            or not np.isfinite(low).all() or not np.isfinite(high).all()):
        raise ValueError("Expected finite float32 28/42 margins and boolean foreground.")
    weights = area_weights()
    mapped = (weights @ high.astype(np.float64).reshape(42, 42) @ weights.T).ravel()
    original = low.astype(np.float64)
    # Preserve this operation order, including cancellation and the final cast.
    mixed = (original + 0.5 * (mapped - original)).astype(np.float32)
    return np.where(foreground, mixed, low.min())


def native_tail(native: np.ndarray) -> float:
    """Average the top eight native values after promotion to float64.

    Args:
        native: Finite float32 array with shape (784,).

    Returns:
        Float64 mean of the largest eight values, including tied values.

    Raises:
        ValueError: The native field has an incompatible shape or dtype.
    """
    if native.shape != (784,) or native.dtype != np.float32 or not np.isfinite(native).all():
        raise ValueError("Native tail requires a finite float32 784-element array.")
    promoted = native.astype(np.float64)
    return float(np.partition(promoted, 776)[-8:].mean(dtype=np.float64))


def monotone_transport(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Project source tie blocks onto the target empirical distribution.

    Args:
        target: Finite float32 array defining the target distribution.
        source: Finite float32 array of the same shape, defining spatial order.

    Returns:
        Float32 array with source ordering and target block barycenters.
        Source ties remain tied. Target histogram equality can be lost through
        tie blocks or float32 rounding.

    Raises:
        ValueError: Inputs have incompatible shapes, dtypes, or finite support.
    """
    if (target.shape != source.shape or not source.size
            or target.dtype != np.float32 or source.dtype != np.float32
            or not np.isfinite(target).all() or not np.isfinite(source).all()):
        raise ValueError("Transport requires same-shaped finite float32 arrays.")
    if target.tobytes() == source.tobytes():
        return target.copy()
    target_ordered = np.sort(target.ravel().astype(np.float64))
    order = np.argsort(source.ravel(), kind="stable")
    source_ordered = source.ravel()[order]
    starts = np.r_[0, np.flatnonzero(source_ordered[1:] != source_ordered[:-1]) + 1]
    sizes = np.diff(np.r_[starts, source.size])
    means = np.add.reduceat(target_ordered, starts, dtype=np.float64) / sizes
    output = np.empty(source.size, np.float64)
    output[order] = np.repeat(means, sizes)
    return output.astype(np.float32).reshape(source.shape)


def postprocess(native: np.ndarray, height: int = 518, width: int = 518) -> np.ndarray:
    """Resize on CPU with bilinear interpolation and apply Gaussian sigma four.

    Args:
        native: Float32 values on a 28-by-28 grid, flattened or two-dimensional.
        height: Output height. KSDD uses 173, 172, and 173 band rows.
        width: Output width in pixels.

    Returns:
        Float32 array with shape (height, width).
    """
    import torch
    from scipy.ndimage import gaussian_filter

    tensor = torch.from_numpy(np.asarray(native, dtype=np.float32)).reshape(1, 1, 28, 28)
    resized = torch.nn.functional.interpolate(
        tensor, size=(height, width), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    return gaussian_filter(resized, sigma=4.0)


def stitch_bands(native: np.ndarray) -> np.ndarray:
    """Postprocess and assemble exactly three KSDD band fields.

    Args:
        native: Float32 margins with shape (3, 784).

    Returns:
        Float32 518-by-518 frame, blurred separately within each band.

    Raises:
        ValueError: The number of bands or native shape differs from the protocol.
    """
    if native.shape != (3, 784):
        raise ValueError("KSDD frame assembly requires three 784-element band fields.")
    return np.concatenate([
        postprocess(native[index], height=height)
        for index, height in enumerate((173, 172, 173))
    ], axis=0)


def predict_observations(
    low: np.ndarray,
    high: np.ndarray,
    foreground: np.ndarray,
    global_score: np.ndarray,
    geometry: str = "ordinary",
    target_map: np.ndarray | None = None,
) -> Prediction:
    """Compute image scores and anchored maps from two-scale margin observations.

    Args:
        low: Float32 array with shape (bands, 784).
        high: Float32 array with shape (bands, 1764).
        foreground: Boolean array with shape (bands, 784).
        global_score: Numeric array with shape (bands,), using 448 observations.
        geometry: ordinary, ksdd-three-bands, or classification-only.
        target_map: Optional explicit anchoring distribution A for cache replay.
            When omitted, A is constructed from the current 448-resolution field.

    Returns:
        AMT1 native fields, scores, and applicable final maps.

    Raises:
        ValueError: Geometry or band counts are incompatible.
    """
    if geometry not in ("ordinary", "ksdd-three-bands", "classification-only"):
        raise ValueError(f"Unknown geometry {geometry!r}.")
    bands = 3 if geometry == "ksdd-three-bands" else 1
    if low.shape != (bands, 784) or high.shape != (bands, 1764):
        raise ValueError("Observation shapes do not match the selected geometry.")
    native = np.stack([fuse(low[i], high[i], foreground[i]) for i in range(bands)])
    image_scores = [float(0.5 * np.float64(global_score[i])
                          + 0.5 * np.float64(native_tail(native[i]))) for i in range(bands)]
    baseline_score = float(np.max(global_score))
    if geometry == "classification-only":
        return Prediction(max(image_scores), baseline_score, native, None, None, None)
    official_native = np.stack([
        np.where(foreground[i], low[i], low[i].min()) for i in range(bands)
    ])
    if bands == 3:
        unanchored = stitch_bands(native)
        official = stitch_bands(official_native)
    else:
        unanchored = postprocess(native[0])
        official = postprocess(official_native[0])
    if target_map is not None:
        official = np.asarray(target_map)
    anchored = monotone_transport(official, unanchored)
    return Prediction(max(image_scores), baseline_score, native, unanchored, anchored, official)
