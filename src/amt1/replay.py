# SPDX-License-Identifier: Apache-2.0
"""Public fixed-observation reproduction with per-category numerical certificates."""
from pathlib import Path
import time

import numpy as np

from .core import predict_observations
from .io import read_json, sha256, write_json


def replay_category(manifest_path: str | Path, artifact_root: str | Path,
                    output_root: str | Path, tolerance: float = 1e-7) -> dict:
    """Replay every admitted image from AMT1-only cached observations.

    Args:
        manifest_path: Path to an external portable category manifest.
        artifact_root: Directory containing benchmark/<dataset>/<category>.
        output_root: Independent destination for per-category certificates.
        tolerance: Maximum absolute error allowed against the reference arrays.

    Returns:
        Counts, observed maximum errors, and exact-equality counts.

    Raises:
        ValueError: Cached identities, numerical errors, or applicability differ.
    """
    started = time.perf_counter()
    manifest = read_json(manifest_path)
    key = manifest["key"]
    folder = Path(artifact_root) / "benchmark" / key
    with np.load(folder / "observations.npz") as archive:
        arrays = {name: archive[name] for name in archive.files}
    classification_only = manifest["geometry"] == "classification-only"
    target = None if classification_only else np.load(folder / "A.npy", mmap_mode="r")
    reference = None if classification_only else np.load(folder / "Q.npy", mmap_mode="r")
    errors = dict(J=0.0, G=0.0, native=0.0, Q=0.0)
    exact = dict(J=0, native=0, Q=0)
    rows = []
    for index, sample in enumerate(manifest["samples"]):
        if index != sample["index"]:
            raise ValueError("Sample order differs from the admitted manifest.")
        units = arrays["frame_unit_indices"][index]
        prediction = predict_observations(
            arrays["448_D0"][units], arrays["672_D0"][units],
            arrays["F"][units], arrays["G"][units], geometry=manifest["geometry"],
            target_map=None if classification_only else np.asarray(target[index]),
        )
        j_error = abs(prediction.image_score - arrays["J_reference"][index])
        g_error = abs(prediction.baseline_score - arrays["G_reference"][index])
        native_error = float(np.max(np.abs(prediction.native.astype(np.float64)
                                          - arrays["native_reference"][units].astype(np.float64))))
        q_error = 0.0 if classification_only else float(np.max(np.abs(
            prediction.anomaly_map.astype(np.float64) - reference[index].astype(np.float64))))
        current = dict(J=float(j_error), G=float(g_error), native=native_error, Q=q_error)
        for name, error in current.items():
            errors[name] = max(errors[name], error)
        exact["J"] += int(prediction.image_score == arrays["J_reference"][index])
        exact["native"] += int(prediction.native.tobytes()
                               == arrays["native_reference"][units].tobytes())
        if not classification_only:
            exact["Q"] += int(prediction.anomaly_map.tobytes() == reference[index].tobytes())
        rows.append(dict(id=sample["id"], index=index, J=prediction.image_score,
                         G=prediction.baseline_score, maximum_errors=current))
    status = "passed" if max(errors.values()) <= tolerance else "failed"
    report = dict(status=status, key=key, count=len(rows), units=len(arrays["448_D0"]),
                  maximum_errors=errors, exact_counts=exact, tolerance=tolerance,
                  rows=rows, classification_only=classification_only,
                  manifest_sha256=sha256(manifest_path),
                  method="Two-scale margins recomputed to native/J/U/Q; official A explicitly supplied",
                  measured_metrics=manifest["measured_metrics"],
                  measured_metrics_source="Audited existing point result, not rescored in cache replay",
                  seconds=time.perf_counter() - started)
    write_json(Path(output_root) / (key.replace("/", "__") + ".json"), report)
    if status != "passed":
        raise ValueError(f"Cache replay numerical mismatch for {key}: {errors}")
    return report
