# SPDX-License-Identifier: Apache-2.0
"""Portable manifests, checksums, and atomically completed prediction files."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def read_json(path: str | Path) -> dict:
    """Read a UTF-8 JSON object from an explicit path.

    Args:
        path: Existing JSON file.

    Returns:
        Parsed JSON object.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: dict) -> None:
    """Atomically publish finite JSON metadata.

    Args:
        path: Destination file.
        value: JSON-serializable metadata with finite numeric values.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def sha256(path: str | Path) -> str:
    """Hash file bytes using bounded memory.

    Args:
        path: Existing file.

    Returns:
        Lowercase SHA256 hex digest.
    """
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def save_prediction(path: str | Path, prediction, identity: dict | None = None) -> None:
    """Publish an NPZ prediction followed by its content-bound completion marker.

    Args:
        path: New output NPZ path. Existing completed outputs are preserved.
        prediction: amt1.core.Prediction returned by the predictor.
        identity: Optional canonical sample ID, index, and manifest SHA256.

    Raises:
        FileExistsError: The prediction destination already exists.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    arrays = dict(J=np.asarray(prediction.image_score), G=np.asarray(prediction.baseline_score),
                  native=prediction.native)
    for name, value in (("A", prediction.baseline_map), ("U", prediction.unanchored_map),
                        ("Q", prediction.anomaly_map)):
        if value is not None:
            arrays[name] = value
    try:
        with temporary.open("xb") as stream:
            np.savez(stream, **arrays)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(destination.with_suffix(".complete.json"),
               dict(status="completed", sha256=sha256(destination),
                    bytes=destination.stat().st_size, identity=identity))


def checked_relative(root: str | Path, relative: str) -> Path:
    """Resolve a portable relative path while enforcing the supplied root.

    Args:
        root: Explicit dataset or artifact root.
        relative: Root-relative file name.

    Returns:
        Absolute resolved path within root.

    Raises:
        ValueError: A path escapes root or is absolute.
    """
    base = Path(root).resolve()
    path = Path(relative)
    candidate = (base / path).resolve()
    if path.is_absolute() or not candidate.is_relative_to(base):
        raise ValueError(f"Expected a relative path within the supplied root: {relative!r}")
    return candidate
