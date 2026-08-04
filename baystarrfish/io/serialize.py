"""JSON and array serialisation helpers with crash-safe writes.

Fits run for tens of hours; an interrupted write that leaves a complete-looking
file is worse than no file, because the next stage happily reads it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

__all__ = ["atomic_save_array", "input_fingerprint", "jsonable", "write_json"]


def jsonable(obj):
    """Recursively convert NumPy scalars and arrays to JSON-serialisable types."""
    if isinstance(obj, dict):
        return {key: jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(value) for value in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_json(path: Path | str, payload: dict) -> None:
    """Write ``payload`` sorted and indented, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_save_array(path: Path | str, array: np.ndarray) -> None:
    """Avoid leaving a complete-looking partial .npy after interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array)
    tmp.replace(path)


def input_fingerprint(path: Path | str) -> dict:
    """Cheap, reproducible identity record without hashing a multi-GB file.

    Size plus mtime is enough to detect that the input changed between a fit and
    a downstream stage, which is the failure this guards against; it is not a
    content hash and does not pretend to be.
    """
    path = Path(path).resolve()
    stat = path.stat()
    payload = f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "metadata_sha256": hashlib.sha256(payload).hexdigest(),
    }
