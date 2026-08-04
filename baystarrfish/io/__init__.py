"""Serialisation of fits and their provenance.

Free of JAX and NumPyro, like :mod:`baystarrfish.data`, so the plotting
environment can read a fit produced on the GPU node.
"""

from __future__ import annotations

from .results import (
    decode_strings,
    fit_tag,
    load_gamma,
    load_posterior_samples,
    read_fit,
    write_fit,
)
from .serialize import atomic_save_array, input_fingerprint, jsonable, write_json

__all__ = [
    "atomic_save_array",
    "decode_strings",
    "fit_tag",
    "input_fingerprint",
    "jsonable",
    "load_gamma",
    "load_posterior_samples",
    "read_fit",
    "write_fit",
    "write_json",
]
