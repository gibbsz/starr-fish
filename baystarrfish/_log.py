"""Unbuffered progress printing.

Fits run for tens of hours under slurm with stdout redirected to a file; without
``flush=True`` the log only appears when the job ends, which is exactly when it
stops being useful.
"""

from __future__ import annotations

__all__ = ["log"]


def log(message: str) -> None:
    print(message, flush=True)
