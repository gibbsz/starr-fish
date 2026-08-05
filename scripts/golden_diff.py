#!/usr/bin/env python
"""Prove the baystarrfish refactor is numerically inert.

Fits the same model twice -- once from the commit where
``bayesian_hierarchical.py`` is byte-identical to the version that produced every
published result, and once from HEAD -- then compares the activity table, the
global parameter draws and the ``log_gamma`` draws.

Why a tolerance and not bit-equality
------------------------------------
SVI is deterministic given a seed, but GPU float reductions are not associative
and their order depends on the physical device and allocation. Refitting the full
production model on a different L40S reproduced ``gamma.csv`` to 1e-12 relative
and the scalar draws to 1e-13 -- same code, same seed, same data. Demanding
bit-equality would fail on hardware differences, while a real sample-site-order
or rng change moves things by orders of magnitude. Integer columns and the
``prior_dominated`` flag are still compared exactly: nothing may reclassify them.

Usage
-----
::

    python scripts/golden_diff.py                      # reduced scale, ~minutes
    python scripts/golden_diff.py --full               # the real 3.3 GB input
    python scripts/golden_diff.py --compare-only A B   # two existing fit dirs

A full production fit is ~13 minutes on an L40S; reduced scale exists to avoid
needing the 3.3 GB input, not to save hours.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "79d7aea"  # "Move the Bayesian model into a root package"
TAG = "subclass_joint_copy_number_dropout_svi"
RTOL = 1e-9
BAYES_JAX = Path("/gpfs/commons/home/guojiezhong/miniconda3/envs/bayes-jax/bin/python")

REDUCED = ["--max-cells", "20000", "--max-cres", "40", "--steps", "800",
           "--num-posterior", "50"]
COMMON = ["--level", "subclass", "--channel", "joint",
          "--infection-model", "copy_number_dropout",
          "--activity-model", "direct", "--negative-control-mode", "ordinary",
          "--seed", "0"]


def compare(pre: Path, post: Path, tag: str = TAG) -> None:
    """Raise unless the two fit directories agree within tolerance."""
    a = pd.read_csv(pre / f"{tag}_gamma.csv")
    b = pd.read_csv(post / f"{tag}_gamma.csv")
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        raise SystemExit(f"gamma.csv shape/columns differ: {a.shape} vs {b.shape}")

    worst = 0.0
    for column in a.columns:
        x, y = a[column], b[column]
        if pd.api.types.is_float_dtype(x):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            rel = np.abs(xv - yv) / np.maximum(np.abs(xv), 1e-300)
            worst = max(worst, float(np.nanmax(rel)))
            if np.nanmax(rel) >= RTOL:
                raise SystemExit(f"{column}: relative drift {np.nanmax(rel):.3e}")
        elif not (x == y).all():
            raise SystemExit(f"{column}: exact-valued column changed")
    print(f"  gamma.csv           max relative drift {worst:.2e}   "
          f"({len(a):,} rows x {a.shape[1]} cols; integer columns exact)")

    with np.load(pre / f"{tag}_scalar_samples.npz") as x, \
         np.load(post / f"{tag}_scalar_samples.npz") as y:
        if set(x.files) != set(y.files):
            raise SystemExit(f"scalar sites differ: {sorted(x.files)} vs {sorted(y.files)}")
        worst = max(
            float(np.nanmax(np.abs(x[k] - y[k]) / np.maximum(np.abs(x[k]), 1e-300)))
            for k in x.files
        )
        n_sites = len(x.files)
    if worst >= RTOL:
        raise SystemExit(f"scalar draws drifted by {worst:.3e}")
    print(f"  scalar_samples.npz  max relative drift {worst:.2e}   ({n_sites} sites)")

    with np.load(pre / f"{tag}_posterior_samples.npz", allow_pickle=True) as x, \
         np.load(post / f"{tag}_posterior_samples.npz", allow_pickle=True) as y:
        drift = float(np.nanmax(np.abs(x["log_gamma"] - y["log_gamma"])))
        shape = x["log_gamma"].shape
    if drift >= 1e-4:  # stored as float32
        raise SystemExit(f"log_gamma draws drifted by {drift:.3e}")
    print(f"  log_gamma draws     max abs drift {drift:.2e}   {shape} (float32)")

    print(f"\nGOLDEN DIFF CLEAN -- agreement to <{RTOL:g} relative; "
          "the refactor is numerically inert.")


def fit(python: Path, cwd: Path, outdir: Path, extra: list[str]) -> None:
    subprocess.run(
        [str(python), "revision/bayesian_vs_fold_change/code/run_bayes.py",
         *COMMON, *extra, "--outdir", str(outdir)],
        cwd=cwd, check=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--compare-only", nargs=2, metavar=("PRE", "POST"), type=Path,
                        help="compare two existing fit directories and exit")
    parser.add_argument("--full", action="store_true",
                        help="fit at production scale instead of the reduced smoke scale")
    parser.add_argument("--outdir", type=Path, default=Path("/scratch/baystarrfish_golden"))
    parser.add_argument("--python", type=Path, default=BAYES_JAX)
    parser.add_argument("--tag", default=TAG)
    args = parser.parse_args(argv)

    if args.compare_only:
        compare(*args.compare_only, tag=args.tag)
        return 0

    scale = [] if args.full else REDUCED
    worktree = args.outdir / "baseline_checkout"
    args.outdir.mkdir(parents=True, exist_ok=True)
    if not worktree.exists():
        print(f"[golden] creating worktree at {BASELINE_COMMIT}")
        subprocess.run(["git", "-C", str(REPO), "worktree", "add", "--detach",
                        str(worktree), BASELINE_COMMIT], check=True)

    print(f"[golden] fitting PRE  (worktree @ {BASELINE_COMMIT})")
    fit(args.python, worktree, args.outdir / "pre", scale)
    print("[golden] fitting POST (HEAD)")
    fit(args.python, REPO, args.outdir / "post", scale)

    compare(args.outdir / "pre", args.outdir / "post", tag=args.tag)
    print(f"\n[golden] reclaim the baseline worktree with:\n"
          f"   git worktree remove {worktree}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
