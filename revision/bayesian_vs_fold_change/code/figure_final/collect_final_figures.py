#!/usr/bin/env python3
"""Assemble the curated manuscript figure set from the working figure directory.

Every plotting script writes into ``results/figures/work/``. This script owns the
definition of what counts as final and copies exactly that set into
``results/figures/final/``. Nothing else should write to the final directory, so
the curated set can always be rebuilt from a fresh run plus this list.

Each entry names a PDF. Alongside it, any file sharing the PDF's stem is treated
as a sidecar and copied too, searched in both the working figure directory (that
is how the ``*_pairs.csv`` tables travel with the stripe-count figures) and
``results/tables/`` (where ``--dump-values`` writes the plotted heatmap matrix).
The producers' shared manifests are copied separately for provenance.

Two of the figures below come from the standalone heatmap jobs in this directory
rather than from run_newnew_figures.sh; submit those before collecting:

    sbatch code/figure_final/submit_joint_dropout_activity_heatmap.slurm
    sbatch code/figure_final/submit_on_target_activity_heatmap.slurm
    python code/figure_final/collect_final_figures.py            # copy
    python code/figure_final/collect_final_figures.py --dry-run   # report only
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Final

# The shared analysis layer lives in the parent code/ directory.
_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from analysis_utils import (  # noqa: E402
    ANALYSIS_DIR,
    FIGURES_FINAL,
    FIGURES_WORK,
    log,
    write_json,
)

# The manuscript figure set. Adding a figure here is the only step needed to
# promote it; its sidecars follow automatically.
FINAL_FIGURES: Final[tuple[str, ...]] = (
    "method_activity_correlation_t7_gt50.pdf",
    "method_activity_correlation_cellgt1000.pdf",
    "method_activity_correlation_t7nanoporegt1000.pdf",
    "method_activity_stripe_count_diagnostics_horizontal_bayesian_minus0p5.pdf",
    "method_activity_stripe_count_diagnostics_vertical_bootstrap_0.pdf",
    "method_activity_t7_filter_pr_curves_t7_ge50.pdf",
    "method_activity_t7_filter_precision_recall.pdf",
    "joint_dropout_activity_heatmap_t7_ge50_joint_plus_dropout.pdf",
    "joint_dropout_activity_heatmap_on_target_t7_ge50_joint_plus_dropout.pdf",
)

# Run manifests of the producing scripts. These record the input fits and
# thresholds behind the figures above, so they belong with the curated set even
# though no single figure owns them. Their names are not prefixes of any figure
# stem, so they cannot be picked up by the sidecar scan.
PROVENANCE_MANIFESTS: Final[tuple[str, ...]] = (
    "method_activity_correlation_manifest.json",
    "method_activity_stripe_count_diagnostics_manifest.json",
    "method_activity_t7_filter_precision_recall_manifest.json",
    "joint_dropout_activity_heatmap_manifest.json",
    "joint_dropout_activity_heatmap_on_target_manifest.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=FIGURES_WORK)
    parser.add_argument("--final-dir", type=Path, default=FIGURES_FINAL)
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=ANALYSIS_DIR / "results" / "tables",
        help="Also searched for sidecars sharing a final figure's stem.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be copied without writing anything.",
    )
    return parser.parse_args()


def sidecars(search_dirs: list[Path], figure: str) -> list[Path]:
    """Files sharing the figure's stem, excluding the figure itself."""
    stem = Path(figure).stem
    found: list[Path] = []
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        found.extend(
            sorted(
                path
                for path in directory.iterdir()
                if path.is_file()
                and path.name != figure
                and path.name.startswith(stem)
            )
        )
    return found


def resolve(
    work_dir: Path, search_dirs: list[Path]
) -> tuple[list[Path], list[str]]:
    """Return the files to copy and the names of any declared finals missing."""
    selected: list[Path] = []
    missing: list[str] = []
    for figure in FINAL_FIGURES:
        source = work_dir / figure
        if not source.is_file():
            missing.append(figure)
            continue
        selected.append(source)
        selected.extend(sidecars(search_dirs, figure))
    for manifest in PROVENANCE_MANIFESTS:
        path = work_dir / manifest
        if path.is_file():
            selected.append(path)
        else:
            log(f"[final] provenance manifest absent, skipping: {manifest}")
    # dedupe on filename while preserving order: a sidecar can also be a
    # declared manifest, and the two search directories can overlap.
    seen: set[str] = set()
    unique = [p for p in selected if not (p.name in seen or seen.add(p.name))]
    return unique, missing


def main() -> None:
    args = parse_args()
    if not args.work_dir.is_dir():
        raise SystemExit(f"working figure directory not found: {args.work_dir}")

    search_dirs = [args.work_dir, args.tables_dir]
    selected, missing = resolve(args.work_dir, search_dirs)
    if missing:
        # A silently absent final figure is the failure mode worth being loud
        # about: the curated set would look complete but not be. The two heatmap
        # figures come from standalone jobs, so name them in the hint.
        raise SystemExit(
            f"declared final figures missing from {args.work_dir}:\n  "
            + "\n  ".join(missing)
            + "\n\nThe joint_dropout_activity_heatmap* figures come from the "
            "standalone jobs in code/figure_final/; submit those first."
        )

    if args.dry_run:
        for path in selected:
            log(f"[final] would copy {path.name}  (from {path.parent.name}/)")
        log(f"[final] {len(selected)} files ({len(FINAL_FIGURES)} figures + sidecars)")
        return

    args.final_dir.mkdir(parents=True, exist_ok=True)
    for path in selected:
        shutil.copy2(path, args.final_dir / path.name)
        log(f"[final] {path.name}")

    write_json(
        args.final_dir / "final_figure_set_manifest.json",
        {
            "work_dir": str(args.work_dir),
            "tables_dir": str(args.tables_dir),
            "figures": list(FINAL_FIGURES),
            "provenance_manifests": list(PROVENANCE_MANIFESTS),
            "copied": [p.name for p in selected],
            "n_copied": len(selected),
        },
    )
    log(
        f"[final] wrote {len(selected)} files to {args.final_dir} "
        f"({len(FINAL_FIGURES)} figures + sidecars)"
    )


if __name__ == "__main__":
    main()
