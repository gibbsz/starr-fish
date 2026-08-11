"""One normalised-activity spatial map per cCRE in the published heatmap.

    python revision/Bayes_OldData/visualization/plot_all_ccre_spatial.py

For every cCRE shown in
``results/figures/final/joint_dropout_activity_heatmap_t7_ge50_joint_plus_dropout.pdf``
this draws the Gamma-conjugate per-cell activity normalised by the cell type's
negative-control mean, and uses the same tables the heatmap and its calls came
from to decide what to show:

* **excluded** -- cell types where the quantity is not defined for that cCRE. The
  cells stay in the grey background layer, so the section outline is complete;
  only their *value* is withheld, and they do not set the colour scale;
* **highlighted** -- cell types called significant, each given its own colour on
  top of the value map;
* everything else keeps the default value ramp.

What "not defined" means here
-----------------------------
The heatmap's own criterion is the target T7 >= 50 filter, which is *weaker* than
what a normalised activity needs: dividing by a control reference also requires
the cell type's pooled control T7 to clear 50. 59 of the heatmap's 103 subclasses
have defined activities but no reference, so their normalised value is NaN. By
default they are excluded along with the heatmap's own NaNs; either way their
cells remain visible as grey tissue, so this only decides whether a value is
claimed for them. ``--keep-unreferenced`` restores the literal heatmap rule.

Cost
----
The copy-number reconstruction runs **once** over every cCRE the heatmap needs
plus the 7 controls (~219 of 389 columns), not once per figure. Rendering then
dominates: roughly 10-20 s per figure at 408,621 cells, so a full sweep of 212
cCREs is 1-2 hours. Use ``--limit`` or ``--cres`` for a smoke test first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from baystarrfish._log import log
from baystarrfish.data import CountData
from baystarrfish.inference.copy_number import infer_copy_number_from_fit
from baystarrfish.plotting import plot_spatial

REPO = Path(__file__).resolve().parents[3]
TABLES = REPO / "revision/bayesian_vs_fold_change/results/tables"
HEATMAP_VALUES = TABLES / "joint_dropout_activity_heatmap_t7_ge50_joint_plus_dropout_values.csv"
TESTS = TABLES / "joint_dropout_direct_activity_mean_negative_control_tests_t7_ge50.csv.gz"
FIT_DIR = REPO / "revision/Bayes_OldData/bayesian"
OUTDIR = REPO / "revision/Bayes_OldData/visualization/figures"

MODE = "activity_posterior_normalized"


def _load_layout(
    values_path: Path, tests_path: Path, significance: str
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """``(defined, significant, referenced)`` -- two subclass x cCRE masks and a set.

    ``defined`` is the heatmap's own non-NaN pattern; ``significant`` marks the
    calls; ``referenced`` is the set of cell types that had a usable
    negative-control reference, which is exactly the set the test table covers.
    """
    values = pd.read_csv(values_path).set_index("subclass")
    defined = values.notna()

    tests = pd.read_csv(tests_path)
    if significance not in tests.columns:
        raise KeyError(
            f"{tests_path.name} has no column {significance!r}; it has "
            f"{[c for c in tests.columns if 'signif' in c or c.startswith('q')]}"
        )
    flag = tests[significance]
    if flag.dtype != bool:
        flag = flag.astype(float) < 0.05
    # Pivot on int, not bool: an object-dtype frame makes fillna downcast, which
    # pandas warns about and will change behaviour on.
    significant = (
        tests.assign(_hit=flag.astype(int))
        .pivot_table(index="group", columns="cre", values="_hit", aggfunc="max")
        .reindex(index=defined.index, columns=defined.columns)
        .fillna(0)
        .astype(bool)
    )
    return defined, significant, set(tests["group"].astype(str))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fit-dir", type=Path, default=FIT_DIR)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--heatmap-values", type=Path, default=HEATMAP_VALUES)
    parser.add_argument("--tests", type=Path, default=TESTS)
    parser.add_argument("--significance", default="significant_q",
                        help="boolean column of the tests table marking calls; a "
                             "numeric column is thresholded at 0.05")
    parser.add_argument("--cres", nargs="+", default=None,
                        help="only these cCREs (default: every one in the heatmap)")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many figures -- for a smoke test")
    parser.add_argument("--keep-unreferenced", action="store_true",
                        help="exclude only the heatmap's own NaNs, leaving cell "
                             "types that have no control reference in the figure "
                             "(they render grey, since their value is NaN)")
    parser.add_argument("--max-draws", type=int, default=200)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"],
                        choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--no-log", action="store_true",
                        help="linear colour ramp; the default is log1p because "
                             "the fold change spans several orders of magnitude")
    parser.add_argument("--overwrite", action="store_true",
                        help="redraw cCREs whose figures already exist")
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    defined, significant, referenced = _load_layout(
        args.heatmap_values, args.tests, args.significance
    )
    log(f"[sweep] heatmap: {defined.shape[0]} subclasses x {defined.shape[1]} cCREs, "
        f"{int(defined.to_numpy().sum()):,} defined cells; "
        f"{len(referenced)} subclasses have a control reference")

    manifest = json.loads((args.fit_dir / "run_manifest.json").read_text())
    config = manifest.get("config", {})
    controls = [str(c) for c in config.get("annotated_negative_control_cre", [])]

    data = CountData.from_h5ad(
        Path(config["input"]["path"]),
        section=manifest.get("section", "all"),
        negative_control_mode=manifest.get("negative_control_mode", "ordinary"),
    )
    present = {str(name) for name in data.cre_names}
    wanted = [c for c in defined.columns if c in present]
    dropped = sorted(set(defined.columns) - present)
    if dropped:
        log(f"[sweep] {len(dropped)} heatmap cCRE(s) absent from the data, skipped: "
            f"{dropped[:5]}")
    if args.cres:
        unknown = sorted(set(args.cres) - set(wanted))
        if unknown:
            raise SystemExit(f"cCRE(s) {unknown} are not in the heatmap or the data")
        wanted = [c for c in wanted if c in set(args.cres)]

    # One reconstruction for every column the sweep needs, rather than one per
    # figure: the h5ad load and the posterior pass are the expensive parts and
    # neither depends on which cCRE is being drawn.
    columns = wanted + [c for c in controls if c not in wanted]
    import dataclasses

    names = [str(n) for n in data.cre_names]
    index = np.array([names.index(c) for c in columns], dtype=np.int64)
    subset = dataclasses.replace(
        data,
        t7=np.asarray(data.t7)[:, index],
        cre=np.asarray(data.cre)[:, index],
        lib_size_log=np.asarray(data.lib_size_log)[index],
        cre_names=list(columns),
        negative_control_mask=(
            None if data.negative_control_mask is None
            else np.asarray(data.negative_control_mask)[index]
        ),
    )
    log(f"[sweep] reconstructing {len(columns)} of {len(names)} cCRE columns "
        f"({len(wanted)} to draw + {len(columns) - len(wanted)} controls)")
    matrix = infer_copy_number_from_fit(
        subset, args.fit_dir, return_activity_normalized=True,
        max_draws=args.max_draws,
    )

    all_subclasses = set(np.unique(np.asarray(subset.subclass).astype(str)))
    args.outdir.mkdir(parents=True, exist_ok=True)
    records = []
    drawn = 0
    for cre in wanted:
        column = defined[cre]
        keep = set(column.index[column.to_numpy()].astype(str))
        if not args.keep_unreferenced:
            keep &= referenced
        keep &= all_subclasses
        excluded = sorted(all_subclasses - keep)
        highlight = sorted(
            set(significant.index[significant[cre].to_numpy()].astype(str)) & keep
        )

        stem = f"{cre}_{MODE}_spatial"
        outputs = [args.outdir / f"{stem}.{ext}" for ext in args.formats]
        record = {
            "cre": cre, "n_celltypes_kept": len(keep),
            "n_celltypes_excluded": len(excluded),
            "n_celltypes_highlighted": len(highlight),
            "highlighted": ",".join(highlight),
        }
        if not keep:
            log(f"[sweep] {cre}: no cell type has a defined normalised activity, skipped")
            records.append({**record, "status": "skipped_undefined"})
            continue
        if not args.overwrite and all(p.exists() for p in outputs):
            records.append({**record, "status": "existing"})
            continue
        if len(highlight) > 20:
            log(f"[sweep] {cre}: {len(highlight)} significant cell types -- the "
                "palette cycles after 20, so hues repeat")

        fig = plot_spatial(
            subset, MODE, cre=cre, activity=matrix,
            celltypes=highlight or None,
            exclude_celltypes=excluded or None,
            log=not args.no_log,
        )
        for path in outputs:
            fig.savefig(path, dpi=args.dpi, facecolor="black", bbox_inches="tight")
        plt.close(fig)
        drawn += 1
        records.append({**record, "status": "drawn"})
        log(f"[sweep] {cre}: {len(keep)} cell types kept, {len(highlight)} highlighted "
            f"-> {outputs[0].name}")
        if args.limit is not None and drawn >= args.limit:
            log(f"[sweep] stopping at --limit {args.limit}")
            break

    summary = args.outdir / "sweep_manifest.csv"
    pd.DataFrame(records).to_csv(summary, index=False)
    log(f"[sweep] {drawn} figure(s) drawn; manifest at {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
