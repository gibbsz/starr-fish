#!/usr/bin/env python3
"""Build a non-genic cCRE-by-cell-type RNA signal matrix.

The script:

1. reads cCRE coordinates from ``STARRFISH_in_vivo/Data/CRE.bed`` (the
   project default) or from ``uns['CRE_info']`` in an h5ad;
2. removes every cCRE that overlaps a gene body in a supplied mm10
   GTF/GFF/BED annotation;
3. discovers only unstranded, unsmoothed ``*.RPKM.bw`` tracks in
   ``STARRFISH_in_vivo/Data/BICCN_RNA_bw``; and
4. writes the mean per-base RPKM signal over each retained cCRE as a
   cCRE-by-cell-type CSV matrix.

For GTF/GFF input, only rows whose feature is ``gene`` are used and the
1-based inclusive annotation coordinates are converted to BED coordinates.
BED input is assumed to contain gene-body intervals in standard 0-based,
half-open coordinates.

Example
-------
python revision/Data/Enhancer_RNA.py \
    --gene-annotation /path/to/gencode.vM25.annotation.gtf.gz
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
import tempfile
from bisect import bisect_left
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, TextIO

import numpy as np
import pandas as pd
import pyBigWig


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CRE_BED = REPO_ROOT / "STARRFISH_in_vivo" / "Data" / "CRE.bed"
DEFAULT_BIGWIG_DIR = (
    REPO_ROOT / "STARRFISH_in_vivo" / "Data" / "BICCN_RNA_bw"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "Enhancer_RNA.csv"

NEUTRAL_BIGWIG_RE = re.compile(r"^(?P<cell_type>.+)\.RPKM\.bw$")
STRANDED_MARKERS = (".forward-strand.", ".reverse-strand.")


@dataclass(frozen=True)
class CCRE:
    """A cCRE interval in 0-based, half-open coordinates."""

    identifier: str
    chrom: str
    start: int
    end: int
    category: str = ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exclude gene-body-overlapping cCREs and extract their mean RNA "
            "RPKM bigWig signal into a cCRE x cell-type matrix."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--cre-bed",
        type=Path,
        help=(
            "BED-like cCRE file. If neither cCRE source option is given, "
            f"use {DEFAULT_CRE_BED}."
        ),
    )
    source.add_argument(
        "--h5ad",
        type=Path,
        help=(
            "h5ad containing a coordinate table in uns['CRE_info']. The "
            "current revision h5ad has no such table, so CRE.bed is the "
            "project default."
        ),
    )
    parser.add_argument(
        "--cre-id-column",
        type=int,
        help=(
            "1-based cCRE ID column for --cre-bed. Auto-detection uses column "
            "5 when present, then column 4, then chrom:start-end."
        ),
    )
    parser.add_argument(
        "--gene-annotation",
        type=Path,
        required=True,
        help=(
            "mm10 gene annotation: GTF/GFF/GFF3 (optionally gzip-compressed), "
            "or a BED of gene-body intervals."
        ),
    )
    parser.add_argument(
        "--gene-feature",
        default="gene",
        help="Feature name selected from column 3 of GTF/GFF input.",
    )
    parser.add_argument(
        "--bigwig-dir",
        type=Path,
        default=DEFAULT_BIGWIG_DIR,
        help="Directory containing BICCN RNA bigWig tracks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output cCRE-by-cell-type CSV matrix.",
    )
    parser.add_argument(
        "--excluded-bed",
        type=Path,
        help=(
            "Optional BED file recording cCREs excluded due to a gene-body "
            "overlap."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel bigWig readers; use 1 for serial execution.",
    )
    parser.add_argument(
        "--float-format",
        default="%.6g",
        help="Floating-point format used in the output CSV.",
    )
    args = parser.parse_args(argv)

    if args.cre_bed is None and args.h5ad is None:
        args.cre_bed = DEFAULT_CRE_BED
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.cre_id_column is not None and args.cre_id_column < 1:
        parser.error("--cre-id-column is 1-based and must be at least 1")
    if args.h5ad is not None and args.cre_id_column is not None:
        parser.error("--cre-id-column applies only to --cre-bed")
    return args


def _open_text(path: Path) -> TextIO:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(mode="rt", encoding="utf-8")


def _validate_ccres(ccres: Iterable[CCRE], source: Path) -> list[CCRE]:
    result = list(ccres)
    if not result:
        raise ValueError(f"No valid cCRE coordinates were found in {source}")

    seen: set[str] = set()
    duplicates: set[str] = set()
    for cre in result:
        if cre.start < 0 or cre.end <= cre.start:
            raise ValueError(
                f"Invalid cCRE interval {cre.identifier}: "
                f"{cre.chrom}:{cre.start}-{cre.end}"
            )
        if cre.identifier in seen:
            duplicates.add(cre.identifier)
        seen.add(cre.identifier)
    if duplicates:
        preview = ", ".join(sorted(duplicates)[:10])
        raise ValueError(f"Duplicate cCRE IDs in {source}: {preview}")
    return result


def load_ccres_from_bed(path: Path, id_column: int | None = None) -> list[CCRE]:
    """Read a BED-like cCRE file while preserving its input row order."""

    if not path.is_file():
        raise FileNotFoundError(f"cCRE BED does not exist: {path}")

    ccres: list[CCRE] = []
    with _open_text(path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith(("#", "track ", "browser ")):
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                fields = line.split()
            if len(fields) < 3:
                raise ValueError(
                    f"{path}:{line_number} has fewer than three columns"
                )
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number} has non-integer BED coordinates"
                ) from exc

            if id_column is not None:
                index = id_column - 1
                if index >= len(fields):
                    raise ValueError(
                        f"{path}:{line_number} has no column {id_column}"
                    )
                identifier = fields[index]
            elif len(fields) >= 5:
                # CRE.bed stores its ID in column 5; column 4 is a category.
                identifier = fields[4]
            elif len(fields) >= 4:
                identifier = fields[3]
            else:
                identifier = f"{fields[0]}:{start}-{end}"

            category = fields[3] if len(fields) >= 4 else ""
            ccres.append(
                CCRE(
                    identifier=str(identifier),
                    chrom=fields[0],
                    start=start,
                    end=end,
                    category=category,
                )
            )
    return _validate_ccres(ccres, path)


def _first_matching_column(
    columns: Iterable[object], candidates: Sequence[str]
) -> object | None:
    by_lower = {str(column).lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def load_ccres_from_h5ad(path: Path) -> list[CCRE]:
    """Read cCRE coordinates from an h5ad ``uns['CRE_info']`` table."""

    if not path.is_file():
        raise FileNotFoundError(f"h5ad does not exist: {path}")
    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError("anndata is required when --h5ad is used") from exc

    adata = ad.read_h5ad(path, backed="r")
    try:
        if "CRE_info" not in adata.uns:
            raise ValueError(
                f"{path} has no uns['CRE_info'] coordinate table; "
                f"use --cre-bed {DEFAULT_CRE_BED}"
            )
        info = adata.uns["CRE_info"]
        if not isinstance(info, pd.DataFrame):
            info = pd.DataFrame(info)
        info = info.copy()
    finally:
        if adata.file is not None:
            adata.file.close()

    chrom_column = _first_matching_column(
        info.columns, ("Chromosome", "Chrom", "chrom", "chr")
    )
    start_column = _first_matching_column(
        info.columns, ("Start", "start", "chromStart")
    )
    end_column = _first_matching_column(
        info.columns, ("End", "end", "chromEnd")
    )
    coordinate_column = _first_matching_column(
        info.columns, ("enh", "coordinate", "coordinates", "region")
    )
    id_column = _first_matching_column(
        info.columns, ("CRE_ID", "cCRE", "cre_id", "name", "Name")
    )
    category_column = _first_matching_column(
        info.columns, ("labeling_type", "category", "type")
    )

    ccres: list[CCRE] = []
    for row_number, (index, row) in enumerate(info.iterrows(), start=1):
        identifier = str(row[id_column]) if id_column is not None else str(index)
        category = (
            str(row[category_column]) if category_column is not None else ""
        )

        try:
            if (
                chrom_column is not None
                and start_column is not None
                and end_column is not None
            ):
                chrom = str(row[chrom_column])
                start = int(row[start_column])
                end = int(row[end_column])
            elif coordinate_column is not None:
                match = re.fullmatch(
                    r"([^:]+):(\d+)[\-\N{MINUS SIGN}](\d+)",
                    str(row[coordinate_column]),
                )
                if match is None:
                    continue
                chrom, start_text, end_text = match.groups()
                start, end = int(start_text), int(end_text)
            else:
                raise ValueError(
                    "uns['CRE_info'] has neither Chrom/Start/End columns nor "
                    "a parseable coordinate column"
                )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid coordinate in {path} uns['CRE_info'] row {row_number}"
            ) from exc

        # Negative controls/blanks have no genomic interval and are skipped.
        if chrom in {"", "nan", "None"}:
            continue
        ccres.append(CCRE(identifier, chrom, start, end, category))
    return _validate_ccres(ccres, path)


def _annotation_format(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith((".gtf", ".gff", ".gff3")):
        return "gff"
    if name.endswith((".bed", ".bed3", ".bed6")):
        return "bed"
    raise ValueError(
        f"Cannot infer annotation format from {path}; expected "
        ".gtf[.gz], .gff[.gz], .gff3[.gz], or .bed[.gz]"
    )


def _normalize_chromosome(chrom: str, use_chr_prefix: bool) -> str:
    """Normalize common mouse chromosome labels to match the cCREs."""

    chrom = chrom.strip()
    if use_chr_prefix:
        if chrom == "MT":
            return "chrM"
        if not chrom.startswith("chr") and re.fullmatch(
            r"(?:[0-9]+|X|Y|M)", chrom
        ):
            return f"chr{chrom}"
    else:
        if chrom == "chrM":
            return "MT"
        if chrom.startswith("chr") and re.fullmatch(
            r"chr(?:[0-9]+|X|Y|M)", chrom
        ):
            return chrom[3:]
    return chrom


def load_gene_bodies(
    path: Path, cre_chromosomes: Iterable[str], feature: str = "gene"
) -> dict[str, list[tuple[int, int]]]:
    """Load and merge gene-body intervals by chromosome."""

    if not path.is_file():
        raise FileNotFoundError(f"Gene annotation does not exist: {path}")
    annotation_format = _annotation_format(path)
    cre_chromosomes = set(cre_chromosomes)
    use_chr_prefix = sum(c.startswith("chr") for c in cre_chromosomes) > (
        len(cre_chromosomes) / 2
    )
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    selected_rows = 0

    with _open_text(path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith(("#", "track ", "browser ")):
                continue
            fields = line.split("\t")
            if annotation_format == "gff":
                if len(fields) < 9:
                    raise ValueError(
                        f"{path}:{line_number} has fewer than nine GTF/GFF "
                        "columns"
                    )
                if fields[2] != feature:
                    continue
                try:
                    # GTF/GFF is 1-based inclusive; BED/bigWig is 0-based
                    # half-open.
                    start = int(fields[3]) - 1
                    end = int(fields[4])
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{line_number} has invalid GTF/GFF coordinates"
                    ) from exc
                chrom = fields[0]
            else:
                if len(fields) < 3:
                    fields = line.split()
                if len(fields) < 3:
                    raise ValueError(
                        f"{path}:{line_number} has fewer than three BED columns"
                    )
                try:
                    start = int(fields[1])
                    end = int(fields[2])
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{line_number} has invalid BED coordinates"
                    ) from exc
                chrom = fields[0]

            chrom = _normalize_chromosome(chrom, use_chr_prefix)
            if chrom not in cre_chromosomes:
                continue
            if start < 0 or end <= start:
                raise ValueError(
                    f"{path}:{line_number} has invalid interval "
                    f"{chrom}:{start}-{end}"
                )
            intervals[chrom].append((start, end))
            selected_rows += 1

    if selected_rows == 0:
        raise ValueError(
            f"No {feature!r} intervals in {path} matched the cCRE chromosomes. "
            "Check that the annotation is for mm10/GRCm38."
        )

    merged: dict[str, list[tuple[int, int]]] = {}
    for chrom, chrom_intervals in intervals.items():
        chrom_intervals.sort()
        merged_intervals: list[list[int]] = []
        for start, end in chrom_intervals:
            if not merged_intervals or start > merged_intervals[-1][1]:
                merged_intervals.append([start, end])
            elif end > merged_intervals[-1][1]:
                merged_intervals[-1][1] = end
        merged[chrom] = [(start, end) for start, end in merged_intervals]
    return merged


def split_gene_body_overlaps(
    ccres: Sequence[CCRE],
    gene_bodies: dict[str, list[tuple[int, int]]],
) -> tuple[list[CCRE], list[CCRE]]:
    """Split cCREs according to whether they overlap any gene-body base."""

    starts = {
        chrom: [start for start, _ in intervals]
        for chrom, intervals in gene_bodies.items()
    }
    retained: list[CCRE] = []
    excluded: list[CCRE] = []
    for cre in ccres:
        chrom_intervals = gene_bodies.get(cre.chrom, [])
        chrom_starts = starts.get(cre.chrom, [])
        candidate = bisect_left(chrom_starts, cre.end) - 1
        overlaps = (
            candidate >= 0
            and chrom_intervals[candidate][1] > cre.start
        )
        (excluded if overlaps else retained).append(cre)
    return retained, excluded


def discover_bigwigs(directory: Path) -> list[tuple[str, Path]]:
    """Return only neutral-strand, non-s300 RPKM bigWig tracks."""

    if not directory.is_dir():
        raise NotADirectoryError(f"bigWig directory does not exist: {directory}")

    result: list[tuple[str, Path]] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        name = path.name
        match = NEUTRAL_BIGWIG_RE.fullmatch(name)
        if match is None:
            continue
        if ".s300." in name or any(marker in name for marker in STRANDED_MARKERS):
            continue
        result.append((match.group("cell_type"), path))

    if not result:
        raise FileNotFoundError(
            f"No neutral-strand, non-s300 *.RPKM.bw tracks found in {directory}"
        )
    cell_types = [cell_type for cell_type, _ in result]
    if len(cell_types) != len(set(cell_types)):
        duplicates = sorted(
            name for name in set(cell_types) if cell_types.count(name) > 1
        )
        raise ValueError(f"Duplicate cell-type bigWigs: {duplicates[:10]}")
    return result


def mean_signal_for_bigwig(
    bigwig: tuple[str, Path], ccres: Sequence[CCRE]
) -> tuple[str, np.ndarray]:
    """Extract mean per-base signal for all cCREs from one bigWig."""

    cell_type, path = bigwig
    values = np.empty(len(ccres), dtype=np.float32)
    try:
        with pyBigWig.open(str(path)) as bw:
            chrom_sizes = bw.chroms()
            for index, cre in enumerate(ccres):
                if cre.chrom not in chrom_sizes:
                    raise ValueError(
                        f"{cre.chrom} is absent from {path.name} "
                        f"(needed by {cre.identifier})"
                    )
                if cre.end > chrom_sizes[cre.chrom]:
                    raise ValueError(
                        f"{cre.identifier} extends beyond {cre.chrom} in "
                        f"{path.name}: {cre.end} > {chrom_sizes[cre.chrom]}"
                    )
                signal_sum = bw.stats(
                    cre.chrom,
                    cre.start,
                    cre.end,
                    type="sum",
                    exact=True,
                )[0]
                # A missing bigWig block represents zero signal. Dividing the
                # exact sum by the full cCRE length therefore gives the mean
                # across all bases, including uncovered bases.
                values[index] = (
                    0.0
                    if signal_sum is None
                    else float(signal_sum) / (cre.end - cre.start)
                )
    except Exception as exc:
        raise RuntimeError(f"Failed while reading {path}") from exc
    return cell_type, values


def extract_matrix(
    bigwigs: Sequence[tuple[str, Path]],
    ccres: Sequence[CCRE],
    workers: int,
) -> pd.DataFrame:
    """Extract one matrix column per bigWig, in deterministic file order."""

    if not ccres:
        raise ValueError("No non-gene-body cCREs remain for signal extraction")

    values_by_cell_type: dict[str, np.ndarray] = {}
    total = len(bigwigs)
    report_every = max(1, total // 20)

    if workers == 1:
        for completed, bigwig in enumerate(bigwigs, start=1):
            cell_type, values = mean_signal_for_bigwig(bigwig, ccres)
            values_by_cell_type[cell_type] = values
            if completed == 1 or completed % report_every == 0 or completed == total:
                print(f"[bigWig] completed {completed}/{total}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, total)) as executor:
            futures = {
                executor.submit(mean_signal_for_bigwig, bigwig, ccres): bigwig
                for bigwig in bigwigs
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                cell_type, values = future.result()
                values_by_cell_type[cell_type] = values
                if (
                    completed == 1
                    or completed % report_every == 0
                    or completed == total
                ):
                    print(f"[bigWig] completed {completed}/{total}", flush=True)

    ordered = {
        cell_type: values_by_cell_type[cell_type]
        for cell_type, _ in bigwigs
    }
    matrix = pd.DataFrame(
        ordered,
        index=pd.Index([cre.identifier for cre in ccres], name="cCRE"),
    )
    return matrix


def write_matrix_atomic(
    matrix: pd.DataFrame, output: Path, float_format: str
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output_mode = (output.stat().st_mode & 0o777) if output.exists() else 0o664
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            matrix.to_csv(handle, float_format=float_format)
        temp_path.chmod(output_mode)
        os.replace(temp_path, output)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def write_excluded_bed(ccres: Sequence[CCRE], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for cre in ccres:
            fields = (
                cre.chrom,
                str(cre.start),
                str(cre.end),
                cre.identifier,
                cre.category,
            )
            handle.write("\t".join(fields).rstrip("\t") + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.h5ad is not None:
            print(f"[cCRE] reading {args.h5ad}", flush=True)
            ccres = load_ccres_from_h5ad(args.h5ad)
        else:
            print(f"[cCRE] reading {args.cre_bed}", flush=True)
            ccres = load_ccres_from_bed(args.cre_bed, args.cre_id_column)
        print(f"[cCRE] loaded {len(ccres):,} intervals", flush=True)

        print(f"[genes] reading {args.gene_annotation}", flush=True)
        gene_bodies = load_gene_bodies(
            args.gene_annotation,
            (cre.chrom for cre in ccres),
            feature=args.gene_feature,
        )
        retained, excluded = split_gene_body_overlaps(ccres, gene_bodies)
        print(
            f"[genes] excluded {len(excluded):,} gene-body-overlapping cCREs; "
            f"retained {len(retained):,}",
            flush=True,
        )
        if args.excluded_bed is not None:
            write_excluded_bed(excluded, args.excluded_bed)
            print(f"[genes] wrote {args.excluded_bed}", flush=True)

        bigwigs = discover_bigwigs(args.bigwig_dir)
        print(
            f"[bigWig] found {len(bigwigs):,} neutral-strand, non-s300 tracks",
            flush=True,
        )
        matrix = extract_matrix(bigwigs, retained, args.workers)
        write_matrix_atomic(matrix, args.output, args.float_format)
        print(
            f"[output] wrote {args.output} with shape "
            f"{matrix.shape[0]:,} x {matrix.shape[1]:,}",
            flush=True,
        )
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
