#!/usr/bin/env python3
"""Compare HOCOMOCO motifs in significant versus non-significant cCREs.

The input is the whole-dataset joint-dropout mean-negative-control test table
from ``revision/bayesian_vs_fold_change``.  Each cell type's significant cCREs
are compared with the other cCREs that were eligible for the same test in that
cell type.  The assayed 200-bp cCRE inserts are scanned against HOCOMOCO v14
CORE with FIMO.  Each cell type uses a separate zero-order DNA background
estimated from all of its finite-q cCREs (significant and non-significant
together).  Motifs retain their HOCOMOCO mouse gene annotations and, when the
TF is in the expression panel, cell-type mean expression and detection
fraction.  Enrichment uses a one-sided Fisher exact test for binary motif
presence and reports both within-cell-type and global BH FDR.  The workflow
also exports cell-type by motif matrices formed by multiplying each valid
cCRE's posterior mean effect versus negative controls by its FIMO score and by
correlating those two quantities over all T7>=50 cCREs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path

for env_name, dirname in (
    ("MPLCONFIGDIR", "matplotlib-starrfish-tf-motif"),
    ("XDG_CACHE_HOME", "xdg-cache-starrfish-tf-motif"),
):
    if env_name in os.environ:
        continue
    for cache_root in (os.environ.get("TMPDIR"), tempfile.gettempdir(), "/tmp"):
        if not cache_root:
            continue
        cache_dir = Path(cache_root) / dirname
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        os.environ[env_name] = str(cache_dir)
        break

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.stats import fisher_exact, rankdata, t as student_t


HERE = Path(__file__).resolve()
ANALYSIS_DIR = HERE.parents[1]
REPO = HERE.parents[3]

DEFAULT_TESTS = (
    REPO
    / "revision"
    / "bayesian_vs_fold_change"
    / "results"
    / "tables"
    / "joint_dropout_mean_negative_control_tests.csv.gz"
)
DEFAULT_CRE_INFO = REPO / "STARRFISH_in_vivo" / "results" / "cre_info.csv"
DEFAULT_H5AD = (
    REPO
    / "revision"
    / "Data"
    / "scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE_NEWNEW.h5ad"
)
DEFAULT_HOCOMOCO_MEME = (
    ANALYSIS_DIR / "resources" / "H14CORE_meme_format.meme"
)
DEFAULT_HOCOMOCO_ANNOTATION = (
    ANALYSIS_DIR / "resources" / "H14CORE-MOUSE_annotation.jsonl"
)
DEFAULT_FIMO = Path(
    "/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN/bin/fimo"
)
DEFAULT_FASTA_GET_MARKOV = DEFAULT_FIMO.with_name("fasta-get-markov")
DEFAULT_RESULTS = ANALYSIS_DIR / "results"
EXPECTED_METHOD = "Joint+dropout mean controls"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", type=Path, default=DEFAULT_TESTS)
    parser.add_argument(
        "--cre-info",
        type=Path,
        default=DEFAULT_CRE_INFO,
        help="cCRE metadata containing the actual assayed insert sequences.",
    )
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--hocomoco-meme", type=Path, default=DEFAULT_HOCOMOCO_MEME)
    parser.add_argument(
        "--hocomoco-annotation",
        type=Path,
        default=DEFAULT_HOCOMOCO_ANNOTATION,
    )
    parser.add_argument("--fimo-bin", type=Path, default=DEFAULT_FIMO)
    parser.add_argument(
        "--fasta-get-markov-bin",
        type=Path,
        default=DEFAULT_FASTA_GET_MARKOV,
        help=(
            "MEME Suite utility used to estimate a separate zero-order DNA "
            "background from all finite-q cCRE sequences in each cell type."
        ),
    )
    parser.add_argument("--fimo-pvalue", type=float, default=1e-4)
    parser.add_argument(
        "--fimo-jobs",
        type=int,
        default=2,
        help="Number of cell-type FIMO scans to run concurrently.",
    )
    parser.add_argument(
        "--force-rescan",
        action="store_true",
        help="Ignore a compatible cached FIMO scan and rescan all tested cCREs.",
    )
    parser.add_argument(
        "--skip-expression",
        action="store_true",
        help="Do not aggregate HOCOMOCO TF expression from the h5ad.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--expected-method",
        default=EXPECTED_METHOD,
        help="Refuse to analyze a table with a different method label; use an empty string to disable.",
    )
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument(
        "--motif-q-cutoff",
        type=float,
        default=0.05,
        help="Within-cell-type motif FDR used for summaries and figures.",
    )
    parser.add_argument("--min-significant-ccres", type=int, default=1)
    parser.add_argument("--min-nonsignificant-ccres", type=int, default=1)
    parser.add_argument("--top-n-per-cell-type", type=int, default=10)
    parser.add_argument("--plot-top-n", type=int, default=40)
    parser.add_argument("--heatmap-top-motifs", type=int, default=20)
    parser.add_argument("--activity-heatmap-top-motifs", type=int, default=100)
    parser.add_argument(
        "--activity-heatmap-min-valid-ccres",
        type=int,
        default=20,
    )
    parser.add_argument("--expression-chunk-size", type=int, default=50_000)
    return parser.parse_args()


def bh_fdr(p_values: pd.Series | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving NaNs."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return adjusted

    p = values[valid]
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    ranked_q = ranked * n / np.arange(1, n + 1)
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    ranked_q = np.clip(ranked_q, 0.0, 1.0)
    q = np.empty_like(ranked_q)
    q[order] = ranked_q
    adjusted[valid] = q
    return adjusted


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tests(path: Path, q_cutoff: float, expected_method: str) -> pd.DataFrame:
    tests = pd.read_csv(path)
    required = {
        "method",
        "group",
        "class",
        "cre",
        "q_right",
        "target_t7_total",
        "effect_vs_mean_control_mean",
    }
    missing = sorted(required - set(tests.columns))
    if missing:
        raise ValueError(f"Test table is missing required columns: {missing}")

    tests = tests.copy()
    for column in ("method", "group", "class", "cre"):
        tests[column] = tests[column].astype(str)
    tests["q_right"] = pd.to_numeric(tests["q_right"], errors="coerce")
    tests["target_t7_total"] = pd.to_numeric(
        tests["target_t7_total"], errors="coerce"
    )
    tests["effect_vs_mean_control_mean"] = pd.to_numeric(
        tests["effect_vs_mean_control_mean"], errors="coerce"
    )
    methods = sorted(tests["method"].dropna().unique())
    if expected_method and methods != [expected_method]:
        raise ValueError(
            f"Expected only method {expected_method!r}; found {methods}. "
            "Pass --expected-method '' only if this is intentional."
        )
    if tests.duplicated(["group", "cre"]).any():
        examples = tests.loc[
            tests.duplicated(["group", "cre"], keep=False), ["group", "cre"]
        ].head()
        raise ValueError(f"Expected one eligible test per cell type/cCRE; duplicates include:\n{examples}")

    tests["is_significant"] = tests["q_right"].le(q_cutoff)
    tests["has_finite_q"] = np.isfinite(tests["q_right"])
    return tests


def serialize_annotation_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def load_hocomoco_annotation(path: Path) -> pd.DataFrame:
    """Flatten HOCOMOCO mouse annotation into joinable motif/TF metadata."""
    rows = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            master = record["masterlist_info"]
            mouse = master["species"]["MOUSE"]
            rows.append(
                {
                    "motif": str(record["name"]),
                    "hocomoco_tf": str(record["tf"]),
                    "mouse_gene_symbol": str(mouse["gene_symbol"]),
                    "mouse_gene_synonyms": serialize_annotation_value(
                        mouse.get("gene_synonyms")
                    ),
                    "mouse_uniprot_id": serialize_annotation_value(
                        mouse.get("uniprot_id")
                    ),
                    "mouse_uniprot_ac": serialize_annotation_value(
                        mouse.get("uniprot_ac")
                    ),
                    "mouse_mgi": serialize_annotation_value(mouse.get("mgi")),
                    "mouse_entrez": serialize_annotation_value(mouse.get("entrez")),
                    "tfclass_id": serialize_annotation_value(master.get("tfclass_id")),
                    "tfclass_superclass": serialize_annotation_value(
                        master.get("tfclass_superclass")
                    ),
                    "tfclass_class": serialize_annotation_value(
                        master.get("tfclass_class")
                    ),
                    "tfclass_family": serialize_annotation_value(
                        master.get("tfclass_family")
                    ),
                    "tfclass_subfamily": serialize_annotation_value(
                        master.get("tfclass_subfamily")
                    ),
                    "hocomoco_collection": str(record["collection"]),
                    "motif_subtype": int(record["subtype_order"]),
                    "motif_datatype": str(record["datatype"]),
                    "motif_quality": str(record["quality"]),
                    "motif_length": int(record["length"]),
                    "motif_consensus": str(record["consensus"]),
                    "hocomoco_url": (
                        "https://hocomoco14.autosome.org/motif/"
                        + str(record["name"])
                    ),
                }
            )
    annotation = pd.DataFrame(rows)
    if annotation.empty:
        raise ValueError(f"No HOCOMOCO annotations found in {path}")
    if annotation["motif"].duplicated().any():
        raise ValueError("HOCOMOCO annotation contains duplicate motif identifiers")
    return annotation


def load_cre_info(path: Path) -> pd.DataFrame:
    cre_info = pd.read_csv(path, index_col=0)
    cre_info.index = cre_info.index.astype(str)
    if "sequence" not in cre_info:
        raise ValueError(f"{path} does not contain the assayed cCRE 'sequence' column")
    if cre_info.index.duplicated().any():
        raise ValueError("cCRE metadata contains duplicate identifiers")
    return cre_info


def valid_ccres_by_group(
    tests: pd.DataFrame,
    min_target_t7: float = 50.0,
) -> dict[str, list[str]]:
    """Return every finite-q, T7-eligible cCRE in deterministic order."""
    valid = tests.loc[
        tests["has_finite_q"] & tests["target_t7_total"].ge(min_target_t7),
        ["group", "cre"],
    ]
    return {
        str(group): sorted(frame["cre"].astype(str).unique())
        for group, frame in valid.groupby("group", sort=True)
        if not frame.empty
    }


def fimo_scan_signature(
    grouped_ccres: dict[str, list[str]],
    cre_info_path: Path,
    hocomoco_meme: Path,
    annotation_path: Path,
    fimo_pvalue: float,
) -> dict:
    grouped_text = "".join(
        f"{group}\t{cre}\n"
        for group, ccres in grouped_ccres.items()
        for cre in ccres
    )
    tested_digest = hashlib.sha256(
        grouped_text.encode("utf-8")
    ).hexdigest()
    return {
        "valid_cell_type_ccre_pairs_sha256": tested_digest,
        "n_valid_cell_type_ccre_pairs": sum(map(len, grouped_ccres.values())),
        "cre_info_sha256": sha256(cre_info_path),
        "hocomoco_meme_sha256": sha256(hocomoco_meme),
        "hocomoco_annotation_sha256": sha256(annotation_path),
        "fimo_pvalue": float(fimo_pvalue),
        "background_model": (
            "cell-type-specific zero-order DNA Markov model estimated by "
            "fasta-get-markov from all finite-q, target-T7>=50 cCREs in that "
            "cell type; "
            "reverse complements combined; pseudocount=0.1"
        ),
        "score": "-log10(best FIMO hit p-value); zero means no hit",
    }


def group_directory_name(group: str) -> str:
    """Create a readable, collision-resistant directory name for a cell type."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", group).strip("._-") or "cell_type"
    digest = hashlib.sha256(group.encode("utf-8")).hexdigest()[:10]
    return f"{stem[:80]}-{digest}"


def write_fasta(
    path: Path,
    ccres: list[str],
    cre_info: pd.DataFrame,
) -> pd.Series:
    sequences = cre_info.loc[ccres, "sequence"].fillna("").astype(str)
    bad = sequences.str.strip().eq("") | ~sequences.str.fullmatch(
        r"[ACGTNacgtn]+", na=False
    )
    if bad.any():
        raise ValueError(
            "Missing or invalid assayed sequences for cCREs: "
            + ", ".join(sequences.index[bad][:10])
        )
    with path.open("w") as handle:
        for cre, sequence in sequences.items():
            handle.write(f">{cre}\n{sequence.upper()}\n")
    return sequences.str.upper()


def read_zero_order_background(path: Path) -> dict[str, float]:
    """Read A/C/G/T probabilities from fasta-get-markov order-zero output."""
    frequencies: dict[str, float] = {}
    with path.open() as handle:
        for line in handle:
            fields = line.split()
            if len(fields) != 2 or fields[0] not in {"A", "C", "G", "T"}:
                continue
            frequencies[fields[0]] = float(fields[1])
    if set(frequencies) != {"A", "C", "G", "T"}:
        raise ValueError(f"Could not parse a DNA zero-order background from {path}")
    if not np.isclose(sum(frequencies.values()), 1.0, atol=1e-5):
        raise ValueError(f"Background frequencies in {path} do not sum to one")
    return frequencies


def parse_fimo_text(path: Path) -> pd.DataFrame:
    """Parse MEME Suite 4.x FIMO --text output, including an empty result."""
    names = [
        "motif_id",
        "cre",
        "start",
        "stop",
        "strand",
        "score",
        "p_value",
        "fimo_q_value",
        "matched_sequence",
    ]
    try:
        return pd.read_csv(
            path,
            sep="\t",
            comment="#",
            header=None,
            names=names,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=names)


def scan_hocomoco_cell_type(
    group: str,
    ccres: list[str],
    cre_info: pd.DataFrame,
    annotation: pd.DataFrame,
    scan_root: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict]:
    """Estimate one cell-type background and run FIMO on the same valid cCREs."""
    group_dir = scan_root / group_directory_name(group)
    group_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = group_dir / "all_valid_ccres.fa"
    background_path = group_dir / "all_valid_ccres.markov0.bg"
    raw_fimo_path = group_dir / "fimo_raw.tsv"
    sequences = write_fasta(fasta_path, ccres, cre_info)

    background_command = [
        str(args.fasta_get_markov_bin),
        "-m",
        "0",
        "-dna",
        "-pseudo",
        "0.1",
        "-nostatus",
        str(fasta_path),
        str(background_path),
    ]
    background_run = subprocess.run(
        background_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if background_run.returncode != 0:
        raise RuntimeError(
            f"fasta-get-markov failed for {group!r} with exit code "
            f"{background_run.returncode}:\n{background_run.stderr[-4000:]}"
        )
    frequencies = read_zero_order_background(background_path)

    command = [
        str(args.fimo_bin),
        "--text",
        "--verbosity",
        "1",
        "--thresh",
        str(args.fimo_pvalue),
        "--bgfile",
        str(background_path),
        str(args.hocomoco_meme),
        str(fasta_path),
    ]
    with raw_fimo_path.open("w") as output:
        completed = subprocess.run(
            command,
            stdout=output,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"FIMO failed for {group!r} with exit code {completed.returncode}:\n"
            f"{completed.stderr[-4000:]}"
        )

    hits = parse_fimo_text(raw_fimo_path)
    allowed_motifs = set(annotation["motif"])
    hits = hits.loc[hits["motif_id"].astype(str).isin(allowed_motifs)].copy()
    hits["group"] = group
    hits["cre"] = hits["cre"].astype(str)
    hits["motif_id"] = hits["motif_id"].astype(str)
    hits["p_value"] = pd.to_numeric(hits["p_value"], errors="coerce")
    hits["score"] = pd.to_numeric(hits["score"], errors="coerce")
    hits = hits.loc[
        hits["p_value"].notna() & hits["p_value"].le(args.fimo_pvalue)
    ].copy()
    hits["neg_log10_p_value"] = -np.log10(
        hits["p_value"].clip(lower=np.nextafter(0.0, 1.0))
    )
    hits = hits.sort_values(
        ["cre", "motif_id", "p_value", "score"],
        ascending=[True, True, True, False],
    ).drop_duplicates(["cre", "motif_id"], keep="first")

    modeled_bases = int(
        sum(sequence.count(base) for sequence in sequences for base in "ACGT")
    )
    total_bases = int(sequences.str.len().sum())
    background = {
        "group": group,
        "n_valid_ccres": len(ccres),
        "total_sequence_bases": total_bases,
        "modeled_acgt_bases": modeled_bases,
        "ambiguous_bases": total_bases - modeled_bases,
        "frequency_A": frequencies["A"],
        "frequency_C": frequencies["C"],
        "frequency_G": frequencies["G"],
        "frequency_T": frequencies["T"],
        "background_file": str(background_path.resolve()),
        "fasta_file": str(fasta_path.resolve()),
    }
    return hits, background


def scan_hocomoco(
    tests: pd.DataFrame,
    cre_info: pd.DataFrame,
    annotation: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Run group-specific FIMO scans with signature-checked combined caching."""
    intermediate = args.results_dir / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    scan_root = intermediate / "fimo_by_cell_type"
    hits_path = intermediate / "hocomoco_celltype_fimo_best_hits.csv.gz"
    matrix_path = intermediate / "hocomoco_celltype_motif_matrix.csv.gz"
    backgrounds_path = intermediate / "fimo_background_by_cell_type.csv"
    cache_manifest_path = intermediate / "hocomoco_celltype_scan_manifest.json"

    grouped_ccres = valid_ccres_by_group(tests)
    if not grouped_ccres:
        raise ValueError("No finite-q cCREs are available for FIMO scanning")
    signature = fimo_scan_signature(
        grouped_ccres,
        args.cre_info,
        args.hocomoco_meme,
        args.hocomoco_annotation,
        args.fimo_pvalue,
    )
    if (
        not args.force_rescan
        and matrix_path.exists()
        and hits_path.exists()
        and backgrounds_path.exists()
        and cache_manifest_path.exists()
    ):
        cached_signature = json.loads(cache_manifest_path.read_text())
        if cached_signature == signature:
            motif = pd.read_csv(matrix_path, index_col=[0, 1])
            motif.index = pd.MultiIndex.from_tuples(
                [(str(group), str(cre)) for group, cre in motif.index],
                names=["group", "cre"],
            )
            hits = pd.read_csv(hits_path)
            backgrounds = pd.read_csv(backgrounds_path)
            print(f"[HOCOMOCO] reused compatible FIMO cache: {matrix_path}")
            return motif, hits, backgrounds, signature

    tested_ccres = sorted(
        {cre for ccres in grouped_ccres.values() for cre in ccres}
    )
    missing = sorted(set(tested_ccres) - set(cre_info.index))
    if missing:
        raise ValueError(
            f"{len(missing)} tested cCREs are absent from cCRE metadata: {missing[:10]}"
        )
    if args.fimo_jobs < 1:
        raise ValueError("--fimo-jobs must be at least 1")
    if not args.fimo_bin.is_file():
        raise FileNotFoundError(f"FIMO executable not found: {args.fimo_bin}")
    if not args.fasta_get_markov_bin.is_file():
        raise FileNotFoundError(
            f"fasta-get-markov executable not found: {args.fasta_get_markov_bin}"
        )
    scan_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[HOCOMOCO] scanning {sum(map(len, grouped_ccres.values()))} valid "
        f"cell-type/cCRE pairs in {len(grouped_ccres)} cell types against "
        f"H14CORE at p<={args.fimo_pvalue:g}; each cell type uses its own "
        "all-valid-cCRE zero-order background",
        flush=True,
    )
    group_hits: list[pd.DataFrame] = []
    background_rows: list[dict] = []
    completed_count = 0
    with ThreadPoolExecutor(
        max_workers=min(args.fimo_jobs, len(grouped_ccres))
    ) as executor:
        futures = {
            executor.submit(
                scan_hocomoco_cell_type,
                group,
                ccres,
                cre_info,
                annotation,
                scan_root,
                args,
            ): group
            for group, ccres in grouped_ccres.items()
        }
        for future in as_completed(futures):
            group = futures[future]
            hits_for_group, background = future.result()
            group_hits.append(hits_for_group)
            background_rows.append(background)
            completed_count += 1
            print(
                f"[HOCOMOCO] completed {completed_count}/{len(grouped_ccres)}: "
                f"{group} ({len(grouped_ccres[group])} valid cCREs, "
                f"{len(hits_for_group)} best motif hits)",
                flush=True,
            )

    hits = pd.concat(group_hits, ignore_index=True)
    hits = hits.sort_values(["group", "cre", "motif_id"]).reset_index(drop=True)
    hits = hits.merge(
        annotation,
        left_on="motif_id",
        right_on="motif",
        how="left",
        validate="many_to_one",
    )
    hits.to_csv(hits_path, index=False)
    backgrounds = pd.DataFrame(background_rows).sort_values("group").reset_index(
        drop=True
    )
    backgrounds.to_csv(backgrounds_path, index=False)

    if hits.empty:
        raise ValueError(
            "FIMO found no HOCOMOCO mouse motif hits at the requested threshold"
        )
    motif = hits.pivot(
        index=["group", "cre"],
        columns="motif_id",
        values="neg_log10_p_value",
    ).fillna(0.0)
    expected_index = pd.MultiIndex.from_tuples(
        [
            (group, cre)
            for group, ccres in grouped_ccres.items()
            for cre in ccres
        ],
        names=["group", "cre"],
    )
    motif = motif.reindex(index=expected_index, fill_value=0.0)
    motif = motif.reindex(sorted(motif.columns), axis=1)
    motif.columns.name = None
    motif.to_csv(matrix_path)
    cache_manifest_path.write_text(
        json.dumps(signature, indent=2, sort_keys=True) + "\n"
    )
    return motif, hits, backgrounds, signature


def aggregate_tf_expression(
    h5ad_path: Path,
    groups: list[str],
    annotation: pd.DataFrame,
    chunk_size: int,
) -> tuple[pd.DataFrame, set[str]]:
    """Aggregate panel expression for HOCOMOCO TF genes by Bayesian cell type."""
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        var_names = pd.Index(adata.var_names.astype(str))
        hocomoco_genes = set(annotation["mouse_gene_symbol"].astype(str))
        gene_indices = np.flatnonzero(var_names.isin(hocomoco_genes))
        panel_genes = var_names[gene_indices].tolist()
        if not panel_genes:
            return pd.DataFrame(), set()

        if "subclass_name" not in adata.obs:
            raise ValueError(f"{h5ad_path} lacks obs['subclass_name']")
        labels = (
            adata.obs["subclass_name"]
            .astype(str)
            .str.replace(r"^\d+\s+", "", regex=True)
            .str.replace("/", "-", regex=False)
            .to_numpy()
        )
        missing_groups = sorted(set(groups) - set(labels))
        if missing_groups:
            raise ValueError(
                "Bayesian cell types absent from h5ad subclass labels: "
                + ", ".join(missing_groups)
            )

        group_index = {group: index for index, group in enumerate(groups)}
        sums = np.zeros((len(groups), len(panel_genes)), dtype=np.float64)
        detected = np.zeros_like(sums)
        n_cells = np.zeros(len(groups), dtype=np.int64)
        n_obs = adata.n_obs
        for start in range(0, n_obs, chunk_size):
            stop = min(start + chunk_size, n_obs)
            block = np.asarray(adata.X[start:stop, gene_indices], dtype=np.float64)
            block_labels = labels[start:stop]
            for group in np.unique(block_labels):
                index = group_index.get(str(group))
                if index is None:
                    continue
                mask = block_labels == group
                n_cells[index] += int(mask.sum())
                sums[index] += block[mask].sum(axis=0)
                detected[index] += np.count_nonzero(block[mask] > 0, axis=0)

        rows = []
        for group, group_position in group_index.items():
            count = int(n_cells[group_position])
            if count == 0:
                continue
            for gene_position, gene in enumerate(panel_genes):
                rows.append(
                    {
                        "group": group,
                        "mouse_gene_symbol": gene,
                        "tf_expression_mean": (
                            sums[group_position, gene_position] / count
                        ),
                        "tf_expression_fraction_detected": (
                            detected[group_position, gene_position] / count
                        ),
                        "tf_expression_n_cells": count,
                    }
                )
        return pd.DataFrame(rows), set(panel_genes)
    finally:
        adata.file.close()


def load_cell_type_number_prefixes(
    h5ad_path: Path,
    groups: list[str],
) -> pd.Series:
    """Recover the original numbered subclass prefix for plot ordering."""
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if "subclass_name" not in adata.obs:
            raise ValueError(f"{h5ad_path} lacks obs['subclass_name']")
        raw_labels = pd.Index(adata.obs["subclass_name"].astype(str).unique())
    finally:
        adata.file.close()

    wanted = set(groups)
    rows = []
    for raw_label in raw_labels:
        match = re.match(r"^(\d+)\s+(.+)$", str(raw_label))
        if match is None:
            continue
        normalized = match.group(2).replace("/", "-")
        if normalized in wanted:
            rows.append(
                {
                    "group": normalized,
                    "numbered_prefix": int(match.group(1)),
                }
            )
    mapping = pd.DataFrame(rows)
    if mapping.empty:
        raise ValueError("No numbered h5ad subclass labels matched cell types")
    duplicates = mapping.loc[
        mapping.duplicated("group", keep=False)
    ].sort_values(["group", "numbered_prefix"])
    if not duplicates.empty:
        raise ValueError(
            "Cell types map to multiple numbered h5ad prefixes:\n"
            + duplicates.to_string(index=False)
        )
    prefixes = mapping.set_index("group")["numbered_prefix"].reindex(groups)
    if prefixes.isna().any():
        raise ValueError(
            "Cell types lack a numbered h5ad subclass prefix: "
            + ", ".join(prefixes.index[prefixes.isna()])
        )
    prefixes = prefixes.astype(int)
    prefixes.name = "numbered_prefix"
    return prefixes


def add_tf_annotation_and_expression(
    frame: pd.DataFrame,
    annotation: pd.DataFrame,
    expression: pd.DataFrame,
    panel_genes: set[str],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.merge(annotation, on="motif", how="left", validate="many_to_one")
    output["tf_gene_in_expression_panel"] = output["mouse_gene_symbol"].isin(
        panel_genes
    )
    if not expression.empty and "group" in output:
        output = output.merge(
            expression,
            on=["group", "mouse_gene_symbol"],
            how="left",
            validate="many_to_one",
        )
    return output


def collapse_motifs_to_tf_genes(
    motif: pd.DataFrame,
    annotation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse alternative HOCOMOCO models to one any-hit score per mouse TF."""
    annotation_index = annotation.set_index("motif")
    mapped = annotation_index.reindex(motif.columns)
    if mapped["mouse_gene_symbol"].isna().any():
        missing = mapped.index[mapped["mouse_gene_symbol"].isna()].tolist()
        raise ValueError(f"Motifs lack mouse gene annotation: {missing[:10]}")
    gene_matrix = motif.T.groupby(mapped["mouse_gene_symbol"], sort=True).max().T
    gene_matrix.index = motif.index
    gene_matrix.index.names = motif.index.names

    gene_metadata = (
        annotation.loc[annotation["motif"].isin(motif.columns)]
        .groupby("mouse_gene_symbol", sort=True)
        .agg(
            n_hocomoco_motifs=("motif", "nunique"),
            hocomoco_motifs=(
                "motif",
                lambda values: ";".join(sorted(set(map(str, values)))),
            ),
            hocomoco_tf_ids=(
                "hocomoco_tf",
                lambda values: ";".join(sorted(set(map(str, values)))),
            ),
            mouse_gene_synonyms=(
                "mouse_gene_synonyms",
                lambda values: ";".join(
                    sorted({str(value) for value in values if str(value)})
                ),
            ),
            mouse_uniprot_ids=(
                "mouse_uniprot_id",
                lambda values: ";".join(
                    sorted({str(value) for value in values if str(value)})
                ),
            ),
            mouse_uniprot_acs=(
                "mouse_uniprot_ac",
                lambda values: ";".join(
                    sorted({str(value) for value in values if str(value)})
                ),
            ),
            mouse_mgi_ids=(
                "mouse_mgi",
                lambda values: ";".join(
                    sorted({str(value) for value in values if str(value)})
                ),
            ),
            mouse_entrez_ids=(
                "mouse_entrez",
                lambda values: ";".join(
                    sorted({str(value) for value in values if str(value)})
                ),
            ),
            tfclass_families=(
                "tfclass_family",
                lambda values: ";".join(
                    sorted({str(value) for value in values if str(value)})
                ),
            ),
        )
        .reset_index()
    )
    return gene_matrix, gene_metadata


def compare_tf_genes(
    tests: pd.DataFrame,
    gene_matrix: pd.DataFrame,
    gene_metadata: pd.DataFrame,
    expression: pd.DataFrame,
    panel_genes: set[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare any HOCOMOCO motif hit per TF and attach matching expression."""
    enrichment, _ = compare_motifs(
        tests,
        gene_matrix,
        0.0,
        args.min_significant_ccres,
        args.min_nonsignificant_ccres,
    )
    if enrichment.empty:
        return enrichment, pd.DataFrame(), pd.DataFrame()
    enrichment = enrichment.rename(
        columns={
            "motif": "mouse_gene_symbol",
            "significant_score_mean": "significant_best_motif_score_mean",
            "nonsignificant_score_mean": "nonsignificant_best_motif_score_mean",
            "significant_ccres_with_motif": "significant_ccres_with_tf_motif",
        }
    )
    enrichment = enrichment.merge(
        gene_metadata,
        on="mouse_gene_symbol",
        how="left",
        validate="many_to_one",
    )
    enrichment["tf_gene_in_expression_panel"] = enrichment[
        "mouse_gene_symbol"
    ].isin(panel_genes)
    if not expression.empty:
        enrichment = enrichment.merge(
            expression,
            on=["group", "mouse_gene_symbol"],
            how="left",
            validate="many_to_one",
        )
    enrichment = enrichment.sort_values(
        ["group", "q_value_cell_type", "p_value", "fraction_difference"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    top = enrichment.groupby("group", sort=False, group_keys=False).head(
        args.top_n_per_cell_type
    ).copy()
    top["rank_within_cell_type"] = top.groupby("group", sort=False).cumcount() + 1

    if expression.empty:
        supported = enrichment.iloc[0:0].copy()
    else:
        supported = enrichment.loc[
            enrichment["tf_gene_in_expression_panel"]
            & enrichment["tf_expression_fraction_detected"].fillna(0).gt(0)
        ].copy()
    supported = supported.groupby("group", sort=False, group_keys=False).head(
        args.top_n_per_cell_type
    )
    supported["rank_within_cell_type"] = (
        supported.groupby("group", sort=False).cumcount() + 1
    )
    return enrichment, top, supported


def compare_motifs(
    tests: pd.DataFrame,
    motif: pd.DataFrame,
    motif_score_cutoff: float,
    min_significant_ccres: int,
    min_nonsignificant_ccres: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run matched significant-vs-non-significant comparisons per cell type."""
    if not isinstance(motif.index, pd.MultiIndex) or list(motif.index.names) != [
        "group",
        "cre",
    ]:
        raise ValueError(
            "Motif matrix must have a ('group', 'cre') MultiIndex because "
            "FIMO p-values use cell-type-specific sequence backgrounds"
        )
    rows: list[dict] = []
    summaries: list[dict] = []

    for group, group_frame in tests.groupby("group", sort=True):
        if group not in motif.index.get_level_values("group"):
            group_motif = motif.iloc[0:0].copy()
            group_motif.index = pd.Index([], name="cre")
        else:
            group_motif = motif.xs(group, level="group", drop_level=True)
        motif_index = set(group_motif.index)
        eligible = group_frame.loc[
            group_frame["has_finite_q"] & group_frame["cre"].isin(motif_index)
        ].copy()
        significant = eligible.loc[eligible["is_significant"]]
        nonsignificant = eligible.loc[~eligible["is_significant"]]
        classes = sorted(eligible["class"].dropna().unique())
        cell_class = ";".join(classes)
        summary = {
            "group": group,
            "class": cell_class,
            "n_eligible_tests": int(len(group_frame)),
            "n_motif_covered_tests": int(len(eligible)),
            "n_significant_ccres": int(len(significant)),
            "n_nonsignificant_ccres": int(len(nonsignificant)),
            "n_motifs_tested": 0,
            "n_enriched_motifs_q_le_cutoff": 0,
            "skip_reason": "",
        }
        if len(significant) < min_significant_ccres:
            summary["skip_reason"] = "too_few_significant_ccres"
            summaries.append(summary)
            continue
        if len(nonsignificant) < min_nonsignificant_ccres:
            summary["skip_reason"] = "too_few_nonsignificant_ccres"
            summaries.append(summary)
            continue

        significant_ids = significant["cre"].tolist()
        nonsignificant_ids = nonsignificant["cre"].tolist()
        significant_scores = group_motif.loc[significant_ids]
        nonsignificant_scores = group_motif.loc[nonsignificant_ids]
        significant_presence = significant_scores.gt(motif_score_cutoff)
        nonsignificant_presence = nonsignificant_scores.gt(motif_score_cutoff)

        for motif_name in group_motif.columns:
            target_mask = significant_presence[motif_name].to_numpy(dtype=bool)
            background_mask = nonsignificant_presence[motif_name].to_numpy(dtype=bool)
            target_present = int(target_mask.sum())
            target_absent = int(len(target_mask) - target_present)
            background_present = int(background_mask.sum())
            background_absent = int(len(background_mask) - background_present)
            odds_ratio, p_value = fisher_exact(
                [
                    [target_present, target_absent],
                    [background_present, background_absent],
                ],
                alternative="greater",
            )
            target_fraction = target_present / len(target_mask)
            background_fraction = background_present / len(background_mask)
            if background_fraction == 0:
                enrichment_ratio = math.inf if target_fraction > 0 else math.nan
            else:
                enrichment_ratio = target_fraction / background_fraction
            log2_odds_ratio_pseudocount = math.log2(
                ((target_present + 0.5) * (background_absent + 0.5))
                / ((target_absent + 0.5) * (background_present + 0.5))
            )
            rows.append(
                {
                    "group": group,
                    "class": cell_class,
                    "motif": motif_name,
                    "n_significant_ccres": len(significant_ids),
                    "n_nonsignificant_ccres": len(nonsignificant_ids),
                    "significant_present": target_present,
                    "significant_absent": target_absent,
                    "nonsignificant_present": background_present,
                    "nonsignificant_absent": background_absent,
                    "significant_fraction": target_fraction,
                    "nonsignificant_fraction": background_fraction,
                    "fraction_difference": target_fraction - background_fraction,
                    "enrichment_ratio": enrichment_ratio,
                    "odds_ratio": odds_ratio,
                    "log2_odds_ratio_pseudocount": log2_odds_ratio_pseudocount,
                    "significant_score_mean": float(significant_scores[motif_name].mean()),
                    "nonsignificant_score_mean": float(nonsignificant_scores[motif_name].mean()),
                    "p_value": p_value,
                    "significant_ccres_with_motif": ";".join(
                        np.asarray(significant_ids, dtype=str)[target_mask]
                    ),
                }
            )
        summary["n_motifs_tested"] = int(group_motif.shape[1])
        summaries.append(summary)

    enrichment = pd.DataFrame(rows)
    summary = pd.DataFrame(summaries)
    if enrichment.empty:
        return enrichment, summary

    enrichment["q_value_cell_type"] = np.nan
    for _, index in enrichment.groupby("group", sort=False).groups.items():
        enrichment.loc[index, "q_value_cell_type"] = bh_fdr(
            enrichment.loc[index, "p_value"]
        )
    enrichment["q_value_global"] = bh_fdr(enrichment["p_value"])
    enrichment = enrichment.sort_values(
        ["group", "q_value_cell_type", "p_value", "fraction_difference", "motif"],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)
    return enrichment, summary


def annotate_significant_ccres(
    tests: pd.DataFrame,
    motif: pd.DataFrame,
    metadata: pd.DataFrame,
    motif_score_cutoff: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return wide and long motif annotations for significant cCRE-cell-type pairs."""
    significant = tests.loc[
        tests["has_finite_q"]
        & tests["is_significant"]
    ].copy()
    base_columns = [
        column
        for column in (
            "method",
            "group",
            "class",
            "cre",
            "n_cells",
            "target_t7_total",
            "activity_mean",
            "effect_vs_mean_control_mean",
            "posterior_probability_above_mean_control",
            "p_right",
            "q_right",
        )
        if column in significant.columns
    ]
    significant = significant.loc[:, base_columns]
    motif_frame = motif.rename_axis(index=["group", "cre"]).reset_index()
    metadata_frame = metadata.rename_axis("cre").reset_index()
    wide = significant.merge(metadata_frame, on="cre", how="left", validate="many_to_one")
    wide = wide.merge(
        motif_frame,
        on=["group", "cre"],
        how="inner",
        validate="one_to_one",
    )

    identifier_columns = base_columns + [
        column for column in metadata.columns if column not in base_columns
    ]
    long = wide.melt(
        id_vars=identifier_columns,
        value_vars=list(motif.columns),
        var_name="motif",
        value_name="motif_score",
    )
    long = long.loc[long["motif_score"].gt(motif_score_cutoff)].copy()
    long = long.sort_values(
        ["group", "q_right", "cre", "motif_score", "motif"],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)
    long["motif_rank_within_ccre"] = (
        long.groupby(["group", "cre"], sort=False).cumcount() + 1
    )
    return wide, long


def aggregate_weighted_motif_activity(
    tests: pd.DataFrame,
    motif: pd.DataFrame,
    min_target_t7: float = 50.0,
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """Aggregate effect-weighted FIMO scores over all T7-eligible cCREs.

    For each valid cell-type/cCRE pair and motif, the contribution is
    ``effect_vs_mean_control_mean * -log10(best FIMO p-value)``. A missing
    motif hit already has score zero in ``motif``. Negative Bayesian effects
    are retained rather than clipped.
    """
    required = {
        "group",
        "cre",
        "has_finite_q",
        "target_t7_total",
        "effect_vs_mean_control_mean",
    }
    missing_columns = sorted(required - set(tests.columns))
    if missing_columns:
        raise ValueError(
            "Cannot calculate weighted motif activity; test table is missing "
            f"columns: {missing_columns}"
        )
    if not isinstance(motif.index, pd.MultiIndex) or list(motif.index.names) != [
        "group",
        "cre",
    ]:
        raise ValueError("Motif matrix must have a ('group', 'cre') MultiIndex")

    valid = tests.loc[
        tests["has_finite_q"]
        & pd.to_numeric(tests["target_t7_total"], errors="coerce").ge(
            min_target_t7
        ),
        ["group", "cre", "effect_vs_mean_control_mean"],
    ].copy()
    valid["effect_vs_mean_control_mean"] = pd.to_numeric(
        valid["effect_vs_mean_control_mean"], errors="coerce"
    )
    if valid["effect_vs_mean_control_mean"].isna().any():
        raise ValueError(
            "Valid T7>=50 cCREs contain missing effect_vs_mean_control_mean"
        )
    effects = valid.set_index(["group", "cre"])[
        "effect_vs_mean_control_mean"
    ]
    missing_pairs = motif.index.difference(effects.index)
    extra_pairs = effects.index.difference(motif.index)
    if len(missing_pairs) or len(extra_pairs):
        raise ValueError(
            "Weighted motif activity requires the FIMO matrix and valid "
            f"T7>=50 set to match exactly; missing={len(missing_pairs)}, "
            f"extra={len(extra_pairs)}"
        )
    effects = effects.reindex(motif.index)

    presence = motif.gt(0.0)
    contributions = motif.mul(effects, axis=0)
    grouped_contributions = contributions.groupby(level="group", sort=True)
    grouped_presence = presence.groupby(level="group", sort=True)
    grouped_scores = motif.groupby(level="group", sort=True)
    matrices = {
        "activity_mean": grouped_contributions.mean(),
        "activity_sum": grouped_contributions.sum(),
        "occurrence_fraction": grouped_presence.mean(),
        "hit_count": grouped_presence.sum().astype(int),
        "matching_score_mean_all": grouped_scores.mean(),
        "matching_score_mean_present": motif.where(presence)
        .groupby(level="group", sort=True)
        .mean(),
    }
    activity_row_mean = matrices["activity_mean"].mean(axis=1)
    activity_row_sd = matrices["activity_mean"].std(axis=1, ddof=0)
    if activity_row_sd.le(0).any():
        zero_variance_groups = activity_row_sd.index[
            activity_row_sd.le(0)
        ].tolist()
        raise ValueError(
            "Cannot z-score motif activities for cell types with zero motif "
            f"variance: {zero_variance_groups}"
        )
    matrices["activity_zscore"] = matrices["activity_mean"].sub(
        activity_row_mean,
        axis=0,
    ).div(activity_row_sd, axis=0)
    n_valid_ccres = effects.groupby(level="group", sort=True).size()
    n_valid_ccres.name = "n_valid_ccres"
    return matrices, n_valid_ccres


def columnwise_correlation(
    activity: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Pearson correlations and two-sided p-values for score columns."""
    n_ccres = len(activity)
    correlations = np.full(scores.shape[1], np.nan, dtype=float)
    p_values = np.full(scores.shape[1], np.nan, dtype=float)
    if n_ccres < 3:
        return correlations, p_values

    centered_activity = activity - activity.mean()
    centered_scores = scores - scores.mean(axis=0)
    activity_ss = float(np.dot(centered_activity, centered_activity))
    score_ss = np.einsum("ij,ij->j", centered_scores, centered_scores)
    defined = (score_ss > 0.0) & (activity_ss > 0.0)
    if not defined.any():
        return correlations, p_values

    correlations[defined] = (
        centered_scores[:, defined].T @ centered_activity
    ) / np.sqrt(score_ss[defined] * activity_ss)
    correlations[defined] = np.clip(correlations[defined], -1.0, 1.0)
    degrees_freedom = n_ccres - 2
    with np.errstate(divide="ignore", invalid="ignore"):
        test_statistic = np.abs(correlations[defined]) * np.sqrt(
            degrees_freedom
            / np.maximum(
                1.0 - correlations[defined] ** 2,
                np.finfo(float).tiny,
            )
        )
    p_values[defined] = 2.0 * student_t.sf(test_statistic, degrees_freedom)
    return correlations, p_values


def correlate_motif_scores_with_activity(
    tests: pd.DataFrame,
    motif: pd.DataFrame,
    min_target_t7: float = 50.0,
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """Correlate cCRE activity with every motif score within each cell type.

    Pearson correlations use the original values; Spearman correlations use
    average ranks, including tied zero scores for cCREs without a qualifying
    FIMO hit. Both analyses use all finite-q, T7-eligible cCREs.
    """
    required = {
        "group",
        "cre",
        "has_finite_q",
        "target_t7_total",
        "effect_vs_mean_control_mean",
    }
    missing_columns = sorted(required - set(tests.columns))
    if missing_columns:
        raise ValueError(
            "Cannot calculate motif-activity correlations; test table is "
            f"missing columns: {missing_columns}"
        )
    if not isinstance(motif.index, pd.MultiIndex) or list(motif.index.names) != [
        "group",
        "cre",
    ]:
        raise ValueError("Motif matrix must have a ('group', 'cre') MultiIndex")

    valid = tests.loc[
        tests["has_finite_q"]
        & pd.to_numeric(tests["target_t7_total"], errors="coerce").ge(
            min_target_t7
        ),
        ["group", "cre", "effect_vs_mean_control_mean"],
    ].copy()
    valid["effect_vs_mean_control_mean"] = pd.to_numeric(
        valid["effect_vs_mean_control_mean"], errors="coerce"
    )
    if valid["effect_vs_mean_control_mean"].isna().any():
        raise ValueError(
            "Valid T7>=50 cCREs contain missing effect_vs_mean_control_mean"
        )
    effects = valid.set_index(["group", "cre"])[
        "effect_vs_mean_control_mean"
    ]
    missing_pairs = motif.index.difference(effects.index)
    extra_pairs = effects.index.difference(motif.index)
    if len(missing_pairs) or len(extra_pairs):
        raise ValueError(
            "Motif-activity correlations require the FIMO matrix and valid "
            f"T7>=50 set to match exactly; missing={len(missing_pairs)}, "
            f"extra={len(extra_pairs)}"
        )

    row_data: dict[str, list[pd.Series]] = {
        "pearson_r": [],
        "pearson_p_value": [],
        "pearson_q_value_cell_type": [],
        "spearman_rho": [],
        "spearman_p_value": [],
        "spearman_q_value_cell_type": [],
        "hit_count": [],
    }
    n_valid_rows: dict[str, int] = {}
    for group, group_scores in motif.groupby(level="group", sort=True):
        group_scores = group_scores.droplevel("group")
        activity = (
            effects.xs(group, level="group")
            .reindex(group_scores.index)
            .to_numpy(dtype=float)
        )
        scores = group_scores.to_numpy(dtype=float)
        n_valid_rows[str(group)] = len(activity)

        pearson_r, pearson_p = columnwise_correlation(activity, scores)
        spearman_rho, spearman_p = columnwise_correlation(
            rankdata(activity, method="average"),
            rankdata(scores, axis=0, method="average"),
        )
        values = {
            "pearson_r": pearson_r,
            "pearson_p_value": pearson_p,
            "pearson_q_value_cell_type": bh_fdr(pearson_p),
            "spearman_rho": spearman_rho,
            "spearman_p_value": spearman_p,
            "spearman_q_value_cell_type": bh_fdr(spearman_p),
            "hit_count": np.count_nonzero(scores > 0.0, axis=0),
        }
        for key, value in values.items():
            row_data[key].append(
                pd.Series(value, index=motif.columns, name=group)
            )

    matrices = {
        key: pd.DataFrame(rows) for key, rows in row_data.items()
    }
    matrices["hit_count"] = matrices["hit_count"].astype(int)
    for matrix in matrices.values():
        matrix.index.name = "group"
        matrix.columns.name = None
    n_valid_ccres = pd.Series(n_valid_rows, name="n_valid_ccres", dtype=int)
    n_valid_ccres.index.name = "group"
    return matrices, n_valid_ccres


def make_motif_activity_correlation_long(
    matrices: dict[str, pd.DataFrame],
    n_valid_ccres: pd.Series,
    annotation: pd.DataFrame,
    expression: pd.DataFrame,
    panel_genes: set[str],
) -> pd.DataFrame:
    """Add motif/gene metadata to the cell-type motif correlations."""
    column_names = {
        "pearson_r": "activity_matching_score_pearson_r",
        "pearson_p_value": "activity_matching_score_pearson_p",
        "pearson_q_value_cell_type": (
            "activity_matching_score_pearson_q_cell_type"
        ),
        "spearman_rho": "activity_matching_score_spearman_rho",
        "spearman_p_value": "activity_matching_score_spearman_p",
        "spearman_q_value_cell_type": (
            "activity_matching_score_spearman_q_cell_type"
        ),
        "hit_count": "n_valid_ccres_with_motif",
    }
    stacked = []
    for key, output_name in column_names.items():
        series = matrices[key].rename_axis(index="group", columns="motif").stack(
            future_stack=True
        )
        stacked.append(series.rename(output_name))
    long = pd.concat(stacked, axis=1).reset_index()
    long["n_valid_ccres"] = long["group"].map(n_valid_ccres).astype(int)
    long = add_tf_annotation_and_expression(
        long,
        annotation,
        expression,
        panel_genes,
    )
    return long.sort_values(
        ["group", "activity_matching_score_pearson_r", "motif"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)


def make_motif_activity_long(
    matrices: dict[str, pd.DataFrame],
    n_valid_ccres: pd.Series,
    annotation: pd.DataFrame,
    expression: pd.DataFrame,
    panel_genes: set[str],
) -> pd.DataFrame:
    """Combine motif-activity summaries with HOCOMOCO and TF-expression data."""
    column_names = {
        "activity_mean": "motif_activity_mean_per_valid_ccre",
        "activity_sum": "motif_activity_sum",
        "activity_zscore": "motif_activity_zscore_within_cell_type",
        "occurrence_fraction": "motif_occurrence_fraction",
        "hit_count": "n_valid_ccres_with_motif",
        "matching_score_mean_all": "motif_matching_score_mean_all_valid_ccres",
        "matching_score_mean_present": (
            "motif_matching_score_mean_motif_positive_ccres"
        ),
    }
    stacked = []
    for key, output_name in column_names.items():
        series = matrices[key].rename_axis(index="group", columns="motif").stack(
            future_stack=True
        )
        stacked.append(series.rename(output_name))
    long = pd.concat(stacked, axis=1).reset_index()
    long["n_valid_ccres"] = long["group"].map(n_valid_ccres).astype(int)
    long = add_tf_annotation_and_expression(
        long,
        annotation,
        expression,
        panel_genes,
    )
    return long.sort_values(
        ["group", "motif_activity_mean_per_valid_ccre", "motif"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def finish_summaries(
    enrichment: pd.DataFrame,
    summary: pd.DataFrame,
    motif_q_cutoff: float,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if enrichment.empty:
        return summary, pd.DataFrame(), pd.DataFrame()

    enriched = enrichment["q_value_cell_type"].le(motif_q_cutoff)
    enriched_counts = enriched.groupby(enrichment["group"]).sum().astype(int)
    summary = summary.copy()
    summary["n_enriched_motifs_q_le_cutoff"] = (
        summary["group"].map(enriched_counts).fillna(0).astype(int)
    )

    top = enrichment.groupby("group", sort=False, group_keys=False).head(top_n).copy()
    top["rank_within_cell_type"] = top.groupby("group", sort=False).cumcount() + 1

    recurrence = (
        enrichment.groupby("motif", sort=False)
        .agg(
            n_cell_types_tested=("group", "nunique"),
            n_cell_types_enriched_q_le_cutoff=(
                "q_value_cell_type",
                lambda values: int(values.le(motif_q_cutoff).sum()),
            ),
            n_cell_types_global_q_le_cutoff=(
                "q_value_global",
                lambda values: int(values.le(motif_q_cutoff).sum()),
            ),
            minimum_p_value=("p_value", "min"),
            minimum_q_value_cell_type=("q_value_cell_type", "min"),
            minimum_q_value_global=("q_value_global", "min"),
            median_fraction_difference=("fraction_difference", "median"),
            maximum_fraction_difference=("fraction_difference", "max"),
            median_log2_odds_ratio_pseudocount=(
                "log2_odds_ratio_pseudocount",
                "median",
            ),
        )
        .reset_index()
    )
    recurrence = recurrence.sort_values(
        [
            "n_cell_types_enriched_q_le_cutoff",
            "minimum_q_value_cell_type",
            "maximum_fraction_difference",
            "motif",
        ],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    return summary, top, recurrence


def plot_top_enrichments(
    enrichment: pd.DataFrame,
    figures_dir: Path,
    motif_q_cutoff: float,
    top_n: int,
) -> None:
    if enrichment.empty:
        return
    significant_only = enrichment["q_value_cell_type"].le(motif_q_cutoff)
    plot_frame = enrichment.loc[significant_only].copy()
    if plot_frame.empty:
        plot_frame = enrichment.sort_values(
            ["p_value", "fraction_difference"],
            ascending=[True, False],
        ).head(top_n)
        score_values = plot_frame["p_value"]
        x_label = "−log10(raw one-sided Fisher p-value)"
        reference = 0.05
    else:
        plot_frame = plot_frame.sort_values(
            ["q_value_cell_type", "p_value", "fraction_difference"],
            ascending=[True, True, False],
        ).head(top_n)
        score_values = plot_frame["q_value_cell_type"]
        x_label = "−log10(within-cell-type motif q-value)"
        reference = motif_q_cutoff
    if plot_frame.empty:
        return

    scores = -np.log10(
        score_values.clip(lower=np.nextafter(0.0, 1.0))
    )
    labels = []
    for row in plot_frame.itertuples(index=False):
        gene = getattr(row, "mouse_gene_symbol", "")
        labels.append(f"{row.group} | {gene} | {row.motif}")
    colors = np.where(plot_frame["fraction_difference"].ge(0), "#3264a8", "#9aa0a6")
    figure_height = max(5.5, 0.25 * len(plot_frame) + 1.2)
    fig, axis = plt.subplots(figsize=(10.5, figure_height), constrained_layout=True)
    y = np.arange(len(plot_frame))
    axis.barh(y, scores, color=colors)
    axis.set_yticks(y)
    axis.set_yticklabels(labels, fontsize=7)
    axis.invert_yaxis()
    axis.axvline(
        -math.log10(reference), color="#b33b33", linestyle="--", linewidth=1
    )
    axis.set_xlabel(x_label)
    if significant_only.any():
        axis.set_title("HOCOMOCO motifs enriched in significant cCREs")
    else:
        axis.set_title(
            "Top HOCOMOCO motif comparisons "
            f"(none significant at within-cell-type FDR {motif_q_cutoff:g})"
        )
    axis.grid(axis="x", color="0.9", linewidth=0.6)
    for suffix in (".png", ".pdf"):
        fig.savefig(figures_dir / f"top_motif_enrichments{suffix}", dpi=220)
    plt.close(fig)


def plot_enrichment_heatmap(
    enrichment: pd.DataFrame,
    recurrence: pd.DataFrame,
    figures_dir: Path,
    motif_q_cutoff: float,
    top_n_motifs: int,
) -> None:
    if enrichment.empty or recurrence.empty:
        return
    significant = enrichment.loc[
        enrichment["q_value_cell_type"].le(motif_q_cutoff)
        & enrichment["fraction_difference"].gt(0)
    ].copy()
    if significant.empty:
        return
    top_motifs = recurrence.loc[
        recurrence["n_cell_types_enriched_q_le_cutoff"].gt(0), "motif"
    ].head(top_n_motifs)
    if top_motifs.empty:
        return
    groups = sorted(significant.loc[significant["motif"].isin(top_motifs), "group"].unique())
    if not groups:
        return

    effect = enrichment.pivot(index="motif", columns="group", values="fraction_difference")
    q_values = enrichment.pivot(index="motif", columns="group", values="q_value_cell_type")
    effect = effect.reindex(index=top_motifs, columns=groups)
    q_values = q_values.reindex(index=top_motifs, columns=groups)
    display = effect.where(q_values.le(motif_q_cutoff), 0.0)
    max_abs = max(float(np.nanmax(np.abs(display.to_numpy()))), 1e-6)

    figure_width = max(9.0, 0.29 * len(groups) + 3.0)
    figure_height = max(5.0, 0.31 * len(top_motifs) + 1.8)
    fig, axis = plt.subplots(
        figsize=(figure_width, figure_height), constrained_layout=True
    )
    image = axis.imshow(
        display.to_numpy(),
        aspect="auto",
        cmap="Blues",
        vmin=0.0,
        vmax=max_abs,
        interpolation="nearest",
    )
    axis.set_xticks(np.arange(len(groups)))
    axis.set_xticklabels(groups, rotation=90, fontsize=7)
    axis.set_yticks(np.arange(len(top_motifs)))
    axis.set_yticklabels(top_motifs, fontsize=8)
    axis.set_title("Significant motif enrichment by cell type")
    axis.set_xlabel("Cell type")
    axis.set_ylabel("HOCOMOCO motif")
    colorbar = fig.colorbar(image, ax=axis, shrink=0.8)
    colorbar.set_label("Significant − non-significant motif frequency")
    for suffix in (".png", ".pdf"):
        fig.savefig(figures_dir / f"motif_enrichment_heatmap{suffix}", dpi=220)
    plt.close(fig)


class MotifDisplay(NamedTuple):
    """Cell type by motif matrix with the linkages used to order it."""

    values: pd.DataFrame
    motif_linkage: np.ndarray | None
    cell_type_linkage: np.ndarray | None


def prepare_motif_display(
    activity_matrix: pd.DataFrame,
    cell_type_number_prefixes: pd.Series,
    n_valid_ccres: pd.Series,
    top_n_motifs: int,
    min_valid_ccres: int,
    selection: str = "variability",
    cluster_cell_types: bool = False,
) -> MotifDisplay | None:
    """Select the top motifs and order both axes for the activity heatmaps.

    Motifs are always hierarchically clustered (average linkage, correlation
    distance).  Cell types keep their numbered-prefix order unless
    ``cluster_cell_types`` is set, in which case they are clustered the same
    way.  Returns ``None`` when nothing is left to plot.
    """
    if activity_matrix.empty or top_n_motifs < 1:
        return None
    retained_cell_types = n_valid_ccres.loc[
        n_valid_ccres.ge(min_valid_ccres)
    ].index
    activity_matrix = activity_matrix.reindex(retained_cell_types)
    if activity_matrix.empty:
        return None
    if selection == "variability":
        ranking = activity_matrix.std(axis=0, ddof=0)
    elif selection == "max_absolute":
        ranking = activity_matrix.abs().max(axis=0)
    else:
        raise ValueError(f"Unknown heatmap motif selection method: {selection}")
    finite_counts = activity_matrix.notna().sum(axis=0)
    plot_variability = activity_matrix.fillna(0.0).std(axis=0, ddof=0)
    ranking = ranking.loc[
        finite_counts.ge(2) & plot_variability.gt(0.0)
    ].sort_values(ascending=False)
    selected = ranking.head(top_n_motifs).index
    if selected.empty:
        return None
    cell_type_order = (
        cell_type_number_prefixes.reindex(activity_matrix.index)
        .sort_values(kind="stable")
        .index
    )
    display = activity_matrix.reindex(index=cell_type_order, columns=selected)
    if display.shape[1] > 1:
        motif_linkage = linkage(
            np.nan_to_num(display.T.to_numpy(dtype=float), nan=0.0),
            method="average",
            metric="correlation",
            optimal_ordering=True,
        )
        display = display.iloc[:, leaves_list(motif_linkage)]
    else:
        motif_linkage = None

    cell_type_linkage = None
    if cluster_cell_types and display.shape[0] > 1:
        cell_type_values = np.nan_to_num(
            display.to_numpy(dtype=float),
            nan=0.0,
        )
        if np.ptp(cell_type_values, axis=1).min() > 0.0:
            cell_type_linkage = linkage(
                cell_type_values,
                method="average",
                metric="correlation",
                optimal_ordering=True,
            )
            display = display.iloc[leaves_list(cell_type_linkage), :]
    return MotifDisplay(display, motif_linkage, cell_type_linkage)


class HeatmapPanel(NamedTuple):
    """One stacked heatmap panel sharing the figure's row and column order."""

    values: np.ndarray
    title: str
    y_label: str
    colorbar_label: str
    vmin: float
    vmax: float
    color_map_name: str = "RdBu_r"


def draw_stacked_clustered_heatmaps(
    panels: list[HeatmapPanel],
    row_labels: list[str],
    column_labels: list[str],
    figures_dir: Path,
    output_stem: str,
    x_label: str,
    column_linkage: np.ndarray | None = None,
    row_linkage: np.ndarray | None = None,
) -> None:
    """Render one or more matrices that share row and column order.

    A single column dendrogram is drawn above the stack; the row dendrogram,
    when supplied, is repeated beside every panel so each stays readable.
    Only the bottom panel carries the column tick labels.
    """
    if not panels:
        raise ValueError("At least one heatmap panel is required.")
    n_panels = len(panels)
    n_rows, n_columns = panels[0].values.shape
    for panel in panels:
        if panel.values.shape != (n_rows, n_columns):
            raise ValueError("All stacked panels must share the same shape.")
    heatmap_cell_size = 0.22
    row_dendrogram_width = 1.4 if row_linkage is not None else 0.0
    fig = plt.figure(
        figsize=(
            max(13.0, heatmap_cell_size * n_columns + 5.0 + row_dendrogram_width),
            max(
                10.0 * n_panels,
                heatmap_cell_size * n_rows * n_panels + 3.0 + 1.5 * (n_panels - 1),
            ),
        ),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        1 + n_panels,
        2 if row_linkage is not None else 1,
        height_ratios=[1.2] + [8.0] * n_panels,
        width_ratios=[1.0, 10.0] if row_linkage is not None else [1.0],
    )
    heatmap_column = 1 if row_linkage is not None else 0
    dendrogram_axis = fig.add_subplot(grid[0, heatmap_column])
    if column_linkage is not None:
        dendrogram(
            column_linkage,
            ax=dendrogram_axis,
            no_labels=True,
            color_threshold=0,
            above_threshold_color="#4d4d4d",
        )
    dendrogram_axis.set_axis_off()

    for index, panel in enumerate(panels):
        axis = fig.add_subplot(grid[1 + index, heatmap_column])
        colorbar_axes = [axis]
        if row_linkage is not None:
            row_dendrogram_axis = fig.add_subplot(grid[1 + index, 0])
            dendrogram(
                row_linkage,
                ax=row_dendrogram_axis,
                orientation="left",
                no_labels=True,
                color_threshold=0,
                above_threshold_color="#4d4d4d",
            )
            # scipy places the first leaf at the bottom; imshow row 0 is at
            # the top, so flip the axis to keep leaves and rows aligned.
            row_dendrogram_axis.invert_yaxis()
            row_dendrogram_axis.set_axis_off()
            colorbar_axes.insert(0, row_dendrogram_axis)
        if n_panels == 1:
            colorbar_axes.insert(0, dendrogram_axis)

        color_map = plt.get_cmap(panel.color_map_name).copy()
        color_map.set_bad("#d9d9d9")
        image = axis.imshow(
            np.ma.masked_invalid(panel.values),
            aspect="auto",
            cmap=color_map,
            vmin=panel.vmin,
            vmax=panel.vmax,
            interpolation="nearest",
        )
        axis.set_xticks(np.arange(n_columns))
        if index == n_panels - 1:
            axis.set_xticklabels(
                column_labels,
                rotation=90,
                ha="center",
                va="top",
                fontsize=6,
            )
            axis.set_xlabel(x_label)
        else:
            axis.set_xticklabels([])
        axis.set_yticks(np.arange(n_rows))
        axis.set_yticklabels(row_labels, fontsize=7)
        axis.set_ylabel(panel.y_label)
        axis.set_title(panel.title)
        colorbar = fig.colorbar(image, ax=colorbar_axes, shrink=0.82)
        colorbar.set_label(panel.colorbar_label)

    for suffix in (".png", ".pdf"):
        fig.savefig(figures_dir / f"{output_stem}{suffix}", dpi=220)
    plt.close(fig)


def draw_clustered_heatmap(
    values: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    figures_dir: Path,
    output_stem: str,
    title: str,
    x_label: str,
    y_label: str,
    colorbar_label: str,
    vmin: float,
    vmax: float,
    column_linkage: np.ndarray | None = None,
    row_linkage: np.ndarray | None = None,
    color_map_name: str = "RdBu_r",
) -> None:
    """Render a single matrix with optional column and row dendrograms."""
    draw_stacked_clustered_heatmaps(
        [
            HeatmapPanel(
                values,
                title,
                y_label,
                colorbar_label,
                vmin,
                vmax,
                color_map_name,
            )
        ],
        row_labels,
        column_labels,
        figures_dir,
        output_stem,
        x_label,
        column_linkage=column_linkage,
        row_linkage=row_linkage,
    )


def motif_axis_labels(
    motifs: pd.Index,
    annotation: pd.DataFrame,
) -> list[str]:
    """Format ``mouse gene | motif`` labels for heatmap columns."""
    annotation_index = annotation.set_index("motif")
    return [
        f"{annotation_index.at[motif_name, 'mouse_gene_symbol']} | {motif_name}"
        for motif_name in motifs
    ]


def cell_type_axis_labels(
    groups: pd.Index,
    cell_type_number_prefixes: pd.Series,
) -> list[str]:
    """Format ``NNN group`` labels for heatmap rows."""
    return [
        f"{int(cell_type_number_prefixes.at[group]):03d} {group}"
        for group in groups
    ]


def plot_weighted_motif_activity_heatmap(
    activity_matrix: pd.DataFrame,
    annotation: pd.DataFrame,
    cell_type_number_prefixes: pd.Series,
    n_valid_ccres: pd.Series,
    figures_dir: Path,
    top_n_motifs: int,
    min_valid_ccres: int,
    output_stem: str,
    title: str,
    colorbar_label: str,
    selection: str = "variability",
    fixed_color_limit: float | None = None,
    cluster_cell_types: bool = False,
) -> None:
    """Plot selected motif values with clustered columns."""
    prepared = prepare_motif_display(
        activity_matrix,
        cell_type_number_prefixes,
        n_valid_ccres,
        top_n_motifs,
        min_valid_ccres,
        selection=selection,
        cluster_cell_types=cluster_cell_types,
    )
    if prepared is None:
        return
    display = prepared.values
    values = display.to_numpy(dtype=float)
    finite_abs = np.abs(values[np.isfinite(values)])
    color_limit = fixed_color_limit
    if color_limit is None:
        color_limit = (
            max(float(np.quantile(finite_abs, 0.98)), 1e-6)
            if finite_abs.size
            else 1.0
        )
    row_ordering = (
        "hierarchically clustered"
        if prepared.cell_type_linkage is not None
        else "ordered by numbered prefix"
    )
    draw_clustered_heatmap(
        values,
        cell_type_axis_labels(display.index, cell_type_number_prefixes),
        motif_axis_labels(display.columns, annotation),
        figures_dir,
        output_stem,
        title,
        "Hierarchically clustered mouse TF | HOCOMOCO motif",
        f"Cell type with ≥{min_valid_ccres} valid cCREs, {row_ordering}",
        colorbar_label,
        -color_limit,
        color_limit,
        column_linkage=prepared.motif_linkage,
        row_linkage=prepared.cell_type_linkage,
    )


def main() -> None:
    args = parse_args()
    tables_dir = args.results_dir / "tables"
    figures_dir = args.results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    tests = load_tests(args.tests, args.q_cutoff, args.expected_method)
    annotation = load_hocomoco_annotation(args.hocomoco_annotation)
    cre_info = load_cre_info(args.cre_info)
    valid_t7_mask = tests["has_finite_q"] & tests["target_t7_total"].ge(50.0)
    tested_ccres = sorted(set(tests.loc[valid_t7_mask, "cre"]))
    motif, fimo_hits, fimo_backgrounds, scan_signature = scan_hocomoco(
        tests, cre_info, annotation, args
    )
    valid_pairs = pd.MultiIndex.from_frame(
        tests.loc[valid_t7_mask, ["group", "cre"]],
        names=["group", "cre"],
    )
    missing_motif_pairs = valid_pairs.difference(motif.index)
    cell_type_number_prefixes = load_cell_type_number_prefixes(
        args.h5ad,
        sorted(tests["group"].unique()),
    )

    if args.skip_expression:
        expression = pd.DataFrame()
        expression_panel_genes: set[str] = set()
    else:
        print("[expression] aggregating HOCOMOCO TF genes by cell type", flush=True)
        expression, expression_panel_genes = aggregate_tf_expression(
            args.h5ad,
            sorted(tests["group"].unique()),
            annotation,
            args.expression_chunk_size,
        )

    motif_activity_matrices, motif_activity_n_valid = (
        aggregate_weighted_motif_activity(tests, motif, min_target_t7=50.0)
    )
    motif_activity_long = make_motif_activity_long(
        motif_activity_matrices,
        motif_activity_n_valid,
        annotation,
        expression,
        expression_panel_genes,
    )
    motif_correlation_matrices, motif_correlation_n_valid = (
        correlate_motif_scores_with_activity(
            tests,
            motif,
            min_target_t7=50.0,
        )
    )
    if not motif_correlation_n_valid.equals(motif_activity_n_valid):
        raise ValueError(
            "Weighted activity and correlation analyses used different valid "
            "cCRE counts"
        )
    motif_correlation_long = make_motif_activity_correlation_long(
        motif_correlation_matrices,
        motif_correlation_n_valid,
        annotation,
        expression,
        expression_panel_genes,
    )

    enrichment, summary = compare_motifs(
        tests,
        motif,
        0.0,
        args.min_significant_ccres,
        args.min_nonsignificant_ccres,
    )
    enrichment = add_tf_annotation_and_expression(
        enrichment, annotation, expression, expression_panel_genes
    )
    ccre_metadata_columns = [
        column
        for column in (
            "enh",
            "label",
            "labeling_type",
            "subclass_num",
            "best_subclass",
            "Chrom",
            "Start",
            "End",
        )
        if column in cre_info.columns
    ]
    wide, long = annotate_significant_ccres(
        tests, motif, cre_info.loc[:, ccre_metadata_columns], 0.0
    )
    long = add_tf_annotation_and_expression(
        long, annotation, expression, expression_panel_genes
    )
    gene_matrix, gene_metadata = collapse_motifs_to_tf_genes(motif, annotation)
    tf_gene_enrichment, top_tf_genes, top_expression_supported = compare_tf_genes(
        tests,
        gene_matrix,
        gene_metadata,
        expression,
        expression_panel_genes,
        args,
    )
    summary, top, recurrence = finish_summaries(
        enrichment,
        summary,
        args.motif_q_cutoff,
        args.top_n_per_cell_type,
    )
    recurrence = add_tf_annotation_and_expression(
        recurrence, annotation, pd.DataFrame(), expression_panel_genes
    )

    enrichment_path = tables_dir / "motif_enrichment_by_cell_type.csv.gz"
    summary_path = tables_dir / "cell_type_comparison_summary.csv"
    top_path = tables_dir / "top_motifs_by_cell_type.csv"
    recurrence_path = tables_dir / "motif_recurrence_summary.csv"
    wide_path = tables_dir / "significant_ccre_motif_profiles.csv.gz"
    long_path = tables_dir / "significant_ccre_motifs_long.csv.gz"
    sets_path = tables_dir / "ccre_sets_by_cell_type.csv"
    annotation_path = tables_dir / "hocomoco_motif_to_mouse_gene.csv.gz"
    expression_path = tables_dir / "hocomoco_tf_expression_by_cell_type.csv.gz"
    hits_path = tables_dir / "hocomoco_fimo_best_hits.csv.gz"
    fimo_backgrounds_path = tables_dir / "fimo_background_by_cell_type.csv"
    motif_activity_path = (
        tables_dir / "cell_type_by_motif_activity.csv.gz"
    )
    motif_activity_sum_path = (
        tables_dir / "cell_type_by_motif_activity_sum.csv.gz"
    )
    motif_activity_zscore_path = (
        tables_dir / "cell_type_by_motif_activity_zscore.csv.gz"
    )
    motif_occurrence_path = (
        tables_dir / "cell_type_by_motif_occurrence_fraction.csv.gz"
    )
    motif_hit_count_path = (
        tables_dir / "cell_type_by_motif_hit_count.csv.gz"
    )
    motif_matching_score_all_path = (
        tables_dir / "cell_type_by_motif_matching_score_mean_all.csv.gz"
    )
    motif_matching_score_present_path = (
        tables_dir / "cell_type_by_motif_matching_score_mean_present.csv.gz"
    )
    motif_activity_long_path = (
        tables_dir / "motif_activity_by_cell_type_long.csv.gz"
    )
    motif_pearson_path = (
        tables_dir / "cell_type_by_motif_activity_pearson_r.csv.gz"
    )
    motif_pearson_p_path = (
        tables_dir / "cell_type_by_motif_activity_pearson_p.csv.gz"
    )
    motif_pearson_q_path = (
        tables_dir / "cell_type_by_motif_activity_pearson_q_cell_type.csv.gz"
    )
    motif_spearman_path = (
        tables_dir / "cell_type_by_motif_activity_spearman_rho.csv.gz"
    )
    motif_spearman_p_path = (
        tables_dir / "cell_type_by_motif_activity_spearman_p.csv.gz"
    )
    motif_spearman_q_path = (
        tables_dir / "cell_type_by_motif_activity_spearman_q_cell_type.csv.gz"
    )
    motif_correlation_long_path = (
        tables_dir / "motif_activity_correlation_by_cell_type_long.csv.gz"
    )
    cell_type_number_prefixes_path = (
        tables_dir / "cell_type_number_prefixes.csv"
    )
    tf_gene_enrichment_path = (
        tables_dir / "tf_gene_motif_enrichment_by_cell_type.csv.gz"
    )
    top_tf_genes_path = tables_dir / "top_tf_genes_by_cell_type.csv"
    top_expression_supported_path = (
        tables_dir / "top_expression_supported_tf_genes_by_cell_type.csv"
    )
    gene_metadata_path = tables_dir / "hocomoco_mouse_tf_gene_metadata.csv.gz"
    gene_matrix_path = (
        args.results_dir / "intermediate" / "hocomoco_tf_gene_motif_matrix.csv.gz"
    )

    enrichment.to_csv(enrichment_path, index=False)
    summary.to_csv(summary_path, index=False)
    top.to_csv(top_path, index=False)
    recurrence.to_csv(recurrence_path, index=False)
    wide.to_csv(wide_path, index=False)
    long.to_csv(long_path, index=False)
    annotation.to_csv(annotation_path, index=False)
    expression.to_csv(expression_path, index=False)
    fimo_hits.to_csv(hits_path, index=False)
    fimo_backgrounds.to_csv(fimo_backgrounds_path, index=False)
    motif_activity_matrices["activity_mean"].to_csv(motif_activity_path)
    motif_activity_matrices["activity_sum"].to_csv(motif_activity_sum_path)
    motif_activity_matrices["activity_zscore"].to_csv(
        motif_activity_zscore_path
    )
    motif_activity_matrices["occurrence_fraction"].to_csv(
        motif_occurrence_path
    )
    motif_activity_matrices["hit_count"].to_csv(motif_hit_count_path)
    motif_activity_matrices["matching_score_mean_all"].to_csv(
        motif_matching_score_all_path
    )
    motif_activity_matrices["matching_score_mean_present"].to_csv(
        motif_matching_score_present_path
    )
    motif_activity_long.to_csv(motif_activity_long_path, index=False)
    motif_correlation_matrices["pearson_r"].to_csv(motif_pearson_path)
    motif_correlation_matrices["pearson_p_value"].to_csv(motif_pearson_p_path)
    motif_correlation_matrices["pearson_q_value_cell_type"].to_csv(
        motif_pearson_q_path
    )
    motif_correlation_matrices["spearman_rho"].to_csv(motif_spearman_path)
    motif_correlation_matrices["spearman_p_value"].to_csv(
        motif_spearman_p_path
    )
    motif_correlation_matrices["spearman_q_value_cell_type"].to_csv(
        motif_spearman_q_path
    )
    motif_correlation_long.to_csv(motif_correlation_long_path, index=False)
    (
        cell_type_number_prefixes.rename_axis("group")
        .reset_index()
        .assign(
            numbered_cell_type=lambda frame: frame.apply(
                lambda row: (
                    f"{int(row['numbered_prefix']):03d} {row['group']}"
                ),
                axis=1,
            )
        )
        .sort_values("numbered_prefix")
        .to_csv(cell_type_number_prefixes_path, index=False)
    )
    tf_gene_enrichment.to_csv(tf_gene_enrichment_path, index=False)
    top_tf_genes.to_csv(top_tf_genes_path, index=False)
    top_expression_supported.to_csv(top_expression_supported_path, index=False)
    gene_metadata.to_csv(gene_metadata_path, index=False)
    gene_matrix.to_csv(gene_matrix_path)
    tests.loc[
        valid_t7_mask,
        ["method", "group", "class", "cre", "q_right", "is_significant"],
    ].to_csv(sets_path, index=False)

    plot_top_enrichments(
        enrichment, figures_dir, args.motif_q_cutoff, args.plot_top_n
    )
    plot_enrichment_heatmap(
        enrichment,
        recurrence,
        figures_dir,
        args.motif_q_cutoff,
        args.heatmap_top_motifs,
    )
    plot_weighted_motif_activity_heatmap(
        motif_activity_matrices["activity_mean"],
        annotation,
        cell_type_number_prefixes,
        motif_activity_n_valid,
        figures_dir,
        args.activity_heatmap_top_motifs,
        args.activity_heatmap_min_valid_ccres,
        "weighted_motif_activity_heatmap",
        "Top variable motif activities across valid T7≥50 cCREs",
        "Mean [activity − negative-control mean] × [−log10(best FIMO p)]",
    )
    plot_weighted_motif_activity_heatmap(
        motif_activity_matrices["activity_zscore"],
        annotation,
        cell_type_number_prefixes,
        motif_activity_n_valid,
        figures_dir,
        args.activity_heatmap_top_motifs,
        args.activity_heatmap_min_valid_ccres,
        "weighted_motif_activity_zscore_heatmap",
        "Within-cell-type z-scored motif activities",
        "Motif activity z-score within cell type",
        cluster_cell_types=True,
    )
    plot_weighted_motif_activity_heatmap(
        motif_correlation_matrices["pearson_r"],
        annotation,
        cell_type_number_prefixes,
        motif_correlation_n_valid,
        figures_dir,
        args.activity_heatmap_top_motifs,
        args.activity_heatmap_min_valid_ccres,
        "motif_activity_pearson_correlation_heatmap",
        "cCRE activity versus motif matching score: Pearson correlation",
        "Pearson r across all valid T7≥50 cCREs",
        selection="max_absolute",
        fixed_color_limit=1.0,
    )
    plot_weighted_motif_activity_heatmap(
        motif_correlation_matrices["spearman_rho"],
        annotation,
        cell_type_number_prefixes,
        motif_correlation_n_valid,
        figures_dir,
        args.activity_heatmap_top_motifs,
        args.activity_heatmap_min_valid_ccres,
        "motif_activity_spearman_correlation_heatmap",
        "cCRE activity versus motif matching score: Spearman correlation",
        "Spearman ρ across all valid T7≥50 cCREs",
        selection="max_absolute",
        fixed_color_limit=1.0,
    )

    manifest = {
        "analysis": (
            "Whole-dataset, per-cell-type comparison of joint-dropout "
            "mean-negative-control significant versus non-significant cCREs"
        ),
        "inputs": {
            "tests": str(args.tests.resolve()),
            "tests_sha256": sha256(args.tests),
            "cre_info": str(args.cre_info.resolve()),
            "cre_info_sha256": sha256(args.cre_info),
            "assayed_sequence_column": "sequence",
            "hocomoco_meme": str(args.hocomoco_meme.resolve()),
            "hocomoco_meme_sha256": sha256(args.hocomoco_meme),
            "hocomoco_annotation": str(args.hocomoco_annotation.resolve()),
            "hocomoco_annotation_sha256": sha256(args.hocomoco_annotation),
            "fimo_bin": str(args.fimo_bin.resolve()),
            "fasta_get_markov_bin": str(args.fasta_get_markov_bin.resolve()),
            "expression_h5ad": (
                None if args.skip_expression else str(args.h5ad.resolve())
            ),
        },
        "definitions": {
            "significant": f"q_right <= {args.q_cutoff:g}",
            "enrichment_background": (
                "finite-q, target-T7>=50 motif-covered cCREs tested in the "
                "same cell type with q_right above the significance cutoff"
            ),
            "fimo_sequence_background": (
                "separate zero-order DNA Markov frequencies for each cell "
                "type, estimated by fasta-get-markov from all finite-q, "
                "target-T7>=50 cCRE sequences in that cell type (significant and "
                "non-significant together), with reverse complements combined "
                "and pseudocount 0.1; passed to FIMO with --bgfile"
            ),
            "motif_space": (
                "HOCOMOCO v14 CORE motifs with mouse gene annotation"
            ),
            "motif_present": (
                "at least one FIMO hit in the actual assayed cCRE insert "
                f"with p <= {args.fimo_pvalue:g}"
            ),
            "motif_score": (
                "-log10 of the best FIMO hit p-value for a cCRE/motif pair; "
                "zero means no hit"
            ),
            "weighted_motif_activity": (
                "for every finite-q cCRE/cell-type pair with target_t7_total "
                ">= 50, effect_vs_mean_control_mean (posterior mean activity "
                "minus the draw-wise negative-control mean) multiplied by "
                "-log10(best FIMO p); score zero for no motif hit; negative "
                "effects are retained"
            ),
            "primary_cell_type_by_motif_activity_matrix": (
                "mean weighted motif contribution over all valid T7>=50 cCREs "
                "in each cell type; the sum matrix is also exported"
            ),
            "cell_type_by_motif_activity_zscore": (
                "within each cell type, subtract the mean activity over all "
                "HOCOMOCO motifs and divide by the population standard "
                "deviation over all motifs (ddof=0)"
            ),
            "motif_activity_correlations": (
                "within each cell type, Pearson r and Spearman rho between "
                "effect_vs_mean_control_mean and -log10(best FIMO p) over all "
                "finite-q, target-T7>=50 cCREs; score zero represents no "
                "qualifying motif hit; two-sided p-values and BH FDR across "
                "all nonconstant motif models are reported per cell type"
            ),
            "motif_activity_heatmap_ordering": (
                "cell-type rows with at least the configured number of valid "
                "cCREs, ordered by the original numbered h5ad subclass "
                "prefix; selected motif columns ordered by average-linkage "
                "hierarchical clustering with correlation distance"
            ),
            "motif_correlation_heatmap_selection": (
                "top configured number of motifs ranked by maximum absolute "
                "correlation across retained cell types; Pearson and Spearman "
                "figures are selected separately; undefined correlations are "
                "gray and treated as zero only for column clustering"
            ),
            "tf_expression": (
                None
                if args.skip_expression
                else (
                    "mean X and fraction X>0 for the HOCOMOCO mouse gene in "
                    "each standardized h5ad subclass; unavailable genes are "
                    "marked tf_gene_in_expression_panel=False"
                )
            ),
            "test": "one-sided Fisher exact test, alternative=greater",
            "multiple_testing": (
                "BH within each cell type (q_value_cell_type) and across all "
                "cell-type/motif tests (q_value_global)"
            ),
            "analysis_scope": "whole dataset; no anatomical-section split",
        },
        "parameters": {
            "expected_method": args.expected_method,
            "q_cutoff": args.q_cutoff,
            "fimo_pvalue": args.fimo_pvalue,
            "fimo_jobs": args.fimo_jobs,
            "motif_activity_min_target_t7": 50.0,
            "activity_heatmap_top_motifs": (
                args.activity_heatmap_top_motifs
            ),
            "activity_heatmap_min_valid_ccres": (
                args.activity_heatmap_min_valid_ccres
            ),
            "motif_q_cutoff": args.motif_q_cutoff,
            "min_significant_ccres": args.min_significant_ccres,
            "min_nonsignificant_ccres": args.min_nonsignificant_ccres,
            "top_n_per_cell_type": args.top_n_per_cell_type,
            "skip_expression": args.skip_expression,
        },
        "counts": {
            "eligible_test_rows": int(len(tests)),
            "cell_types": int(tests["group"].nunique()),
            "significant_ccre_cell_type_pairs": int(tests["is_significant"].sum()),
            "cell_types_with_significant_ccres": int(
                tests.loc[tests["is_significant"], "group"].nunique()
            ),
            "tested_unique_ccres": int(len(tested_ccres)),
            "valid_ccre_cell_type_pairs": int(len(valid_pairs)),
            "motif_covered_valid_ccre_cell_type_pairs": int(len(motif)),
            "valid_ccre_cell_type_pairs_missing_motif_data": int(
                len(missing_motif_pairs)
            ),
            "cell_type_specific_fimo_backgrounds": int(len(fimo_backgrounds)),
            "hocomoco_mouse_motifs_annotated": int(len(annotation)),
            "hocomoco_mouse_tf_genes_annotated": int(
                annotation["mouse_gene_symbol"].nunique()
            ),
            "hocomoco_motifs_with_fimo_hits": int(motif.shape[1]),
            "fimo_best_hits": int(len(fimo_hits)),
            "motif_activity_cell_types": int(
                motif_activity_matrices["activity_mean"].shape[0]
            ),
            "motif_activity_motifs": int(
                motif_activity_matrices["activity_mean"].shape[1]
            ),
            "motif_activity_valid_ccre_cell_type_pairs": int(
                motif_activity_n_valid.sum()
            ),
            "motif_activity_pearson_defined_tests": int(
                motif_correlation_matrices["pearson_r"].notna().sum().sum()
            ),
            "motif_activity_spearman_defined_tests": int(
                motif_correlation_matrices["spearman_rho"].notna().sum().sum()
            ),
            "motif_activity_heatmap_cell_types": int(
                motif_activity_n_valid.ge(
                    args.activity_heatmap_min_valid_ccres
                ).sum()
            ),
            "motif_activity_heatmap_excluded_cell_types": int(
                motif_activity_n_valid.lt(
                    args.activity_heatmap_min_valid_ccres
                ).sum()
            ),
            "hocomoco_tf_genes_in_expression_panel": int(
                len(expression_panel_genes)
            ),
            "cell_type_motif_tests": int(len(enrichment)),
            "hocomoco_tf_genes_with_fimo_hits": int(gene_matrix.shape[1]),
            "cell_type_tf_gene_tests": int(len(tf_gene_enrichment)),
            "within_cell_type_q_le_cutoff": int(
                enrichment["q_value_cell_type"].le(args.motif_q_cutoff).sum()
            )
            if not enrichment.empty
            else 0,
            "global_q_le_cutoff": int(
                enrichment["q_value_global"].le(args.motif_q_cutoff).sum()
            )
            if not enrichment.empty
            else 0,
            "tf_gene_within_cell_type_q_le_cutoff": int(
                tf_gene_enrichment["q_value_cell_type"]
                .le(args.motif_q_cutoff)
                .sum()
            )
            if not tf_gene_enrichment.empty
            else 0,
            "tf_gene_global_q_le_cutoff": int(
                tf_gene_enrichment["q_value_global"].le(args.motif_q_cutoff).sum()
            )
            if not tf_gene_enrichment.empty
            else 0,
        },
        "outputs": {
            "enrichment": str(enrichment_path.resolve()),
            "cell_type_summary": str(summary_path.resolve()),
            "top_motifs": str(top_path.resolve()),
            "motif_recurrence": str(recurrence_path.resolve()),
            "significant_profiles": str(wide_path.resolve()),
            "significant_motifs_long": str(long_path.resolve()),
            "ccre_sets": str(sets_path.resolve()),
            "hocomoco_annotation": str(annotation_path.resolve()),
            "hocomoco_tf_expression": str(expression_path.resolve()),
            "fimo_best_hits": str(hits_path.resolve()),
            "fimo_background_by_cell_type": str(
                fimo_backgrounds_path.resolve()
            ),
            "cell_type_by_motif_activity": str(
                motif_activity_path.resolve()
            ),
            "cell_type_by_motif_activity_sum": str(
                motif_activity_sum_path.resolve()
            ),
            "cell_type_by_motif_activity_zscore": str(
                motif_activity_zscore_path.resolve()
            ),
            "cell_type_by_motif_occurrence_fraction": str(
                motif_occurrence_path.resolve()
            ),
            "cell_type_by_motif_hit_count": str(
                motif_hit_count_path.resolve()
            ),
            "cell_type_by_motif_matching_score_mean_all": str(
                motif_matching_score_all_path.resolve()
            ),
            "cell_type_by_motif_matching_score_mean_present": str(
                motif_matching_score_present_path.resolve()
            ),
            "motif_activity_by_cell_type_long": str(
                motif_activity_long_path.resolve()
            ),
            "cell_type_by_motif_activity_pearson_r": str(
                motif_pearson_path.resolve()
            ),
            "cell_type_by_motif_activity_pearson_p": str(
                motif_pearson_p_path.resolve()
            ),
            "cell_type_by_motif_activity_pearson_q_cell_type": str(
                motif_pearson_q_path.resolve()
            ),
            "cell_type_by_motif_activity_spearman_rho": str(
                motif_spearman_path.resolve()
            ),
            "cell_type_by_motif_activity_spearman_p": str(
                motif_spearman_p_path.resolve()
            ),
            "cell_type_by_motif_activity_spearman_q_cell_type": str(
                motif_spearman_q_path.resolve()
            ),
            "motif_activity_correlation_by_cell_type_long": str(
                motif_correlation_long_path.resolve()
            ),
            "cell_type_number_prefixes": str(
                cell_type_number_prefixes_path.resolve()
            ),
            "tf_gene_enrichment": str(tf_gene_enrichment_path.resolve()),
            "top_tf_genes": str(top_tf_genes_path.resolve()),
            "top_expression_supported_tf_genes": str(
                top_expression_supported_path.resolve()
            ),
            "hocomoco_mouse_tf_gene_metadata": str(gene_metadata_path.resolve()),
            "hocomoco_tf_gene_motif_matrix": str(gene_matrix_path.resolve()),
        },
        "fimo_scan_signature": scan_signature,
    }
    manifest_path = args.results_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps(manifest["counts"], indent=2))
    print(f"[TFMotif] wrote results to {args.results_dir}")


if __name__ == "__main__":
    main()
