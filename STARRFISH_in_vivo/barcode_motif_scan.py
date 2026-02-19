#!/usr/bin/env python3
"""
Scan barcode sequences from Supplementary Table 6 for TF motifs using FIMO
against HOCOMOCOv11 full MOUSE motif database at multiple p-value thresholds.
"""

import os
import subprocess
import pandas as pd
from io import StringIO
from pathlib import Path

# --- Configuration ---
EXCEL_PATH = "Data/Supplementary Tables.xlsx"
SHEET_NAME = "Supplementary Table 6"
MOTIF_DB = "/gpfs/commons/groups/ren_lab/guojiezhong/BICAN/Data/source/meme-5.4.1/motif_databases/MOUSE/HOCOMOCOv11_full_MOUSE_mono_meme_format.meme"
FIMO_BIN = "/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN/bin/fimo"
OUTPUT_BASE = "results/barcode_motif_scan"
PVAL_THRESHOLDS = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]


def read_barcode_sequences(excel_path, sheet_name):
    """Read barcode sequences from Supplementary Table 6."""
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=1)
    df = df.dropna(subset=["Barcode sequence (5'-3')"])
    print(f"Read {len(df)} barcode sequences from {sheet_name}")
    return df


def write_fasta(df, fasta_path):
    """Write barcode sequences to a FASTA file."""
    with open(fasta_path, "w") as f:
        for _, row in df.iterrows():
            lib = row["Library/experiment"].replace("/", "_").replace(" ", "-")
            eid = row["Enhancer ID"]
            seq = row["Barcode sequence (5'-3')"]
            f.write(f">{lib}|{eid}\n{seq.upper()}\n")
    print(f"Wrote {len(df)} sequences to {fasta_path}")


def run_fimo(fasta_path, motif_db, out_dir, thresh):
    """Run FIMO with a given p-value threshold."""
    fimo_out = os.path.join(out_dir, "fimo_raw")
    os.makedirs(fimo_out, exist_ok=True)

    cmd = [
        FIMO_BIN,
        "--thresh", str(thresh),
        "--oc", fimo_out,
        motif_db,
        fasta_path,
    ]
    print(f"  Running FIMO (p < {thresh}) ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: FIMO returned {result.returncode}")
        if result.stderr:
            print(f"  stderr: {result.stderr[:500]}")
        return None

    # FIMO 4.x outputs fimo.txt, FIMO 5.x outputs fimo.tsv
    tsv_path = os.path.join(fimo_out, "fimo.tsv")
    txt_path = os.path.join(fimo_out, "fimo.txt")
    hit_path = tsv_path if os.path.exists(tsv_path) else txt_path
    if os.path.exists(hit_path):
        with open(hit_path) as f:
            lines = [l for l in f if not l.startswith("##")]
        if not lines:
            return None
        lines[0] = lines[0].lstrip("#")
        hits = pd.read_csv(StringIO("".join(lines)), sep="\t")
        hits = hits.dropna(how="all")
        col_map = {
            "pattern name": "motif_id",
            "sequence name": "sequence_name",
            "matched sequence": "matched_sequence",
        }
        hits = hits.rename(columns=col_map)
        if "motif_alt_id" not in hits.columns:
            hits["motif_alt_id"] = hits["motif_id"]
        if len(hits) > 0:
            hits[["library", "enhancer_id"]] = hits["sequence_name"].str.split(
                "|", n=1, expand=True
            )
        else:
            hits["library"] = []
            hits["enhancer_id"] = []
        return hits
    return None


def save_results(hits, thresh, out_dir):
    """Save FIMO results and summaries for a given threshold."""
    thresh_label = f"{thresh:.0e}"
    thresh_dir = os.path.join(out_dir, f"thresh_{thresh_label}")
    os.makedirs(thresh_dir, exist_ok=True)

    hits = hits.sort_values(["sequence_name", "p-value"])
    hits.to_csv(os.path.join(thresh_dir, "all_motif_hits.tsv"), sep="\t", index=False)

    if len(hits) > 0:
        summary = (
            hits.groupby(["library", "enhancer_id"])
            .agg(
                n_hits=("motif_id", "count"),
                n_unique_motifs=("motif_id", "nunique"),
                top_motifs=("motif_id", lambda x: ",".join(x.unique()[:5])),
                best_pval=("p-value", "min"),
            )
            .reset_index()
            .sort_values("n_hits", ascending=False)
        )
        summary.to_csv(os.path.join(thresh_dir, "motif_summary_per_barcode.tsv"), sep="\t", index=False)

        motif_counts = (
            hits.groupby(["motif_id", "motif_alt_id"])
            .agg(
                n_sequences=("sequence_name", "nunique"),
                n_hits=("sequence_name", "count"),
                mean_pval=("p-value", "mean"),
            )
            .reset_index()
            .sort_values("n_sequences", ascending=False)
        )
        motif_counts.to_csv(os.path.join(thresh_dir, "motif_frequency_summary.tsv"), sep="\t", index=False)

    return {
        "pval_threshold": thresh_label,
        "total_hits": len(hits),
        "unique_motifs": hits["motif_id"].nunique() if len(hits) > 0 else 0,
        "sequences_with_hits": hits["sequence_name"].nunique() if len(hits) > 0 else 0,
    }


def main():
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    df = read_barcode_sequences(EXCEL_PATH, SHEET_NAME)

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    fasta_path = os.path.join(OUTPUT_BASE, "barcode_sequences.fa")
    write_fasta(df, fasta_path)

    print(f"\nMotif database: {Path(MOTIF_DB).name}")

    # Run FIMO at each p-value threshold
    summaries = []
    for thresh in PVAL_THRESHOLDS:
        print(f"\n{'='*60}")
        print(f"p-value threshold: {thresh:.0e}")
        print(f"{'='*60}")

        hits = run_fimo(fasta_path, MOTIF_DB, OUTPUT_BASE, thresh)
        if hits is not None and len(hits) > 0:
            s = save_results(hits, thresh, OUTPUT_BASE)
            print(f"  {s['total_hits']} hits, {s['unique_motifs']} motifs, "
                  f"{s['sequences_with_hits']} sequences")
        else:
            s = {"pval_threshold": f"{thresh:.0e}", "total_hits": 0,
                 "unique_motifs": 0, "sequences_with_hits": 0}
            print(f"  No hits found.")
        summaries.append(s)

    # Print comparison table
    print(f"\n{'='*60}")
    print("Summary across p-value thresholds")
    print(f"{'='*60}")
    summary_df = pd.DataFrame(summaries)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(OUTPUT_BASE, "threshold_comparison.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
