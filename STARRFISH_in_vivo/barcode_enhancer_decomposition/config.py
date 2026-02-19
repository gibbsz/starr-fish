"""Default configuration for barcode-enhancer decomposition analysis."""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic")

# Real data paths (relative to the STARRFISH_in_vivo root)
_INVIVO_ROOT = os.path.dirname(PROJECT_ROOT)
ACTIVITY_CSV = os.path.join(_INVIVO_ROOT, "results", "intrinsic_activity.csv")
SEQUENCES_XLSX = os.path.join(_INVIVO_ROOT, "Data", "Supplementary Tables.xlsx")
SEQUENCES_SHEET = "Supplementary Table 6"
LIBRARY_FILTER = "400CRE/in vivo"
CELLTYPE_COUNTS_CSV = os.path.join(_INVIVO_ROOT, "results", "expr3",
                                    "celltype_number.csv")
MIN_CELLS = 1000              # only keep cell types with >= this many cells

# ── Coordinate-based motif scanning (legacy / optional) ──────────────────
CRE_BED = os.path.join(_INVIVO_ROOT, "Data", "CRE.bed")
MOTIF_BED = os.path.join(_INVIVO_ROOT, "Data", "annotation",
                          "mm10.all_motifs.v1.0.bed.gz")

# ── Distance / kernel method ──────────────────────────────────────────────
# IMPORTANT: The same method is used for BOTH barcode and enhancer to avoid
# bias in the decomposition.  "kmer" works on raw sequences; "motif" runs
# FIMO against a TF-motif database and uses cosine distance/kernel on the
# resulting motif-score vectors.
METHOD = "kmer"               # "kmer" or "motif" — applied to BOTH
KMER_K = 6                    # k for k-mer distance / kernel (single-k legacy)
KMER_KS = [6, 10, 14]        # k values combined into one distance/kernel
BOTH_STRANDS = True           # consider reverse complement and take min distance

# ── FIMO motif scanning (used when METHOD = "motif") ─────────────────────
FIMO_BIN = "/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN/bin/fimo"
MOTIF_DB_MOUSE = (
    "/gpfs/commons/groups/ren_lab/guojiezhong/BICAN/Data/source/"
    "meme-5.4.1/motif_databases/MOUSE/HOCOMOCOv11_full_MOUSE_mono_meme_format.meme"
)
FIMO_PVAL = 1e-4

# ── Pre-computed motif matrices (optional overrides) ─────────────────────
ENHANCER_MOTIF_CSV = os.path.join(_INVIVO_ROOT, "results", "CRE_motif.csv")
BARCODE_MOTIF_CSV = os.path.join(RESULTS_DIR, "barcode_motif.csv")

# ── Partial Mantel test ────────────────────────────────────────────────────
MANTEL_PERMUTATIONS = 9999

# ── Variance decomposition ─────────────────────────────────────────────────
KERNEL_TYPE = "kmer"          # "kmer" or "rbf" (only used when METHOD="kmer")
REML_MAX_ITER = 500
REML_TOL = 1e-6
FDR_ALPHA = 0.05
N_BOOTSTRAP = 100             # Number of bootstrap iterations for CI estimation
N_JOBS = -1                   # Number of parallel jobs for bootstrap (-1 = all cores)

# ── Synthetic data defaults ────────────────────────────────────────────────
SYNTH_N_CONSTRUCTS = 400
SYNTH_N_CELLTYPES = 300
SYNTH_BARCODE_LEN = 20
SYNTH_ENHANCER_LEN = 200
SYNTH_SIGMA2_BARCODE = 0.1
SYNTH_SIGMA2_ENHANCER = 1.0
SYNTH_SIGMA2_NOISE = 0.2
SYNTH_N_MOTIFS_BARCODE = 3
SYNTH_N_MOTIFS_ENHANCER = 10
