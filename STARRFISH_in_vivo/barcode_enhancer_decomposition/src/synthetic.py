"""Generate synthetic data with known ground truth."""

import numpy as np
import pandas as pd


def _random_dna(length, rng):
    """Generate a random DNA sequence of given length."""
    return "".join(rng.choice(list("ACGT"), size=length))


def _random_motif(length, rng):
    """Generate a random motif."""
    return _random_dna(length, rng)


def _embed_motifs(sequences, motifs, embed_prob, rng):
    """Embed motifs into a random subset of sequences.

    For each motif, with probability `embed_prob` per sequence,
    insert the motif at a random position (overwrite).
    """
    out = list(sequences)
    for motif in motifs:
        mlen = len(motif)
        for i in range(len(out)):
            if rng.random() < embed_prob:
                seq = list(out[i])
                pos = rng.integers(0, len(seq) - mlen + 1)
                seq[pos:pos + mlen] = list(motif)
                out[i] = "".join(seq)
    return out


def _count_motifs(sequences, motifs):
    """Count occurrences of each motif in each sequence.

    Returns shape (n_sequences, n_motifs).
    """
    n = len(sequences)
    m = len(motifs)
    counts = np.zeros((n, m))
    for j, motif in enumerate(motifs):
        for i, seq in enumerate(sequences):
            start = 0
            while True:
                pos = seq.find(motif, start)
                if pos == -1:
                    break
                counts[i, j] += 1
                start = pos + 1
    return counts


def generate_synthetic_data(
    n_constructs=400,
    n_celltypes=300,
    barcode_len=20,
    enhancer_len=200,
    sigma2_barcode_true=0.1,
    sigma2_enhancer_true=1.0,
    sigma2_noise_true=0.2,
    n_motifs_barcode=3,
    n_motifs_enhancer=10,
    seed=42,
):
    """Generate synthetic data with known variance components.

    Returns
    -------
    activity : pd.DataFrame
        Shape (n_constructs, n_celltypes).
    barcode_seqs : list of str
    enhancer_seqs : list of str
    ground_truth : dict
        Contains true variance components and intermediate values.
    """
    rng = np.random.default_rng(seed)

    # 1. Generate random DNA sequences
    barcode_seqs = [_random_dna(barcode_len, rng) for _ in range(n_constructs)]
    enhancer_seqs = [_random_dna(enhancer_len, rng) for _ in range(n_constructs)]

    # 2. Barcode motifs and embedding
    bc_motifs = [_random_motif(rng.integers(4, 7), rng)
                 for _ in range(n_motifs_barcode)]
    barcode_seqs = _embed_motifs(barcode_seqs, bc_motifs, embed_prob=0.4, rng=rng)
    bc_counts = _count_motifs(barcode_seqs, bc_motifs)  # (n_constructs, n_motifs_bc)

    # Barcode activity: mostly cell-type invariant, small noise
    bc_weights = rng.standard_normal(n_motifs_barcode)  # shared across cell types
    barcode_activity_base = bc_counts @ bc_weights       # (n_constructs,)
    # Add small cell-type noise
    bc_celltype_noise = rng.standard_normal((n_constructs, n_celltypes)) * 0.05
    barcode_activity = barcode_activity_base[:, None] + bc_celltype_noise

    # Scale to target variance
    if barcode_activity.var() > 0:
        barcode_activity *= np.sqrt(sigma2_barcode_true / barcode_activity.var())

    # 3. Enhancer motifs and embedding
    enh_motifs = [_random_motif(rng.integers(6, 11), rng)
                  for _ in range(n_motifs_enhancer)]
    enhancer_seqs = _embed_motifs(enhancer_seqs, enh_motifs, embed_prob=0.5, rng=rng)
    enh_counts = _count_motifs(enhancer_seqs, enh_motifs)

    # Cell-type-specific weights from a low-rank structure (simulate TF patterns)
    n_factors = 5
    U = rng.standard_normal((n_motifs_enhancer, n_factors))
    V = rng.standard_normal((n_factors, n_celltypes))
    enh_weights = U @ V  # (n_motifs_enhancer, n_celltypes)
    enhancer_activity = enh_counts @ enh_weights  # (n_constructs, n_celltypes)

    # Scale to target variance
    if enhancer_activity.var() > 0:
        enhancer_activity *= np.sqrt(sigma2_enhancer_true / enhancer_activity.var())

    # 4. Noise
    noise = rng.standard_normal((n_constructs, n_celltypes))
    noise *= np.sqrt(sigma2_noise_true)

    # 5. Total activity
    total = barcode_activity + enhancer_activity + noise

    # Build DataFrame
    construct_ids = [f"CRE{i+1:03d}" for i in range(n_constructs)]
    celltype_ids = [f"CellType{j+1:03d}" for j in range(n_celltypes)]
    activity = pd.DataFrame(total, index=construct_ids, columns=celltype_ids)

    # Ground truth
    actual_var_bc = barcode_activity.var()
    actual_var_enh = enhancer_activity.var()
    actual_var_noise = noise.var()
    total_var = actual_var_bc + actual_var_enh + actual_var_noise
    ground_truth = {
        "sigma2_barcode": actual_var_bc,
        "sigma2_enhancer": actual_var_enh,
        "sigma2_noise": actual_var_noise,
        "prop_barcode": actual_var_bc / total_var,
        "prop_enhancer": actual_var_enh / total_var,
        "prop_noise": actual_var_noise / total_var,
        "barcode_motifs": bc_motifs,
        "enhancer_motifs": enh_motifs,
    }

    print(f"Synthetic data generated: {n_constructs} constructs × {n_celltypes} cell types")
    print(f"  True variance proportions: "
          f"barcode={ground_truth['prop_barcode']:.3f}, "
          f"enhancer={ground_truth['prop_enhancer']:.3f}, "
          f"noise={ground_truth['prop_noise']:.3f}")

    return activity, barcode_seqs, enhancer_seqs, ground_truth
