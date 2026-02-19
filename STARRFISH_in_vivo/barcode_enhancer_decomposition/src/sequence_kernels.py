"""Sequence distance and kernel computation.

For all pairwise distances, both strands are considered: for each pair (i, j),
we compute distance(seq_i, seq_j) and distance(seq_i, revcomp(seq_j)),
and take the minimum.
"""

import numpy as np
from collections import Counter
from itertools import product
from tqdm import tqdm


# -- Helpers -----------------------------------------------------------------

_COMP = str.maketrans("ACGTacgt", "TGCAtgca")


def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence."""
    return seq.translate(_COMP)[::-1]


def _kmer_counts(seq, k):
    """Return a Counter of k-mers in seq."""
    c = Counter()
    seq_upper = seq.upper()
    for i in range(len(seq_upper) - k + 1):
        c[seq_upper[i:i + k]] += 1
    return c


def _kmer_vector(seq, k, all_kmers):
    """Return k-mer frequency vector for a sequence."""
    counts = _kmer_counts(seq, k)
    return np.array([counts.get(km, 0) for km in all_kmers], dtype=np.float64)


def _all_kmers(k):
    """Generate all possible DNA k-mers in sorted order."""
    return ["".join(p) for p in product("ACGT", repeat=k)]


# -- Distance matrices ------------------------------------------------------

def levenshtein_distance_matrix(sequences, both_strands=True, normalized=True):
    """Compute pairwise Levenshtein distance matrix.

    When both_strands=True, for each pair computes
    min(d(s_i, s_j), d(s_i, rc(s_j))) — the smallest distance
    considering both strands.
    """
    import Levenshtein as lev

    n = len(sequences)
    seqs_upper = [s.upper() for s in sequences]
    rc_seqs = [reverse_complement(s) for s in seqs_upper] if both_strands else None

    D = np.zeros((n, n))
    for i in tqdm(range(n), desc="Levenshtein distance"):
        for j in range(i + 1, n):
            d_fwd = lev.distance(seqs_upper[i], seqs_upper[j])
            if both_strands:
                d_rc = lev.distance(seqs_upper[i], rc_seqs[j])
                d = min(d_fwd, d_rc)
            else:
                d = d_fwd
            if normalized:
                max_len = max(len(seqs_upper[i]), len(seqs_upper[j]))
                d = d / max_len if max_len > 0 else 0.0
            D[i, j] = d
            D[j, i] = d
    return D


def kmer_distance_matrix(sequences, k=6, ks=None, both_strands=True):
    """Compute pairwise k-mer Jaccard distance matrix.

    Jaccard distance = 1 - |intersection| / |union| on k-mer sets.
    When both_strands=True, takes min(d(s_i, s_j), d(s_i, rc(s_j))).

    Parameters
    ----------
    sequences : list of str
    k : int
        Single k-mer size (used when ks is None).
    ks : list of int or None
        Multiple k-mer sizes to combine. When provided, k-mer sets from all
        sizes are unioned into a single set per sequence before computing
        Jaccard distance.  This gives one overall distance that captures
        sequence similarity at multiple resolutions.
    both_strands : bool
    """
    k_values = ks if ks is not None else [k]
    n = len(sequences)
    seqs_upper = [s.upper() for s in sequences]

    # Build k-mer sets (union across all k values)
    kmer_sets = []
    for s in seqs_upper:
        combined = set()
        for kv in k_values:
            combined.update(_kmer_counts(s, kv).keys())
        kmer_sets.append(combined)

    if both_strands:
        rc_seqs = [reverse_complement(s) for s in seqs_upper]
        rc_kmer_sets = []
        for s in rc_seqs:
            combined = set()
            for kv in k_values:
                combined.update(_kmer_counts(s, kv).keys())
            rc_kmer_sets.append(combined)

    ks_label = ",".join(str(kv) for kv in k_values)
    D = np.zeros((n, n))
    for i in tqdm(range(n), desc=f"K-mer Jaccard distance (k={ks_label})"):
        for j in range(i + 1, n):
            inter_fwd = len(kmer_sets[i] & kmer_sets[j])
            union_fwd = len(kmer_sets[i] | kmer_sets[j])
            d_fwd = 1.0 - inter_fwd / union_fwd if union_fwd > 0 else 0.0

            if both_strands:
                inter_rc = len(kmer_sets[i] & rc_kmer_sets[j])
                union_rc = len(kmer_sets[i] | rc_kmer_sets[j])
                d_rc = 1.0 - inter_rc / union_rc if union_rc > 0 else 0.0
                d = min(d_fwd, d_rc)
            else:
                d = d_fwd

            D[i, j] = d
            D[j, i] = d
    return D


def compute_distance_matrix(sequences, method="kmer", k=6, ks=None,
                            both_strands=True):
    """Dispatch to the appropriate distance function.

    When ks is provided (list of k values), computes a single combined
    distance using k-mer sets from all specified sizes.
    """
    if method == "levenshtein":
        return levenshtein_distance_matrix(sequences, both_strands=both_strands)
    elif method == "kmer":
        return kmer_distance_matrix(sequences, k=k, ks=ks,
                                    both_strands=both_strands)
    else:
        raise ValueError(f"Unknown distance method: {method}")


# -- Kernel matrices --------------------------------------------------------

def _multi_kmer_counts(seq, ks):
    """Return a Counter of k-mers across multiple k values."""
    c = Counter()
    seq_upper = seq.upper()
    for k in ks:
        for i in range(len(seq_upper) - k + 1):
            c[seq_upper[i:i + k]] += 1
    return c


def kmer_kernel(sequences, k=6, ks=None, both_strands=True):
    """Compute normalized k-mer kernel (cosine similarity on k-mer vectors).

    K_ij = dot(v_i, v_j) / (||v_i|| * ||v_j||)
    When both_strands=True, takes max(K(s_i, s_j), K(s_i, rc(s_j)))
    — the highest similarity considering both strands.

    Parameters
    ----------
    sequences : list of str
    k : int
        Single k-mer size (used when ks is None).
    ks : list of int or None
        Multiple k-mer sizes to combine.  When provided, k-mer count
        vectors from all sizes are concatenated into a single feature
        vector per sequence before computing cosine similarity.  Uses a
        sparse (observed-only) vocabulary so memory stays manageable
        even for large k.
    both_strands : bool
    """
    seqs_upper = [s.upper() for s in sequences]
    n = len(seqs_upper)

    if ks is not None:
        # Multi-k: use observed vocabulary (sparse) to handle large k
        fwd_counts = [_multi_kmer_counts(s, ks) for s in seqs_upper]
        if both_strands:
            rc_seqs = [reverse_complement(s) for s in seqs_upper]
            rc_counts = [_multi_kmer_counts(s, ks) for s in rc_seqs]
            all_kmers_set = set()
            for c in fwd_counts + rc_counts:
                all_kmers_set.update(c.keys())
        else:
            all_kmers_set = set()
            for c in fwd_counts:
                all_kmers_set.update(c.keys())

        vocab = sorted(all_kmers_set)
        vocab_idx = {km: i for i, km in enumerate(vocab)}
        d = len(vocab)

        vecs = np.zeros((n, d))
        for i, counts in enumerate(fwd_counts):
            for km, cnt in counts.items():
                vecs[i, vocab_idx[km]] = cnt

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        vecs_normed = vecs / norms
        K_fwd = vecs_normed @ vecs_normed.T

        if both_strands:
            rc_vecs = np.zeros((n, d))
            for i, counts in enumerate(rc_counts):
                for km, cnt in counts.items():
                    rc_vecs[i, vocab_idx[km]] = cnt
            rc_norms = np.linalg.norm(rc_vecs, axis=1, keepdims=True)
            rc_norms = np.maximum(rc_norms, 1e-10)
            rc_vecs_normed = rc_vecs / rc_norms
            K_rc = vecs_normed @ rc_vecs_normed.T
            K = np.maximum(K_fwd, K_rc)
        else:
            K = K_fwd
    else:
        # Single-k: use full enumerated vocabulary
        all_km = _all_kmers(k)
        vecs = np.array([_kmer_vector(s, k, all_km) for s in seqs_upper])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        vecs_normed = vecs / norms
        K_fwd = vecs_normed @ vecs_normed.T

        if both_strands:
            rc_seqs = [reverse_complement(s) for s in seqs_upper]
            rc_vecs = np.array([_kmer_vector(s, k, all_km) for s in rc_seqs])
            rc_norms = np.linalg.norm(rc_vecs, axis=1, keepdims=True)
            rc_norms = np.maximum(rc_norms, 1e-10)
            rc_vecs_normed = rc_vecs / rc_norms
            K_rc = vecs_normed @ rc_vecs_normed.T
            K = np.maximum(K_fwd, K_rc)
        else:
            K = K_fwd

    np.fill_diagonal(K, 1.0)
    return K


def rbf_kernel_from_distance(D, sigma=None):
    """Compute RBF kernel from a distance matrix.

    K_ij = exp(-D_ij^2 / (2 * sigma^2))
    Uses median heuristic for sigma if not specified.
    """
    if sigma is None:
        upper_tri = D[np.triu_indices_from(D, k=1)]
        sigma = np.median(upper_tri)
        if sigma < 1e-10:
            sigma = 1.0
    K = np.exp(-D ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(K, 1.0)
    return K


def center_kernel(K):
    """Center a kernel matrix: K_c = H @ K @ H, where H = I - 11^T/n."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def kmer_feature_matrix(sequences, k=6, ks=None, both_strands=True):
    """Compute k-mer feature vectors for sequences.

    Returns the normalized feature matrix rather than the kernel.
    When both_strands=True, uses the forward sequence.

    Parameters
    ----------
    sequences : list of str
    k : int
        k-mer size (used when ks is None).
    ks : list of int or None
        Multiple k-mer sizes to combine.
    both_strands : bool
        Consider reverse complement (uses forward strand for feature matrix).

    Returns
    -------
    vecs : np.ndarray, shape (n, n_features)
        Normalized k-mer count feature vectors.
    vocab : list
        List of k-mers corresponding to columns (for reference).
    """
    seqs_upper = [s.upper() for s in sequences]
    n = len(seqs_upper)

    if ks is not None:
        # Multi-k: use observed vocabulary (sparse)
        fwd_counts = [_multi_kmer_counts(s, ks) for s in seqs_upper]
        all_kmers_set = set()
        for c in fwd_counts:
            all_kmers_set.update(c.keys())

        vocab = sorted(all_kmers_set)
        vocab_idx = {km: i for i, km in enumerate(vocab)}
        d = len(vocab)

        vecs = np.zeros((n, d))
        for i, counts in enumerate(fwd_counts):
            for km, cnt in counts.items():
                vecs[i, vocab_idx[km]] = cnt
    else:
        # Single-k: use full enumerated vocabulary
        all_km = _all_kmers(k)
        vecs = np.array([_kmer_vector(s, k, all_km) for s in seqs_upper])
        vocab = all_km

    # Normalize rows to unit vectors
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    vecs_normed = vecs / norms

    return vecs_normed, vocab


def kernel_from_feature_matrix(X, center=True):
    """Compute cosine similarity kernel from a feature matrix.

    Parameters
    ----------
    X : np.ndarray, shape (n, n_features)
        Feature matrix with rows normalized to unit vectors.
    center : bool
        If True, center the kernel matrix.

    Returns
    -------
    K : np.ndarray, shape (n, n)
        Cosine similarity kernel matrix.
    """
    K = X @ X.T
    np.fill_diagonal(K, 1.0)

    if center:
        K = center_kernel(K)

    return K


def distance_from_feature_matrix(X):
    """Compute cosine distance from a feature matrix.

    Parameters
    ----------
    X : np.ndarray, shape (n, n_features)
        Feature matrix (rows should be unit vectors or will be normalized).

    Returns
    -------
    D : np.ndarray, shape (n, n)
        Pairwise cosine distance matrix.
    """
    from scipy.spatial.distance import pdist, squareform

    X_float = X.astype(np.float64)
    # Normalize rows to unit vectors
    norms = np.linalg.norm(X_float, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    X_normed = X_float / norms

    D = squareform(pdist(X_normed, metric="cosine"))
    # Replace NaN (from zero vectors) with max distance
    D = np.nan_to_num(D, nan=1.0)
    return D


def compute_kernel(sequences, method="kmer", k=6, ks=None, both_strands=True,
                   center=True):
    """Compute a kernel matrix from sequences.

    Parameters
    ----------
    sequences : list of str
    method : str
        'kmer' for normalized k-mer kernel, 'rbf' for RBF on Levenshtein.
    k : int
        k-mer size (used when ks is None).
    ks : list of int or None
        Multiple k-mer sizes to combine into a single kernel.
    both_strands : bool
        Consider reverse complement and use the best similarity.
    center : bool
        If True, center the kernel matrix. Set to False when the kernel
        will be subset and re-centered per cell type.

    Returns
    -------
    K : np.ndarray, shape (n, n).
    """
    if method == "kmer":
        K = kmer_kernel(sequences, k=k, ks=ks, both_strands=both_strands)
    elif method == "rbf":
        D = levenshtein_distance_matrix(sequences, both_strands=both_strands)
        K = rbf_kernel_from_distance(D)
    else:
        raise ValueError(f"Unknown kernel method: {method}")

    if center:
        K = center_kernel(K)
    return K
