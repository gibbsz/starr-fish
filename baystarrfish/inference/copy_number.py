"""How many AAV genomes infected each cell, for each cCRE.

The latent copy number ``k_ij`` is the quantity the model is built around and the
one thing it never samples: it is marginalised analytically so that inference
stays in continuous parameters. This module puts it back, producing the
**cell x cCRE matrix of posterior mean copies** ``E[k_ij | t7_ij, cre_ij]``.

Why this is not just "the counts"
---------------------------------
A cell with ``t7 = 0`` and ``cre = 0`` is not known to be uninfected -- with
dropout, or simply a low per-copy rate, an infected cell often reads zero. And a
cell with ``t7 = 12`` did not receive twelve genomes; it received some ``k`` whose
posterior depends on ``beta_t7``, the dispersion, that cCRE's library abundance
and that cell type's infection rate. The matrix returned here is the model's
answer to both, with the arbitrary-scale problem already resolved.

How it stays tractable
----------------------
``P(k | obs)`` depends on the observation only through ``(cell type, cCRE, t7,
cre)``. On the real data 99.85% of pairs are ``(0, 0)``, so:

1. compute ``E[k | 0, 0]`` once per (cell type, cCRE) -- a 328 x 389 grid;
2. broadcast that baseline across every cell of each type;
3. compute the unique nonzero ``(cell type, cCRE, t7, cre)`` patterns -- about a
   million rather than 160 million -- and scatter them over the baseline.

The result is exact, not approximate: identical observations in the same cell
type have identical posteriors.

Required posterior sites
------------------------
This needs ``log_rho`` and ``log_a`` in addition to ``log_gamma``, because the
copy number depends on the infection rate and the library abundance, not just on
activity. The production fit was written with the default
``--posterior-sites log_gamma`` and therefore **cannot** be used as-is; see
:func:`load_copy_number_draws` for the error message and the exact refit command.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .._log import log
from .posterior_k import posterior_k_moments

__all__ = [
    "CopyNumberMatrix",
    "MissingPosteriorSites",
    "REQUIRED_POSTERIOR_SITES",
    "infer_copy_number",
    "infer_copy_number_from_fit",
    "load_copy_number_draws",
]

#: Sites the copy-number posterior needs beyond the scalars.
REQUIRED_POSTERIOR_SITES = ("log_rho", "log_a", "log_gamma")

#: Scalars every copy-number fit provides.
_REQUIRED_SCALARS = ("beta_t7", "phi_t7", "phi_cre")

#: Present only for a ``copy_number_dropout`` fit; omitting them where they exist
#: silently evaluates the wrong model, so they are carried through when found.
_OPTIONAL_SCALARS = ("p_drop_t7", "p_drop_cre")


class MissingPosteriorSites(KeyError):
    """A fit does not carry the draws a copy-number reconstruction needs.

    Subclasses ``KeyError`` so existing handlers still catch it, but overrides
    ``__str__``: ``KeyError`` renders its argument with ``repr``, which escapes
    the newlines out of a multi-line message and turns the suggested command into
    one unreadable line.
    """

    def __str__(self) -> str:
        return str(self.args[0])


@dataclass(frozen=True)
class CopyNumberMatrix:
    """Posterior AAV copy number per (cell, cCRE).

    Attributes
    ----------
    copies : (n_cells, n_cre)
        ``E[k_ij | observed counts]``. Continuous, because it is a posterior
        mean over an integer quantity -- 0.02 means "almost certainly zero
        copies", not "a fiftieth of a virus". It is *not* a probability of
        infection, and thresholding it is not an infection call; ``P(k >= 1)``
        would have to be accumulated from the posterior weights instead.
    sd : (n_cells, n_cre) or None
        Posterior sd of the same quantity, combining the spread of ``k`` at fixed
        parameters with the spread induced by parameter uncertainty.
    """

    copies: np.ndarray
    sd: np.ndarray | None
    obs_names: np.ndarray | None
    cre_names: list[str]
    kmax: int
    level: str
    infection_model: str

    @property
    def n_cells(self) -> int:
        return int(self.copies.shape[0])

    @property
    def n_cre(self) -> int:
        return int(self.copies.shape[1])

    def to_frame(self) -> pd.DataFrame:
        """Dense DataFrame of copies, cells x cCREs.

        Materialises the whole matrix as float64 -- roughly 1.3 GB for the full
        dataset. Prefer :attr:`copies` and :attr:`cre_names` for anything large.
        """
        return pd.DataFrame(self.copies, index=self.obs_names, columns=self.cre_names)

    def total_per_cell(self) -> np.ndarray:
        """Expected AAV genomes per cell, summed over cCREs."""
        return self.copies.sum(axis=1)

    def write_npz(self, path: Path | str) -> Path:
        """Save compressed, float32, with the axis labels."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "copies": np.asarray(self.copies, dtype=np.float32),
            "cre_names": np.asarray(self.cre_names, dtype=object),
            "kmax": np.asarray(self.kmax),
            "level": np.asarray(self.level, dtype=object),
            "infection_model": np.asarray(self.infection_model, dtype=object),
        }
        if self.sd is not None:
            payload["sd"] = np.asarray(self.sd, dtype=np.float32)
        if self.obs_names is not None:
            payload["obs_names"] = np.asarray(self.obs_names, dtype=object)
        np.savez_compressed(path, **payload)
        return path


def load_copy_number_draws(
    fit_dir: Path | str, tag: str | None = None
) -> tuple[dict[str, np.ndarray], dict]:
    """Load the posterior draws a copy-number reconstruction needs.

    Returns ``(draws, manifest)``. ``draws`` holds ``rho`` and ``a`` on the
    natural scale (the fit stores them as logs) plus ``log_gamma`` and the
    scalars, ready for :func:`baystarrfish.inference.posterior_k_moments`.

    Raises a ``KeyError`` naming the missing sites, because the common case --
    a fit written with the default ``--posterior-sites log_gamma`` -- is not
    recoverable after the fact: ``log_rho`` and ``log_a`` cannot be rebuilt from
    what was saved.
    """
    from ..io.results import fit_tag, load_posterior_samples

    fit_dir = Path(fit_dir)
    tag = fit_tag(fit_dir, tag)
    manifest = json.loads((fit_dir / "run_manifest.json").read_text())

    with np.load(fit_dir / f"{tag}_posterior_samples.npz", allow_pickle=True) as handle:
        available = set(handle.files)
    missing = [site for site in REQUIRED_POSTERIOR_SITES if site not in available]
    if missing:
        raise MissingPosteriorSites(
            f"{fit_dir} has no posterior site(s) {missing}; it stores "
            f"{sorted(available - {'group_names', 'cre_names'})}. The copy number "
            "depends on the infection rate and the library abundance, and neither "
            "can be reconstructed from what was saved. Refit requesting them:\n\n"
            "    python revision/bayesian_vs_fold_change/code/run_bayes.py \\\n"
            "        --posterior-sites log_gamma log_rho log_a [other args as before]\n"
        )

    posterior = load_posterior_samples(fit_dir, tag, sites=list(REQUIRED_POSTERIOR_SITES))
    with np.load(fit_dir / f"{tag}_scalar_samples.npz") as handle:
        scalars = {key: np.asarray(handle[key], dtype=np.float64) for key in handle.files}

    absent = [name for name in _REQUIRED_SCALARS if name not in scalars]
    if absent:
        raise KeyError(f"{fit_dir} scalar samples lack {absent}")

    draws = {
        "rho": np.exp(np.asarray(posterior["log_rho"], dtype=np.float64)),
        "a": np.exp(np.asarray(posterior["log_a"], dtype=np.float64)),
        "log_gamma": np.asarray(posterior["log_gamma"], dtype=np.float64),
        **{name: scalars[name] for name in _REQUIRED_SCALARS},
    }
    is_dropout = manifest.get("config", {}).get("infection_model") == "copy_number_dropout"
    for name in _OPTIONAL_SCALARS:
        if name in scalars:
            draws[name] = scalars[name]
        elif is_dropout:
            raise KeyError(
                f"{fit_dir} is a copy_number_dropout fit but its scalar samples "
                f"lack {name}; evaluating it without dropout would be a different model"
            )
    draws["group_names"] = np.asarray(posterior["group_names"])
    draws["cre_names"] = np.asarray(posterior["cre_names"])
    return draws, manifest


def _align_axes(
    data_cre_names: Sequence[str],
    fit_cre_names: Sequence[str],
    group_labels: np.ndarray,
    fit_group_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Map the data's cCRE and cell-type axes onto the fit's, or explain why not."""
    fit_cre = list(map(str, fit_cre_names))
    position = {name: index for index, name in enumerate(fit_cre)}
    unknown = [name for name in map(str, data_cre_names) if name not in position]
    if unknown:
        raise ValueError(
            f"{len(unknown)} cCRE(s) in the data are absent from the fit "
            f"(e.g. {unknown[:5]}); the blacklist used for the fit must match"
        )
    cre_index = np.array([position[str(name)] for name in data_cre_names], dtype=np.int64)

    fit_groups = list(map(str, fit_group_names))
    group_position = {name: index for index, name in enumerate(fit_groups)}
    labels = np.asarray(group_labels).astype(str)
    absent = sorted(set(labels) - set(group_position))
    if absent:
        raise ValueError(
            f"{len(absent)} cell type(s) are absent from the fit "
            f"(e.g. {absent[:5]}); was the fit run at a different level?"
        )
    group_index = np.array([group_position[label] for label in labels], dtype=np.int64)
    return group_index, cre_index


def _thin_draws(
    draws: Mapping[str, np.ndarray], max_draws: int | None, *, verbose: bool
) -> Mapping[str, np.ndarray]:
    """Keep ``max_draws`` evenly-spaced posterior draws, or all of them."""
    if max_draws is None:
        return draws
    if max_draws < 1:
        raise ValueError("max_draws must be positive")
    n_draws = len(np.asarray(draws["beta_t7"]))
    if max_draws >= n_draws:
        return draws
    keep = np.linspace(0, n_draws - 1, max_draws).round().astype(np.int64)
    if verbose:
        log(f"[copies] thinning {n_draws} posterior draws to {len(keep)}")
    return {
        key: (value[keep] if key not in ("group_names", "cre_names") else value)
        for key, value in draws.items()
    }


def _warn_if_truncated(
    draws: Mapping[str, np.ndarray], kmax: int, group_index, cre_index
) -> None:
    """Warn when the ``0..kmax`` grid does not cover the infection rates.

    ``E[k | obs]`` is a mean over a truncated grid. Where the Poisson rate
    approaches ``kmax`` the truncation bites, and the reported copy number is
    biased low by an amount the number itself gives no hint of.
    """
    import warnings

    from ..model.collapse import kmax_tail_mass

    lam = (
        draws["rho"].mean(axis=0)[np.unique(group_index)][:, None]
        * draws["a"].mean(axis=0)[np.unique(cre_index)][None, :]
    )
    tail = kmax_tail_mass(lam.ravel(), kmax)
    if tail > 1e-6:
        warnings.warn(
            f"the latent grid 0..{kmax} leaves {tail:.2e} Poisson mass in the tail "
            f"(largest rho*a is {lam.max():.1f}); E[k] is truncated and biased low "
            "for the most-infected pairs. Refit with a larger kmax to resolve them.",
            RuntimeWarning,
            stacklevel=3,
        )


def infer_copy_number(
    t7: np.ndarray,
    cre: np.ndarray,
    group_labels: np.ndarray,
    draws: Mapping[str, np.ndarray],
    *,
    kmax: int,
    cre_names: Sequence[str],
    obs_names: Sequence[str] | None = None,
    return_sd: bool = False,
    chunk: int = 400,
    max_draws: int | None = None,
    dtype=np.float32,
    level: str = "subclass",
    infection_model: str = "copy_number",
    verbose: bool = True,
) -> CopyNumberMatrix:
    """Infer ``E[k | obs]`` for every (cell, cCRE) pair.

    Parameters
    ----------
    t7, cre : (n_cells, n_cre) observed counts, aligned to ``cre_names``.
    group_labels : (n_cells,) cell-type labels matching the fit's granularity.
    draws
        As returned by :func:`load_copy_number_draws`; must carry ``group_names``
        and ``cre_names`` so the axes can be aligned rather than assumed.
    kmax
        The truncation used at fit time. Read it from the run manifest; a
        different grid renormalises the posterior and changes the answer.
    max_draws
        Thin the posterior to this many evenly-spaced draws. Cost is linear in
        the draw count and the all-zero baseline alone is
        ``n_cell_types x n_cre`` patterns -- 128k for the production fit --
        which is paid whatever the cell count. A posterior *mean* over 200 draws
        is within Monte Carlo error of one over 1,000, so this is usually the
        difference between minutes and an hour. Evenly spaced rather than random
        so the result is reproducible without a seed.
    dtype
        Output dtype. float32 halves a 636 MB matrix and is far finer than the
        posterior is sharp.
    """
    # obsm count matrices arrive as floats; run_model casts them the same way, so
    # the reconstruction must see exactly the integers the fit saw. Without this
    # the pattern stack below promotes to float and the index columns stop being
    # usable as indices.
    t7 = np.asarray(t7).astype(np.int64)
    cre = np.asarray(cre).astype(np.int64)
    if t7.shape != cre.shape:
        raise ValueError(f"t7 {t7.shape} and cre {cre.shape} must have equal shape")
    n_cells, n_cre = t7.shape
    if n_cre != len(cre_names):
        raise ValueError(f"{n_cre} count columns but {len(cre_names)} cCRE names")
    if len(group_labels) != n_cells:
        raise ValueError(f"{len(group_labels)} labels for {n_cells} cells")

    group_index, cre_index = _align_axes(
        cre_names, draws["cre_names"], group_labels, draws["group_names"]
    )
    n_groups = len(draws["group_names"])
    draws = _thin_draws(draws, max_draws, verbose=verbose)
    _warn_if_truncated(draws, kmax, group_index, cre_index)

    # 1. Baseline: the all-zero observation, once per (cell type, cCRE). Only over
    #    the cell types and cCREs this data actually contains -- a fit covers 328
    #    subclasses and 389 cCREs, and a section or a subsample may use far fewer.
    present_groups, group_slot = np.unique(group_index, return_inverse=True)
    present_cres, cre_slot = np.unique(cre_index, return_inverse=True)
    if verbose:
        log(
            f"[copies] baseline grid: {len(present_groups)} cell types x "
            f"{len(present_cres)} cCREs (fit covers {n_groups} x {len(draws['cre_names'])})"
        )
    grid_group, grid_cre = np.meshgrid(present_groups, present_cres, indexing="ij")
    zeros = np.zeros(grid_group.size, dtype=np.int64)
    baseline = posterior_k_moments(
        zeros, zeros, grid_group.ravel(), grid_cre.ravel(), draws, kmax, chunk=chunk
    )
    shape = (len(present_groups), len(present_cres))
    baseline_mean = baseline.mean.reshape(shape)
    baseline_sd = baseline.sd.reshape(shape)

    # 2. Broadcast it, then correct only where something was actually observed.
    copies = baseline_mean[np.ix_(group_slot, cre_slot)].astype(dtype, copy=True)
    sd = baseline_sd[np.ix_(group_slot, cre_slot)].astype(dtype, copy=True) if return_sd else None

    rows, cols = np.nonzero((t7 > 0) | (cre > 0))
    if len(rows):
        patterns = np.stack(
            [group_index[rows], cre_index[cols], t7[rows, cols], cre[rows, cols]], axis=1
        ).astype(np.int64, copy=False)
        unique, inverse = np.unique(patterns, axis=0, return_inverse=True)
        if verbose:
            log(
                f"[copies] {len(rows):,} nonzero pairs collapse to "
                f"{len(unique):,} unique (cell type, cCRE, t7, cre) patterns"
            )
        moments = posterior_k_moments(
            unique[:, 2], unique[:, 3], unique[:, 0], unique[:, 1], draws, kmax, chunk=chunk
        )
        copies[rows, cols] = moments.mean[inverse].astype(dtype)
        if sd is not None:
            sd[rows, cols] = moments.sd[inverse].astype(dtype)
    elif verbose:
        log("[copies] no nonzero observations; every pair is at the baseline")

    if verbose:
        total = copies.sum(axis=1)
        log(
            f"[copies] expected genomes per cell: median {np.median(total):.3g}, "
            f"range {total.min():.3g}-{total.max():.3g}"
        )
    return CopyNumberMatrix(
        copies=copies,
        sd=sd,
        obs_names=None if obs_names is None else np.asarray(obs_names, dtype=object),
        cre_names=[str(name) for name in cre_names],
        kmax=int(kmax),
        level=level,
        infection_model=infection_model,
    )


def infer_copy_number_from_fit(
    data,
    fit_dir: Path | str,
    *,
    tag: str | None = None,
    return_sd: bool = False,
    chunk: int = 400,
    max_draws: int | None = None,
    dtype=np.float32,
    verbose: bool = True,
) -> CopyNumberMatrix:
    """Infer the copy-number matrix for ``data`` under a fit on disk.

    ``kmax``, the cell-type granularity and the infection model are read from the
    fit's manifest, so the reconstruction cannot silently disagree with the model
    that produced the posterior.

    Parameters
    ----------
    data : CountData
        Must be assembled the same way the fit was -- same blacklist, same
        section -- or the axis alignment raises.
    """
    draws, manifest = load_copy_number_draws(fit_dir, tag)
    config = manifest.get("config", {})
    kmax = config.get("kmax")
    if kmax is None:
        raise KeyError(
            f"{fit_dir} manifest records no kmax; the truncation is part of the "
            "posterior's normalisation and cannot be guessed"
        )
    level = config.get("level", "subclass")
    labels = data.class_ if level == "class" else data.subclass
    return infer_copy_number(
        data.t7,
        data.cre,
        labels,
        draws,
        kmax=int(kmax),
        cre_names=data.cre_names,
        obs_names=data.obs_names,
        return_sd=return_sd,
        chunk=chunk,
        max_draws=max_draws,
        dtype=dtype,
        level=level,
        infection_model=config.get("infection_model", "copy_number"),
        verbose=verbose,
    )


def _main(argv: Sequence[str] | None = None) -> int:
    """``python -m baystarrfish copy-number --fit-dir ... --out ...``"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m baystarrfish copy-number",
        description=__doc__.split("\n")[0],
    )
    parser.add_argument("--fit-dir", type=Path, required=True,
                        help="directory holding run_manifest.json and the posterior draws")
    parser.add_argument("--tag", default=None, help="defaults to the manifest tag")
    parser.add_argument("--out", type=Path, required=True, help="output .npz")
    parser.add_argument("--h5ad", type=Path, default=None,
                        help="input; defaults to the package's configured dataset")
    parser.add_argument("--section", choices=["all", "sec1", "sec2"], default=None,
                        help="defaults to the section recorded in the fit manifest")
    parser.add_argument("--max-cells", type=int, default=None, help="Smoke testing only.")
    parser.add_argument("--max-cres", type=int, default=None, help="Smoke testing only.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=400,
                        help="patterns per block; peak memory is n_draws x chunk x (kmax+1)")
    parser.add_argument("--max-draws", type=int, default=None,
                        help="thin the posterior to N evenly-spaced draws; cost is "
                             "linear in the draw count and 200 is usually within "
                             "Monte Carlo error of the full 1,000")
    parser.add_argument("--with-sd", action="store_true",
                        help="also compute the posterior sd (doubles the output size)")
    args = parser.parse_args(argv)

    from ..data import CountData

    manifest = json.loads((args.fit_dir / "run_manifest.json").read_text())
    section = args.section or manifest.get("section", "all")
    data = CountData.from_h5ad(
        args.h5ad,
        section=section,
        max_cells=args.max_cells,
        max_cres=args.max_cres,
        seed=args.seed,
        negative_control_mode=manifest.get("negative_control_mode", "ordinary"),
    )
    matrix = infer_copy_number_from_fit(
        data, args.fit_dir, tag=args.tag, return_sd=args.with_sd,
        chunk=args.chunk, max_draws=args.max_draws,
    )
    path = matrix.write_npz(args.out)
    log(f"[copies] wrote {matrix.n_cells:,} x {matrix.n_cre} matrix to {path}")
    return 0
