#!/usr/bin/env python3
"""Simulate STARR-FISH counts from the fitted joint dropout model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import anndata as ad

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(os.environ.get("TMPDIR", "/tmp")) / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy import sparse

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from analysis_utils import (  # noqa: E402
    ANALYSIS_DIR,
    DEFAULT_H5AD,
    LIBSIZE_CSV,
    cre_blacklist,
    input_fingerprint,
    log,
    read_and_prepare_adata,
    select_cre_info,
    write_json,
)


SIM_DIR = Path(__file__).resolve().parent
DEFAULT_FIT_DIR = ANALYSIS_DIR / "results" / "bayesian_joint_dropout"
DEFAULT_OUTDIR = SIM_DIR / "results" / "joint_dropout_simulated"
TAG = "subclass_joint_copy_number_dropout_svi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--fit-dir", type=Path, default=DEFAULT_FIT_DIR)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--section", choices=["all", "sec1", "sec2"], default="all")
    parser.add_argument("--max-cells", type=int, default=None, help="Smoke testing only.")
    parser.add_argument("--max-cres", type=int, default=None, help="Smoke testing only.")
    parser.add_argument("--chunk-cells", type=int, default=4096)
    parser.add_argument(
        "--truth-draw",
        type=int,
        default=None,
        help=(
            "Use one posterior draw as truth. Requires log_rho and log_a draws "
            "for coherent infection truth; otherwise use posterior means."
        ),
    )
    return parser.parse_args()


def discover_tag(fit_dir: Path, tag: str | None) -> str:
    if tag:
        return tag
    manifest = fit_dir / "run_manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text())["tag"]
    return TAG


def load_scalar_truth(scalar_path: Path, truth_draw: int | None) -> tuple[dict[str, float], dict[str, int | str]]:
    with np.load(scalar_path, allow_pickle=True) as scalars:
        arrays = {key: np.asarray(scalars[key], dtype=np.float64).reshape(-1) for key in scalars.files}
    if truth_draw is not None:
        n_draws = len(next(iter(arrays.values())))
        if truth_draw < 0 or truth_draw >= n_draws:
            raise ValueError(f"--truth-draw must be in [0, {n_draws}); got {truth_draw}")
        truth = {key: float(values[truth_draw]) for key, values in arrays.items()}
        meta = {"scalar_truth": "posterior_draw", "truth_draw": int(truth_draw)}
    else:
        truth = {key: float(values.mean()) for key, values in arrays.items()}
        meta = {"scalar_truth": "posterior_mean"}
    return truth, meta


def load_gamma_truth(
    fit_dir: Path,
    tag: str,
    posterior: np.lib.npyio.NpzFile,
    groups: pd.Index,
    cres: pd.Index,
    truth_draw: int | None,
) -> tuple[np.ndarray, str]:
    if "log_gamma" in posterior.files:
        log_gamma = np.asarray(posterior["log_gamma"], dtype=np.float64)
        if truth_draw is not None:
            gamma = np.exp(log_gamma[truth_draw])
            source = "posterior_draw_exp_log_gamma"
        else:
            gamma = np.exp(log_gamma).mean(axis=0)
            source = "posterior_mean_exp_log_gamma"
    else:
        gamma_df = pd.read_csv(fit_dir / f"{tag}_gamma.csv")
        gamma = (
            gamma_df.assign(group=gamma_df["group"].astype(str), cre=gamma_df["cre"].astype(str))
            .pivot(index="group", columns="cre", values="gamma_mean")
            .reindex(index=groups, columns=cres)
            .to_numpy(dtype=np.float64)
        )
        source = "gamma_csv_mean"
    if not np.isfinite(gamma).all():
        raise ValueError("gamma truth contains non-finite values")
    return gamma, source


def abundance_from_libsize(cre_names: pd.Index) -> np.ndarray:
    lib = pd.read_csv(LIBSIZE_CSV, index_col=0)
    lib.index = lib.index.astype(str)
    log_lib = np.log1p(lib.reindex(cre_names, fill_value=0)["counts"].to_numpy(dtype=np.float64))
    log_a = log_lib - log_lib.mean()
    return np.exp(log_a)


def load_infection_truth(
    fit_dir: Path,
    tag: str,
    posterior: np.lib.npyio.NpzFile,
    groups: pd.Index,
    cres: pd.Index,
    truth_draw: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if {"log_rho", "log_a"}.issubset(posterior.files):
        log_rho = np.asarray(posterior["log_rho"], dtype=np.float64)
        log_a = np.asarray(posterior["log_a"], dtype=np.float64)
        if truth_draw is not None:
            rho = np.exp(log_rho[truth_draw])
            abundance = np.exp(log_a[truth_draw])
            lam = np.exp(log_rho[truth_draw, :, None] + log_a[truth_draw, None, :])
            source = "posterior_draw_exp_log_rho_plus_log_a"
        else:
            rho = np.exp(log_rho).mean(axis=0)
            abundance = np.exp(log_a).mean(axis=0)
            lam = np.exp(log_rho[:, :, None] + log_a[:, None, :]).mean(axis=0)
            source = "posterior_mean_exp_log_rho_plus_log_a"
        return lam, rho, abundance, source

    if truth_draw is not None:
        raise ValueError(
            "--truth-draw requires posterior log_rho and log_a draws. "
            "Rerun the source fit with --posterior-sites log_gamma log_rho log_a, "
            "or omit --truth-draw."
        )

    rho_df = pd.read_csv(fit_dir / f"{tag}_rho.csv")
    rho = (
        rho_df.assign(group=rho_df["group"].astype(str))
        .set_index("group")
        .reindex(groups)["rho_mean"]
        .to_numpy(dtype=np.float64)
    )
    if not np.isfinite(rho).all():
        missing = groups[~np.isfinite(rho)].tolist()
        raise ValueError(f"rho truth missing groups: {missing[:5]}")
    abundance = abundance_from_libsize(cres)
    lam = rho[:, None] * abundance[None, :]
    source = "rho_csv_mean_times_centered_library_abundance"
    return lam, rho, abundance, source


def load_truth(
    fit_dir: Path,
    tag: str,
    selected_groups: pd.Index,
    selected_cres: pd.Index,
    truth_draw: int | None,
) -> tuple[dict[str, np.ndarray | float], dict]:
    posterior_path = fit_dir / f"{tag}_posterior_samples.npz"
    scalar_path = fit_dir / f"{tag}_scalar_samples.npz"
    if not posterior_path.exists():
        raise FileNotFoundError(posterior_path)
    if not scalar_path.exists():
        raise FileNotFoundError(scalar_path)

    with np.load(posterior_path, allow_pickle=True) as posterior:
        all_groups = pd.Index(posterior["group_names"].astype(str), dtype=str)
        all_cres = pd.Index(posterior["cre_names"].astype(str), dtype=str)
        if not selected_groups.isin(all_groups).all():
            missing = selected_groups[~selected_groups.isin(all_groups)].tolist()
            raise ValueError(f"selected groups absent from fit: {missing[:5]}")
        if not selected_cres.isin(all_cres).all():
            missing = selected_cres[~selected_cres.isin(all_cres)].tolist()
            raise ValueError(f"selected cCREs absent from fit: {missing[:5]}")

        group_pos = all_groups.get_indexer(selected_groups)
        cre_pos = all_cres.get_indexer(selected_cres)
        scalars, scalar_meta = load_scalar_truth(scalar_path, truth_draw)
        gamma_all, gamma_source = load_gamma_truth(
            fit_dir, tag, posterior, all_groups, all_cres, truth_draw
        )
        lam_all, rho_all, abundance_all, infection_source = load_infection_truth(
            fit_dir, tag, posterior, all_groups, all_cres, truth_draw
        )

    truth = {
        "lambda": lam_all[np.ix_(group_pos, cre_pos)],
        "rho": rho_all[group_pos],
        "abundance": abundance_all[cre_pos],
        "gamma": gamma_all[np.ix_(group_pos, cre_pos)],
        "beta_t7": scalars["beta_t7"],
        "phi_t7": scalars["phi_t7"],
        "phi_cre": scalars["phi_cre"],
        "p_drop_t7": scalars.get("p_drop_t7", 0.0),
        "p_drop_cre": scalars.get("p_drop_cre", 0.0),
    }
    meta = {
        **scalar_meta,
        "gamma_source": gamma_source,
        "infection_source": infection_source,
        "fit_dir": str(fit_dir.resolve()),
        "tag": tag,
    }
    return truth, meta


# NOTE: baystarrfish.model.forward.sample_channel is the canonical encoding of
# this draw and is what the model, the posterior predictive check and the
# recovery test all use. These two are distributionally identical but consume the
# rng in a different order -- dropout first here, NB2 first there -- and this one
# samples NB2 only for the surviving entries. Switching would therefore change
# the dataset this script produces for a fixed seed and invalidate the archived
# simulation outputs, so it is kept deliberately. Do not "fix" the duplication
# without re-running the simulation study.
def nb2_sample(rng: np.random.Generator, mean: np.ndarray, conc: float) -> np.ndarray:
    out = np.zeros(mean.shape, dtype=np.int32)
    mask = mean > 0
    if mask.any():
        lam = rng.gamma(shape=conc, scale=mean[mask] / conc)
        out[mask] = rng.poisson(lam).astype(np.int32)
    return out


def sample_channel(
    rng: np.random.Generator,
    k: np.ndarray,
    per_copy: float | np.ndarray,
    phi: float,
    p_drop: float,
) -> np.ndarray:
    out = np.zeros(k.shape, dtype=np.int32)
    positive = k > 0
    if not positive.any():
        return out
    keep = positive.copy()
    if p_drop > 0:
        keep[positive] = rng.random(int(positive.sum())) >= p_drop
    if keep.any():
        mean = np.asarray(per_copy) * k
        out[keep] = nb2_sample(rng, mean[keep], phi)
    return out


def write_truth_tables(
    outdir: Path,
    groups: pd.Index,
    cres: pd.Index,
    truth: dict[str, np.ndarray | float],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outdir / "truth_parameters.npz",
        group_names=groups.to_numpy(dtype=object),
        cre_names=cres.to_numpy(dtype=object),
        lambda_rate=np.asarray(truth["lambda"], dtype=np.float64),
        rho=np.asarray(truth["rho"], dtype=np.float64),
        abundance=np.asarray(truth["abundance"], dtype=np.float64),
        gamma=np.asarray(truth["gamma"], dtype=np.float64),
        beta_t7=np.asarray(truth["beta_t7"], dtype=np.float64),
        phi_t7=np.asarray(truth["phi_t7"], dtype=np.float64),
        phi_cre=np.asarray(truth["phi_cre"], dtype=np.float64),
        p_drop_t7=np.asarray(truth["p_drop_t7"], dtype=np.float64),
        p_drop_cre=np.asarray(truth["p_drop_cre"], dtype=np.float64),
    )

    gg, cc = np.meshgrid(np.arange(len(groups)), np.arange(len(cres)), indexing="ij")
    pd.DataFrame(
        {
            "group": groups.to_numpy()[gg.ravel()],
            "cre": cres.to_numpy()[cc.ravel()],
            "gamma_true": np.asarray(truth["gamma"]).ravel(),
        }
    ).to_csv(outdir / "truth_gamma.csv", index=False)
    pd.DataFrame(
        {
            "group": groups.to_numpy()[gg.ravel()],
            "cre": cres.to_numpy()[cc.ravel()],
            "lambda_true": np.asarray(truth["lambda"]).ravel(),
        }
    ).to_csv(outdir / "truth_infection.csv", index=False)
    pd.DataFrame(
        {"group": groups.to_numpy(), "rho_true": np.asarray(truth["rho"], dtype=np.float64)}
    ).to_csv(outdir / "truth_rho.csv", index=False)
    pd.DataFrame(
        {
            "parameter": ["beta_t7", "phi_t7", "phi_cre", "p_drop_t7", "p_drop_cre"],
            "truth": [
                truth["beta_t7"],
                truth["phi_t7"],
                truth["phi_cre"],
                truth["p_drop_t7"],
                truth["p_drop_cre"],
            ],
        }
    ).to_csv(outdir / "truth_scalars.csv", index=False)


def h5ad_safe_cre_info(cre_info: pd.DataFrame) -> pd.DataFrame:
    """Return a CRE_info copy with object columns writable by h5ad."""
    safe = cre_info.copy()
    for col in safe.columns:
        is_category = isinstance(safe[col].dtype, pd.CategoricalDtype)
        if pd.api.types.is_object_dtype(safe[col]) or is_category:
            safe[col] = safe[col].where(safe[col].notna(), "").astype(str)
    return safe


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = discover_tag(args.fit_dir, args.tag)
    rng = np.random.default_rng(args.seed)

    adata = read_and_prepare_adata(
        args.h5ad,
        section=args.section,
        max_cells=args.max_cells,
        max_cres=None,
        seed=args.seed,
    )
    cre_info = adata.uns["CRE_info"].copy()
    all_cres = pd.Index(cre_info.index.astype(str), dtype=str)
    blacklist = set(cre_blacklist(all_cres))
    selected_cre_info = select_cre_info(cre_info, args.max_cres)
    fitted_cres = pd.Index(selected_cre_info.index.astype(str), dtype=str)
    fitted_cres = fitted_cres[~fitted_cres.isin(blacklist)]
    groups = pd.Index(np.unique(adata.obs["subclass"].astype(str)), dtype=str)

    truth, truth_meta = load_truth(args.fit_dir, tag, groups, fitted_cres, args.truth_draw)
    write_truth_tables(args.outdir, groups, fitted_cres, truth)

    subclass_to_idx = {name: idx for idx, name in enumerate(groups)}
    group_idx = np.array(
        [subclass_to_idx[name] for name in adata.obs["subclass"].astype(str)],
        dtype=np.int64,
    )
    fit_col_idx = all_cres.get_indexer(fitted_cres)
    if (fit_col_idx < 0).any():
        raise AssertionError("fitted cCREs are not a subset of output cCRE columns")

    n_cells = adata.n_obs
    n_all_cres = len(all_cres)
    t7_out = np.zeros((n_cells, n_all_cres), dtype=np.int32)
    cre_out = np.zeros((n_cells, n_all_cres), dtype=np.int32)

    log(
        f"[simulate] generating {n_cells:,} cells x {len(fitted_cres):,} fitted cCREs "
        f"({n_all_cres:,} output cCRE columns)"
    )
    lam_truth = np.asarray(truth["lambda"], dtype=np.float64)
    gamma_truth = np.asarray(truth["gamma"], dtype=np.float64)
    beta_t7 = float(truth["beta_t7"])
    phi_t7 = float(truth["phi_t7"])
    phi_cre = float(truth["phi_cre"])
    p_drop_t7 = float(truth["p_drop_t7"])
    p_drop_cre = float(truth["p_drop_cre"])

    latent_nonzero = 0
    for start in range(0, n_cells, args.chunk_cells):
        stop = min(start + args.chunk_cells, n_cells)
        g = group_idx[start:stop]
        lam = lam_truth[g, :]
        k = rng.poisson(lam).astype(np.int32)
        latent_nonzero += int((k > 0).sum())
        t7 = sample_channel(rng, k, beta_t7, phi_t7, p_drop_t7)
        cre = sample_channel(rng, k, gamma_truth[g, :], phi_cre, p_drop_cre)
        t7_out[start:stop, fit_col_idx] = t7
        cre_out[start:stop, fit_col_idx] = cre
        if start == 0 or stop == n_cells or stop // args.chunk_cells % 20 == 0:
            log(f"[simulate] rows {start:,}-{stop:,}")

    obs = adata.obs.copy()
    var = adata.var.copy()
    sim = ad.AnnData(
        X=sparse.csr_matrix((n_cells, var.shape[0]), dtype=np.float32),
        obs=obs,
        var=var,
    )
    sim.obsm["T7CRE"] = pd.DataFrame(t7_out, index=adata.obs_names.copy(), columns=all_cres)
    sim.obsm["CRE"] = pd.DataFrame(cre_out, index=adata.obs_names.copy(), columns=all_cres)
    if "X_spatial" in adata.obsm:
        sim.obsm["X_spatial"] = adata.obsm["X_spatial"].copy()
    sim.uns["CRE_info"] = h5ad_safe_cre_info(cre_info)
    sim.uns["simulation"] = {
        "model": "subclass_joint_copy_number_dropout",
        "truth_dir": str(args.outdir.resolve()),
        "seed": int(args.seed),
    }
    out_h5ad = args.outdir / "simulated_joint_dropout.h5ad"
    sim.write_h5ad(out_h5ad, compression="gzip")

    write_json(
        args.outdir / "truth_manifest.json",
        {
            "model": "subclass_joint_copy_number_dropout",
            "input": input_fingerprint(args.h5ad),
            "simulated_h5ad": str(out_h5ad.resolve()),
            "n_cells": int(n_cells),
            "n_output_cres": int(n_all_cres),
            "n_fitted_cres": int(len(fitted_cres)),
            "n_groups": int(len(groups)),
            "section": args.section,
            "max_cells": args.max_cells,
            "max_cres": args.max_cres,
            "seed": args.seed,
            "latent_nonzero_pairs": int(latent_nonzero),
            "truth": truth_meta,
            "scalars": {
                "beta_t7": beta_t7,
                "phi_t7": phi_t7,
                "phi_cre": phi_cre,
                "p_drop_t7": p_drop_t7,
                "p_drop_cre": p_drop_cre,
            },
        },
    )
    log(f"[simulate] wrote {out_h5ad}")


if __name__ == "__main__":
    main()
