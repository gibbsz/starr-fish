"""Array-level drivers -- the public entry points of the package.

:func:`run_model` fits one of the registered models in a single stage.
:func:`run_decoupled_model` is the two-stage ablation: fit the T7 channel alone,
then fit the cCRE channel conditional on the stage-1 infection posterior,
integrating over ``log lambda`` by Gauss-Hermite quadrature.

Both take plain NumPy arrays and return a plain dict -- no AnnData, no STARRFISH
object, no 97 GB pickle. See :class:`baystarrfish.data.CountData` for the
AnnData-to-arrays adapter and :func:`baystarrfish.io.write_fit` to serialise the
result.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

# jax_enable_x64 must be set before the first jnp array; see baystarrfish._jax_setup.
from .. import _jax_setup as _jax_setup  # noqa: F401

import jax.numpy as jnp

from ..model.collapse import (
    build_sufficient_stats,
    choose_kmax,
    count_kmax_truncated,
    kmax_tail_mass,
    summarize_evidence,
)
from ..model.likelihood import gauss_hermite_rule
from ..model.models import (
    model_cre_conditional_subclass,
    model_cre_conditional_subclass_no_dropout,
    model_t7_full,
    model_t7_full_dropout,
)
from ..model.priors import ModelPriors
from ..model.registry import MODEL_FAMILIES
from .fit import fit_nuts, fit_svi
from .initialize import init_cre_from_moments, init_from_moments
from .ppc import posterior_predictive_check, posterior_predictive_check_decoupled
from .summarize import (
    summarize_binary_infection,
    summarize_log_lambda_posterior,
    summarize_lognormal_infection,
    summarize_posterior,
)


def _prepare_grouping(subclass_labels, class_labels, level: str):
    sub = np.asarray(subclass_labels).astype(str)
    cls = np.asarray(class_labels).astype(str)
    sub_cats, sub_idx = np.unique(sub, return_inverse=True)
    cls_cats, cls_idx_cell = np.unique(cls, return_inverse=True)
    mapping = pd.DataFrame({"sub": sub_idx, "cls": cls_idx_cell}).drop_duplicates()
    if mapping["sub"].duplicated().any():
        raise ValueError("subclass does not nest cleanly within class")
    class_of_sub = mapping.sort_values("sub")["cls"].to_numpy().astype(np.int64)

    if level == "class":
        return cls_idx_cell, len(cls_cats), list(cls_cats), None, None
    return sub_idx, len(sub_cats), list(sub_cats), class_of_sub, len(cls_cats)


def run_model(t7, cre, subclass_labels, class_labels, lib_size_log, cre_names, *,
              level="class", channel="joint", method="svi", kmax=None,
              priors: "ModelPriors" = None, num_steps=20000, lr=5e-3,
              guide="AutoNormal", num_warmup=1000, num_samples=1000, num_chains=2,
              num_posterior=1000, seed=0, verbose=True, negative_control_mask=None,
              infection_model="copy_number", activity_model="hierarchical",
              posterior_sites_to_return=None) -> dict:
    """Fit the model from plain arrays (no STARRFISH object, no 97GB pickle).

    Parameters
    ----------
    t7, cre : (n_cells, n_cre) integer count matrices (cre may be ignored for channel='t7').
    subclass_labels, class_labels : (n_cells,) cell-type label arrays (subclass nested in class).
    lib_size_log : (n_cre,) log1p library abundance aligned to ``cre_names`` (centered internally).
    cre_names : sequence of CRE identifiers, length n_cre, in the count-matrix column order.

    Returns the same result dict described in ``STARRFISH.bayesian_activity_test``.
    """
    if infection_model not in MODEL_FAMILIES:
        raise ValueError(
            f"unsupported infection_model={infection_model}; available {sorted(MODEL_FAMILIES)}")
    model_family = MODEL_FAMILIES[infection_model]
    if activity_model not in {"hierarchical", "direct"}:
        raise ValueError(
            f"unsupported activity_model={activity_model}; available ['direct', 'hierarchical']"
        )
    if (level, channel) not in model_family:
        raise ValueError(
            f"unsupported (level, channel)=({level}, {channel}); available {sorted(model_family)}")
    priors = priors or ModelPriors()

    t7 = np.asarray(t7).astype(np.int64)
    cre = np.asarray(cre).astype(np.int64)
    n_cre = len(cre_names)
    if negative_control_mask is not None:
        negative_control_mask = np.asarray(negative_control_mask, dtype=bool)
        if negative_control_mask.shape != (n_cre,):
            raise ValueError("negative_control_mask must have shape (n_cre,)")
        if not negative_control_mask.any():
            negative_control_mask = None
    if activity_model == "direct" and negative_control_mask is not None:
        raise ValueError(
            "direct activity requires ordinary negative controls; pass no "
            "negative_control_mask"
        )
    lib_centered = np.asarray(lib_size_log, dtype=np.float64)
    lib_centered = lib_centered - lib_centered.mean()

    sub = np.asarray(subclass_labels).astype(str)
    cls = np.asarray(class_labels).astype(str)
    sub_cats, sub_idx = np.unique(sub, return_inverse=True)
    cls_cats, cls_idx_cell = np.unique(cls, return_inverse=True)
    mapping = pd.DataFrame({"sub": sub_idx, "cls": cls_idx_cell}).drop_duplicates()
    if mapping["sub"].duplicated().any():
        raise ValueError("subclass does not nest cleanly within class")
    class_of_sub = mapping.sort_values("sub")["cls"].to_numpy().astype(np.int64)

    if level == "class":
        group_idx, n_group, group_names = cls_idx_cell, len(cls_cats), list(cls_cats)
        class_of_group, n_class = None, None
    else:
        group_idx, n_group, group_names = sub_idx, len(sub_cats), list(sub_cats)
        class_of_group, n_class = class_of_sub, len(cls_cats)

    channels = {"t7": t7} if channel == "t7" else {"t7": t7, "cre": cre}
    stats = build_sufficient_stats(channels, group_idx, n_group, n_cre,
                                   class_of_group=class_of_group, n_class=n_class)
    evidence = summarize_evidence(stats)
    if verbose:
        print("[run_model] evidence audit:", evidence["totals"])
        print(f"[run_model] collapsed rows: {len(stats.weight)} (naive cells*cre = {t7.size})")

    init_values = init_from_moments(
        stats,
        lib_centered,
        priors,
        level,
        channel,
        negative_control_mask=negative_control_mask,
        activity_model=activity_model,
    )
    is_copy_number = infection_model in {"copy_number", "copy_number_dropout"}
    if infection_model == "copy_number_dropout":
        init_values["p_drop_t7"] = np.float64(
            priors.p_drop_t7_alpha / (priors.p_drop_t7_alpha + priors.p_drop_t7_beta)
        )
        if channel == "joint":
            init_values["p_drop_cre"] = np.float64(
                priors.p_drop_cre_alpha
                / (priors.p_drop_cre_alpha + priors.p_drop_cre_beta)
            )
    if is_copy_number:
        if kmax is None:
            pp = evidence["per_pair"]
            p_t7 = pp["n_t7_pos"].to_numpy() / np.maximum(
                stats.n_per_group[pp["group"].to_numpy()], 1)
            lam_max = float(np.max(-np.log1p(-np.clip(p_t7, 0, 1 - 1e-9)))) if len(p_t7) else 0.1
            kmax = choose_kmax(
                lam_max, int(max(t7.max(), cre.max())), float(init_values["beta_t7"]))
        n_trunc = count_kmax_truncated(stats, kmax, float(init_values["beta_t7"]))
    else:
        # Retain the common model call signature; no copy-number grid is used.
        kmax = 1
        n_trunc = 0
    if verbose:
        if is_copy_number:
            print(f"[run_model] Kmax={kmax}; cells with counts implying k>Kmax (truncated): {n_trunc}")
        else:
            print("[run_model] binary infection model; no copy-number truncation")

    model = model_family[(level, channel)]
    if channel == "joint":
        model = partial(model, activity_model=activity_model)
    sj = stats.to_jax()
    if method == "svi":
        samples, info = fit_svi(model, sj, lib_centered, kmax, priors,
                                negative_control_mask=negative_control_mask, init_values=init_values,
                                num_steps=num_steps, lr=lr, guide=guide,
                                num_posterior=num_posterior, seed=seed)
        diagnostics = {"losses": np.asarray(info["losses"])}
    else:
        samples, info = fit_nuts(model, sj, lib_centered, kmax, priors,
                                 negative_control_mask=negative_control_mask, init_values=init_values,
                                 num_warmup=num_warmup, num_samples=num_samples,
                                 num_chains=num_chains, seed=seed)
        diagnostics = {"method": "nuts"}

    summary = summarize_posterior(samples, stats, evidence, cre_names, group_names, level)
    if infection_model == "binary":
        summary["infection"] = summarize_binary_infection(samples, cre_names, group_names)
    ppc = posterior_predictive_check(
        samples, sj, lib_centered, kmax, level, seed=seed, infection_model=infection_model)

    rho_mean = np.exp(np.asarray(samples["log_rho"]).mean(0))
    a_mean = np.exp(np.asarray(samples["log_a"]).mean(0))
    max_infection_rate = float(np.max(rho_mean[:, None] * a_mean[None, :]))
    diagnostics["max_infection_rate"] = max_infection_rate
    diagnostics["max_infection_probability"] = float(-np.expm1(-max_infection_rate))
    diagnostics["kmax_tail_mass"] = (
        kmax_tail_mass(max_infection_rate, kmax) if is_copy_number else None
    )
    diagnostics["n_kmax_truncated"] = n_trunc

    scalar_sites = ["beta_t7", "phi_t7", "phi_cre", "mu_rho", "sigma_u", "sigma_w",
                    "tau_a", "mu_alpha", "sigma_alpha", "sigma_eta", "sigma_delta",
                    "mu_gamma", "sigma_gamma",
                    "alpha_neg", "log_gamma_neg", "p_drop_t7", "p_drop_cre"]
    scalar_samples = {k: np.asarray(samples[k]) for k in scalar_sites if k in samples}

    negative_control_cre = []
    if negative_control_mask is not None:
        negative_control_cre = np.asarray(cre_names)[negative_control_mask].tolist()

    posterior_samples = {}
    if posterior_sites_to_return:
        requested_sites = list(posterior_sites_to_return)
        if "all" in requested_sites:
            posterior_samples = {k: np.asarray(v) for k, v in samples.items()}
        else:
            if verbose:
                missing_sites = sorted(set(requested_sites) - set(samples))
                if missing_sites:
                    print(f"[run_model] requested posterior sites not found: {missing_sites}")
            posterior_samples = {
                k: np.asarray(samples[k])
                for k in requested_sites
                if k in samples
            }

    reported_kmax = kmax if is_copy_number else None
    result = {"summary": summary, "evidence": evidence, "ppc": ppc, "diagnostics": diagnostics,
            "scalar_samples": scalar_samples, "kmax": reported_kmax, "group_names": group_names,
            "cre_names": list(cre_names),
            "config": dict(level=level, channel=channel, method=method, kmax=reported_kmax,
                           infection_model=infection_model, activity_model=activity_model,
                           num_steps=num_steps, lr=lr, guide=guide, num_warmup=num_warmup,
                           num_samples=num_samples, num_chains=num_chains, seed=seed,
                           negative_control_cre=negative_control_cre)}
    if posterior_samples:
        result["posterior_samples"] = posterior_samples
    return result


def run_decoupled_model(
    t7,
    cre,
    subclass_labels,
    class_labels,
    lib_size_log,
    cre_names,
    *,
    kmax=None,
    priors: "ModelPriors" = None,
    steps_t7=20000,
    steps_cre=20000,
    lr=5e-3,
    guide="AutoNormal",
    num_posterior=1000,
    seed=0,
    verbose=True,
    negative_control_mask=None,
    infection_quadrature_points: int = 7,
    posterior_sites_to_return=None,
    dropout_model: str = "zero_inflated",
) -> dict:
    """Fit the two-stage T7 infection / CRE activity model."""
    if dropout_model not in {"zero_inflated", "none"}:
        raise ValueError("dropout_model must be 'zero_inflated' or 'none'")
    priors = priors or ModelPriors()
    t7 = np.asarray(t7).astype(np.int64)
    cre = np.asarray(cre).astype(np.int64)
    n_cre = len(cre_names)
    if negative_control_mask is not None:
        negative_control_mask = np.asarray(negative_control_mask, dtype=bool)
        if negative_control_mask.shape != (n_cre,):
            raise ValueError("negative_control_mask must have shape (n_cre,)")
        if not negative_control_mask.any():
            negative_control_mask = None
    lib_centered = np.asarray(lib_size_log, dtype=np.float64)
    lib_centered = lib_centered - lib_centered.mean()

    group_idx, n_group, group_names, class_of_group, n_class = _prepare_grouping(
        subclass_labels, class_labels, "subclass"
    )
    t7_stats = build_sufficient_stats(
        {"t7": t7}, group_idx, n_group, n_cre,
        class_of_group=class_of_group, n_class=n_class,
    )
    cre_stats = build_sufficient_stats(
        {"cre": cre}, group_idx, n_group, n_cre,
        class_of_group=class_of_group, n_class=n_class,
    )
    joint_stats = build_sufficient_stats(
        {"t7": t7, "cre": cre}, group_idx, n_group, n_cre,
        class_of_group=class_of_group, n_class=n_class,
    )
    t7_evidence = summarize_evidence(t7_stats)
    cre_evidence = summarize_evidence(cre_stats)
    joint_evidence = summarize_evidence(joint_stats)
    if verbose:
        print("[run_decoupled_model] joint evidence audit:", joint_evidence["totals"])
        print(
            "[run_decoupled_model] collapsed rows: "
            f"t7={len(t7_stats.weight)} cre={len(cre_stats.weight)} "
            f"joint={len(joint_stats.weight)}"
        )

    init_t7 = init_from_moments(t7_stats, lib_centered, priors, "subclass", "t7")
    if dropout_model == "zero_inflated":
        init_t7["p_drop_t7"] = np.float64(
            priors.p_drop_t7_alpha / (priors.p_drop_t7_alpha + priors.p_drop_t7_beta)
        )
    if kmax is None:
        pp = t7_evidence["per_pair"]
        p_t7 = pp["n_t7_pos"].to_numpy() / np.maximum(
            t7_stats.n_per_group[pp["group"].to_numpy()], 1
        )
        lam_max = (
            float(np.max(-np.log1p(-np.clip(p_t7, 0, 1 - 1e-9))))
            if len(p_t7)
            else 0.1
        )
        kmax = choose_kmax(lam_max, int(t7.max()), float(init_t7["beta_t7"]))
    n_trunc = count_kmax_truncated(t7_stats, kmax, float(init_t7["beta_t7"]))
    if verbose:
        print(f"[run_decoupled_model] T7 Kmax={kmax}; T7 rows implying k>Kmax: {n_trunc}")

    t7_model = model_t7_full_dropout if dropout_model == "zero_inflated" else model_t7_full
    t7_samples, t7_info = fit_svi(
        t7_model,
        t7_stats.to_jax(),
        lib_centered,
        kmax,
        priors,
        init_values=init_t7,
        num_steps=steps_t7,
        lr=lr,
        guide=guide,
        num_posterior=num_posterior,
        seed=seed,
    )
    log_lambda_mean, log_lambda_sd = summarize_log_lambda_posterior(t7_samples)
    gh_nodes, gh_log_weights = gauss_hermite_rule(infection_quadrature_points)
    log_lambda_mean_j = jnp.asarray(log_lambda_mean)
    log_lambda_sd_j = jnp.asarray(log_lambda_sd)
    gh_nodes_j = jnp.asarray(gh_nodes)
    gh_log_weights_j = jnp.asarray(gh_log_weights)

    cre_conditional = (
        model_cre_conditional_subclass
        if dropout_model == "zero_inflated"
        else model_cre_conditional_subclass_no_dropout
    )

    def cre_model(stats, lib_size_centered, kmax, negative_control_mask, priors, observe=True):
        return cre_conditional(
            stats,
            log_lambda_mean_j,
            log_lambda_sd_j,
            gh_nodes_j,
            gh_log_weights_j,
            kmax,
            negative_control_mask,
            priors,
            observe,
        )

    init_cre = init_cre_from_moments(cre_stats, priors, negative_control_mask)
    if dropout_model == "none":
        init_cre.pop("p_drop_cre", None)
    cre_samples, cre_info = fit_svi(
        cre_model,
        cre_stats.to_jax(),
        lib_centered,
        kmax,
        priors,
        negative_control_mask=negative_control_mask,
        init_values=init_cre,
        num_steps=steps_cre,
        lr=lr,
        guide=guide,
        num_posterior=num_posterior,
        seed=seed + 10_000,
    )

    t7_summary = summarize_posterior(
        t7_samples, t7_stats, t7_evidence, cre_names, group_names, "subclass"
    )
    cre_summary = summarize_posterior(
        cre_samples, cre_stats, cre_evidence, cre_names, group_names, "subclass",
        prior_dominated_basis="cre",
    )
    summary = {
        "rho": t7_summary["rho"],
        "infection": summarize_lognormal_infection(log_lambda_mean, log_lambda_sd, cre_names, group_names),
        "gamma": cre_summary["gamma"],
    }
    if "delta_mean" in cre_summary:
        summary["delta_mean"] = cre_summary["delta_mean"]

    ppc = posterior_predictive_check_decoupled(
        t7_samples, cre_samples, joint_stats, seed=seed
    )
    rate_mean = np.exp(log_lambda_mean + 0.5 * np.square(log_lambda_sd))
    diagnostics = {
        "losses_t7": np.asarray(t7_info["losses"]),
        "losses_cre": np.asarray(cre_info["losses"]),
        "loss_t7_all_finite": bool(np.isfinite(t7_info["losses"]).all()),
        "loss_cre_all_finite": bool(np.isfinite(cre_info["losses"]).all()),
        "max_infection_rate_mean": float(np.max(rate_mean)),
        "max_log_infection_rate_sd": float(np.max(log_lambda_sd)),
        "kmax_tail_mass_at_mean_rate": kmax_tail_mass(rate_mean, kmax),
        "n_kmax_truncated_t7": n_trunc,
    }

    scalar_sites_t7 = [
        "beta_t7", "phi_t7", "p_drop_t7", "mu_rho", "sigma_u", "sigma_w", "tau_a",
    ]
    scalar_sites_cre = [
        "phi_cre", "p_drop_cre", "mu_alpha", "sigma_alpha", "sigma_eta",
        "sigma_delta", "alpha_neg", "log_gamma_neg",
    ]
    scalar_samples = {
        key: np.asarray(t7_samples[key]) for key in scalar_sites_t7 if key in t7_samples
    }
    scalar_samples.update({
        key: np.asarray(cre_samples[key]) for key in scalar_sites_cre if key in cre_samples
    })

    posterior_samples = {}
    if posterior_sites_to_return:
        requested_sites = list(posterior_sites_to_return)
        if "all" in requested_sites:
            posterior_samples = {k: np.asarray(v) for k, v in cre_samples.items()}
        else:
            if verbose:
                missing_sites = sorted(set(requested_sites) - set(cre_samples))
                if missing_sites:
                    print(f"[run_decoupled_model] requested CRE posterior sites not found: {missing_sites}")
            posterior_samples = {
                key: np.asarray(cre_samples[key])
                for key in requested_sites
                if key in cre_samples
            }

    infection_posterior_samples = {
        key: np.asarray(t7_samples[key])
        for key in ("log_rho", "log_a", "beta_t7", "phi_t7", "p_drop_t7")
        if key in t7_samples
    }
    negative_control_cre = []
    if negative_control_mask is not None:
        negative_control_cre = np.asarray(cre_names)[negative_control_mask].tolist()

    result = {
        "summary": summary,
        "evidence": joint_evidence,
        "t7_evidence": t7_evidence,
        "cre_evidence": cre_evidence,
        "ppc": ppc,
        "diagnostics": diagnostics,
        "scalar_samples": scalar_samples,
        "posterior_samples": posterior_samples,
        "infection_posterior_samples": infection_posterior_samples,
        "log_lambda_mean": log_lambda_mean,
        "log_lambda_sd": log_lambda_sd,
        "kmax": kmax,
        "group_names": group_names,
        "cre_names": list(cre_names),
        "config": dict(
            level="subclass",
            channel="cre",
            method="svi",
            model_variant="decoupled_t7_cre",
            infection_model="copy_number",
            dropout_model=dropout_model,
            kmax=kmax,
            steps_t7=steps_t7,
            steps_cre=steps_cre,
            lr=lr,
            guide=guide,
            num_posterior=num_posterior,
            seed=seed,
            infection_quadrature_points=infection_quadrature_points,
            negative_control_cre=negative_control_cre,
        ),
    }
    return result


__all__ = [
    "run_model",
    "run_decoupled_model",
]
