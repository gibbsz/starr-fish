"""Prior hyper-parameters of the BAYSTARRFISH generative model.

Every prior is weakly informative and specified on the log scale where the
parameter is positive. See ``docs`` in ``README_BAYSTARRFISH.md`` for the table
of values and the dropout sensitivity analysis.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelPriors:
    """Hyperparameters of the priors (sensible rare-infection defaults)."""
    mu_rho_loc: float = -6.0
    mu_rho_scale: float = 2.0
    sigma_u_scale: float = 1.0          # class-level infection sd
    sigma_w_scale: float = 1.0          # subclass-level infection sd
    tau_a_scale: float = 0.5            # abundance noise around lib_size
    beta_t7_loc: float = 0.0            # LogNormal(loc, scale) for per-copy T7
    beta_t7_scale: float = 1.0
    phi_t7_scale: float = 5.0           # HalfNormal dispersion
    phi_cre_scale: float = 5.0
    mu_alpha_scale: float = 3.0         # CRE baseline activity mean
    sigma_alpha_scale: float = 2.0
    sigma_eta_scale: float = 1.0        # class-level activity sd
    sigma_delta_scale: float = 1.0      # subclass-level activity sd
    mu_gamma_scale: float = 3.0         # direct activity global mean
    sigma_gamma_scale: float = 2.0      # direct activity global sd
    p_drop_t7_alpha: float = 1.0
    p_drop_t7_beta: float = 9.0
    p_drop_cre_alpha: float = 1.0
    p_drop_cre_beta: float = 9.0


__all__ = [
    "ModelPriors",
]
