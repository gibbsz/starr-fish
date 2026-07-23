# Bayesian Priors

This note documents the priors used by the decoupled Bayesian model in
`STARRFISH_in_vivo/STARRFISH/bayesian_hierarchical.py` and the sensitivity
checks that are most relevant for interpreting the current results.

## Current Model

The decoupled model has two stages:

1. Fit T7 counts to infer infection/copy-number parameters.
2. Condition on the T7 infection posterior to infer cCRE activity from cCRE counts.

The latent cCRE activity is `gamma = exp(log_gamma)`, so every modeled
cCRE-celltype pair has a positive latent value. If a pair is weakly supported by
data, its posterior value is mostly determined by the hierarchical activity
prior rather than becoming exactly zero.

## Current Priors

### Infection Rate

These priors control the expected AAV infection rate across classes and
subclasses.

| Parameter | Current prior | Meaning |
|---|---:|---|
| `mu_rho` | `Normal(-6, 2)` | Global log infection-rate baseline |
| `sigma_u` | `HalfNormal(1)` | Class-level infection variation |
| `sigma_w` | `HalfNormal(1)` | Subclass-level infection variation |

Sensitivity checks:

| Label | Suggested change | Purpose |
|---|---|---|
| `infection_low` | `mu_rho_loc = -7` | Test lower prior infection rate |
| `infection_high` | `mu_rho_loc = -5` | Test higher prior infection rate |
| `infection_less_subclass_var` | `sigma_w_scale = 0.5` | More shared infection across subclasses |
| `infection_more_subclass_var` | `sigma_w_scale = 2` | More flexible subclass infection |

### Nanopore / Library Abundance

The model uses nanopore read counts as an informative prior for cCRE abundance.

| Parameter | Current prior | Meaning |
|---|---:|---|
| `tau_a` | `HalfNormal(0.5)` | Noise around centered log nanopore abundance |

Sensitivity checks:

| Label | Suggested change | Purpose |
|---|---|---|
| `nanopore_strong` | `tau_a_scale = 0.2` | Trust nanopore counts more |
| `nanopore_weak` | `tau_a_scale = 1.0` or `2.0` | Let T7 data override nanopore counts more |

### T7 Per-Copy Expression

These priors control the relationship between inferred copy number and observed
T7 counts.

| Parameter | Current prior | Meaning |
|---|---:|---|
| `beta_t7` | `LogNormal(0, 1)` | Mean T7 expression per infected copy |
| `phi_t7` | `HalfNormal(5)` | T7 negative-binomial dispersion |

Sensitivity checks:

| Label | Suggested change | Purpose |
|---|---|---|
| `t7_lower_per_copy` | `beta_t7_loc = -0.7` | More copies needed to explain T7 counts |
| `t7_higher_per_copy` | `beta_t7_loc = 0.7` | Fewer copies needed to explain T7 counts |
| `t7_tighter_per_copy` | `beta_t7_scale = 0.5` | Less prior uncertainty in per-copy T7 |
| `t7_less_overdispersed` | `phi_t7_scale = 2` | Less T7 count noise |
| `t7_more_overdispersed` | `phi_t7_scale = 10` | More T7 count noise |

### cCRE Activity

These priors control the cCRE activity floor and the amount of sharing across
classes/subclasses.

| Parameter | Current prior | Meaning |
|---|---:|---|
| `mu_alpha` | `Normal(0, 3)` | Global cCRE activity baseline |
| `sigma_alpha` | `HalfNormal(2)` | cCRE-level baseline activity variation |
| `sigma_eta` | `HalfNormal(1)` | Class-level activity variation |
| `sigma_delta` | `HalfNormal(1)` | Subclass-level activity variation |
| `phi_cre` | `HalfNormal(5)` | cCRE negative-binomial dispersion |

Sensitivity checks:

| Label | Suggested change | Purpose |
|---|---|---|
| `activity_strong_shrinkage` | `sigma_alpha_scale = 1`, `sigma_eta_scale = 0.5`, `sigma_delta_scale = 0.5` | Pull weakly supported pairs harder toward background |
| `activity_weak_shrinkage` | `sigma_alpha_scale = 4`, `sigma_eta_scale = 2`, `sigma_delta_scale = 2` | Let sparse pairs deviate more |
| `cre_less_overdispersed` | `phi_cre_scale = 2` | Less cCRE count noise |
| `cre_more_overdispersed` | `phi_cre_scale = 10` | More cCRE count noise |

## Dropout Priors

The decoupled model includes one universal T7 dropout rate and one universal
cCRE dropout rate.

| Parameter | Current prior | Meaning |
|---|---:|---|
| `p_drop_t7` | `Beta(1, 9)` | Global T7 measurement dropout probability |
| `p_drop_cre` | `Beta(1, 9)` | Global cCRE measurement dropout probability |

We tested stronger dropout priors:

| Label | Prior mean | T7 posterior mean | cCRE posterior mean |
|---|---:|---:|---:|
| `default_beta_1_9` | 0.10 | 0.00047 | 0.00087 |
| `moderate_beta_2_5` | 0.286 | 0.00092 | 0.00197 |
| `high_beta_5_5` | 0.50 | 0.00231 | 0.00532 |
| `strongly_high_beta_8_2` | 0.80 | 0.00375 | 0.00869 |

The posterior dropout rates remain low even under high-dropout priors. Bootstrap
vs Bayesian correlations also barely changed:

| Filter | Default `r` | Strongly high `r` |
|---|---:|---:|
| Complete | 0.8581 | 0.8586 |
| Total T7 > 100 | 0.97850 | 0.97857 |

This suggests the current model explains zeros mostly through infection rate,
low activity, dispersion, and `k = 0`, rather than through explicit measurement
dropout.

## Recommended Next Sensitivity Runs

The highest-value next checks are:

1. `activity_strong_shrinkage`
2. `activity_weak_shrinkage`
3. `infection_low`
4. `infection_high`
5. `nanopore_strong`
6. `nanopore_weak`

These priors are more likely than dropout priors to affect the Bayesian activity
floor, the number of prior-dominated pairs, and the correlation with bootstrap.
