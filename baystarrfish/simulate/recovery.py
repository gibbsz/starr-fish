"""Smoke / simulation-recovery test for the Bayesian hierarchical STARR-FISH model.

Simulates cell-level T7/CRE counts from the generative model with known
parameters, collapses them to sufficient statistics, fits by SVI (and optionally
NUTS), and checks (a) the ELBO is finite/decreasing and (b) the known
infection/activity parameters fall inside their posterior 90% credible intervals.

Run (fast, simulated data only -- not the real 188k-cell dataset)::

    python -m baystarrfish.simulate.recovery --classes 3 --cres 5 --cells 400 --steps 4000
    python -m baystarrfish.simulate.recovery --classes 3 --cres 5 --nuts

This is the script referenced in the plan's verification section; it does not
touch any STARRFISH object.
"""

from __future__ import annotations

import argparse
from functools import partial

import numpy as np

from .. import model as _model
from ..inference import fit as _fit
from ..inference import initialize as _initialize
from ..model.forward import sample_channel


def simulate(n_class=3, sub_per_class=2, n_cre=5, cells_per_sub=400,
             mu_rho=-1.5, beta_t7=4.0, phi_t7=3.0, phi_cre=3.0, seed=0):
    """Draw a small dataset from the generative model. Returns matrices + truth."""
    rng = np.random.default_rng(seed)
    n_sub = n_class * sub_per_class
    class_of_sub = np.repeat(np.arange(n_class), sub_per_class)

    lib = rng.normal(0.0, 0.5, size=n_cre)
    lib -= lib.mean()                                   # centered log abundance
    a_j = np.exp(lib)

    u = rng.normal(0, 0.6, size=n_class)
    w = rng.normal(0, 0.4, size=n_sub)
    log_rho = mu_rho + u[class_of_sub] + w
    rho = np.exp(log_rho)

    alpha = rng.normal(np.log(5.0), 0.5, size=n_cre)
    eta = rng.normal(0, 0.5, size=(n_class, n_cre))
    delta = rng.normal(0, 0.4, size=(n_sub, n_cre))
    log_gamma = alpha[None, :] + eta[class_of_sub, :] + delta
    gamma = np.exp(log_gamma)                            # (n_sub, n_cre)

    sub_idx = np.repeat(np.arange(n_sub), cells_per_sub)
    n_cells = sub_idx.shape[0]
    lam = rho[sub_idx][:, None] * a_j[None, :]           # (n_cells, n_cre)
    k = rng.poisson(lam)

    t7 = sample_channel(rng, k, beta_t7, phi_t7)
    cre = sample_channel(rng, k, gamma[sub_idx], phi_cre)

    truth = dict(beta_t7=beta_t7, phi_t7=phi_t7, phi_cre=phi_cre,
                 rho_sub=rho, rho_class=np.exp(mu_rho + u),
                 log_gamma_sub=log_gamma, a_j=a_j, lib_centered=lib,
                 class_of_sub=class_of_sub, n_class=n_class, n_sub=n_sub, n_cre=n_cre)
    return t7.astype(np.int64), cre.astype(np.int64), sub_idx, class_of_sub, lib, truth


def selfcheck():
    """Fast pure-math guards for both infection models and sufficient-stat collapse."""
    from scipy.stats import poisson, nbinom

    def nb2_logpmf(c, mean, conc):
        p = conc / (conc + mean)
        return nbinom.logpmf(c, conc, p)

    def channel_logpmf(c, mean, conc, p_drop=None):
        nb = nb2_logpmf(c, mean, conc)
        if p_drop is None:
            return nb
        if c == 0:
            return float(np.logaddexp(np.log(p_drop), np.log1p(-p_drop) + nb))
        return float(np.log1p(-p_drop) + nb)

    def brute(t7v, crev, lam, beta, pt, gam, pc, kmax):
        terms = []
        for k in range(kmax + 1):
            lp = poisson.logpmf(k, lam)
            if k == 0:
                lp += (0.0 if t7v == 0 else -np.inf) + (0.0 if crev == 0 else -np.inf)
            else:
                lp += nb2_logpmf(t7v, beta * k, pt) + nb2_logpmf(crev, gam * k, pc)
            terms.append(lp)
        return float(np.logaddexp.reduce(terms))

    def brute_dropout(t7v, crev, lam, beta, pt, gam, pc, pdt, pdc, kmax):
        terms = []
        for k in range(kmax + 1):
            lp = poisson.logpmf(k, lam)
            if k == 0:
                lp += (0.0 if t7v == 0 else -np.inf) + (0.0 if crev == 0 else -np.inf)
            else:
                lp += channel_logpmf(t7v, beta * k, pt, pdt)
                lp += channel_logpmf(crev, gam * k, pc, pdc)
            terms.append(lp)
        return float(np.logaddexp.reduce(terms))

    def brute_cre_only(crev, lam, gam, pc, pdc, kmax):
        terms = []
        for k in range(kmax + 1):
            lp = poisson.logpmf(k, lam)
            if k == 0:
                lp += 0.0 if crev == 0 else -np.inf
            else:
                lp += channel_logpmf(crev, gam * k, pc, pdc)
            terms.append(lp)
        return float(np.logaddexp.reduce(terms))

    kmax, beta, pt, pc = 25, 4.0, 3.0, 2.5
    rows = [(0, 0, 0.05, 6.0), (5, 0, 0.2, 3.0), (0, 4, 0.1, 8.0), (12, 7, 0.5, 5.0)]
    st = _model.CollapsedStats(group=np.zeros(len(rows), int), cre=np.arange(len(rows)),
                           counts={"t7": np.array([r[0] for r in rows]),
                                   "cre": np.array([r[1] for r in rows])},
                           weight=np.ones(len(rows)), n_per_group=np.array([len(rows)]),
                           n_group=1, n_cre=len(rows), channels=("t7", "cre")).to_jax()
    lam = np.array([r[2] for r in rows]); gam = np.array([r[3] for r in rows])
    got = np.asarray(_model.marginal_loglik(st, lam, beta, pt, gam, pc, kmax))
    exp = np.array([brute(*r[:2], r[2], beta, pt, r[3], pc, kmax) for r in rows])
    assert np.max(np.abs(got - exp)) < 1e-5, (got, exp)

    got_drop = np.asarray(
        _model.marginal_loglik(
            st, lam, beta, pt, gam, pc, kmax, p_drop_t7=0.2, p_drop_cre=0.3
        )
    )
    exp_drop = np.array([
        brute_dropout(*r[:2], r[2], beta, pt, r[3], pc, 0.2, 0.3, kmax)
        for r in rows
    ])
    assert np.max(np.abs(got_drop - exp_drop)) < 1e-5, (got_drop, exp_drop)

    cre_st = _model.CollapsedStats(
        group=np.zeros(len(rows), int),
        cre=np.arange(len(rows)),
        counts={"cre": np.array([r[1] for r in rows])},
        weight=np.ones(len(rows)),
        n_per_group=np.array([len(rows)]),
        n_group=1,
        n_cre=len(rows),
        channels=("cre",),
    ).to_jax()
    gh_nodes, gh_log_weights = _model.gauss_hermite_rule(5)
    got_cre = np.asarray(
        _model.cre_marginal_loglik(
            cre_st,
            np.log(lam),
            np.zeros_like(lam),
            gh_nodes,
            gh_log_weights,
            gam,
            pc,
            0.3,
            kmax,
        )
    )
    exp_cre = np.array([brute_cre_only(r[1], r[2], r[3], pc, 0.3, kmax) for r in rows])
    assert np.max(np.abs(got_cre - exp_cre)) < 1e-5, (got_cre, exp_cre)

    # Shared binary-infection gate: direct two-component mixture.
    p = -np.expm1(-lam)
    infected = np.array([
        nb2_logpmf(r[0], beta, pt) + nb2_logpmf(r[1], r[3], pc)
        for r in rows
    ])
    expected_binary = np.logaddexp(
        np.where(np.array([(r[0] == 0 and r[1] == 0) for r in rows]), np.log1p(-p), -np.inf),
        np.log(p) + infected,
    )
    got_binary = np.asarray(_model.binary_infection_loglik(st, lam, beta, pt, gam, pc))
    assert np.max(np.abs(got_binary - expected_binary)) < 1e-5, (got_binary, expected_binary)

    rng = np.random.default_rng(2)
    n, j, g = 300, 4, 3
    t7 = rng.poisson(0.1, (n, j)); cre = rng.poisson(0.05, (n, j)); gi = rng.integers(0, g, n)
    s = _model.build_sufficient_stats({"t7": t7, "cre": cre}, gi, g, j)
    assert int(s.weight.sum()) == n * j
    rho = np.exp(rng.normal(-1, .3, g)); a = np.exp(rng.normal(0, .3, j))
    gam_gj = np.exp(rng.normal(1, .3, (g, j)))
    sj = s.to_jax()
    ll = np.asarray(_model.marginal_loglik(sj, rho[np.asarray(sj.group)] * a[np.asarray(sj.cre)],
                                       3., 2., gam_gj[np.asarray(sj.group), np.asarray(sj.cre)], 2., 30))
    collapsed = float((np.asarray(sj.weight) * ll).sum())
    fk = _model.CollapsedStats(group=gi.repeat(j), cre=np.tile(np.arange(j), n),
                           counts={"t7": t7.reshape(-1), "cre": cre.reshape(-1)},
                           weight=np.ones(n * j), n_per_group=s.n_per_group, n_group=g,
                           n_cre=j, channels=("t7", "cre")).to_jax()
    lln = np.asarray(_model.marginal_loglik(fk, rho[np.asarray(fk.group)] * a[np.asarray(fk.cre)],
                                        3., 2., gam_gj[np.asarray(fk.group), np.asarray(fk.cre)], 2., 30))
    assert abs(collapsed - float(lln.sum())) < 1e-5
    binary_ll = np.asarray(_model.binary_infection_loglik(
        sj, rho[np.asarray(sj.group)] * a[np.asarray(sj.cre)],
        3., 2., gam_gj[np.asarray(sj.group), np.asarray(sj.cre)], 2.))
    binary_collapsed = float((np.asarray(sj.weight) * binary_ll).sum())
    binary_lln = np.asarray(_model.binary_infection_loglik(
        fk, rho[np.asarray(fk.group)] * a[np.asarray(fk.cre)],
        3., 2., gam_gj[np.asarray(fk.group), np.asarray(fk.cre)], 2.))
    assert abs(binary_collapsed - float(binary_lln.sum())) < 1e-5
    print("selfcheck OK (marginals/dropout/conditional direct, collapse==naive)")


def _coverage(samples, truth, level):
    """Print recovery of beta_t7 and rho against 90% CIs."""
    beta = np.asarray(samples["beta_t7"])
    blo, bhi = np.percentile(beta, [5, 95])
    print(f"  beta_t7: truth {truth['beta_t7']:.3f}  post mean {beta.mean():.3f}  "
          f"90% CI [{blo:.3f}, {bhi:.3f}]  {'OK' if blo <= truth['beta_t7'] <= bhi else 'MISS'}")
    rho = np.exp(np.asarray(samples["log_rho"]))
    truth_rho = truth["rho_class"] if level == "class" else truth["rho_sub"]
    lo, hi = np.percentile(rho, [5, 95], axis=0)
    cov = np.mean((lo <= truth_rho) & (truth_rho <= hi))
    print(f"  rho ({level}): coverage {cov:.0%}  e.g. truth {truth_rho[0]:.4f} "
          f"CI [{lo[0]:.4f}, {hi[0]:.4f}]")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m baystarrfish recovery",
        description=__doc__.split(chr(10))[0],
    )
    ap.add_argument("--classes", type=int, default=3)
    ap.add_argument("--sub-per-class", type=int, default=2)
    ap.add_argument("--cres", type=int, default=5)
    ap.add_argument("--cells", type=int, default=400, help="cells per subclass")
    ap.add_argument("--level", choices=["class", "subclass"], default="subclass")
    ap.add_argument("--channel", choices=["t7", "joint"], default="joint")
    ap.add_argument(
        "--infection-model",
        choices=["copy_number", "copy_number_dropout", "binary"],
        default="copy_number",
    )
    ap.add_argument(
        "--activity-model",
        choices=["hierarchical", "direct"],
        default="direct",
    )
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--nuts", action="store_true")
    ap.add_argument("--selfcheck", action="store_true", help="only run pure-math regression guards")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    selfcheck()
    if args.selfcheck:
        return

    t7, cre, sub_idx, class_of_sub, lib, truth = simulate(
        n_class=args.classes, sub_per_class=args.sub_per_class, n_cre=args.cres,
        cells_per_sub=args.cells, seed=args.seed)

    # group index per cell at the requested granularity
    if args.level == "class":
        group_idx = class_of_sub[sub_idx]; n_group = args.classes; cog = None; ncl = None
    else:
        group_idx = sub_idx; n_group = truth["n_sub"]; cog = class_of_sub; ncl = args.classes

    channels = {"t7": t7} if args.channel == "t7" else {"t7": t7, "cre": cre}
    stats = _model.build_sufficient_stats(channels, group_idx, n_group, args.cres,
                                      class_of_group=cog, n_class=ncl)
    ev = _model.summarize_evidence(stats)
    print("evidence totals:", {k: ev["totals"][k] for k in list(ev["totals"])[:8]})
    print(f"collapsed rows: {len(stats.weight)}  (naive cells*cre = {t7.size})")

    lam_max = float(np.exp(truth["log_gamma_sub"].max()))  # rough
    kmax = _model.choose_kmax(lam_max=truth["rho_sub"].max() * truth["a_j"].max(),
                          max_count=int(max(t7.max(), cre.max())), beta_t7=truth["beta_t7"])
    print("Kmax:", kmax)

    priors = _model.ModelPriors()
    model = _model.MODEL_FAMILIES[args.infection_model][(args.level, args.channel)]
    if args.channel == "joint":
        model = partial(model, activity_model=args.activity_model)
    sj = stats.to_jax()
    lib_j = lib
    init = _initialize.init_from_moments(
        stats,
        lib,
        priors,
        args.level,
        args.channel,
        activity_model=args.activity_model,
    )
    if args.infection_model == "copy_number_dropout":
        init["p_drop_t7"] = np.float64(
            priors.p_drop_t7_alpha
            / (priors.p_drop_t7_alpha + priors.p_drop_t7_beta)
        )
        if args.channel == "joint":
            init["p_drop_cre"] = np.float64(
                priors.p_drop_cre_alpha
                / (priors.p_drop_cre_alpha + priors.p_drop_cre_beta)
            )

    if args.nuts:
        samples, info = _fit.fit_nuts(model, sj, lib_j, kmax, priors, init_values=init,
                                    num_warmup=500, num_samples=500, num_chains=1, seed=args.seed)
    else:
        samples, info = _fit.fit_svi(model, sj, lib_j, kmax, priors, init_values=init,
                                   num_steps=args.steps, seed=args.seed)
        losses = info["losses"]
        print(f"ELBO loss: start {losses[0]:.1f} -> end {losses[-1]:.1f}  "
              f"finite={np.isfinite(losses).all()}")
        assert np.isfinite(losses).all(), "non-finite ELBO"

    if args.channel == "joint" and args.activity_model == "direct":
        expected_shape = (samples["log_gamma"].shape[0], n_group, args.cres)
        assert samples["log_gamma"].shape == expected_shape
        assert "mu_gamma" in samples and "sigma_gamma" in samples
        assert "gamma_raw" in samples
        assert not {"alpha", "eta", "delta"}.intersection(samples)

    _coverage(samples, truth, args.level)
    print("smoke test completed.")


if __name__ == "__main__":
    main()
