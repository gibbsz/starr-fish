"""End-to-end simulation recovery: does the fit find parameters it generated?

Marked slow -- this runs real SVI. Everything else in the suite checks pieces;
this is the only test that exercises the whole path, so it is what catches a
regression that is individually invisible in every component.

    pytest tests -m slow
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_svi_recovers_the_generating_parameters():
    from baystarrfish.simulate import recovery

    t7, cre, sub_idx, class_of_sub, lib, truth = recovery.simulate(
        n_class=2, sub_per_class=2, n_cre=4, cells_per_sub=300, seed=0
    )
    subclass_labels = np.array([f"sub{i}" for i in sub_idx])
    class_labels = np.array([f"cls{class_of_sub[i]}" for i in sub_idx])

    import baystarrfish as bsf

    result = bsf.fit(
        t7, cre, subclass_labels, class_labels, lib,
        [f"CRE{i:03d}" for i in range(4)],
        level="subclass", channel="joint", method="svi",
        num_steps=4000, num_posterior=200, seed=0, verbose=False,
    )

    losses = np.asarray(result["diagnostics"]["losses"])
    assert np.isfinite(losses).all(), "ELBO went non-finite"
    assert losses[-1] < losses[0], "ELBO did not improve"

    # beta_t7 is the best-identified global parameter; if the fit is wired
    # correctly it lands near the truth, and if it is not, it does not.
    beta = np.asarray(result["scalar_samples"]["beta_t7"])
    lo, hi = np.quantile(beta, [0.025, 0.975])
    assert lo < truth["beta_t7"] < hi, (lo, truth["beta_t7"], hi)


def test_the_bundled_selfcheck_passes():
    """recovery.selfcheck() pins both infection models and the stat collapse."""
    from baystarrfish.simulate import recovery

    recovery.selfcheck()
