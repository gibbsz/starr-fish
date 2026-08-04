"""Simulation studies against the BAYSTARRFISH generative model.

The forward sampler itself lives in :mod:`baystarrfish.model.forward` -- it is
part of the model, not of any one study, and both the posterior predictive check
and the recovery test draw from it so there is exactly one encoding of the
data-generating process.

:mod:`baystarrfish.simulate.recovery` simulates from known parameters, fits, and
checks that the truth falls inside the posterior credible intervals::

    python -m baystarrfish.simulate.recovery --classes 3 --cres 5 --cells 400 --steps 4000
"""

from __future__ import annotations

from ..model.forward import nb2_sample, sample_channel, sample_latent_multiplier

__all__ = ["nb2_sample", "sample_channel", "sample_latent_multiplier"]
