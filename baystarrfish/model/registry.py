"""(level, channel) -> model function lookup, keyed by infection model family.

``MODEL_FAMILIES`` is the entry point used by
:func:`baystarrfish.inference.run.run_model` to resolve ``infection_model``,
``level`` and ``channel`` to a concrete model function.

The two decoupled stage-2 models (``model_cre_conditional_subclass`` and its
no-dropout variant) are deliberately absent: they are reached only through
:func:`baystarrfish.inference.run.run_decoupled_model`, which fits T7 first and
then conditions the cCRE channel on the stage-1 infection posterior.
"""

from __future__ import annotations

from .models import (
    model_binary_classlevel,
    model_binary_full,
    model_binary_t7_classlevel,
    model_classlevel,
    model_classlevel_dropout,
    model_full,
    model_full_dropout,
    model_t7_classlevel,
    model_t7_full,
    model_t7_full_dropout,
)


MODELS = {
    ("class", "t7"): model_t7_classlevel,
    ("subclass", "t7"): model_t7_full,
    ("class", "joint"): model_classlevel,
    ("subclass", "joint"): model_full,
}


COPY_NUMBER_DROPOUT_MODELS = {
    ("class", "t7"): model_t7_classlevel,
    ("subclass", "t7"): model_t7_full_dropout,
    ("class", "joint"): model_classlevel_dropout,
    ("subclass", "joint"): model_full_dropout,
}


BINARY_INFECTION_MODELS = {
    ("class", "t7"): model_binary_t7_classlevel,
    ("class", "joint"): model_binary_classlevel,
    ("subclass", "joint"): model_binary_full,
}


MODEL_FAMILIES = {
    "copy_number": MODELS,
    "copy_number_dropout": COPY_NUMBER_DROPOUT_MODELS,
    "binary": BINARY_INFECTION_MODELS,
}


__all__ = [
    "MODELS",
    "COPY_NUMBER_DROPOUT_MODELS",
    "BINARY_INFECTION_MODELS",
    "MODEL_FAMILIES",
]
