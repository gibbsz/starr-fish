"""``CountData`` -- everything a fit consumes, assembled once and validated.

Before this existed, each of the three fitting runners repeated the same ~80
lines: read the AnnData, canonicalise cCRE names, blacklist, find the negative
controls, pull the two count matrices, load the nanopore counts, and optionally
build a pooled control column. They could drift, and one of them silently did.

The class carries the provenance (blacklist, control names, cell counts,
``cre_info``) alongside the arrays, because those are exactly the side tables
:func:`baystarrfish.io.write_fit` records next to a fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .._log import log
from .anndata import read_and_prepare_adata
from .controls import (
    NEGATIVE_CONTROL_MODES,
    NegativeControlMode,
    build_pooled_negative_control,
    cre_blacklist,
    negative_control_names,
)
from .libsize import library_counts

__all__ = ["CountData"]


@dataclass(frozen=True)
class CountData:
    """Aligned T7 / cCRE counts plus the metadata a fit and its outputs need.

    Attributes
    ----------
    t7, cre : (n_cells, n_cre) int64
        The two transcript readouts of the same construct: ``t7`` is the
        constitutive channel that calibrates infection, ``cre`` the
        enhancer-driven channel. Neither is a DNA measurement.
    subclass, class_ : (n_cells,) str
        Cell-type labels; subclass must nest cleanly within class.
    lib_size_log : (n_cre,) float64
        ``log1p`` nanopore library counts, aligned to ``cre_names``.
    negative_control_mask : (n_cre,) bool or None
        The mask handed to the *model*. ``None`` in ``ordinary`` mode, where the
        controls are ordinary cCREs and the contrast happens downstream.
    """

    t7: np.ndarray
    cre: np.ndarray
    subclass: np.ndarray
    class_: np.ndarray
    lib_size_log: np.ndarray
    cre_names: list[str]
    negative_control_mask: np.ndarray | None
    negative_controls: list[str]
    negative_control_mode: str
    blacklist: list[str] = field(default_factory=list)
    cre_info: pd.DataFrame | None = None
    subclass_cell_counts: pd.Series | None = None
    pooled_negative_control: dict | None = None
    section: str = "all"
    source: str | None = None

    def __post_init__(self) -> None:
        n_cells, n_cre = self.t7.shape
        if self.cre.shape != self.t7.shape:
            raise ValueError(
                f"t7 {self.t7.shape} and cre {self.cre.shape} must have equal shape"
            )
        if n_cre != len(self.cre_names):
            raise ValueError(
                f"{n_cre} count columns but {len(self.cre_names)} cCRE names"
            )
        if self.lib_size_log.shape != (n_cre,):
            raise ValueError(
                f"lib_size_log has shape {self.lib_size_log.shape}, expected {(n_cre,)}"
            )
        for name, labels in (("subclass", self.subclass), ("class_", self.class_)):
            if labels.shape != (n_cells,):
                raise ValueError(
                    f"{name} has shape {labels.shape}, expected {(n_cells,)}"
                )
        if self.negative_control_mask is not None:
            if self.negative_control_mask.shape != (n_cre,):
                raise ValueError(
                    "negative_control_mask has shape "
                    f"{self.negative_control_mask.shape}, expected {(n_cre,)}"
                )
        if self.negative_control_mode not in NEGATIVE_CONTROL_MODES:
            raise ValueError(
                f"unsupported negative_control_mode={self.negative_control_mode}; "
                f"available {sorted(NEGATIVE_CONTROL_MODES)}"
            )

    # ---- shape helpers --------------------------------------------------- #

    @property
    def n_cells(self) -> int:
        return int(self.t7.shape[0])

    @property
    def n_cre(self) -> int:
        return int(self.t7.shape[1])

    @property
    def n_subclasses(self) -> int:
        return int(np.unique(self.subclass).size)

    @property
    def n_classes(self) -> int:
        return int(np.unique(self.class_).size)

    def to_run_kwargs(self) -> dict:
        """The keyword arguments :func:`baystarrfish.fit` expects.

        Usage::

            fit = bsf.fit(**data.to_run_kwargs(), level="subclass",
                          infection_model="copy_number_dropout")
        """
        return {
            "t7": self.t7,
            "cre": self.cre,
            "subclass_labels": self.subclass,
            "class_labels": self.class_,
            "lib_size_log": self.lib_size_log,
            "cre_names": self.cre_names,
            "negative_control_mask": self.negative_control_mask,
        }

    def side_tables(self) -> dict[str, pd.DataFrame | pd.Series]:
        """Provenance tables written alongside a fit by ``io.write_fit``."""
        tables: dict[str, pd.DataFrame | pd.Series] = {}
        if self.cre_info is not None:
            tables["cre_info"] = self.cre_info
        tables["cre_blacklist"] = pd.Series(self.blacklist, name="cre")
        tables["negative_controls"] = pd.Series(self.negative_controls, name="cre")
        if self.subclass_cell_counts is not None:
            tables["subclass_cell_counts"] = self.subclass_cell_counts
        if self.pooled_negative_control is not None:
            tables["pooled_negative_control_definition"] = pd.DataFrame(
                {
                    "pooled_cre": self.pooled_negative_control["name"],
                    "constituent_cre": self.pooled_negative_control[
                        "constituent_cre"
                    ],
                }
            )
        return tables

    # ---- constructors ---------------------------------------------------- #

    @classmethod
    def from_anndata(
        cls,
        adata: ad.AnnData,
        *,
        negative_control_mode: NegativeControlMode = "ordinary",
        libsize_csv: Path | str | None = None,
        section: str = "all",
        source: str | None = None,
    ) -> CountData:
        """Assemble from an AnnData already passed through ``read_and_prepare_adata``."""
        if negative_control_mode not in NEGATIVE_CONTROL_MODES:
            raise ValueError(
                f"unsupported negative_control_mode={negative_control_mode}; "
                f"available {sorted(NEGATIVE_CONTROL_MODES)}"
            )
        cre_info = adata.uns["CRE_info"].copy()
        blacklist = cre_blacklist(cre_info.index)
        cre_names = [
            name for name in cre_info.index.astype(str) if name not in blacklist
        ]
        negative_controls = negative_control_names(cre_info, blacklist)
        annotated_mask = np.isin(cre_names, negative_controls)

        t7 = adata.obsm["T7CRE"].loc[:, cre_names].to_numpy()
        cre = adata.obsm["CRE"].loc[:, cre_names].to_numpy()
        counts = library_counts(cre_names, libsize_csv)

        pooled: dict | None = None
        if negative_control_mode == "pooled":
            model_mask: np.ndarray | None = annotated_mask
        elif negative_control_mode == "ordinary":
            model_mask = None
        else:
            (
                t7,
                cre,
                counts,
                cre_names,
                model_mask,
                pooled,
            ) = build_pooled_negative_control(
                t7, cre, counts, cre_names, annotated_mask, negative_controls
            )

        return cls(
            t7=np.asarray(t7),
            cre=np.asarray(cre),
            subclass=adata.obs["subclass"].astype(str).to_numpy(),
            class_=adata.obs["class"].astype(str).to_numpy(),
            lib_size_log=np.log1p(counts),
            cre_names=cre_names,
            negative_control_mask=model_mask,
            negative_controls=negative_controls,
            negative_control_mode=negative_control_mode,
            blacklist=blacklist,
            cre_info=cre_info,
            subclass_cell_counts=pd.Series(
                adata.obs["subclass"].value_counts(), name="n_cells"
            ),
            pooled_negative_control=pooled,
            section=section,
            source=source,
        )

    @classmethod
    def from_h5ad(
        cls,
        path: Path | str | None = None,
        *,
        section: str = "all",
        max_cells: int | None = None,
        max_cres: int | None = None,
        seed: int = 0,
        negative_control_mode: NegativeControlMode = "ordinary",
        libsize_csv: Path | str | None = None,
    ) -> CountData:
        """Read the input from disk and assemble in one call.

        ``max_cells`` / ``max_cres`` shrink a smoke run without materialising the
        full object; negative controls are always retained by ``max_cres``.
        """
        adata = read_and_prepare_adata(
            path, section=section, max_cells=max_cells, max_cres=max_cres, seed=seed
        )
        data = cls.from_anndata(
            adata,
            negative_control_mode=negative_control_mode,
            libsize_csv=libsize_csv,
            section=section,
            source=str(path) if path is not None else None,
        )
        log(
            f"[input] {data.n_cells:,} cells x {data.n_cre} cCREs, "
            f"{data.n_subclasses} subclasses, {len(data.negative_controls)} "
            f"negative controls, mode={negative_control_mode}"
        )
        return data
