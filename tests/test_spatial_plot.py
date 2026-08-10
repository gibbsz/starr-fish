"""Spatial maps: the four modes, and the conventions that make them readable.

Figures are hard to assert on, so these check the things that would actually go
wrong -- the wrong array plotted, the background layer missing, the size ramp
inverted, a region filter dropping the wrong cells -- by reading back the
artists matplotlib produced rather than by eyeballing a PNG.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib", reason="the `plots` extra")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from baystarrfish.data import CountData  # noqa: E402
from baystarrfish.plotting import (  # noqa: E402
    MODE_COLORS,
    SPATIAL_MODES,
    plot_spatial,
    spatial_values,
)

N_CELL, N_CRE = 120, 4
CRE_NAMES = [f"CRE{i:03d}" for i in range(1, N_CRE + 1)]


@pytest.fixture
def data(rng):
    types = np.array(["Pvalb", "Sst", "L2-3 IT"] * (N_CELL // 3))
    return CountData(
        t7=rng.poisson(0.8, (N_CELL, N_CRE)),
        cre=rng.poisson(0.5, (N_CELL, N_CRE)),
        subclass=types,
        class_=np.where(types == "L2-3 IT", "Glut", "GABA"),
        lib_size_log=np.log1p(rng.poisson(50, N_CRE)),
        cre_names=list(CRE_NAMES),
        negative_control_mask=None,
        negative_controls=[CRE_NAMES[-1]],
        negative_control_mode="ordinary",
        obs_names=np.array([f"cell{i}" for i in range(N_CELL)], dtype=object),
        spatial=rng.uniform(0, 100, size=(N_CELL, 2)),
    )


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _collections(fig):
    return fig.axes[0].collections


# ---- the grammar every mode shares ---------------------------------------- #


@pytest.mark.parametrize("mode", SPATIAL_MODES)
def test_every_mode_draws_on_black_with_a_grey_background_layer(mode, data):
    kwargs = {"cre": CRE_NAMES[0]} if mode in {"cre", "t7"} else {}
    if mode == "copy_number":
        kwargs = {"cre": CRE_NAMES[0], "copies": np.abs(data.cre).astype(float)}
    fig = plot_spatial(data, mode, **kwargs)
    ax = fig.axes[0]
    assert ax.get_facecolor() == matplotlib.colors.to_rgba("black")
    assert fig.get_facecolor() == matplotlib.colors.to_rgba("black")
    background = ax.collections[0]
    # Every cell, grey, underneath -- so the section outline is always visible.
    assert background.get_offsets().shape == (N_CELL, 2)
    np.testing.assert_allclose(
        background.get_facecolor()[0][:3], matplotlib.colors.to_rgb("grey"), atol=1e-6
    )
    assert ax.get_aspect() == 1.0
    assert ax.get_xticks().size == 0 and ax.get_yticks().size == 0


@pytest.mark.parametrize("mode", SPATIAL_MODES)
def test_every_mode_accepts_an_external_axes(mode, data):
    kwargs = {"cre": CRE_NAMES[0]} if mode in {"cre", "t7"} else {}
    if mode == "copy_number":
        kwargs = {"cre": CRE_NAMES[0], "copies": np.abs(data.cre).astype(float)}
    fig, ax = plt.subplots()
    assert plot_spatial(data, mode, ax=ax, **kwargs) is fig
    assert ax.get_facecolor() == matplotlib.colors.to_rgba("black")


def test_unknown_mode_and_level_are_rejected(data):
    with pytest.raises(ValueError, match="unknown mode"):
        plot_spatial(data, "nonsense")
    with pytest.raises(ValueError, match="unknown level"):
        plot_spatial(data, "celltype", level="nope")


def test_missing_coordinates_are_reported(data, rng):
    import dataclasses

    flat = dataclasses.replace(data, spatial=None)
    with pytest.raises(ValueError, match="no spatial coordinates"):
        plot_spatial(flat, "celltype")


# ---- mode 1: cell type ----------------------------------------------------- #


def test_celltype_mode_draws_one_layer_per_requested_type(data):
    fig = plot_spatial(data, "celltype", celltypes=["Pvalb", "Sst"])
    assert len(_collections(fig)) == 1 + 2  # background + two types
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels == ["Pvalb", "Sst"]


def test_celltype_mode_colours_the_right_cells(data):
    fig = plot_spatial(data, "celltype", celltypes=["Sst"])
    drawn = _collections(fig)[1].get_offsets()
    expected = data.spatial[data.subclass == "Sst"]
    np.testing.assert_allclose(np.sort(drawn, axis=0), np.sort(expected, axis=0))


def test_celltype_mode_can_use_the_class_level(data):
    fig = plot_spatial(data, "celltype", level="class")
    labels = {t.get_text() for t in fig.axes[0].get_legend().get_texts()}
    assert labels == {"GABA", "Glut"}


def test_a_palette_mapping_is_honoured(data):
    fig = plot_spatial(data, "celltype", celltypes=["Pvalb"],
                       palette={"Pvalb": "#123456"})
    np.testing.assert_allclose(
        _collections(fig)[1].get_facecolor()[0][:3],
        matplotlib.colors.to_rgb("#123456"), atol=1e-6,
    )


def test_an_absent_cell_type_is_skipped_not_crashed(data):
    fig = plot_spatial(data, "celltype", celltypes=["Pvalb", "NoSuchType"])
    assert len(_collections(fig)) == 1 + 1


# ---- modes 2-4: values ----------------------------------------------------- #


def test_cre_and_t7_modes_read_different_matrices(data):
    np.testing.assert_array_equal(
        spatial_values(data, "cre", cre=CRE_NAMES[1]), data.cre[:, 1]
    )
    np.testing.assert_array_equal(
        spatial_values(data, "t7", cre=CRE_NAMES[1]), data.t7[:, 1]
    )


def test_copy_number_mode_reads_the_supplied_matrix(data, rng):
    copies = rng.uniform(0, 5, size=(N_CELL, N_CRE))
    np.testing.assert_allclose(
        spatial_values(data, "copy_number", cre=CRE_NAMES[2], copies=copies),
        copies[:, 2],
    )


def test_copy_number_mode_accepts_a_CopyNumberMatrix(data, rng):
    from baystarrfish.inference.copy_number import CopyNumberMatrix

    matrix = CopyNumberMatrix(
        copies=rng.uniform(0, 5, size=(N_CELL, N_CRE)), sd=None, p_infected=None,
        obs_names=data.obs_names, cre_names=list(CRE_NAMES), kmax=60,
        level="subclass", infection_model="copy_number_dropout",
    )
    np.testing.assert_allclose(
        spatial_values(data, "copy_number", cre=CRE_NAMES[0], copies=matrix),
        matrix.copies[:, 0],
    )
    assert plot_spatial(data, "copy_number", cre=CRE_NAMES[0], copies=matrix) is not None


def test_aggregate_sums_or_averages_across_ccres(data):
    np.testing.assert_allclose(
        spatial_values(data, "t7", aggregate="sum"), data.t7.sum(axis=1)
    )
    np.testing.assert_allclose(
        spatial_values(data, "cre", aggregate="mean"), data.cre.mean(axis=1)
    )


def test_value_modes_need_exactly_one_of_cre_or_aggregate(data):
    with pytest.raises(ValueError, match="either cre=|aggregate="):
        spatial_values(data, "cre")
    with pytest.raises(ValueError, match="not both"):
        spatial_values(data, "cre", cre=CRE_NAMES[0], aggregate="sum")
    with pytest.raises(ValueError, match="unknown cCRE"):
        spatial_values(data, "cre", cre="CRE999")
    with pytest.raises(ValueError, match="unknown aggregate"):
        spatial_values(data, "cre", aggregate="median")


def test_copy_number_without_copies_says_so(data):
    with pytest.raises(ValueError, match="needs copies="):
        spatial_values(data, "copy_number", cre=CRE_NAMES[0])


def test_celltype_has_no_per_cell_value(data):
    with pytest.raises(ValueError, match="categorical"):
        spatial_values(data, "celltype")


def test_only_cells_with_signal_are_overlaid(data):
    fig = plot_spatial(data, "cre", cre=CRE_NAMES[0])
    drawn = _collections(fig)[1].get_offsets()
    assert len(drawn) == int((data.cre[:, 0] > 0).sum())
    np.testing.assert_allclose(
        np.sort(drawn, axis=0), np.sort(data.spatial[data.cre[:, 0] > 0], axis=0)
    )


def test_dot_size_and_opacity_increase_with_the_value(data):
    values = np.zeros((N_CELL, N_CRE))
    values[:, 0] = np.linspace(0, 10, N_CELL)
    fig = plot_spatial(data, "copy_number", cre=CRE_NAMES[0], copies=values,
                       size_min=5, size_max=30)
    layer = _collections(fig)[1]
    sizes, alphas = layer.get_sizes(), layer.get_alpha()
    # Drawn in ascending order, so both ramps must be non-decreasing.
    assert np.all(np.diff(sizes) >= -1e-9)
    assert np.all(np.diff(np.asarray(alphas)) >= -1e-9)
    assert sizes.min() >= 5 - 1e-9 and sizes.max() <= 30 + 1e-9


def test_strongest_cells_are_drawn_last(data):
    """plot_gene draws in array order, which buries strong cells under weak ones."""
    values = np.zeros((N_CELL, N_CRE))
    values[:, 0] = np.linspace(10, 0.1, N_CELL)  # descending in array order
    fig = plot_spatial(data, "copy_number", cre=CRE_NAMES[0], copies=values)
    sizes = _collections(fig)[1].get_sizes()
    assert sizes[-1] == sizes.max()


def test_vmax_defaults_to_a_robust_upper_end(data):
    """One extreme cell must not flatten the rest to invisibility."""
    values = np.zeros((N_CELL, N_CRE))
    values[:, 0] = 1.0
    values[0, 0] = 10_000.0
    fig = plot_spatial(data, "copy_number", cre=CRE_NAMES[0], copies=values)
    sizes = _collections(fig)[1].get_sizes()
    assert sizes.max() == pytest.approx(sizes.min(), rel=1e-6)  # the bulk saturates


def test_explicit_vmax_is_respected(data):
    values = np.zeros((N_CELL, N_CRE))
    values[:, 0] = np.linspace(0.1, 10, N_CELL)
    small = plot_spatial(data, "copy_number", cre=CRE_NAMES[0], copies=values, vmax=1000)
    assert _collections(small)[1].get_sizes().max() < 10


def test_log_scaling_compresses_the_range(data):
    values = np.zeros((N_CELL, N_CRE))
    values[:, 0] = np.geomspace(0.01, 1000, N_CELL)
    linear = _collections(plot_spatial(data, "copy_number", cre=CRE_NAMES[0],
                                       copies=values))[1].get_sizes()
    logged = _collections(plot_spatial(data, "copy_number", cre=CRE_NAMES[0],
                                       copies=values, log=True))[1].get_sizes()
    assert logged.std() > linear.std()


def test_each_value_mode_has_its_own_hue(data):
    assert len({MODE_COLORS[m] for m in ("cre", "t7", "copy_number")}) == 3


def test_a_scale_bar_axes_is_added_for_value_modes_only(data):
    assert len(plot_spatial(data, "cre", cre=CRE_NAMES[0]).axes) == 2
    assert len(plot_spatial(data, "cre", cre=CRE_NAMES[0],
                            show_scale_bar=False).axes) == 1
    assert len(plot_spatial(data, "celltype").axes) == 1


def test_mismatched_value_matrix_is_rejected(data, rng):
    with pytest.raises(ValueError, match="rows for"):
        spatial_values(data, "copy_number", cre=CRE_NAMES[0],
                       copies=rng.uniform(size=(N_CELL - 1, N_CRE)))


# ---- cropping -------------------------------------------------------------- #


def test_region_filters_both_layers_consistently(data):
    fig = plot_spatial(data, "cre", cre=CRE_NAMES[0], x_region=(0, 50))
    background = _collections(fig)[0].get_offsets()
    assert len(background) == int((data.spatial[:, 0] < 50).sum())
    assert background[:, 0].max() < 50


def test_an_empty_region_is_reported(data):
    with pytest.raises(ValueError, match="no cells"):
        plot_spatial(data, "celltype", x_region=(1e6, 2e6))


def test_flips_and_transpose_move_the_coordinates(data):
    plain = _collections(plot_spatial(data, "celltype"))[0].get_offsets()
    flipped = _collections(plot_spatial(data, "celltype", flipx=-1))[0].get_offsets()
    np.testing.assert_allclose(np.asarray(flipped)[:, 0], -np.asarray(plain)[:, 0])
    swapped = _collections(plot_spatial(data, "celltype", transpose=-1))[0].get_offsets()
    np.testing.assert_allclose(np.asarray(swapped)[:, 0], np.asarray(plain)[:, 1])


def test_title_defaults_name_the_mode_and_the_ccre(data):
    assert CRE_NAMES[0] in plot_spatial(data, "cre", cre=CRE_NAMES[0]).axes[0].get_title()
    assert "T7" in plot_spatial(data, "t7", cre=CRE_NAMES[0]).axes[0].get_title()
    assert "E[k" in plot_spatial(
        data, "copy_number", cre=CRE_NAMES[0], copies=np.abs(data.cre).astype(float)
    ).axes[0].get_title()
    assert plot_spatial(data, "celltype", title="mine").axes[0].get_title() == "mine"


def test_from_anndata_carries_the_spatial_coordinates(rng):
    import anndata as ad

    n = 6
    adata = ad.AnnData(
        np.zeros((n, 2)),
        obs=pd.DataFrame(
            {"subclass": ["a"] * n, "class": ["A"] * n,
             "labeling_type": ["x"] * n},
            index=[f"c{i}" for i in range(n)],
        ),
    )
    adata.obsm["X_spatial"] = rng.uniform(size=(n, 2))
    assert adata.obsm["X_spatial"].shape == (n, 2)


# ---- the copy-number mode is nonzero everywhere, unlike the count modes ---- #


def test_evidence_mask_selects_cells_with_a_read_in_either_channel(data):
    from baystarrfish.plotting import evidence_mask

    got = evidence_mask(data, cre=CRE_NAMES[0], aggregate=None)
    want = (data.t7[:, 0] > 0) | (data.cre[:, 0] > 0)
    np.testing.assert_array_equal(got, want)
    pooled = evidence_mask(data, cre=None, aggregate="sum")
    np.testing.assert_array_equal(
        pooled, (data.t7.sum(axis=1) > 0) | (data.cre.sum(axis=1) > 0)
    )


def test_copy_number_draws_only_evidence_bearing_cells(data, rng):
    """E[k|obs] is defined everywhere; drawing all of it paints the whole tissue."""
    from baystarrfish.plotting import evidence_mask

    copies = rng.uniform(0.01, 5.0, size=(N_CELL, N_CRE))  # nonzero everywhere
    fig = plot_spatial(data, "copy_number", cre=CRE_NAMES[0], copies=copies)
    drawn = len(_collections(fig)[1].get_offsets())
    expected = int(evidence_mask(data, cre=CRE_NAMES[0], aggregate=None).sum())
    assert drawn == expected < N_CELL


def test_min_value_overrides_the_default_visibility_rule(data, rng):
    copies = np.tile(np.linspace(0, 10, N_CELL)[:, None], (1, N_CRE))
    fig = plot_spatial(data, "copy_number", cre=CRE_NAMES[0], copies=copies,
                       min_value=5.0)
    drawn = len(_collections(fig)[1].get_offsets())
    assert drawn == int((copies[:, 0] > 5.0).sum())


def test_min_value_also_applies_to_the_count_modes(data):
    fig = plot_spatial(data, "t7", cre=CRE_NAMES[0], min_value=1.0)
    assert len(_collections(fig)[1].get_offsets()) == int((data.t7[:, 0] > 1).sum())
