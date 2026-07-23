#!/usr/bin/env python3

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from plot_section_reproducibility import (
    activity_correlation_tables,
    call_metrics,
    compute_mean_control_tests,
    plot_activity_correlation,
    plot_activity_correlation_violins,
    plot_reproducibility_by_celltype,
    plot_test_concordance,
    posterior_mean_control_centered_activity,
    restrict_to_shared_eligibility,
)


class MeanControlTestTests(unittest.TestCase):
    def test_uses_draw_wise_mean_of_all_seven_controls(self) -> None:
        groups = np.asarray(["group_a"])
        cre_names = np.asarray(
            [*(f"control_{index}" for index in range(7)), "target_high", "target_low"]
        )
        log_gamma = np.zeros((4, 1, 9), dtype=np.float32)
        log_gamma[:, 0, :7] = np.arange(7, dtype=np.float32)
        log_gamma[:, 0, 7] = 4.0
        log_gamma[:, 0, 8] = 2.0
        t7_totals = np.full((1, 9), 100.0)

        result = compute_mean_control_tests(
            log_gamma=log_gamma,
            groups=groups,
            cre_names=cre_names,
            negative_controls=cre_names[:7].tolist(),
            t7_totals=t7_totals,
            t7_threshold=50,
            effect_threshold=0,
            section="sec1",
        ).set_index("cre")

        self.assertTrue((result["n_negative_controls"] == 7).all())
        self.assertTrue(np.allclose(result["mean_negative_control_activity_mean"], 3.0))
        self.assertEqual(
            result.loc["target_high", "activity_vs_mean_control_mean"], 1.0
        )
        self.assertEqual(
            result.loc["target_low", "activity_vs_mean_control_mean"], -1.0
        )
        self.assertEqual(result.loc["target_high", "p_right"], 0.0)
        self.assertEqual(result.loc["target_low", "p_right"], 1.0)
        centered = posterior_mean_control_centered_activity(
            log_gamma,
            np.arange(7),
            np.asarray([7, 8]),
        )
        self.assertTrue(np.allclose(centered, [[1.0, -1.0]]))


class ConcordanceMetricTests(unittest.TestCase):
    def test_call_contingency_and_metrics(self) -> None:
        frame = pd.DataFrame(
            {
                "significant_sec1": [True, True, False, False],
                "significant_sec2": [True, False, True, False],
            }
        )
        result = call_metrics(frame)

        self.assertEqual(result["n_both_significant"], 1)
        self.assertEqual(result["n_sec1_only"], 1)
        self.assertEqual(result["n_sec2_only"], 1)
        self.assertEqual(result["n_both_nonsignificant"], 1)
        self.assertEqual(result["concordance"], 0.5)
        self.assertAlmostEqual(result["significant_jaccard"], 1 / 3)
        self.assertEqual(result["cohen_kappa"], 0.0)
        self.assertEqual(result["n_valid_ccres"], 4)
        self.assertEqual(result["n_reproducible_significant_ccres"], 1)
        self.assertEqual(result["n_reproducible_nonsignificant_ccres"], 1)
        self.assertEqual(result["n_reproducible_ccres"], 2)
        self.assertEqual(result["reproducibility"], 0.5)

    def test_shared_t7_universe_is_used_for_both_sections_and_activity(self) -> None:
        tests = {
            "sec1": pd.DataFrame(
                {
                    "section": "sec1",
                    "group": ["group_a", "group_a", "group_a"],
                    "cre": ["shared_a", "shared_b", "sec1_only"],
                    "target_t7_total": [50.0, 60.0, 70.0],
                    "negative_control_t7_total": [80.0, 80.0, 80.0],
                    "p_right": [0.01, 0.04, 0.0001],
                }
            ),
            "sec2": pd.DataFrame(
                {
                    "section": "sec2",
                    "group": ["group_a", "group_a", "group_a"],
                    "cre": ["shared_a", "shared_b", "sec2_only"],
                    "target_t7_total": [55.0, 65.0, 75.0],
                    "negative_control_t7_total": [90.0, 90.0, 90.0],
                    "p_right": [0.02, 0.05, 0.0001],
                }
            ),
        }
        filtered, shared = restrict_to_shared_eligibility(tests)

        expected_keys = {("group_a", "shared_a"), ("group_a", "shared_b")}
        for section in ("sec1", "sec2"):
            keys = set(
                filtered[section][["group", "cre"]].itertuples(index=False, name=None)
            )
            self.assertEqual(keys, expected_keys)
        self.assertEqual(len(shared), 2)
        self.assertTrue(shared["shared_t7_eligible"].all())
        self.assertTrue(np.allclose(filtered["sec1"]["q_right"], [0.02, 0.04]))
        self.assertTrue(np.allclose(filtered["sec2"]["q_right"], [0.04, 0.05]))

        activity = {
            "sec1": pd.DataFrame(
                [[1.0, 2.0, 3.0]],
                index=["group_a"],
                columns=["shared_a", "shared_b", "not_eligible"],
            ),
            "sec2": pd.DataFrame(
                [[1.1, 2.1, 3.1]],
                index=["group_a"],
                columns=["shared_a", "shared_b", "not_eligible"],
            ),
        }
        activity_pair, _ = activity_correlation_tables(activity, shared, min_points=2)
        self.assertEqual(set(activity_pair["cre"]), {"shared_a", "shared_b"})

    def test_both_requested_figure_pairs_are_written(self) -> None:
        pair = pd.DataFrame(
            {
                "activity_sec1": np.linspace(-1, 1, 20),
                "activity_sec2": np.linspace(-0.9, 1.1, 20),
                "measured": True,
            }
        )
        correlations = pd.DataFrame(
            [
                {
                    "axis": "all_pairs",
                    "unit": "all",
                    "n_points": 20,
                    "spearman": 1.0,
                    "pearson": 1.0,
                },
                {
                    "axis": "within_subclass_across_ccres",
                    "unit": "group_a",
                    "n_points": 10,
                    "spearman": 0.8,
                    "pearson": 0.85,
                },
                {
                    "axis": "within_subclass_across_ccres",
                    "unit": "group_b",
                    "n_points": 10,
                    "spearman": 0.7,
                    "pearson": 0.75,
                },
                {
                    "axis": "across_subclasses_per_ccre",
                    "unit": "cre_a",
                    "n_points": 10,
                    "spearman": 0.6,
                    "pearson": 0.65,
                },
                {
                    "axis": "across_subclasses_per_ccre",
                    "unit": "cre_b",
                    "n_points": 10,
                    "spearman": 0.5,
                    "pearson": 0.55,
                },
            ]
        )
        call_frame = pd.DataFrame(
            {
                "significant_sec1": [True, True, False, False],
                "significant_sec2": [True, False, True, False],
            }
        )
        metrics = call_metrics(call_frame)
        overall = pd.DataFrame([{"unit": "all", **metrics}])
        by_group = pd.DataFrame(
            [
                {"group": "group_a", **metrics},
                {"group": "group_b", **metrics},
            ]
        )

        with TemporaryDirectory() as tempdir:
            figures = Path(tempdir)
            plot_activity_correlation(pair, correlations, figures, 50)
            plot_activity_correlation_violins(correlations, figures, 50)
            plot_test_concordance(overall, by_group, figures, 0.05, 50)
            plot_reproducibility_by_celltype(overall, by_group, figures, 0.05, 50)
            for stem in (
                "section_bayesian_activity_correlation",
                "section_bayesian_activity_correlation_violins",
                "section_bayesian_test_concordance",
                "section_bayesian_reproducibility_by_celltype",
            ):
                self.assertTrue((figures / f"{stem}.pdf").exists())
                self.assertTrue((figures / f"{stem}.png").exists())


if __name__ == "__main__":
    unittest.main()
