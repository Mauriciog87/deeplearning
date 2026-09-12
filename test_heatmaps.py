import tempfile
import subprocess
import sys
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from src.utils.heatmaps import (
    CATEGORY_LABELS,
    HeatmapConfig,
    build_session_drift_matrix,
    build_number_residuals,
    build_rolling_category_residuals,
    build_rolling_number_residuals,
    build_table_heatmap_matrix,
    build_wheel_sector_residuals,
    detect_heatmap_anomalies,
    generate_heatmap_report,
)


class HeatmapTest(unittest.TestCase):
    def test_table_heatmap_matrix_has_expected_layout(self):
        numbers = list(range(37)) * 3

        matrix = build_table_heatmap_matrix(numbers)

        self.assertEqual(matrix.shape, (3, 13))
        self.assertTrue(np.isnan(matrix[0, 0]))
        self.assertTrue(np.isnan(matrix[2, 0]))
        self.assertAlmostEqual(matrix[1, 0], 0.0)

    def test_uniform_numbers_have_zero_residuals(self):
        numbers = list(range(37)) * 5

        residuals = build_number_residuals(numbers)

        self.assertTrue(np.allclose(residuals, np.zeros(37)))

    def test_biased_number_has_positive_residual(self):
        numbers = list(range(37)) * 2 + [7] * 60

        residuals = build_number_residuals(numbers)

        self.assertGreater(residuals[7], 0.0)
        self.assertLess(residuals[0], 0.0)

    def test_rolling_matrices_respect_window_and_step(self):
        numbers = list(range(37)) * 4

        number_matrix, number_labels = build_rolling_number_residuals(
            numbers,
            window_size=30,
            step_size=15,
        )
        category_matrix, category_labels, feature_labels = build_rolling_category_residuals(
            numbers,
            window_size=30,
            step_size=15,
        )

        expected_windows = ((len(numbers) - 30) // 15) + 1
        self.assertEqual(number_matrix.shape, (expected_windows, 37))
        self.assertEqual(category_matrix.shape, (expected_windows, len(CATEGORY_LABELS)))
        self.assertEqual(len(number_labels), expected_windows)
        self.assertEqual(len(category_labels), expected_windows)
        self.assertEqual(feature_labels, CATEGORY_LABELS)

    def test_wheel_sector_residuals_match_sector_count(self):
        numbers = list(range(37)) * 3

        residuals, labels = build_wheel_sector_residuals(numbers, sector_count=12)

        self.assertEqual(len(residuals), 12)
        self.assertEqual(len(labels), 12)
        self.assertTrue(np.allclose(residuals, np.zeros(12)))

    def test_generate_report_creates_expected_pngs(self):
        numbers = list(range(37)) * 5
        with tempfile.TemporaryDirectory() as temp_dir:
            config = HeatmapConfig(
                window_size=50,
                step_size=25,
                output_dir=temp_dir,
                show=False,
                dpi=80,
            )

            result = generate_heatmap_report(numbers, config, source_label="test")

            self.assertEqual(result.total_spins, len(numbers))
            self.assertTrue(any('independent spins' in warning for warning in result.warnings))
            self.assertEqual(
                set(result.output_paths),
                {
                    "number_table",
                    "wheel_sector",
                    "rolling_number",
                    "rolling_category",
                    "anomaly_summary",
                },
            )
            for path in result.output_paths.values():
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)

    def test_short_dataset_raises_error(self):
        with self.assertRaises(ValueError):
            generate_heatmap_report(list(range(20)), HeatmapConfig(show=False))

    def test_short_rolling_dataset_warns_and_keeps_static_outputs(self):
        numbers = list(range(37)) * 2
        with tempfile.TemporaryDirectory() as temp_dir:
            config = HeatmapConfig(
                window_size=100,
                step_size=25,
                output_dir=temp_dir,
                show=False,
                dpi=80,
            )

            result = generate_heatmap_report(numbers, config, source_label="short rolling")

            self.assertEqual(
                set(result.output_paths),
                {"number_table", "wheel_sector", "anomaly_summary"},
            )
            self.assertTrue(result.warnings)

    def test_fair_dataset_has_no_strong_anomalies(self):
        numbers = list(range(37)) * 10

        anomalies = detect_heatmap_anomalies(
            numbers,
            window_size=500,
            step_size=100,
            z_threshold=3.0,
        )

        self.assertEqual(anomalies, [])

    def test_biased_dataset_flags_number_anomaly(self):
        numbers = list(range(37)) * 2 + [7] * 120

        anomalies = detect_heatmap_anomalies(
            numbers,
            window_size=500,
            step_size=100,
            z_threshold=2.5,
        )

        self.assertTrue(any(item.feature == "number_7" and item.z_score > 0 for item in anomalies))
        self.assertTrue(all(0.0 <= item.p_value <= 1.0 for item in anomalies))
        self.assertTrue(all(0.0 <= item.q_value <= 1.0 for item in anomalies))
        self.assertTrue(any(
            item.feature == "number_7" and item.fdr_significant
            for item in anomalies
        ))

    def test_anomaly_threshold_is_respected(self):
        numbers = list(range(37)) * 2 + [7] * 120

        anomalies = detect_heatmap_anomalies(
            numbers,
            window_size=500,
            step_size=100,
            z_threshold=99.0,
        )

        self.assertEqual(anomalies, [])

    def test_session_drift_matrix_is_symmetric(self):
        sessions = {
            "fair_a": list(range(37)) * 3,
            "fair_b": list(reversed(range(37))) * 3,
            "biased": list(range(37)) * 2 + [7] * 60,
        }

        matrix, labels, results = build_session_drift_matrix(sessions)

        self.assertEqual(matrix.shape, (3, 3))
        self.assertEqual(labels, ["fair_a", "fair_b", "biased"])
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertTrue(np.allclose(np.diag(matrix), np.zeros(3)))
        self.assertLess(matrix[0, 1], matrix[0, 2])
        self.assertTrue(results)

    def test_generate_report_creates_drift_heatmap_for_sessions(self):
        numbers = list(range(37)) * 5
        sessions = {
            "fair": numbers,
            "biased": list(range(37)) * 2 + [7] * 80,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            config = HeatmapConfig(
                window_size=50,
                step_size=25,
                output_dir=temp_dir,
                show=False,
                dpi=80,
            )

            result = generate_heatmap_report(
                numbers,
                config,
                source_label="global",
                session_numbers=sessions,
            )

            self.assertIn("session_drift", result.output_paths)
            self.assertTrue(Path(result.output_paths["session_drift"]).exists())
            self.assertTrue(result.drift_results)

    def test_cli_help_exposes_new_heatmap_flags(self):
        completed = subprocess.run(
            [sys.executable, "roulette_cli.py", "heatmaps", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("--z-threshold", completed.stdout)
        self.assertIn("--fdr-alpha", completed.stdout)
        self.assertIn("--include-drift", completed.stdout)
        self.assertIn("--no-anomalies", completed.stdout)


if __name__ == "__main__":
    unittest.main()
