"""
Test Suite for Score Parameter Optimizer & Regularized Generalization Benchmark
"""

import unittest
from backend.ml.parameter_optimizer import (
    ScoreParameterOptimizer,
    generate_historical_calibration_matches,
    FORMAT_PARAMETER_CONFIGS,
    evaluate_parameter_set
)

class TestScoreParameterOptimizer(unittest.TestCase):
    def setUp(self):
        self.matches = generate_historical_calibration_matches(seed=42)

    def test_historical_dataset_generation(self):
        self.assertEqual(len(self.matches), 600)
        t20_count = sum(1 for m in self.matches if m["format"] == "T20I")
        odi_count = sum(1 for m in self.matches if m["format"] == "ODI")
        test_count = sum(1 for m in self.matches if m["format"] == "TEST")
        self.assertEqual(t20_count, 250)
        self.assertEqual(odi_count, 230)
        self.assertEqual(test_count, 120)

    def test_out_of_time_t20_optimization(self):
        optimizer = ScoreParameterOptimizer(lambda_reg=0.04)
        res = optimizer.optimize_format("T20I", self.matches)
        
        self.assertIn("parameters", res)
        self.assertIn("train_metrics", res)
        self.assertIn("out_of_time_test_metrics_2026", res)
        
        test_metrics = res["out_of_time_test_metrics_2026"]
        # Error on unseen out-of-time test matches must decrease or hold steady without exploding
        self.assertLessEqual(test_metrics["optimized_mae"], test_metrics["baseline_mae"] * 1.05)
        # 80% CI Empirical Coverage must stay well-calibrated (between 70% and 95%)
        self.assertGreaterEqual(test_metrics["ci_coverage_pct"], 70.0)
        self.assertLessEqual(test_metrics["ci_coverage_pct"], 95.0)

if __name__ == "__main__":
    unittest.main()
