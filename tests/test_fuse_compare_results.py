import unittest

from scripts.fuse_compare_results import paired_statistics


class PairedStatisticsTests(unittest.TestCase):
    def test_patient_is_unit_and_pairs_are_preserved(self):
        # Patient 2 has repeated frame-derived records collapsed upstream to one
        # patient score; only the matched patient-level deltas enter inference.
        first = {"p1": 0.80, "p2": 0.90, "p3": 0.70}
        second = {"p1": 0.70, "p2": 0.85, "p3": 0.75}
        result = paired_statistics(first, second, bootstrap_resamples=2_000, seed=42)
        self.assertEqual(result["patient_ids"], ["p1", "p2", "p3"])
        self.assertEqual(result["n_paired_patients"], 3)
        self.assertAlmostEqual(result["mean_difference"], 1 / 30)
        self.assertEqual(result["bootstrap_resamples"], 2_000)
        self.assertEqual(result["bootstrap_seed"], 42)
        self.assertEqual(len(result["patient_bootstrap_mean_difference_95ci"]), 2)
        self.assertGreaterEqual(result["wilcoxon_two_sided"]["p_value"], 0.0)
        self.assertLessEqual(result["wilcoxon_two_sided"]["p_value"], 1.0)

    def test_rejects_unpaired_small_cohort(self):
        with self.assertRaisesRegex(ValueError, "two paired patients"):
            paired_statistics({"p1": 0.9}, {"p1": 0.8}, bootstrap_resamples=10)

    def test_rejects_mismatched_patient_sets(self):
        with self.assertRaisesRegex(ValueError, "patient ID sets differ"):
            paired_statistics({"p1": 0.9, "p2": 0.8}, {"p1": 0.85, "p3": 0.7},
                              bootstrap_resamples=10)

    def test_all_zero_differences_have_defined_test_result(self):
        result = paired_statistics({"p1": 0.8, "p2": 0.9}, {"p1": 0.8, "p2": 0.9},
                                   bootstrap_resamples=50)
        self.assertEqual(result["wilcoxon_two_sided"], {"statistic": 0.0, "p_value": 1.0})
        self.assertEqual(result["patient_bootstrap_mean_difference_95ci"], [0.0, 0.0])

    def test_bootstrap_count_must_be_positive_integer(self):
        for count in (0, -1, 1.5):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "positive integer"):
                paired_statistics({"p1": 0.8, "p2": 0.9}, {"p1": 0.7, "p2": 0.8},
                                  bootstrap_resamples=count)


if __name__ == "__main__":
    unittest.main()
