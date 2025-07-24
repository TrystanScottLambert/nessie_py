"""
Testing module for the completeness implementation.
"""

import unittest
import numpy as np

from nessie.helper_funcs import calculate_completeness


class TestCalculateCompleteness(unittest.TestCase):
    """
    Testing Completess implementation in python.
    """

    def test_ra_dec_length_mismatch(self):
        """Crash when ra is not the right length"""
        ra_observed = np.array([10.0, 20.0, 30.0])
        dec_observed = np.array([10.0, 20.0])  # Mismatch

        ra_target = np.array([15.0, 25.0])
        dec_target = np.array([15.0, 25.0])

        ra_eval = np.array([12.0, 18.0])
        dec_eval = np.array([12.0, 18.0])
        radii = np.array([1.0, 1.0])

        with self.assertRaises(ValueError) as context:
            calculate_completeness(
                ra_observed,
                dec_observed,
                ra_target,
                dec_target,
                ra_eval,
                dec_eval,
                radii,
            )
        self.assertIn(
            "ra arrays and dec arrays must be the same length", str(context.exception)
        )

    def test_search_radii_length_mismatch(self):
        """Crash when radii doesn't match the array"""
        ra_observed = np.array([10.0, 20.0])
        dec_observed = np.array([10.0, 20.0])

        ra_target = np.array([15.0, 25.0])
        dec_target = np.array([15.0, 25.0])

        ra_eval = np.array([12.0, 18.0])
        dec_eval = np.array([12.0, 18.0])
        radii = np.array([1.0])  # Too short

        with self.assertRaises(ValueError) as context:
            calculate_completeness(
                ra_observed,
                dec_observed,
                ra_target,
                dec_target,
                ra_eval,
                dec_eval,
                radii,
            )
        self.assertIn("search_radii must be the same length", str(context.exception))

    def test_completeness_actual_call_partial(self):
        """Should return completeness = 0.75 at each evaluation point"""
        ra_target = np.array([20.0] * 4)
        dec_target = np.array([-20.0] * 4)
        ra_observed = np.array([20.0] * 3)
        dec_observed = np.array([-20.0] * 3)
        ra_eval = ra_observed.copy()
        dec_eval = dec_observed.copy()
        radii = np.array([1e-6] * 3)

        result = calculate_completeness(
            ra_observed, dec_observed, ra_target, dec_target, ra_eval, dec_eval, radii
        )
        expected = np.array([0.75] * 3)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_completeness_actual_call_full(self):
        """Should return completeness = 1.0 for full match"""
        ra_target = np.array([20.0, 30.0, 40.0, 50.0])
        dec_target = np.array([-20.0, -18.0, 20.0, 36.0])
        ra_observed = ra_target.copy()
        dec_observed = dec_target.copy()
        ra_eval = ra_observed.copy()
        dec_eval = dec_observed.copy()
        radii = np.array([0.1] * 4)

        result = calculate_completeness(
            ra_observed, dec_observed, ra_target, dec_target, ra_eval, dec_eval, radii
        )
        expected = np.array([1.0] * 4)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_completeness_actual_call_undersample(self):
        """Should still return valid completeness if observed is a subset"""
        ra_target = np.array([20.0, 30.0, 40.0, 50.0])
        dec_target = np.array([-20.0, -18.0, 20.0, 36.0])
        ra_observed = np.array([20.0, 30.0, 40.0])
        dec_observed = np.array([-20.0, -18.0, 20.0])
        ra_eval = ra_observed.copy()
        dec_eval = dec_observed.copy()
        radii = np.array([0.001] * 3)

        result = calculate_completeness(
            ra_observed, dec_observed, ra_target, dec_target, ra_eval, dec_eval, radii
        )
        expected = np.array([1.0] * 3)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)


if __name__ == "__main__":
    unittest.main()
