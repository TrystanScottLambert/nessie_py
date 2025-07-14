"""
Unit test checking the validation functionality in Nessie.
"""

import warnings
import unittest
import numpy as np
from nessie.helper_funcs import validate, ValidationType


class TestValidateFunction(unittest.TestCase):
    """Unit tests for the validate function."""

    # --- RA ---
    def test_ra_valid(self):
        """Check that RA values in valid range pass without error."""
        try:
            validate(np.array([0.0, 180.0, 359.9]), ValidationType.RA)
        except Exception as e:
            self.fail(f"RA validation raised unexpectedly: {e}")

    def test_ra_below_zero_raises(self):
        """Check that negative RA values raise an error."""
        with self.assertRaises(ValueError):
            validate(np.array([-1.0]), ValidationType.RA)

    def test_ra_above_360_raises(self):
        """Check that RA values above 360 raise an error."""
        with self.assertRaises(ValueError):
            validate(np.array([361.0]), ValidationType.RA)

    # --- DEC ---
    def test_dec_valid(self):
        """Check that Declination values in valid range pass without error."""
        try:
            validate(np.array([-90.0, 0.0, 89.9]), ValidationType.DEC)
        except Exception as e:
            self.fail(f"Dec validation raised unexpectedly: {e}")

    def test_dec_below_negative_90_raises(self):
        """Check that Declination values below -90 raise an error."""
        with self.assertRaises(ValueError):
            validate(np.array([-91.0]), ValidationType.DEC)

    def test_dec_above_90_raises(self):
        """Check that Declination values above 90 raise an error."""
        with self.assertRaises(ValueError):
            validate(np.array([91.0]), ValidationType.DEC)

    # --- Redshift ---
    def test_redshift_valid(self):
        """Check that redshift values in valid range pass without error."""
        try:
            validate(np.array([0.0, 1.0, 10.0]), ValidationType.REDSHIFT)
        except Exception as e:
            self.fail(f"Redshift validation raised unexpectedly: {e}")

    def test_redshift_negative_raises(self):
        """Check that negative redshift values raise an error."""
        with self.assertRaises(ValueError):
            validate(np.array([-0.1]), ValidationType.REDSHIFT)

    def test_redshift_large_warns(self):
        """Check that redshift values above 1100 raise a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate(np.array([1200.0]), ValidationType.REDSHIFT)
            self.assertTrue(
                any("redshifts are very large" in str(warn.message) for warn in w)
            )

    # --- Absolute Magnitude ---
    def test_abs_mag_valid(self):
        """Check that valid absolute magnitudes pass without error."""
        try:
            validate(np.array([-21.0, -10.0]), ValidationType.ABS_MAG)
        except Exception as e:
            self.fail(f"Absolute magnitude validation raised unexpectedly: {e}")

    def test_abs_mag_too_faint_warns(self):
        """Check that absolute magnitudes brighter than -4 raise a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate(np.array([-3.0]), ValidationType.ABS_MAG)
            self.assertTrue(
                any("absolute magnitudes look unusual" in str(warn.message) for warn in w)
            )

    def test_abs_mag_too_bright_warns(self):
        """Check that absolute magnitudes fainter than -50 raise a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate(np.array([-51.0]), ValidationType.ABS_MAG)
            self.assertTrue(
                any("absolute magnitudes look unusual" in str(warn.message) for warn in w)
            )

    # --- Completeness ---
    def test_completeness_valid(self):
        """Check that completeness between 0 and 1 passes without error."""
        try:
            validate(np.array([0.85, 0.1]), ValidationType.COMPLETENESS)
        except Exception as e:
            self.fail(f"Completeness validation raised unexpectedly: {e}")

    def test_completeness_not_numeric_raises(self):
        """Check that non-numeric completeness raises a TypeError."""
        with self.assertRaises(TypeError):
            validate(np.array(["not a number", 0.1]), ValidationType.COMPLETENESS)

    def test_completeness_below_zero_raises(self):
        """Check that completeness below 0 raises a ValueError."""
        with self.assertRaises(ValueError):
            validate(np.array([-0.1, 0.1]), ValidationType.COMPLETENESS)

    def test_completeness_above_one_raises(self):
        """Check that completeness above 1 raises a ValueError."""
        with self.assertRaises(ValueError):
            validate(np.array([1.1, 0.1]), ValidationType.COMPLETENESS)

    def test_vel_errors_valid(self):
        """Check that the velocity errors above zero pass without error."""
        try:
            validate(np.array([50., 50.]), ValidationType.VEL_ERR)
        except Exception as e:
            self.fail(f"Velocity Errors failed unexpectedly: {e}")

    def test_vel_errors_warning(self):
        """Checking that there is a warning when the velocity errors seem too large."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate(np.array([2001]), ValidationType.VEL_ERR)
            self.assertTrue(
                any("Warning: velocity errors seem very large. Are units correct?" in str(warn.message) for warn in w)
            )

    def test_vel_errors_below_zero_raises(self):
        """Check that vel_err below 0 raises a ValueError."""
        with self.assertRaises(ValueError):
            validate(np.array([-50, 50]), ValidationType.VEL_ERR)

    # --- B0 ---
    def test_b0_valid(self):
        """Check that valid B0 value passes."""
        try:
            validate(1.5, ValidationType.B0)
        except Exception as e:
            self.fail(f"B0 validation raised unexpectedly: {e}")

    def test_b0_not_numeric_raises(self):
        """Check that non-numeric B0 raises a TypeError."""
        with self.assertRaises(TypeError):
            validate("bad", ValidationType.B0)

    def test_b0_negative_raises(self):
        """Check that negative B0 raises a ValueError."""
        with self.assertRaises(ValueError):
            validate(-1.0, ValidationType.B0)

    # --- R0 ---
    def test_r0_valid(self):
        """Check that valid R0 value passes."""
        try:
            validate(4.2, ValidationType.R0)
        except Exception as e:
            self.fail(f"R0 validation raised unexpectedly: {e}")

    def test_r0_not_numeric_raises(self):
        """Check that non-numeric R0 raises a TypeError."""
        with self.assertRaises(TypeError):
            validate(None, ValidationType.R0)

    def test_r0_negative_raises(self):
        """Check that negative R0 raises a ValueError."""
        with self.assertRaises(ValueError):
            validate(-1.0, ValidationType.R0)

    # --- General Array Validation ---
    def test_nan_in_array_raises(self):
        """Check that arrays containing NaN raise a ValueError."""
        with self.assertRaises(ValueError):
            validate(np.array([1.0, np.nan]), ValidationType.RA)

    def test_inf_in_array_raises(self):
        """Check that arrays containing Inf raise a ValueError."""
        with self.assertRaises(ValueError):
            validate(np.array([1.0, np.inf]), ValidationType.DEC)

    def test_non_numeric_array_raises(self):
        """Check that arrays with non-numeric values raise a TypeError."""
        with self.assertRaises(TypeError):
            validate(np.array(["a", "b"]), ValidationType.REDSHIFT)


if __name__ == "__main__":
    unittest.main()
