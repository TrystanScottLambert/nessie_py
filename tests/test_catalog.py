"""
Unit tests for the RedshiftCatalog implementation
"""

import unittest
import numpy as np

from catalog import RedshiftCatalog
from cosmology import FlatCosmology
from helper_funcs import create_density_function


class TestRedshiftCatalog(unittest.TestCase):
    """
    Testing the RedshiftCatalog class and methods in the catalog module.
    """

    def test_it_runs(self):
        """
        Simple tets to ensure basic funtionality.
        """
        ra_array = np.array([120.0, 120.0, 50.0])
        dec_array = np.array([-34.0, -34.0, 23.0])
        redshift_array = np.array([0.2, 0.2, 0.6])
        velocity_errors = np.array([50.0, 50.0, 50.0])
        absolute_magnitudes = np.array([-18.0, -18.0, -18.0])

        cosmo = FlatCosmology(0.7, 0.3)
        completeness = np.repeat(0.98, len(ra_array))
        density_function = lambda z: np.repeat(0.2, len(z))
        cat = RedshiftCatalog(
            ra_array, dec_array, redshift_array, density_function, cosmo, completeness
        )

        group = cat.get_raw_groups(0.06, 18)
        cat.run_fof(0.06, 18)
        group_catalog = cat.calculate_group_table(absolute_magnitudes, velocity_errors)

        np.testing.assert_array_equal(group["galaxy_id"], np.array([1, 2]))
        np.testing.assert_array_equal(group["group_id"], np.array([1, 1]))
        np.testing.assert_array_equal(cat.group_ids, np.array([1, 1, -1]))
        self.assertEqual(round(group_catalog["ra"][0]), 120.0)
        self.assertEqual(round(group_catalog["dec"][0]), -34.0)
        self.assertEqual(group_catalog["multiplicity"][0], 2)

    def test_completeness_is_automatically_set(self):
        """
        Checking that the completeness is updated appropriately
        """
        ra_array = np.array([120.0, 120.0, 50.0])
        dec_array = np.array([-34.0, -34.0, 23.0])
        redshift_array = np.array([0.2, 0.2, 0.6])
        cosmo = FlatCosmology(0.7, 0.3)
        density_function = lambda z: np.repeat(0.2, len(z))
        cat = RedshiftCatalog(
            ra_array, dec_array, redshift_array, density_function, cosmo
        )
        np.testing.assert_array_equal(cat.completeness, np.ones(len(ra_array)))

    def test_getting_group_ids_works_on_simple_case(self):
        """
        More complete example. Especially to test the 1-indexing between R and python.
        """
        b0 = 100.0
        r0 = 180.0
        ras = np.array(
            [20.0, 20.0, 20.0, -20.0, -20.0, -20.0, 100.0, 100.0, 0.0, 180.0]
        )
        decs = np.array([-50.0, -50.0, -50.0, 0.0, 0.0, 0.0, 90.0, 90.0, 45.0, -45.0])
        redshifts = np.array([0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.2, 0.2])
        cosmo = FlatCosmology(0.7, 0.3)

        random_redshifts = np.random.normal(0.2, 0.1, 20000)
        random_redshifts = random_redshifts[random_redshifts > 0]
        rho_mean = create_density_function(
            random_redshifts, 20000, 0.001, cosmology=cosmo
        )

        cat = RedshiftCatalog(ras, decs, redshifts, rho_mean, cosmo)
        cat.run_fof(b0, r0)

        expected = np.array([1, 1, 1, 2, 2, 2, 3, 3, -1, -1])
        np.testing.assert_array_equal(cat.group_ids, expected)

    def test_comparison_to_mocks_is_working(self):
        """
        Testing that class implementation of the s-score is working.
        """
        group_ids = np.array([1, 1, 1, 2, 2, 2, 2, 3, 3, -1, -1])
        mock_group_ids = np.array([100, 100, 100, 200, 200, -1, -1, -1, -1, -1, -1])
        ras = np.full(11, 120.0)
        decs = np.full(11, 45.0)
        redshifts = np.full(11, 0.2)
        cosmo = FlatCosmology(0.7, 0.3)
        rho_mean = lambda z: np.ones(len(z))

        cat = RedshiftCatalog(ras, decs, redshifts, rho_mean, cosmo)
        cat.group_ids = group_ids
        cat.mock_group_ids = mock_group_ids

        metrics = cat.compare_to_mock()
        self.assertAlmostEqual(metrics, 0.1111, places=3)


if __name__ == "__main__":
    unittest.main()
