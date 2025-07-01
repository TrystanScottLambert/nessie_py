"""
Unit tests for the helper funcs module.
"""

import unittest
import numpy as np
import numpy.testing as npt

from helper_funcs import create_density_function, calculate_s_total
from cosmology import FlatCosmology


class TestDensityFunction(unittest.TestCase):
    """
    Testing the create_density_function in the helper_funcs module.
    """

    cosmo = FlatCosmology(1.0, 0.3)

    def test_against_nessie_r(self):
        """
        Testing against the implementation that is in the R nessie package.
        """
        redshifts = np.array([0.05, 0.1, 0.2, 0.3, 0.4])
        random_zs = np.loadtxt(
            "/Users/00115372/Desktop/GAMA_paper_plotter/gama_g09_randoms.txt",
            skiprows=1,
        )
        func = create_density_function(
            random_zs, len(random_zs) / 400, 0.001453924, self.cosmo
        )
        answers = np.array(
            [0.04775546, 0.024226080, 0.010309411, 0.003870724, 0.001073072]
        )
        result = func(redshifts)
        npt.assert_almost_equal(result, answers, decimal=2)


class TestCalculateSTotal(unittest.TestCase):
    """
    Testing the calculate_s_total function in helper functions.
    """

    def test_simple(self):
        """
        The R nessie package is already testing against the old algorithm. So we need only check
        that this score matches. And the rust code that it is based on is also tested. So we are
        really on testing that this runs.
        """

        measured_ids = np.array([0, 0, 0, 1, 1])
        mock_ids = np.array([0, 0, 0, -1, -1])
        result = calculate_s_total(measured_ids, mock_ids)
        self.assertEqual(result, 0.4)


if __name__ == "__main__":
    unittest.main()
