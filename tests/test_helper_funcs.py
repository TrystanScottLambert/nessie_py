"""
Unit tests for the helper funcs module.
"""

import unittest
import numpy as np
import numpy.testing as npt

from nessie.helper_funcs import (
    create_density_function,
    calculate_s_total,
    remap_ids,
    gen_random_redshifts,
)
from nessie.cosmology import FlatCosmology


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


class TestRemapId(unittest.TestCase):
    """Testing the remap_id function"""

    def test_string(self):
        """Testing that strings are remaped correctly."""
        bad = ["2mass1", "2mass2", -1, "2mass1", "2mass3"]
        good = [1, 2, -1, 1, 3]
        ans = remap_ids(bad)
        for r, a in zip(ans, good):
            self.assertEqual(r, a)

    def test_float(self):
        """Testing that floats work."""
        bad = [1.1, 2.2, -1, 1.1, 3.3]
        good = [1, 2, -1, 1, 3]
        ans = remap_ids(bad)
        for r, a in zip(ans, good):
            self.assertEqual(r, a)

    def test_long_int(self):
        """Testing that long ints work."""
        bad = [10**18, 10**18 + 1, -1, 10**18, 10**18 + 2]
        good = [1, 2, -1, 1, 3]
        ans = remap_ids(bad)
        for r, a in zip(ans, good):
            self.assertEqual(r, a)

    def test_shark(self):
        """Testing that works with shark ids."""
        shark_ids = [21826700000225, 21826700000235, -1, 21826700000225, 21826700000525]
        good = [1, 2, -1, 1, 3]
        ans = remap_ids(shark_ids)
        for r, a in zip(ans, good):
            self.assertEqual(r, a)


class TestGenRandoms(unittest.TestCase):
    """
    Testing that the gen_random_redshifts function.
    """

    def test(self):
        """testing basic run since this is a simple wrapper."""
        cosmo = FlatCosmology(1.0, 0.3)
        redshifts = np.random.normal(0.6, 0.3, 500)
        redshifts = redshifts[redshifts > 0]
        mags = np.random.random(len(redshifts)) * 4 + 15
        z_lim = np.max(redshifts) + 0.1
        maglim = 19.0
        zs = gen_random_redshifts(redshifts, mags, z_lim, maglim, cosmo, iterations=2, n_clone=100)
        self.assertEqual(round(len(zs) / (100*500)), 1)


if __name__ == "__main__":
    unittest.main()
