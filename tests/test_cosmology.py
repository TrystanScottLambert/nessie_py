"""
Testing the comsology functions in test_core_funcs against astropy and celestial.
"""

import unittest
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import numpy.testing as npt

from cosmology import FlatCosmology


class TestCoMovingDistance(unittest.TestCase):
    """
    Testing the comoving_distance method.
    """

    cosmo = FlatCosmology(0.7, 0.3)
    astropy_cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    def test_simple(self):
        """Simple test"""
        redshifts = np.arange(0, 3, 0.1)

        results = self.cosmo.comoving_distance(redshifts)
        answer = self.astropy_cosmo.comoving_distance(redshifts).value
        npt.assert_almost_equal(np.around(results), np.around(answer))


class TestInverseComovingDistance(unittest.TestCase):
    """
    Testing the  z_at_comoving_distances function.
    """

    cosmo = FlatCosmology(0.7, 0.3)

    def test_recovery(self):
        """Check that we get back what we put in."""
        redshifts = np.arange(0, 10, 0.1)
        comoving_distances = self.cosmo.comoving_distance(redshifts)
        returned_redshfits = self.cosmo.z_at_comoving_distances(comoving_distances)
        npt.assert_almost_equal(redshifts, returned_redshfits, decimal=3)


class TestDistanceModulus(unittest.TestCase):
    """
    Testing the dist_mod method in the Cosmology class
    """

    cosmo = FlatCosmology(0.7, 0.3)
    astropy_cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    def test_astropy(self):
        """
        Simple comparison against astropy.
        """
        redshifts = np.arange(0, 5, 0.1)
        result = self.cosmo.dist_mod(redshifts)
        answer = self.astropy_cosmo.distmod(redshifts).value
        npt.assert_array_almost_equal(result, answer, decimal=3)


class TestVirialRadius(unittest.TestCase):
    """
    Testing the virial_radius method in the Cosmology class.
    """

    cosmo = FlatCosmology(0.7, 0.3)

    def test_against_celestial(self):
        """
        This function is just a pure wrapper for the rust function which is tested there.
        So we need only check that it runs correctly. Testing against the celestial call in R.
        """
        mass = 1e12  # default in celestial
        redshifts = np.array([0.0, 0.1, 1.1])
        answers = np.array([0.2062981, 0.2198784, 0.2859294])
        results = self.cosmo.virial_radius(mass, redshifts)
        npt.assert_almost_equal(results, answers, decimal=3)


class TestVirialSigma(unittest.TestCase):
    """
    Testing the velocity_dispersion method in the Cosmology class.
    """

    cosmo = FlatCosmology(0.7, 0.3)

    def test_against_celestial(self):
        """
        This function is just a pure wrapper for the rust function which is tested there.
        So we need only check that it runs correctly. Testing against the celestial call in R.
        """
        mass = 1e12  # default in celestial
        redshifts = np.array([0.0, 0.1, 1.1])
        answers = np.array([204.2154, 207.4633, 251.3717])
        results = self.cosmo.velocity_dispersion(mass, redshifts)
        npt.assert_almost_equal(results, answers, decimal=1)


class TestHgrow(unittest.TestCase):
    """
    Testing the h0_grow function at
    """

    cosmo = FlatCosmology(0.7, 0.3)
    astropy_cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    def test_against_astropy(self):
        """
        Tesing the H(z) function against astropy.
        """
        redshifts = np.arange(0, 10, 0.1)
        answers = self.astropy_cosmo.H(redshifts).value
        results = self.cosmo.h0_grow(redshifts)
        npt.assert_almost_equal(results, answers, decimal=2)


if __name__ == "__main__":
    unittest.main()
