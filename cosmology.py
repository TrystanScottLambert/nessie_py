"""
Cosmology module for handling the Flat LCDM cosmology calculations.
"""

import numpy as np
from nessie_py import (
    comoving_distances_at_z,
    distance_modulus,
    z_at_comoving_distances,
    calculate_max_rvirs,
    calculate_max_sigmas,
    h_at_z,
)


class FlatCosmology:
    """
    Core cosmology class
    """

    def __init__(self, h: float, omega_matter: float) -> None:
        """
        Read in h and Om0 and build cosmology from that.
        """
        self.h = h
        self.omega_m = omega_matter
        self.hubble_constant = 100 * self.h
        self.omega_lambda = 1 - self.omega_m
        self.omega_k = 0.0
        self.omgega_radiation = 0.0

    def comoving_distance(self, redshift: np.ndarray[float]) -> np.ndarray[float]:
        """
        Comoving distance for a vector of redshifts.
        """
        return np.array(
            comoving_distances_at_z(
                redshift,
                self.omega_m,
                self.omega_k,
                self.omega_lambda,
                self.hubble_constant,
            )
        )

    def dist_mod(self, redshift: np.ndarray[float]) -> np.ndarray[float]:
        """
        Distance modulus for an array of redshift values.
        """
        return np.array(
            distance_modulus(
                redshift,
                self.omega_m,
                self.omega_k,
                self.omega_lambda,
                self.hubble_constant,
            )
        )

    def z_at_comoving_distances(self, co_dist: np.ndarray[float]) -> np.ndarray[float]:
        """
        The inverse comoving distance function. Determines the redshift for the array of comoving
        distances.
        """
        return np.array(
            z_at_comoving_distances(
                co_dist,
                self.omega_m,
                self.omega_k,
                self.omega_lambda,
                self.hubble_constant,
            )
        )

    def virial_radius(self, solar_mass: float, redshifts: np.ndarray[float]):
        """
        Computes the virial radius of the halos with the given mass and redshift.
        """
        return np.array(
            calculate_max_rvirs(
                solar_mass,
                redshifts,
                self.omega_m,
                self.omega_k,
                self.omega_lambda,
                self.hubble_constant,
            )
        )

    def velocity_dispersion(self, solar_mass: float, redshifts: np.ndarray[float]):
        """
        Compute the velocity dispersion of halos with given mass and redshifts
        """
        return np.array(
            calculate_max_sigmas(
                solar_mass,
                redshifts,
                self.omega_m,
                self.omega_k,
                self.omega_lambda,
                self.hubble_constant,
            )
        )

    def h0_grow(self, redshifts: np.ndarray[float]) -> np.ndarray[float]:
        """
        Compute the redshift-dependent Hubble parameter H(z) in a flat cosmology
        """
        return np.array(
            h_at_z(
                redshifts,
                self.omega_m,
                self.omega_k,
                self.omega_lambda,
                self.hubble_constant,
            )
        )
