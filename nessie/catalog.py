"""
The Core redshift survey class which handles most the linking assignments and groups.
"""

from typing import Callable
import numpy as np

from nessie_py import create_group_catalog
from .cosmology import FlatCosmology
from .core_funcs import _find_groups
from .helper_funcs import calculate_s_total


class RedshiftCatalog:
    """
    Stores the Redshift Catalog data that is needed for running the group finder.
    """

    def __init__(
        self,
        ra_array: np.ndarray[float],
        dec_array: np.ndarray[float],
        redshift_array: np.ndarray[float],
        density_function: Callable,
        cosmology: FlatCosmology,
        completeness: np.ndarray[float] = None,
    ) -> None:
        self.ra_array = ra_array
        self.dec_array = dec_array
        self.redshift_array = redshift_array
        self.density_function = density_function
        self.cosmology = cosmology
        if completeness is None:
            self.completeness = np.ones(len(self.ra_array))
        else:
            self.completeness = completeness

        self.current_r0 = None
        self.current_b0 = None
        self.group_ids = None
        self.mock_group_ids = None

    def get_raw_groups(self, b0: float, r0: float, max_stellar_mass=1e15) -> dict:
        """
        Generate FoF links between galaxies based on spatial and redshift linking lengths. There is
        very little reason to have to run this yourself. In most cases it is more appropriate to run
        """
        co_dists = self.cosmology.comoving_distance(self.redshift_array)
        linking_lengths = np.array(
            self.density_function(self.redshift_array) ** (-1.0 / 3)
            * (self.completeness) ** (-1.0 / 3)
        )
        gal_rad = b0 * linking_lengths
        max_on_sky_radius = self.cosmology.virial_radius(
            max_stellar_mass, self.redshift_array
        )
        too_wide = np.where(gal_rad > max_on_sky_radius)[0]

        gal_rad[too_wide] = max_on_sky_radius[too_wide]
        linking_lengths_pos = gal_rad / (self.cosmology.h * co_dists)

        r_variable = (
            r0
            * (1 + self.redshift_array)
            / (
                np.sqrt(
                    self.cosmology.omega_m * (1 + self.redshift_array) ** 3
                    + self.cosmology.omega_lambda
                )
            )
        )
        linking_lengths_los = (gal_rad * r_variable) / self.cosmology.h
        max_los_distances = (
            self.cosmology.velocity_dispersion(max_stellar_mass, self.redshift_array)
            * (1 + self.redshift_array)
            / self.cosmology.h0_grow(self.redshift_array)
        )
        too_far = linking_lengths_los > max_los_distances
        linking_lengths_los[too_far] = max_los_distances[too_far]
        groups = _find_groups(
            self.ra_array,
            self.dec_array,
            co_dists,
            linking_lengths_pos,
            linking_lengths_los,
        )
        return groups

    def run_fof(self, b0: float, r0: float, max_stellar_mass=1e15) -> None:
        """
        Run the full Friends-of-Friends (FoF) algorithm and assign group IDs to all galaxies.
        Singleton galaxies (unlinked) are given group ID -1.
        """
        group_links = self.get_raw_groups(b0, r0, max_stellar_mass)
        group_ids = np.ones(len(self.ra_array)) * -1

        group_ids[group_links["galaxy_id"] - 1] = group_links[
            "group_id"
        ]  # removing the 1-indexing
        self.group_ids = np.astype(group_ids, int)
        self.current_r0 = r0
        self.current_b0 = b0

    def calculate_group_table(
        self, absolute_magnitudes: np.ndarray[float], velocity_errors: np.ndarray[float]
    ) -> dict:
        """
        Generate a summary data.frame of group properties based on assigned group IDs.
        Must have run the group finder.
        """
        if self.group_ids is None:
            raise InterruptedError(
                "Algorithm hasn't been run! Make sure to run_fof first."
            )
        group_cat = create_group_catalog(
            self.ra_array,
            self.dec_array,
            self.redshift_array,
            absolute_magnitudes,
            velocity_errors,
            self.group_ids,
            self.cosmology.omega_m,
            self.cosmology.omega_k,
            self.cosmology.omega_lambda,
            self.cosmology.hubble_constant,
        )
        return group_cat

    def compare_to_mock(self, min_group_size=2):
        """
        Compares the current group_ids to a mock known grouping ids. Must have run the group finder
        and set both the mock_group_ids and singleton_id
        """
        if self.group_ids is None:
            raise InterruptedError(
                "No group ids found. Be sure to run the `run_fof` method"
            )

        if self.mock_group_ids is None:
            raise InterruptedError(
                "No mock group ids found. Be sure to set the mock groups ids."
            )

        return calculate_s_total(self.group_ids, self.mock_group_ids, min_group_size)
