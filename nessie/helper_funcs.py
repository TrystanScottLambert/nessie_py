"""
Module for quality of life helper functions which are not core to the algorithm.
"""

import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate
from scipy.integrate import quad
from scipy.interpolate import interp1d

from nessie_py import calculate_s_score
from .cosmology import FlatCosmology


def create_density_function(
    redshifts: np.ndarray,
    total_counts: int,
    survey_fractional_area: float,
    cosmology: FlatCosmology,
    binwidth: int = 40,
    interpolation_n: int = 10_000,
):
    """
    Running density function estimation (rho(z))

    Parameters
    ----------
    redshifts : array_like
        Redshift values used to estimate the redshift distribution.
    total_counts : float
        Total number of objects in the redshift survey.
    survey_fractional_area : float
        Fraction of the sky covered by the survey (relative to 4π steradians).
    cosmology : CustomCosmology
        A cosmology object with methods:
            - comoving_distance(z)
            - z_at_comoving_dist(d)
    binwidth : float
        Bin width in comoving Mpc (default = 40).
    interpolation_n : int
        Number of bins used for interpolation (default = 10,000).

    Returns
    -------
    rho_z_func : callable
        A function rho(z) that gives the running density at a given redshift.
    """
    comoving_distances = cosmology.comoving_distance(redshifts)

    kde = KDEUnivariate(comoving_distances)
    kde.fit(bw=binwidth, fft=True, gridsize=interpolation_n, cut=0)
    kde_x = kde.support
    kde_y = kde.density

    kde_func = interp1d(kde_x, kde_y, bounds_error=False, fill_value="extrapolate")

    # Running integral over each bin
    running_integral = np.array(
        [quad(kde_func, max(x - binwidth / 2, 0), x + binwidth / 2)[0] for x in kde_x]
    )

    # Running comoving volume per bin
    upper_volumes = (4 / 3) * np.pi * (kde_x + binwidth / 2) ** 3
    lower_volumes = (4 / 3) * np.pi * (kde_x - binwidth / 2) ** 3
    running_volume = survey_fractional_area * (upper_volumes - lower_volumes)

    # Convert comoving distance to redshift
    z_vals = cosmology.z_at_comoving_distances(kde_x)
    rho_vals = (total_counts * running_integral) / running_volume

    # Interpolate rho(z)
    rho_z_func = interp1d(
        z_vals, rho_vals, bounds_error=False, fill_value="extrapolate"
    )

    return rho_z_func


def calculate_s_total(
    measured_ids: np.ndarray[int],
    mock_group_ids: np.ndarray[int],
    min_group_size: int = 2,
) -> float:
    """
    Comparing measured groups to mock catalogue groups
    """
    return calculate_s_score(
        np.astype(measured_ids, int),
        np.astype(mock_group_ids, int),
        int(min_group_size),
    )
