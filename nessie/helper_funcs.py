"""
Module for quality of life helper functions which are not core to the algorithm.
"""

from enum import Enum
import warnings

import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate
from scipy.integrate import quad
from scipy.interpolate import interp1d

from nessie_py import calculate_s_score, calc_completeness_rust
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


class ValidationType(Enum):
    """
    All the types of things we can validate
    """

    RA = "ra"
    DEC = "dec"
    REDSHIFT = "redshift"
    ABS_MAG = "absolute_mag"
    COMPLETENESS = "completeness"
    B0 = "b0"
    R0 = "r0"
    VEL_ERR = "vel_err"
    ANGLE = "angle"


def validate(value: np.ndarray[float] | float, valid_type: ValidationType) -> None:
    """
    Validates the given input which can be either an array or a float
    """

    # Checking arrays don't have non-numeric values
    if valid_type in {
        ValidationType.RA,
        ValidationType.DEC,
        ValidationType.ABS_MAG,
        ValidationType.REDSHIFT,
        ValidationType.COMPLETENESS,
        ValidationType.VEL_ERR,
        ValidationType.ANGLE,
    }:
        value = np.asarray(value)
        if not np.issubdtype(value.dtype, np.number):
            raise TypeError(f"{valid_type.value} must be numeric.")
        if np.isnan(value).any():
            raise ValueError(f"{valid_type.value} contains NaNs.")
        if np.isinf(value).any():
            raise ValueError(f"{valid_type.value} contains infinite values.")

    # Match-case for type-specific logic
    match valid_type:
        case ValidationType.RA:
            if np.any((value < 0) | (value > 360)):
                raise ValueError("RA values must be between 0 and 360.")

        case ValidationType.DEC:
            if np.any((value < -90) | (value > 90)):
                raise ValueError("Dec values must be between -90 and 90.")

        case ValidationType.REDSHIFT:
            if np.any(value < 0):
                raise ValueError("Redshifts cannot be negative.")
            if np.any(value > 1100):
                warnings.warn("Warning: redshifts are very large!")

        case ValidationType.ABS_MAG:
            if np.any((value > -4) | (value < -50)):
                warnings.warn("Warning: absolute magnitudes look unusual.")

        case ValidationType.COMPLETENESS:
            if np.any((value < 0) | (value > 1)):
                raise ValueError("Completeness must be between 0 and 1.")

        case ValidationType.VEL_ERR:
            if np.any(value < 0):
                raise ValueError("Velocity errors must be greater than 0.")
            if np.any(value > 2000):
                warnings.warn(
                    "Warning: velocity errors seem very large. Are units correct?"
                )

        case ValidationType.B0 | ValidationType.R0:
            if not isinstance(value, (float, int)):
                raise TypeError(f"{valid_type.value} must be a scalar number.")
            if value < 0:
                raise ValueError(f"{valid_type.value} cannot be negative.")

        case ValidationType.ANGLE:
            if np.any((value < 0) | (value > 360)):
                raise ValueError("Angle must be in degrees between 0 and 360.")

        case _:
            raise ValueError(
                f"Unknown property type: {valid_type.value}. Likely Enum needs updating."
            )


def calculate_completeness(
    ra_observed: np.ndarray[float],
    dec_observed: np.ndarray[float],
    ra_target: np.ndarray[float],
    dec_target: np.ndarray[float],
    ra_eval: np.ndarray[float],
    dec_eval: np.ndarray[float],
    search_radii: np.ndarray[float],
):
    """
    Calculate the completeness of a galaxy survey at specified sky positions.

    The completeness is defined as the ratio of observed to targeted galaxies within a given
    angular radius at each evaluation point. This function is useful for estimating spatial
    completeness and can be used, for example, to adjust linking lengths in structure-finding
    algorithms like Nessie.

    Parameters
    ----------
    ra_observed : array_like
        Right Ascension (in degrees) of galaxies that were actually observed.

    dec_observed : array_like
        Declination (in degrees) of galaxies that were actually observed.

    ra_target : array_like
        Right Ascension (in degrees) of galaxies that were targeted for observation.

    dec_target : array_like
        Declination (in degrees) of galaxies that were targeted for observation.

    ra_evaluate : array_like
        Right Ascension (in degrees) of the positions at which completeness will be evaluated.

    dec_evaluate : array_like
        Declination (in degrees) of the positions at which completeness will be evaluated.

    search_radii : array_like
        Angular search radius (in degrees) for each evaluation point. Must match the length of `ra_evaluate`.

    Returns
    -------
    completeness : ndarray
        Array of completeness values (floats between 0 and 1), one for each evaluation position.
    """
    all_ra_arrays = [ra_observed, ra_target, ra_eval]
    all_dec_arrays = [dec_observed, dec_target, dec_eval]

    for ra_array in all_ra_arrays:
        validate(ra_array, ValidationType.RA)

    for dec_array in all_dec_arrays:
        validate(dec_array, ValidationType.DEC)

    validate(search_radii, ValidationType.ANGLE)

    for ra_array, dec_array in zip(all_ra_arrays, all_dec_arrays):
        if len(ra_array) != len(dec_array):
            raise ValueError("ra arrays and dec arrays must be the same length.")

    if len(search_radii) != len(ra_eval):
        raise ValueError(
            "search_radii must be the same length as the evaluation arrays."
        )

    return calc_completeness_rust(
        ra_observed,
        dec_observed,
        ra_target,
        dec_target,
        ra_eval,
        dec_eval,
        search_radii,
    )
