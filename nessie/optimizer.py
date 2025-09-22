"""
Module to handle all the optimization code.
"""

from scipy.optimize import fmin
from typing import Union, Sequence
from nessie_py import calculate_harmonic_mean
from .catalog import RedshiftCatalog


def optimize_nm(
    redshift_cats: Union[RedshiftCatalog, Sequence[RedshiftCatalog]],
    min_group_size: int,
    b0_guess: float = 0.05,
    r0_guess: float = 30.0,
    max_stellar_mass: float = 1e15,
) -> tuple[float, float]:
    """
    Optimizes the b0 and r0 parameters using Nelder-Mead optimization across multiple
    RedshiftCatalog objects simultaneously.

    Parameters
    ----------
    redshift_cats : RedshiftCatalog or list of RedshiftCatalog
        Single RedshiftCatalog or list of RedshiftCatalog objects for which to optimize the grouping parameters.
    min_group_size : int
        Minimum size of the assigned and mock groups compared when calculating s_total.
    b0_guess : float
        Initial guess for the b0 parameter.
    r0_guess : float
        Initial guess for the r0 parameter.
    max_stellar_mass : float
        Maximum galactic stellar mass present in the data.

    Returns
    -------
    b_opt : float
        Optimized b0 parameter.
    r_opt : float
        Optimized r0 parameter.
    """

    if isinstance(redshift_cats, RedshiftCatalog):
        redshift_cats = [redshift_cats]
    
    for cat in redshift_cats:
        if cat.mock_group_ids is None:
            raise InterruptedError(
                "No mock group ids found in one of the catalogs. Be sure to set the mock groups ids."
            )

    def _calc_s_tot(cat, b0, r0):
        cat.run_fof(b0=b0, r0=r0, max_stellar_mass=max_stellar_mass)
        s_tot = cat.compare_to_mock(min_group_size=min_group_size)
        return s_tot

    def _objective(params):
        b0, r0 = params
        scores = [_calc_s_tot(cat, b0, r0) for cat in redshift_cats]
        fom = calculate_harmonic_mean(scores) if len(redshift_cats) > 1 else scores[0]
        return -fom

    res = fmin(_objective, (b0_guess, r0_guess), xtol=0.1, ftol=0.1, maxiter=50, full_output=True, disp=False)
    b_opt, r_opt = res[0]

    return b_opt, r_opt
