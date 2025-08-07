"""
Module to handle all the optimization code.
"""

from scipy.optimize import fmin
from .catalog import RedshiftCatalog

def optimize_nm(
    redshift_cat: RedshiftCatalog,
    min_group_size: int,
    b0_guess: float = 0.05,
    r0_guess: float = 30.,
    max_stellar_mass: float = 1e15,
) -> tuple[float, float]:
    """
    Optimizes the b0 and r0 parameters using a Nelder-Mead simplex optimization approach to
    maximize the s_tot output for a given RedshiftCatalog object.

    Parameters
    ----------
    redshift_cat : RedshiftCatalog
        RedshiftCatalog object for which to optimize the grouping parameters.
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

    if redshift_cat.mock_group_ids is None:
        raise InterruptedError(
            "No mock group ids found. Be sure to set the mock groups ids."
        )

    def _objective(params):
        b0, r0 = params
        return -_calc_s_tot(redshift_cat, b0, r0)

    def _calc_s_tot(redshift_cat, b0, r0):
        redshift_cat.run_fof(b0=b0, r0=r0, max_stellar_mass=max_stellar_mass)
        s_tot = redshift_cat.compare_to_mock(min_group_size=min_group_size)
        return s_tot

    res = fmin(
        _objective,
        (b0_guess, r0_guess),
        xtol=0.1,
        ftol=0.1,
        maxiter=50,
        full_output=True,
        disp=False,
    )
    b_opt, r_opt = res[0]

    return b_opt, r_opt
