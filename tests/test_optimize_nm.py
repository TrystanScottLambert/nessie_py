"""
Integration test for the optimizer function.
"""

import unittest
import numpy as np

from nessie.catalog import RedshiftCatalog
from nessie.cosmology import FlatCosmology
from nessie.optimizer import optimize_nm
from nessie.helper_funcs import create_density_function


class TestOptimizer(unittest.TestCase):
    """
    Test the optimize_nm function.
    """

    def test_optimizer_consistency(self):
        """
        Check that the optimized parameters from single and multi catalog runs are consistent within 1%.
        """

        INFILE_SDSS = "/Users/00115372/Desktop/nessie_plots/asu.tsv"
        group_id, n_gal, z, ra, dec, mag = np.loadtxt(INFILE_SDSS, unpack=True, skiprows=1)
        group_id[group_id == 0] = -1

        mask = (
          (ra >= (max(ra) + min(ra)) / 2 - 10)
          & (ra <= (max(ra) + min(ra)) / 2 + 10)
          & (dec >= (max(dec) + min(dec)) / 2 - 10)
          & (dec <= (max(dec) + min(dec)) / 2 + 10)
        )
        group_id, n_gal, z, ra, dec, mag = group_id[mask], n_gal[mask], z[mask], ra[mask], dec[mask], mag[mask]

        CAT_AREA = 400 * (np.pi / 180) ** 2 / (4 * np.pi)
        cosmo = FlatCosmology(0.7, 0.3)
        func = create_density_function(z, len(z), CAT_AREA, cosmo)

        red_cat = RedshiftCatalog(ra, dec, z, func, cosmo)
        red_cat.set_completeness()
        red_cat.mock_group_ids = group_id

        b0, r0 = optimize_nm(red_cat, 5)
        b0_multi, r0_multi = optimize_nm([red_cat, red_cat], 5)

        self.assertGreaterEqual(min((b0, b0_multi)) / max((b0, b0_multi)), 0.99)
        self.assertGreaterEqual(min((r0, r0_multi)) / max((r0, r0_multi)), 0.99)

if __name__ == "__main__":
    unittest.main()
