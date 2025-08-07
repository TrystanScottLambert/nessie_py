"""
Integration test testing the optimizer function
"""

from datetime import datetime

import numpy as np

from nessie.catalog import RedshiftCatalog
from nessie.cosmology import FlatCosmology
from nessie.optimizer import optimize_nm
from nessie.helper_funcs import create_density_function


INFILE_SDSS = '/Users/00115372/Desktop/nessie_plots/asu.tsv'
SDSS_AREA = 0.212673
group_id, n_gal, z, ra, dec, mag = np.loadtxt(INFILE_SDSS, unpack=True, skiprows=1)
group_id[group_id == 0] = -1

cosmo = FlatCosmology(0.7, 0.3)

func = create_density_function(z, len(z), SDSS_AREA, cosmo)

red_cat = RedshiftCatalog(ra, dec, z, func, cosmo)
red_cat.set_completeness()
red_cat.mock_group_ids = group_id


start = datetime.now()
b0, r0 = optimize_nm(red_cat, 5)
end = datetime.now()
print("Time taken to optimize is: ", end - start)
