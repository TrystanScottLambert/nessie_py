"""
Testing that the estimated masses from Tempel+2014 are equivalent to GAMA masses scaled with
simulations.
"""

import pandas as pd
import numpy as np

from nessie import RedshiftCatalog
from nessie import FlatCosmology

cosmo = FlatCosmology(h=0.7, omega_matter=0.3)

gama_groups = pd.read_csv("tests/gama_group_catalog_v10.csv")
gama_gals = pd.read_csv("tests/gama_group_galaxies.csv")

ras = np.array(gama_gals["RA"])
decs = np.array(gama_gals["Dec"])
redshifts = np.array(gama_gals["Z"])
mags = np.array(gama_gals["Rpetro"])
absolute_mags = mags - cosmo.dist_mod(redshifts)
group_ids = np.array(gama_gals["GroupID"])

cat = RedshiftCatalog(ras, decs, redshifts, np.nan, cosmo)
cat.set_completeness()
cat.group_ids = group_ids

print("here")
group_catalog = cat.calculate_group_table(absolute_mags, np.repeat(50, len(absolute_mags)))
