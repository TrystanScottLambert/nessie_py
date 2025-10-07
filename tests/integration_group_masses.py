"""
Testing that the estimated masses from Tempel+2014 are equivalent to GAMA masses scaled with
simulations.
"""

import pandas as pd
import numpy as np
import pylab as plt

from nessie import RedshiftCatalog
from nessie import FlatCosmology

cosmo = FlatCosmology(h=1.0, omega_matter=0.25)

gama_groups = pd.read_csv("tests/GAMA_GROUPS_V10.csv")
gama_gals = pd.read_csv("tests/gama_group_galaxies.csv")

ras = np.array(gama_gals["RA"])
decs = np.array(gama_gals["Dec"])
redshifts = np.array(gama_gals["Z"])
mags = np.array(gama_gals["Rpetro"])
absolute_mags = mags - cosmo.dist_mod(redshifts)
group_ids = np.array(gama_gals["GroupID"])
group_ids[group_ids == 0] = -1

cat = RedshiftCatalog(ras, decs, redshifts, np.nan, cosmo)
cat.set_completeness()
cat.group_ids = group_ids


group_catalog = pd.DataFrame(
    cat.calculate_group_table(absolute_mags, np.repeat(50, len(absolute_mags)))
)


# We get the same thing as the v10 catalog using the same cosmology.
group_catalog = group_catalog[group_catalog["multiplicity"] > 2]
gama_groups = gama_groups[gama_groups["Nfof"] > 2]
x = np.linspace(
    np.min(group_catalog["mass_proxy"]), np.max(group_catalog["mass_proxy"]), 1000
)
plt.scatter(group_catalog["mass_proxy"] * 10, gama_groups["MassA"])
plt.plot(x, x, color="k", ls=":", lw=3)
plt.show()

plt.scatter(
    group_catalog["estimated_mass"],
    group_catalog["mass_proxy"] * 10,
    color="r",
    s=1,
    label="A=10",
)
plt.scatter(
    group_catalog["estimated_mass"],
    group_catalog["mass_proxy"] * 5,
    color="g",
    s=1,
    label="A=5",
)
x = np.linspace(
    np.min(group_catalog["estimated_mass"]),
    np.max(group_catalog["estimated_mass"]),
    1000,
)
plt.plot(x, x, color="k")
plt.xlabel("Tempel+2014 estimated masses", fontsize=15)
plt.ylabel("MassA GAMA", fontsize=15)
plt.legend()

plt.show()
