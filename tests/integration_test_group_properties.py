"""
Testing that the group properties are calculated correctly.
"""

import numpy as np
import pandas as pd

from nessie.catalog import RedshiftCatalog
from nessie.cosmology import FlatCosmology
from nessie.helper_funcs import create_density_function


cosmo = FlatCosmology(0.7, 0.3)

SHARK_FILE = (
    "~/Desktop/mock_catalogs/offical_waves_mocks/v0.3.0/wide/waves_wide_gals.parquet"
)

shark_data_frame = pd.read_parquet(SHARK_FILE)
shark_data_frame = shark_data_frame[(shark_data_frame["zobs"] < 0.2) & (shark_data_frame["total_ab_dust_Z_VISTA"] != -999)]

redshift = np.array(shark_data_frame["zobs"])
ra = np.array(shark_data_frame["ra"])
dec = np.array(shark_data_frame["dec"])
ab_mag = np.array(shark_data_frame["total_ab_dust_Z_VISTA"])
vel_errs = np.ones(len(shark_data_frame)) * 50


FRAC_AREA = 1133.86 / 41253
func = create_density_function(redshift, len(redshift), FRAC_AREA, cosmo)

red_cat = RedshiftCatalog(
    shark_data_frame["ra"],
    shark_data_frame["dec"],
    shark_data_frame["zobs"],
    func,
    cosmo,
)

red_cat.run_fof(0.05, 18)
group_ids = red_cat.group_ids

group_cat = red_cat.calculate_group_table(ab_mag, vel_errs)


values = {
    "ra": ra,
    "dec": dec,
    "redshift": redshift,
    "absolute_magnitudes": ab_mag,
    "vel_errs": vel_errs,
    "group_ids": group_ids,
}
buggy_values = pd.DataFrame.from_dict(values)
buggy_values.to_csv("test_group_properties.csv")
