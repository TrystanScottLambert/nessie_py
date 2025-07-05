"""
Simple example script. Testing benchmarks, scores, and other integrations.
"""


import numpy as np
import pandas as pd

from nessie.catalog import RedshiftCatalog
from nessie.cosmology import FlatCosmology
from nessie.helper_funcs import create_density_function, calculate_s_total

cosmo = FlatCosmology(0.7, 0.3)


random_zs = np.loadtxt(
    "/Users/00115372/Desktop/GAMA_paper_plotter/gama_g09_randoms.txt", skiprows=1
)
func = create_density_function(random_zs, len(random_zs) / 400, 0.001453924, cosmo)

data = pd.read_parquet("~/Desktop/GAMA_paper_plotter/mocks/galform_gals_for_R.parquet")
data = data[data["Volume"] == 1]
print("data read in")

ra = np.array(data["RA"])
dec = np.array(data["DEC"])
redshift = np.array(data["Zspec"])

red_cat = RedshiftCatalog(ra, dec, redshift, func, cosmo)
red_cat.run_fof(b0=0.05, r0=18)

red_cat.mock_group_ids = np.array(data["GroupID"])
red_cat.completeness = np.ones(len(ra)) * 0.95

score = red_cat.compare_to_mock(min_group_size=5)
print(score)

another_score = calculate_s_total(red_cat.group_ids, red_cat.mock_group_ids, 5)
print(another_score)
