![nessie](./.github/nessie_logo.png)
# Nessie

A Fast and Flexible Friends-of-Friends Group Finder Based on the GAMA Group Finder in Robotham+2011

The Nessie python package is a tool for constructing galaxy-group catalogs from redshift survey data using the Friends-of-Friends algorithm based on Robotham+2011 and described in Lambert+(in prep). It is meant to exactly replicate the R package Nessie. Both draw from the same rust code.

This package aims to be as user-friendly as possible and requires minimal information. The core functionality can be run on any dataset with R.A., Dec., and redshift information if the appropriate linking lengths are known.

## Installation

### Installing Python

Since this is the Python version of this tool, we assume that the user already has Python installed. You can install the newest version [here](https://www.python.org/downloads/) and a good guide from [real python](https://realpython.com/installing-python/).

### Installing Rust

Since the core functionality of Nessie is written in Rust, the Rust package manager—Cargo—is required. Fortunately, **this is very easy to install.** The [rustlang site](https://www.rust-lang.org/tools/install) should detect your operating system and tell you what command to use to install Rust using `rustup`. This should automatically install `Cargo` as well.

For Unix systems (macOS + Linux), this is as easy as running:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

```

in the terminal.

### Installing Nessie

Nessie can be easily installed using pip. 

```sh
pip install nessie_py

```

and can then be imported as
```python
import nessie
```

Note that we pip install "nessie_py" but import "nessie". That's because the nessie name was taken. 

### Helpful Additional Packages

All dependencies should have been installed automatically. However, if you want to run the unit tests you will need `astropy` to test against.

## Finding Galaxy Groups in Redshift Catalogs

### Setting Up the RedshiftCatalog Object

Nessie works by first setting up a RedshiftCatalog object and then running the group finder on that catalog. This catalog requires R.A., Dec., and redshift coordinates as well as a cosmology and a running density function.

#### Setting a Cosmology

At the moment, Nessie only works for flat cosmologies and one can be created with:

```python
from nessie import FlatCosmology

cosmo = FlatCosmology(h = 0.7, omega_matter = 0.3)

```
**Note**: *Some users might think we should use an astropy.Cosmology object. This would be more generalized however it adds a dependence and there are other cosmology functions that the group finder users which aren't available in astropy. This custom object is also an exact clone of the R package, so things remain consistent between both.*
#### Building a Density Function

The running density function can be created using the n(z) of the data. We include a helper function `create_density_function` which will take a distribution of redshifts and build the appropriate function.

```python
from nessie.helper_funcs import create_density_function

running_density = create_density_function(redshifts, total_counts = len(redshifts), survey_fractional_area = 0.0001, cosmology = cosmo)

```

In the most basic of cases, the redshifts of the actual survey can be used. However, we recommend building an appropriate n(z) that accounts for large-scale structure fluctuations. This can be done by either using a randoms catalog (Cole+2011) or even fitting a skewed normal function and sampling appropriately.

Note that the fractional area is required and not the total area in steradians. i.e., area_in_steradians/4π.

### Running the Group Finder

The final redshift catalog can be built like this:

```python
import numpy as np
from nessie import RedshiftCatalog

ra, dec, zobs = np.loadtxt("some_file.txt", unpack=True)
red_cat = RedshiftCatalog(ra, dec, zobs, running_density, cosmo)

```


#### Completeness

It is possible to account for completeness by passing an array of values between 0 and 1 and using the setter method available in the `RedshiftCatalog` object. A value of 1 indicates that a galaxy lies in a fully complete region, 0 indicates a completely incomplete region, and 0.5 would mean the region is 50% complete. The definition of this completeness array is left to the user
```python
completeness = np.repeat(0.98, len(ra))
red_cat.set_completeness(completeness)

```

If no arguments are passed to the `set_completeness` method then 100% completeness is assumed.
```python
red_cat.set_completeness() # 100% completeness for every galaxy.
```

Using the setter method in this way allows for validation checks.

**Actually calculating completeness:**
Besides manually setting the completeness a `calculate_completeness` method exits to calculate 
which can estimate completeness based on a "target catalog" (i.e., a catalog of RA and Dec representing the galaxies that were planned to be observed). Most surveys should have this available.
```python
target_ra, target_dec = np.loadtxt("/some/target/catalog")
on_sky_radii = np.repeat(0.01, len(ra)) # in degrees

red_cat.calculate_completeness(target_ra, target_dec, on_sky_radii) 

```
In the example above, we set the radius for evaluating completeness to 0.01 degrees for every evaluation point. That is, for each galaxy in the redshift survey, completeness is computed within a 0.01-degree radius. It is also possible to use non-uniform radii on a per-galaxy basis.



**The completeness must be set before the group finder can be run!**

### Running
Once built, and completeness set, the group finder can be run as:

```python
red_cat.run_fof(b0 = 0.05, r0 = 18)

```

This stores the group IDs in the RedshiftCatalog object. So a full example of reading in your data and updating it with the group catalog information may look something like this:

```python
from  nessie  import  FlatCosmology, RedshiftCatalog
from  nessie.helper_funcs  import  create_density_function

# Preparing redshift data
ra, dec, redshifts = np.loadtxt('some_redshift_survey.csv')
cosmo = FlatCosmology(h = 0.7, omega_matter = 0.3)
running_density = create_density_function(redshifts, total_counts = len(redshifts), survey_fractional_area = 0.0001, cosmology = cosmo)

# Running group catalog
red_cat = RedshiftCatalog(ra, dec, redshifts, running_density, cosmo)
red_cat.set_completeness()
red_cat.run_fof(b0 = 0.05, r0 = 18)
group_ids = red_cat.group_ids

```

This would result in numpy arrays with R.A., Dec., redshift, and group_id **where -1 is chosen to mean that that galaxy was not found in any group.**

### Group Catalog

The group catalog is stored as python dictionary that can be written to file in any way that the user wishes later. It does require the absolute magnitudes to be known beforehand and these can be calculated in any way that the user sees fit.

```python
group_catalog_dict = red_cat.calculate_group_table(abs_mags)

```

### Pair Catalog
Besides a Group catalog, a Pair catalog consisting of properties of galaxy-pairs can also be calculated.
```python
pair_catalog_dict = red_cat.calculate_pair_table()
```

## Tuning Against a Mock Catalog
The above example is easy to do if you already know what the linking lengths are, but often the choice of `b0` and `r0` is not clear. A standard practice to overcome this issue is to rely on mock catalogs of known groupings to "tune" the best values. I.e., find the values of `b0` and `r0` that best recover what is known in the mock catalogs.

Obtaining such mock catalogs is beyond the scope of this package. We assume that the user has obtained some in one manner or another or built their own.

Currently, tuning can only be done in the Nessie R package. But we plan on including it here soon. But we do include functionality for comparing to mock catalogs which can be optimized by the user in any way they wish.

### Comparing to Mocks
If known "true" groups are known then they can be set in the RedshiftCatalog object. Then the cost function described in section 3.1 of Robotham+2011 can be calculated trivially. 

Importantly, each RedshiftCatalog object needs a value for the `mock_group_ids` field **where -1 means a galaxy is not in any group.** This has to be manually set.

```python
import numpy as np
mock_group_ids = np.loadtxt("some_mock.txt", usecols=(1), unpack=True)
red_cat.mock_groups_ids = mock_group_ids
score = red_cat.compare_to_mock(min_group_size=5)
```
The minimum number of members in a group can be set for the score too. 

This can then be optimized as the user sees fit.


## Contributing

We welcome contributions to Nessie! Please read our Contributing Guidelines before submitting issues or pull requests.

