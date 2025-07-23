use fof::bijectivity::s_score;
use fof::group_properties::GroupedGalaxyCatalog;
use fof::link_finder::find_links;
use fof::stats::harmonic_mean;
use fof::Cosmology;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

/// Calculate the hubble constant at different redshifts.
#[pyfunction]
fn h_at_z(
    redshift_array: Vec<f64>,
    omega_m: f64,
    omega_k: f64,
    omega_l: f64,
    h0: f64,
) -> PyResult<Vec<f64>> {
    let cosmo = Cosmology {
        omega_m,
        omega_k,
        omega_l,
        h0,
    };
    let result = redshift_array
        .par_iter()
        .map(|z| cosmo.h_at_z(*z))
        .collect();
    Ok(result)
}

/// Calculates multiple comoving distances for multiple redshifts
#[pyfunction]
fn comoving_distances_at_z(
    redshift_array: Vec<f64>,
    omega_m: f64,
    omega_k: f64,
    omega_l: f64,
    h0: f64,
) -> PyResult<Vec<f64>> {
    let cosmo = Cosmology {
        omega_m,
        omega_k,
        omega_l,
        h0,
    };
    let result = redshift_array
        .par_iter()
        .map(|z| cosmo.comoving_distance(*z))
        .collect();
    Ok(result)
}

/// Redshift at some given comoving distances in Mpc.
#[pyfunction]
fn z_at_comoving_distances(
    distances: Vec<f64>,
    omega_m: f64,
    omega_k: f64,
    omega_l: f64,
    h0: f64,
) -> PyResult<Vec<f64>> {
    let cosmo = Cosmology {
        omega_m,
        omega_k,
        omega_l,
        h0,
    };
    let result = distances
        .par_iter()
        .map(|d| cosmo.inverse_codist(*d))
        .collect();
    Ok(result)
}

/// Calculate the Rvir from a given mass for a range of redshift values.
#[pyfunction]
fn calculate_max_rvirs(
    max_solar_mass: f64,
    redshift_array: Vec<f64>,
    omega_m: f64,
    omega_k: f64,
    omega_l: f64,
    h0: f64,
) -> PyResult<Vec<f64>> {
    let cosmo = Cosmology {
        omega_m,
        omega_k,
        omega_l,
        h0,
    };
    let result = redshift_array
        .par_iter()
        .map(|z| cosmo.mvir_to_rvir(max_solar_mass, *z))
        .collect();

    Ok(result)
}

/// Calculate the Sigma from a given mass for a range of redshift values.
#[pyfunction]
fn calculate_max_sigmas(
    max_solar_mass: f64,
    redshift_array: Vec<f64>,
    omega_m: f64,
    omega_k: f64,
    omega_l: f64,
    h0: f64,
) -> PyResult<Vec<f64>> {
    let cosmo = Cosmology {
        omega_m,
        omega_k,
        omega_l,
        h0,
    };
    let result = redshift_array
        .par_iter()
        .map(|z| cosmo.mvir_to_sigma(max_solar_mass, *z))
        .collect();
    Ok(result)
}

/// Distance modulus.
#[pyfunction]
fn distance_modulus(
    redshift_array: Vec<f64>,
    omega_m: f64,
    omega_k: f64,
    omega_l: f64,
    h0: f64,
) -> PyResult<Vec<f64>> {
    let cosmo = Cosmology {
        omega_m,
        omega_k,
        omega_l,
        h0,
    };
    let result = redshift_array
        .par_iter()
        .map(|&z| cosmo.distance_modulus(z))
        .collect();

    Ok(result)
}

/// Link finding
#[pyfunction]
fn fof_links_fast<'py>(
    py: Python<'py>,
    ra_array: Vec<f64>,
    dec_array: Vec<f64>,
    comoving_distances: Vec<f64>,
    linking_lengths_pos: Vec<f64>,
    linking_lengths_los: Vec<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let links = find_links(
        ra_array,
        dec_array,
        comoving_distances,
        linking_lengths_pos,
        linking_lengths_los,
    );
    let i_vec: Vec<usize> = links.iter().map(|(x, _)| *x + 1).collect(); // + 1 for R idx
    let j_vec: Vec<usize> = links.iter().map(|(_, y)| *y + 1).collect();

    let dict = PyDict::new(py);
    dict.set_item("i", i_vec)?;
    dict.set_item("j", j_vec)?;
    Ok(dict)
}

/// Creating the group catalog standard properties.
#[pyfunction]
fn create_group_catalog<'py>(
    py: Python<'py>,
    ra: Vec<f64>,
    dec: Vec<f64>,
    redshift: Vec<f64>,
    absolute_magnitudes: Vec<f64>,
    velocity_errors: Vec<f64>,
    group_ids: Vec<i32>,
    omega_m: f64,
    omega_k: f64,
    omega_l: f64,
    h0: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let catalog = GroupedGalaxyCatalog {
        ra,
        dec,
        redshift,
        absolute_magnitudes,
        velocity_errors,
        group_ids,
    };
    let cosmo = &Cosmology {
        omega_m,
        omega_k,
        omega_l,
        h0,
    };
    let group_catalog = catalog.calculate_group_properties(cosmo);

    let dict = PyDict::new(py);
    dict.set_item("group_id", group_catalog.ids)?;
    dict.set_item("iter_ra", group_catalog.iter_ras)?;
    dict.set_item("iter_dec", group_catalog.iter_decs)?;
    dict.set_item("iter_redshift", group_catalog.iter_redshifts)?;
    dict.set_item("iter_idx", group_catalog.iter_idxs)?;
    dict.set_item("median_redshift", group_catalog.median_redshifts)?;
    dict.set_item("co_dist", group_catalog.distances)?;
    dict.set_item("r50", group_catalog.r50s)?;
    dict.set_item("r100", group_catalog.r100s)?;
    dict.set_item("rsigma", group_catalog.rsigmas)?;
    dict.set_item("multiplicity", group_catalog.multiplicity)?;
    dict.set_item(
        "velocity_dispersion_gap",
        group_catalog.velocity_dispersion_gap,
    )?;
    dict.set_item(
        "velocity_dispersion_gap_err",
        group_catalog.velocity_dispersion_gap_err,
    )?;
    dict.set_item("mass_proxy", group_catalog.raw_masses)?;
    dict.set_item("bcg_idxs", group_catalog.bcg_idxs)?;
    dict.set_item("bcg_ras", group_catalog.bcg_ras)?;
    dict.set_item("bcg_decs", group_catalog.bcg_decs)?;
    dict.set_item("bcg_redshifts", group_catalog.bcg_redshifts)?;
    dict.set_item("center_of_light_ras", group_catalog.col_ras)?;
    dict.set_item("center_of_light_decs", group_catalog.col_decs)?;
    dict.set_item("total_absolute_mag", group_catalog.total_absolute_mags)?;
    dict.set_item("flux_proxies", group_catalog.total_flux_proxies)?;
    Ok(dict)
}

/// Creating the pair catalog standard properties.
#[pyfunction]
fn create_pair_catalog<'py>(
    py: Python<'py>,
    ra: Vec<f64>,
    dec: Vec<f64>,
    redshift: Vec<f64>,
    absolute_magnitudes: Vec<f64>,
    group_ids: Vec<i32>,
) -> PyResult<Bound<'py, PyDict>> {
    let catalog = GroupedGalaxyCatalog {
        ra,
        dec,
        redshift,
        absolute_magnitudes,
        velocity_errors: vec![50.; 1], //dummy variable
        group_ids,
    };

    let pair_catalog = catalog.calculate_pair_properties();

    let dict = PyDict::new(py);
    dict.set_item("pair_id", pair_catalog.ids)?;
    dict.set_item("idx_1", pair_catalog.idx_1)?;
    dict.set_item("idx_2", pair_catalog.idx_2)?;
    dict.set_item("projected_separation", pair_catalog.projected_separation)?;
    dict.set_item("velocity_separation", pair_catalog.velocity_separation)?;
    dict.set_item("ra_bar", pair_catalog.ra_bar)?;
    dict.set_item("dec_bar", pair_catalog.dec_bar)?;
    dict.set_item("redshift_bar", pair_catalog.redshift_bar)?;
    dict.set_item("total_absolute_mag", pair_catalog.total_absolute_mags)?;
    Ok(dict)
}

/// Calculates the Score in robotham+2011
#[pyfunction]
fn calculate_s_score(
    measured_groups: Vec<i32>,
    mock_groups: Vec<i32>,
    min_group_size: usize,
) -> PyResult<f64> {
    let score = s_score(&measured_groups, &mock_groups, min_group_size);
    Ok(score)
}

/// Calculates the harmonic mean in the standard way
#[pyfunction]
fn calculate_harmonic_mean(values: Vec<f64>) -> PyResult<f64> {
    let mean = harmonic_mean(values);
    Ok(mean)
}

/// A Python module implemented in Rust.
#[pymodule]
fn nessie_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(h_at_z, m)?)?;
    m.add_function(wrap_pyfunction!(comoving_distances_at_z, m)?)?;
    m.add_function(wrap_pyfunction!(z_at_comoving_distances, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_max_rvirs, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_max_sigmas, m)?)?;
    m.add_function(wrap_pyfunction!(distance_modulus, m)?)?;
    m.add_function(wrap_pyfunction!(fof_links_fast, m)?)?;
    m.add_function(wrap_pyfunction!(create_group_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_s_score, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_harmonic_mean, m)?)?;
    m.add_function(wrap_pyfunction!(create_pair_catalog, m)?)?;

    Ok(())
}
