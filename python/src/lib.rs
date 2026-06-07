use pyo3::prelude::*;
use pyo3::types::PySequence;

pub mod callable;
pub mod continuous;
pub mod discrete;
pub mod error;
pub mod multivar;

#[cfg(feature = "symbolic")]
pub mod symbolic;

pub use error::{map_build_err, map_sample_err};

pub fn extract_f64_list(seq: &Bound<'_, PySequence>) -> PyResult<Vec<f64>> {
    seq.try_iter()?
        .map(|item| item?.extract())
        .collect()
}

pub fn vec_to_pylist(py: Python<'_>, v: &[f64]) -> PyResult<Py<PyAny>> {
    Ok(pyo3::types::PyList::new(py, v)?.into_any().unbind())
}

pub fn vecs_to_pylist(py: Python<'_>, samples: &[Vec<f64>]) -> PyResult<Py<PyAny>> {
    let rows: PyResult<Vec<_>> = samples
        .iter()
        .map(|row| vec_to_pylist(py, row))
        .collect();
    Ok(pyo3::types::PyList::new(py, rows?)?.into_any().unbind())
}

pub fn optional_seed_rng(seed: u64) -> rand_chacha::ChaCha8Rng {
    use rand::SeedableRng;
    rand_chacha::ChaCha8Rng::seed_from_u64(seed)
}

#[pymodule]
fn _simsam(m: &Bound<'_, PyModule>) -> PyResult<()> {
    error::register(m)?;
    continuous::register(m)?;
    discrete::register(m)?;
    multivar::register(m)?;
    #[cfg(feature = "symbolic")]
    symbolic::register(m)?;
    Ok(())
}
