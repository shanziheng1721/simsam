use pyo3::prelude::*;
use pyo3::types::PySequence;
use simsam::DiscreteSampler;

use crate::error::{map_build_err, map_sample_err};
use crate::{extract_f64_list, optional_seed_rng};

#[pyclass(name = "DiscreteDistribution", module = "_simsam")]
pub struct PyDiscreteDistribution {
    inner: DiscreteSampler,
}

#[pymethods]
impl PyDiscreteDistribution {
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn points(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(pyo3::types::PyList::new(py, self.inner.points())?.into_any().unbind())
    }

    fn pmf(&self, x: f64) -> f64 {
        self.inner.pmf(x)
    }

    fn pmf_at(&self, index: usize) -> f64 {
        self.inner.pmf_at(index)
    }

    fn cdf_at(&self, index: usize) -> f64 {
        self.inner.cdf_at(index)
    }

    fn ppf(&self, u: f64) -> PyResult<f64> {
        self.inner.ppf(u).map_err(map_sample_err)
    }

    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    fn var(&self) -> f64 {
        self.inner.var()
    }

    fn std(&self) -> f64 {
        self.inner.std()
    }

    fn sample(&self) -> PyResult<f64> {
        self.inner.sample().map_err(map_sample_err)
    }

    #[pyo3(signature = (n, seed=None))]
    fn sample_n(&self, py: Python<'_>, n: usize, seed: Option<u64>) -> PyResult<Py<PyAny>> {
        let samples = if let Some(s) = seed {
            let mut rng = optional_seed_rng(s);
            self.inner
                .sample_n_with_rng(&mut rng, n)
                .map_err(map_sample_err)?
        } else {
            self.inner.sample_n(n).map_err(map_sample_err)?
        };
        Ok(pyo3::types::PyList::new(py, samples)?.into_any().unbind())
    }
}

#[pyfunction]
#[pyo3(name = "from_pmf")]
fn py_from_pmf(
    points: &Bound<'_, PySequence>,
    pmf: &Bound<'_, PySequence>,
) -> PyResult<PyDiscreteDistribution> {
    let points = extract_f64_list(points)?;
    let pmf = extract_f64_list(pmf)?;
    let inner = DiscreteSampler::from_pmf(points, pmf).map_err(map_build_err)?;
    Ok(PyDiscreteDistribution { inner })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDiscreteDistribution>()?;
    m.add_function(wrap_pyfunction!(py_from_pmf, m)?)?;
    Ok(())
}
