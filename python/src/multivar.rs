use pyo3::prelude::*;
use pyo3::types::PySequence;
use simsam::{
    CdfMcEstimator, CdfMcOptions, ConditionalFactorSampler, GaussianCopula, GibbsOptions,
    GibbsSamplerNd, HasSupportNd, HmcOptions, HmcSamplerNd, HyperRect, LogPdfNd, MhOptions,
    MetropolisHastingsNd, PdfNd, RejectionOptions, RejectionSamplerNd,
};
use std::sync::Arc;

use crate::callable::{PyConditionalCoord, PyFactorFn, PyNdFn, PyPpfFn};
use crate::error::{map_build_err, map_sample_err};
use crate::{extract_f64_list, optional_seed_rng, vec_to_pylist, vecs_to_pylist};

#[pyclass(name = "HyperRect", module = "_simsam")]
#[derive(Clone)]
pub struct PyHyperRect {
    pub inner: HyperRect,
}

#[pymethods]
impl PyHyperRect {
    #[new]
    fn new(lo: &Bound<'_, PySequence>, hi: &Bound<'_, PySequence>) -> PyResult<Self> {
        let lo = extract_f64_list(lo)?;
        let hi = extract_f64_list(hi)?;
        HyperRect::new(lo, hi)
            .map(|inner| Self { inner })
            .map_err(map_build_err)
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn lo(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        vec_to_pylist(py, &self.inner.lo)
    }

    fn hi(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        vec_to_pylist(py, &self.inner.hi)
    }
}

#[derive(Clone)]
struct PyPdfNd {
    f: Arc<PyNdFn>,
    support: HyperRect,
}

impl PdfNd for PyPdfNd {
    fn pdf(&self, x: &[f64]) -> f64 {
        self.f.call(x)
    }
}

impl LogPdfNd for PyPdfNd {
    fn log_pdf(&self, x: &[f64]) -> f64 {
        let p = self.f.call(x);
        if p > 0.0 {
            p.ln()
        } else {
            f64::NEG_INFINITY
        }
    }
}

impl HasSupportNd for PyPdfNd {
    fn support(&self) -> &HyperRect {
        &self.support
    }
}

fn make_pdf_nd(f: Py<PyAny>, support: HyperRect) -> PyPdfNd {
    let dim = support.dim();
    PyPdfNd {
        f: Arc::new(PyNdFn::new(f, dim)),
        support,
    }
}

#[pyclass(name = "RejectionSampler", module = "_simsam")]
pub struct PyRejectionSampler {
    inner: RejectionSamplerNd<PyPdfNd>,
}

#[pymethods]
impl PyRejectionSampler {
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn pdf_max(&self) -> f64 {
        self.inner.pdf_max()
    }

    fn sample(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let v = self.inner.sample().map_err(map_sample_err)?;
        vec_to_pylist(py, &v)
    }

    #[pyo3(signature = (n, seed=None))]
    fn sample_n(&self, py: Python<'_>, n: usize, seed: Option<u64>) -> PyResult<Py<PyAny>> {
        let mut out = Vec::with_capacity(n);
        if let Some(s) = seed {
            let mut rng = optional_seed_rng(s);
            for _ in 0..n {
                out.push(
                    self.inner
                        .sample_with_rng(&mut rng)
                        .map_err(map_sample_err)?,
                );
            }
        } else {
            for _ in 0..n {
                out.push(self.inner.sample().map_err(map_sample_err)?);
            }
        }
        vecs_to_pylist(py, &out)
    }
}

#[pyfunction]
#[pyo3(name = "rejection_sampler", signature = (f, support, pdf_max, max_trials=100_000))]
fn py_rejection_sampler(
    f: Py<PyAny>,
    support: PyHyperRect,
    pdf_max: f64,
    max_trials: usize,
) -> PyResult<PyRejectionSampler> {
    let pdf = make_pdf_nd(f, support.inner);
    let inner = RejectionSamplerNd::new(
        pdf,
        pdf_max,
        RejectionOptions { max_trials },
    )
    .map_err(map_build_err)?;
    Ok(PyRejectionSampler { inner })
}

#[pyclass(name = "MetropolisHastings", module = "_simsam", unsendable)]
pub struct PyMetropolisHastings {
    inner: MetropolisHastingsNd<PyPdfNd>,
}

#[pymethods]
impl PyMetropolisHastings {
    fn init(&mut self) -> PyResult<()> {
        self.inner.init().map_err(map_sample_err)
    }

    fn step(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let v = self.inner.step().map_err(map_sample_err)?;
        vec_to_pylist(py, v)
    }

    fn sample(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let v = self.inner.sample().map_err(map_sample_err)?;
        vec_to_pylist(py, &v)
    }

    #[pyo3(signature = (n, seed=None))]
    fn sample_n(&mut self, py: Python<'_>, n: usize, seed: Option<u64>) -> PyResult<Py<PyAny>> {
        let samples = if let Some(s) = seed {
            let mut rng = optional_seed_rng(s);
            self.inner
                .sample_n_with_rng(&mut rng, n)
                .map_err(map_sample_err)?
        } else {
            self.inner.sample_n(n).map_err(map_sample_err)?
        };
        vecs_to_pylist(py, &samples)
    }

    fn accept_rate(&self) -> f64 {
        self.inner.accept_rate()
    }
}

#[pyfunction]
#[pyo3(name = "metropolis_hastings", signature = (
    f, support,
    step_size=0.25,
    burn_in=1000,
    thin=1,
    target_accept=0.25,
    adapt_steps=2000,
))]
fn py_metropolis_hastings(
    f: Py<PyAny>,
    support: PyHyperRect,
    step_size: f64,
    burn_in: usize,
    thin: usize,
    target_accept: f64,
    adapt_steps: usize,
) -> PyResult<PyMetropolisHastings> {
    let pdf = make_pdf_nd(f, support.inner);
    let opts = MhOptions {
        step_size,
        burn_in,
        thin,
        target_accept,
        adapt_steps,
        ..MhOptions::default()
    };
    let inner = MetropolisHastingsNd::new(pdf, opts).map_err(map_build_err)?;
    Ok(PyMetropolisHastings { inner })
}

#[pyclass(name = "HmcSampler", module = "_simsam", unsendable)]
pub struct PyHmcSampler {
    inner: HmcSamplerNd<PyPdfNd>,
}

#[pymethods]
impl PyHmcSampler {
    fn init(&mut self) -> PyResult<()> {
        self.inner.init().map_err(map_sample_err)
    }

    fn sample(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let v = self.inner.sample().map_err(map_sample_err)?;
        vec_to_pylist(py, &v)
    }

    #[pyo3(signature = (n, seed=None))]
    fn sample_n(&mut self, py: Python<'_>, n: usize, seed: Option<u64>) -> PyResult<Py<PyAny>> {
        let mut out = Vec::with_capacity(n);
        if let Some(s) = seed {
            let mut rng = optional_seed_rng(s);
            for _ in 0..n {
                out.push(self.inner.sample_with_rng(&mut rng).map_err(map_sample_err)?);
            }
        } else {
            for _ in 0..n {
                out.push(self.inner.sample().map_err(map_sample_err)?);
            }
        }
        vecs_to_pylist(py, &out)
    }
}

#[pyfunction]
#[pyo3(name = "hmc_sampler", signature = (
    f, support,
    step_size=0.05,
    leapfrog_steps=20,
    grad_eps=1e-6,
))]
fn py_hmc_sampler(
    f: Py<PyAny>,
    support: PyHyperRect,
    step_size: f64,
    leapfrog_steps: usize,
    grad_eps: f64,
) -> PyResult<PyHmcSampler> {
    let pdf = make_pdf_nd(f, support.inner);
    let opts = HmcOptions {
        step_size,
        leapfrog_steps,
        grad_eps,
        ..HmcOptions::default()
    };
    let inner = HmcSamplerNd::new(pdf, opts).map_err(map_build_err)?;
    Ok(PyHmcSampler { inner })
}

#[pyclass(name = "GibbsSampler", module = "_simsam", unsendable)]
pub struct PyGibbsSampler {
    inner: GibbsSamplerNd<PyConditionalCoord>,
}

#[pymethods]
impl PyGibbsSampler {
    fn sample(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let v = self.inner.sample().map_err(map_sample_err)?;
        vec_to_pylist(py, &v)
    }

    fn step(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let v = self.inner.step(&mut rand::rng()).map_err(map_sample_err)?;
        vec_to_pylist(py, v)
    }
}

#[pyfunction]
#[pyo3(name = "gibbs_sampler", signature = (sample_coord, support, init, sweeps_per_sample=1))]
fn py_gibbs_sampler(
    sample_coord: Py<PyAny>,
    support: PyHyperRect,
    init: &Bound<'_, PySequence>,
    sweeps_per_sample: usize,
) -> PyResult<PyGibbsSampler> {
    let init = extract_f64_list(init)?;
    let cond = PyConditionalCoord {
        callback: sample_coord,
        support: support.inner.clone(),
    };
    let inner = GibbsSamplerNd::new(
        cond,
        GibbsOptions {
            sweeps_per_sample,
        },
        init,
    )
    .map_err(map_sample_err)?;
    Ok(PyGibbsSampler { inner })
}

#[pyclass(name = "ConditionalFactorizationSampler", module = "_simsam")]
pub struct PyConditionalFactorizationSampler {
    inner: ConditionalFactorSampler<PyFactorFn>,
}

#[pymethods]
impl PyConditionalFactorizationSampler {
    fn sample(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let v = self.inner.sample().map_err(map_sample_err)?;
        vec_to_pylist(py, &v)
    }

    #[pyo3(signature = (n, seed=None))]
    fn sample_n(&self, py: Python<'_>, n: usize, seed: Option<u64>) -> PyResult<Py<PyAny>> {
        let mut out = Vec::with_capacity(n);
        if let Some(s) = seed {
            let mut rng = optional_seed_rng(s);
            for _ in 0..n {
                out.push(
                    self.inner
                        .sample_with_rng(&mut rng)
                        .map_err(map_sample_err)?,
                );
            }
        } else {
            for _ in 0..n {
                out.push(self.inner.sample().map_err(map_sample_err)?);
            }
        }
        vecs_to_pylist(py, &out)
    }
}

#[pyfunction]
#[pyo3(name = "conditional_factorization_sampler")]
fn py_conditional_factorization_sampler(
    sample_i: Py<PyAny>,
    dim: usize,
) -> PyResult<PyConditionalFactorizationSampler> {
    let inner = ConditionalFactorSampler::new(PyFactorFn {
        callback: sample_i,
        dim,
    });
    Ok(PyConditionalFactorizationSampler { inner })
}

#[pyclass(name = "GaussianCopula", module = "_simsam")]
pub struct PyGaussianCopula {
    inner: GaussianCopula,
}

#[pymethods]
impl PyGaussianCopula {
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[pyo3(signature = (ppfs, seed=None))]
    fn sample(&self, py: Python<'_>, ppfs: Vec<Py<PyAny>>, seed: Option<u64>) -> PyResult<Py<PyAny>> {
        let arcs: Vec<Arc<PyPpfFn>> = ppfs
            .into_iter()
            .map(|p| Arc::new(PyPpfFn { callback: p }))
            .collect();
        let boxed: Vec<Box<dyn Fn(f64) -> Result<f64, simsam::SampleError>>> = arcs
            .iter()
            .map(|p| {
                let p = Arc::clone(p);
                Box::new(move |u: f64| p.call(u))
                    as Box<dyn Fn(f64) -> Result<f64, simsam::SampleError>>
            })
            .collect();
        let refs: Vec<&dyn Fn(f64) -> Result<f64, simsam::SampleError>> = boxed
            .iter()
            .map(|b| &**b as &dyn Fn(f64) -> Result<f64, simsam::SampleError>)
            .collect();
        let v = if let Some(s) = seed {
            let mut rng = optional_seed_rng(s);
            self.inner
                .sample_with_ppfs(&mut rng, &refs)
                .map_err(map_sample_err)?
        } else {
            self.inner
                .sample_with_ppfs(&mut rand::rng(), &refs)
                .map_err(map_sample_err)?
        };
        vec_to_pylist(py, &v)
    }
}

#[pyfunction]
#[pyo3(name = "gaussian_copula")]
fn py_gaussian_copula(corr: Vec<Vec<f64>>) -> PyResult<PyGaussianCopula> {
    let inner = GaussianCopula::new(corr)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e:?}")))?;
    Ok(PyGaussianCopula { inner })
}

#[pyclass(name = "CdfMcEstimator", module = "_simsam", unsendable)]
pub struct PyCdfMcEstimator {
    inner: CdfMcEstimator<PyPdfNd>,
}

#[pymethods]
impl PyCdfMcEstimator {
    #[pyo3(signature = (seed=None))]
    fn estimate_normalization(&mut self, seed: Option<u64>) -> PyResult<f64> {
        if let Some(s) = seed {
            let mut rng = optional_seed_rng(s);
            self.inner
                .estimate_normalization_with_rng(&mut rng)
                .map_err(map_sample_err)
        } else {
            self.inner.estimate_normalization().map_err(map_sample_err)
        }
    }

    #[pyo3(signature = (x, seed=None))]
    fn cdf(&mut self, x: &Bound<'_, PySequence>, seed: Option<u64>) -> PyResult<f64> {
        let x = extract_f64_list(x)?;
        if let Some(s) = seed {
            let mut rng = optional_seed_rng(s);
            self.inner
                .cdf_with_rng(&mut rng, &x)
                .map_err(map_sample_err)
        } else {
            self.inner.cdf(&x).map_err(map_sample_err)
        }
    }
}

#[pyfunction]
#[pyo3(name = "cdf_mc_estimator", signature = (
    f, support,
    normalization_samples=50_000,
    cdf_samples=50_000,
))]
fn py_cdf_mc_estimator(
    f: Py<PyAny>,
    support: PyHyperRect,
    normalization_samples: usize,
    cdf_samples: usize,
) -> PyResult<PyCdfMcEstimator> {
    let pdf = make_pdf_nd(f, support.inner);
    let inner = CdfMcEstimator::new(
        pdf,
        CdfMcOptions {
            normalization_samples,
            cdf_samples,
        },
    )
    .map_err(map_build_err)?;
    Ok(PyCdfMcEstimator { inner })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHyperRect>()?;
    m.add_class::<PyRejectionSampler>()?;
    m.add_class::<PyMetropolisHastings>()?;
    m.add_class::<PyHmcSampler>()?;
    m.add_class::<PyGibbsSampler>()?;
    m.add_class::<PyConditionalFactorizationSampler>()?;
    m.add_class::<PyGaussianCopula>()?;
    m.add_class::<PyCdfMcEstimator>()?;
    m.add_function(wrap_pyfunction!(py_rejection_sampler, m)?)?;
    m.add_function(wrap_pyfunction!(py_metropolis_hastings, m)?)?;
    m.add_function(wrap_pyfunction!(py_hmc_sampler, m)?)?;
    m.add_function(wrap_pyfunction!(py_gibbs_sampler, m)?)?;
    m.add_function(wrap_pyfunction!(py_conditional_factorization_sampler, m)?)?;
    m.add_function(wrap_pyfunction!(py_gaussian_copula, m)?)?;
    m.add_function(wrap_pyfunction!(py_cdf_mc_estimator, m)?)?;
    Ok(())
}
