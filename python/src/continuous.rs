use pyo3::prelude::*;
use pyo3::types::PySequence;
use simsam::{
    from_histogram, BuildOptions, Cdf, ContinuousSampler, HasSupport,
    HistogramPdf, IntegratedPdf, Interval, LocScale, Pdf, TdrBuildConfig, Truncated, AffineCdf,
};

use crate::callable::PyFloatFn;
use crate::error::{map_build_err, map_sample_err, parse_tdr_transform};
use crate::{extract_f64_list, optional_seed_rng};
use std::sync::Arc;

#[pyclass(name = "Interval", module = "_simsam")]
#[derive(Clone, Copy)]
pub struct PyInterval {
    pub inner: Interval,
}

#[pymethods]
impl PyInterval {
    #[new]
    fn new(lo: f64, hi: f64) -> PyResult<Self> {
        Interval::new(lo, hi)
            .map(|inner| Self { inner })
            .map_err(map_build_err)
    }

    #[getter]
    fn lo(&self) -> f64 {
        self.inner.lo
    }

    #[getter]
    fn hi(&self) -> f64 {
        self.inner.hi
    }

    fn __repr__(&self) -> String {
        format!("Interval({}, {})", self.inner.lo, self.inner.hi)
    }
}

#[pyclass(name = "BuildOptions", module = "_simsam")]
#[derive(Clone, Copy)]
pub struct PyBuildOptions {
    pub inner: BuildOptions,
}

#[pymethods]
impl PyBuildOptions {
    #[new]
    #[pyo3(signature = (
        quad_tolerance=None,
        ppf_tolerance=None,
        max_iterations=None,
        use_newton=true,
        hermite_grid_size=None,
        tdr_transform=None,
        tdr_construction_points=None,
        tdr_dpdf_rel_step=None,
    ))]
    fn new(
        quad_tolerance: Option<f64>,
        ppf_tolerance: Option<f64>,
        max_iterations: Option<u32>,
        use_newton: bool,
        hermite_grid_size: Option<usize>,
        tdr_transform: Option<&str>,
        tdr_construction_points: Option<usize>,
        tdr_dpdf_rel_step: Option<f64>,
    ) -> PyResult<Self> {
        let mut inner = BuildOptions::default();
        if let Some(v) = quad_tolerance {
            inner.quad_tolerance = v;
        }
        if let Some(v) = ppf_tolerance {
            inner.ppf_tolerance = v;
        }
        if let Some(v) = max_iterations {
            inner.max_iterations = v;
        }
        inner.use_newton = use_newton;
        if let Some(n) = hermite_grid_size {
            inner = inner.with_hermite(n);
        } else if let Some(t) = tdr_transform {
            let mut cfg = TdrBuildConfig::default();
            cfg.transform = parse_tdr_transform(t)?;
            if let Some(n) = tdr_construction_points {
                cfg.construction_points = n;
            }
            if let Some(s) = tdr_dpdf_rel_step {
                cfg.dpdf_rel_step = s;
            }
            inner = inner.with_tdr_config(cfg);
        }
        Ok(Self { inner })
    }

    #[staticmethod]
    fn hermite(grid_size: usize) -> Self {
        Self {
            inner: BuildOptions::default().with_hermite(grid_size),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (transform, construction_points=None))]
    fn tdr(transform: &str, construction_points: Option<usize>) -> PyResult<Self> {
        let mut cfg = TdrBuildConfig::default();
        cfg.transform = parse_tdr_transform(transform)?;
        if let Some(n) = construction_points {
            cfg.construction_points = n;
        }
        Ok(Self {
            inner: BuildOptions::default().with_tdr_config(cfg),
        })
    }
}

#[derive(Clone)]
struct PyPdf {
    f: Arc<PyFloatFn>,
    support: Interval,
}

impl Pdf for PyPdf {
    fn pdf(&self, x: f64) -> f64 {
        self.f.call(x)
    }
}

impl HasSupport for PyPdf {
    fn support(&self) -> Interval {
        self.support
    }
}

#[derive(Clone)]
struct PyCdf {
    f: Arc<PyFloatFn>,
    support: Interval,
}

impl Cdf for PyCdf {
    fn cdf(&self, x: f64) -> f64 {
        self.f.call(x)
    }
}

impl HasSupport for PyCdf {
    fn support(&self) -> Interval {
        self.support
    }
}

#[derive(Clone)]
struct PyDpdf {
    f: Arc<PyFloatFn>,
    support: Interval,
}

impl simsam::Dpdf for PyDpdf {
    fn dpdf(&self, x: f64) -> f64 {
        self.f.call(x)
    }
}

impl HasSupport for PyDpdf {
    fn support(&self) -> Interval {
        self.support
    }
}

enum ContinuousInner {
    Pdf(ContinuousSampler<IntegratedPdf<PyPdf>>),
    Cdf(ContinuousSampler<AffineCdf<PyCdf>>),
    Histogram(ContinuousSampler<IntegratedPdf<HistogramPdf>>),
    LocScale(ContinuousSampler<IntegratedPdf<LocScale<PyPdf>>>),
    Truncated(ContinuousSampler<IntegratedPdf<Truncated<PyPdf>>>),
    #[cfg(feature = "symbolic")]
    Symbolic(ContinuousSampler<IntegratedPdf<crate::symbolic::SymbolicPdfOwned>>),
}

#[cfg_attr(feature = "symbolic", pyclass(name = "ContinuousDistribution", module = "_simsam", unsendable))]
#[cfg_attr(not(feature = "symbolic"), pyclass(name = "ContinuousDistribution", module = "_simsam"))]
pub struct PyContinuousDistribution {
    inner: ContinuousInner,
}

impl PyContinuousDistribution {
    pub(crate) fn wrap_pdf(s: ContinuousSampler<IntegratedPdf<PyPdf>>) -> Self {
        Self {
            inner: ContinuousInner::Pdf(s),
        }
    }

    pub(crate) fn wrap_cdf(s: ContinuousSampler<AffineCdf<PyCdf>>) -> Self {
        Self {
            inner: ContinuousInner::Cdf(s),
        }
    }

    pub(crate) fn wrap_histogram(s: ContinuousSampler<IntegratedPdf<HistogramPdf>>) -> Self {
        Self {
            inner: ContinuousInner::Histogram(s),
        }
    }

    pub(crate) fn wrap_loc_scale(s: ContinuousSampler<IntegratedPdf<LocScale<PyPdf>>>) -> Self {
        Self {
            inner: ContinuousInner::LocScale(s),
        }
    }

    pub(crate) fn wrap_truncated(s: ContinuousSampler<IntegratedPdf<Truncated<PyPdf>>>) -> Self {
        Self {
            inner: ContinuousInner::Truncated(s),
        }
    }

    #[cfg(feature = "symbolic")]
    pub(crate) fn wrap_symbolic(
        s: ContinuousSampler<IntegratedPdf<crate::symbolic::SymbolicPdfOwned>>,
    ) -> Self {
        Self {
            inner: ContinuousInner::Symbolic(s),
        }
    }
}

macro_rules! with_continuous {
    ($self:expr, |$s:ident| $body:expr) => {
        match &$self.inner {
            ContinuousInner::Pdf($s) => $body,
            ContinuousInner::Cdf($s) => $body,
            ContinuousInner::Histogram($s) => $body,
            ContinuousInner::LocScale($s) => $body,
            ContinuousInner::Truncated($s) => $body,
            #[cfg(feature = "symbolic")]
            ContinuousInner::Symbolic($s) => $body,
        }
    };
}

#[pymethods]
impl PyContinuousDistribution {
    fn support(&self) -> PyInterval {
        with_continuous!(self, |s| PyInterval {
            inner: s.support(),
        })
    }

    fn sample(&self) -> PyResult<f64> {
        with_continuous!(self, |s| s.sample().map_err(map_sample_err))
    }

    #[pyo3(signature = (n, seed=None))]
    fn sample_n(&self, py: Python<'_>, n: usize, seed: Option<u64>) -> PyResult<Py<PyAny>> {
        let samples = if let Some(s) = seed {
            let mut rng = optional_seed_rng(s);
            with_continuous!(self, |dist| {
                dist.sample_n_with_rng(&mut rng, n)
                    .map_err(map_sample_err)
            })?
        } else {
            with_continuous!(self, |dist| dist.sample_n(n).map_err(map_sample_err))?
        };
        Ok(pyo3::types::PyList::new(py, samples)?.into_any().unbind())
    }

    fn cdf(&self, x: f64) -> f64 {
        with_continuous!(self, |s| s.cdf(x))
    }

    fn pdf(&self, x: f64) -> Option<f64> {
        with_continuous!(self, |s| s.pdf(x))
    }

    fn ppf(&self, u: f64) -> PyResult<f64> {
        with_continuous!(self, |s| s.ppf(u).map_err(map_sample_err))
    }

    fn sf(&self, x: f64) -> f64 {
        with_continuous!(self, |s| s.sf(x))
    }

    fn isf(&self, q: f64) -> PyResult<f64> {
        with_continuous!(self, |s| s.isf(q).map_err(map_sample_err))
    }

    fn logpdf(&self, x: f64) -> Option<f64> {
        with_continuous!(self, |s| s.logpdf(x))
    }

    fn logcdf(&self, x: f64) -> f64 {
        with_continuous!(self, |s| s.logcdf(x))
    }

    fn logsf(&self, x: f64) -> f64 {
        with_continuous!(self, |s| s.logsf(x))
    }

    fn mean(&self) -> PyResult<f64> {
        with_continuous!(self, |s| s.mean().map_err(map_sample_err))
    }

    fn var(&self) -> PyResult<f64> {
        with_continuous!(self, |s| s.var().map_err(map_sample_err))
    }

    fn std(&self) -> PyResult<f64> {
        with_continuous!(self, |s| s.std().map_err(map_sample_err))
    }

    fn median(&self) -> PyResult<f64> {
        with_continuous!(self, |s| s.median().map_err(map_sample_err))
    }

    fn entropy(&self) -> PyResult<f64> {
        with_continuous!(self, |s| s.entropy().map_err(map_sample_err))
    }

    fn interval(&self, confidence: f64) -> PyResult<(f64, f64)> {
        with_continuous!(self, |s| s.interval(confidence).map_err(map_sample_err))
    }

    fn expect(&self, func: Py<PyAny>) -> PyResult<f64> {
        let f = Arc::new(PyFloatFn::new(func));
        with_continuous!(self, |s| {
            s.expect(|x| f.call(x), simsam::default_quad_tol())
                .map_err(map_sample_err)
        })
    }

    fn uses_hermite_table(&self) -> bool {
        with_continuous!(self, |s| s.uses_hermite_table())
    }

    fn uses_tdr(&self) -> bool {
        with_continuous!(self, |s| s.uses_tdr())
    }
}

fn build_pdf(f: Py<PyAny>, interval: Interval, opts: BuildOptions) -> PyResult<PyContinuousDistribution> {
    let pdf = PyPdf {
        f: Arc::new(PyFloatFn::new(f)),
        support: interval,
    };
    let s = ContinuousSampler::from_pdf(pdf, opts).map_err(map_build_err)?;
    Ok(PyContinuousDistribution::wrap_pdf(s))
}

#[pyfunction]
#[pyo3(name = "from_pdf")]
fn py_from_pdf(f: Py<PyAny>, interval: PyInterval) -> PyResult<PyContinuousDistribution> {
    build_pdf(f, interval.inner, BuildOptions::default())
}

#[pyfunction]
#[pyo3(name = "from_pdf_with_options")]
fn py_from_pdf_with_options(
    f: Py<PyAny>,
    interval: PyInterval,
    options: PyBuildOptions,
) -> PyResult<PyContinuousDistribution> {
    build_pdf(f, interval.inner, options.inner)
}

#[pyfunction]
#[pyo3(name = "from_pdf_dpdf", signature = (f, dpdf, interval, options=None))]
fn py_from_pdf_dpdf(
    f: Py<PyAny>,
    dpdf: Py<PyAny>,
    interval: PyInterval,
    options: Option<PyBuildOptions>,
) -> PyResult<PyContinuousDistribution> {
    let opts = options.map(|o| o.inner).unwrap_or_default();
    let interval = interval.inner;
    let pdf = PyPdf {
        f: Arc::new(PyFloatFn::new(f)),
        support: interval,
    };
    let d = PyDpdf {
        f: Arc::new(PyFloatFn::new(dpdf)),
        support: interval,
    };
    let s = ContinuousSampler::from_pdf_with_dpdf(pdf, d, opts).map_err(map_build_err)?;
    Ok(PyContinuousDistribution::wrap_pdf(s))
}

#[pyfunction]
#[pyo3(name = "from_cdf")]
fn py_from_cdf(f: Py<PyAny>, interval: PyInterval) -> PyResult<PyContinuousDistribution> {
    let cdf = PyCdf {
        f: Arc::new(PyFloatFn::new(f)),
        support: interval.inner,
    };
    let s = ContinuousSampler::from_cdf(cdf, BuildOptions::default()).map_err(map_build_err)?;
    Ok(PyContinuousDistribution::wrap_cdf(s))
}

#[pyfunction]
#[pyo3(name = "from_cdf_with_options")]
fn py_from_cdf_with_options(
    f: Py<PyAny>,
    interval: PyInterval,
    options: PyBuildOptions,
) -> PyResult<PyContinuousDistribution> {
    let cdf = PyCdf {
        f: Arc::new(PyFloatFn::new(f)),
        support: interval.inner,
    };
    let s = ContinuousSampler::from_cdf(cdf, options.inner).map_err(map_build_err)?;
    Ok(PyContinuousDistribution::wrap_cdf(s))
}

#[pyfunction]
#[pyo3(name = "from_histogram", signature = (edges, counts, density=false, options=None))]
fn py_from_histogram(
    edges: &Bound<'_, PySequence>,
    counts: &Bound<'_, PySequence>,
    density: bool,
    options: Option<PyBuildOptions>,
) -> PyResult<PyContinuousDistribution> {
    let edges = extract_f64_list(edges)?;
    let counts = extract_f64_list(counts)?;
    let opts = options.map(|o| o.inner).unwrap_or_default();
    let s = from_histogram(edges, counts, density, opts).map_err(map_build_err)?;
    Ok(PyContinuousDistribution::wrap_histogram(s))
}

#[pyfunction]
#[pyo3(name = "from_pdf_loc_scale", signature = (f, interval, loc, scale, options=None))]
fn py_from_pdf_loc_scale(
    f: Py<PyAny>,
    interval: PyInterval,
    loc: f64,
    scale: f64,
    options: Option<PyBuildOptions>,
) -> PyResult<PyContinuousDistribution> {
    let opts = options.map(|o| o.inner).unwrap_or_default();
    let base = PyPdf {
        f: Arc::new(PyFloatFn::new(f)),
        support: interval.inner,
    };
    let wrapped = LocScale::new(base, loc, scale).map_err(map_build_err)?;
    let s = ContinuousSampler::from_pdf(wrapped, opts).map_err(map_build_err)?;
    Ok(PyContinuousDistribution::wrap_loc_scale(s))
}

#[pyfunction]
#[pyo3(name = "truncated", signature = (f, interval, trunc_interval, options=None))]
fn py_truncated(
    f: Py<PyAny>,
    interval: PyInterval,
    trunc_interval: PyInterval,
    options: Option<PyBuildOptions>,
) -> PyResult<PyContinuousDistribution> {
    let opts = options.map(|o| o.inner).unwrap_or_default();
    let base = PyPdf {
        f: Arc::new(PyFloatFn::new(f)),
        support: interval.inner,
    };
    let trunc = Truncated::new(base, trunc_interval.inner, simsam::default_quad_tol())
        .map_err(map_build_err)?;
    let s = ContinuousSampler::from_pdf(trunc, opts).map_err(map_build_err)?;
    Ok(PyContinuousDistribution::wrap_truncated(s))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyInterval>()?;
    m.add_class::<PyBuildOptions>()?;
    m.add_class::<PyContinuousDistribution>()?;
    m.add_function(wrap_pyfunction!(py_from_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(py_from_pdf_with_options, m)?)?;
    m.add_function(wrap_pyfunction!(py_from_pdf_dpdf, m)?)?;
    m.add_function(wrap_pyfunction!(py_from_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(py_from_cdf_with_options, m)?)?;
    m.add_function(wrap_pyfunction!(py_from_histogram, m)?)?;
    m.add_function(wrap_pyfunction!(py_from_pdf_loc_scale, m)?)?;
    m.add_function(wrap_pyfunction!(py_truncated, m)?)?;
    Ok(())
}
