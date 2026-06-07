use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

/// Thread-safe wrapper around a Python callable `f(x: float) -> float`.
pub struct PyFloatFn {
    pub callback: Py<PyAny>,
}

impl Clone for PyFloatFn {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            callback: self.callback.clone_ref(py),
        })
    }
}

impl PyFloatFn {
    pub fn new(callback: Py<PyAny>) -> Self {
        Self { callback }
    }

    pub fn call(&self, x: f64) -> f64 {
        Python::with_gil(|py| {
            self.callback
                .call1(py, (x,))
                .ok()
                .and_then(|v| v.extract::<f64>(py).ok())
                .unwrap_or(f64::NAN)
        })
    }
}

/// Callable `f(*xs) -> float` or `f(list) -> float`.
pub struct PyNdFn {
    pub callback: Py<PyAny>,
    pub dim: usize,
}

impl Clone for PyNdFn {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            callback: self.callback.clone_ref(py),
            dim: self.dim,
        })
    }
}

impl PyNdFn {
    pub fn new(callback: Py<PyAny>, dim: usize) -> Self {
        Self { callback, dim }
    }

    pub fn call(&self, x: &[f64]) -> f64 {
        Python::with_gil(|py| {
            let bound = self.callback.bind(py);
            if let Ok(args) = PyTuple::new(py, x) {
                if let Ok(v) = bound.call(args, None) {
                    if let Ok(f) = v.extract::<f64>() {
                        return f;
                    }
                }
            }
            if let Ok(list) = PyList::new(py, x) {
                if let Ok(v) = bound.call1((list,)) {
                    if let Ok(f) = v.extract::<f64>() {
                        return f;
                    }
                }
            }
            f64::NAN
        })
    }
}

/// PPF callback for copula: u -> x
pub struct PyPpfFn {
    pub callback: Py<PyAny>,
}

impl Clone for PyPpfFn {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            callback: self.callback.clone_ref(py),
        })
    }
}

impl PyPpfFn {
    pub fn call(&self, u: f64) -> Result<f64, simsam::SampleError> {
        Python::with_gil(|py| {
            self.callback
                .call1(py, (u,))
                .map_err(|_| simsam::SampleError::PpfFailed)?
                .extract::<f64>(py)
                .map_err(|_| simsam::SampleError::PpfFailed)
        })
    }
}

/// Conditional coordinate sampler for Gibbs: `sample_coord(i, state) -> float`.
pub struct PyConditionalCoord {
    pub callback: Py<PyAny>,
    pub support: simsam::HyperRect,
}

impl Clone for PyConditionalCoord {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            callback: self.callback.clone_ref(py),
            support: self.support.clone(),
        })
    }
}

impl simsam::ConditionalSampler for PyConditionalCoord {
    fn dim(&self) -> usize {
        self.support.dim()
    }

    fn support(&self) -> &simsam::HyperRect {
        &self.support
    }

    fn sample_coord<R: rand::Rng + ?Sized>(
        &self,
        _rng: &mut R,
        i: usize,
        state: &[f64],
    ) -> f64 {
        Python::with_gil(|py| {
            let state_list = PyList::new(py, state).ok()?;
            self.callback
                .call1(py, (i, state_list))
                .ok()?
                .extract::<f64>(py)
                .ok()
        })
        .unwrap_or(f64::NAN)
    }
}

/// Factorization callable: `sample_i(i, prefix) -> float`.
pub struct PyFactorFn {
    pub callback: Py<PyAny>,
    pub dim: usize,
}

impl Clone for PyFactorFn {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            callback: self.callback.clone_ref(py),
            dim: self.dim,
        })
    }
}

impl simsam::ConditionalFactorization for PyFactorFn {
    fn dim(&self) -> usize {
        self.dim
    }

    fn sample_i<R: rand::Rng + ?Sized>(&self, _rng: &mut R, i: usize, prefix: &[f64]) -> f64 {
        Python::with_gil(|py| {
            let prefix_list = PyList::new(py, prefix).ok()?;
            self.callback
                .call1(py, (i, prefix_list))
                .ok()?
                .extract::<f64>(py)
                .ok()
        })
        .unwrap_or(f64::NAN)
    }
}
