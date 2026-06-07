use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use simsam::{BuildError, SampleError};

create_exception!(_simsam, BuildErrorPy, PyRuntimeError);
create_exception!(_simsam, SampleErrorPy, PyRuntimeError);

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("BuildError", m.py().get_type::<BuildErrorPy>())?;
    m.add("SampleError", m.py().get_type::<SampleErrorPy>())?;
    Ok(())
}

pub fn map_build_err(e: BuildError) -> PyErr {
    BuildErrorPy::new_err(e.to_string())
}

pub fn map_sample_err(e: SampleError) -> PyErr {
    SampleErrorPy::new_err(e.to_string())
}

pub fn parse_tdr_transform(s: &str) -> PyResult<simsam::TdrTransform> {
    match s.to_ascii_lowercase().as_str() {
        "log" => Ok(simsam::TdrTransform::Log),
        "inv_sqrt" | "invsqrt" | "inv-sqrt" => Ok(simsam::TdrTransform::InvSqrt),
        other => Err(PyValueError::new_err(format!(
            "unknown TDR transform '{other}', expected 'log' or 'inv_sqrt'"
        ))),
    }
}
