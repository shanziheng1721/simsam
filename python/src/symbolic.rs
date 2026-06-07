use pyo3::prelude::*;
use simsam::{ContinuousSampler, HasSupport, Pdf, SymbolicContinuous};

use crate::continuous::{PyBuildOptions, PyContinuousDistribution, PyInterval};
use crate::error::map_build_err;
use std::sync::Arc;

/// Build a symbolic distribution and return a ready-to-use sampler.
///
/// Example: `"3 * x^2"` with variable `"x"`.
#[pyfunction]
#[pyo3(name = "symbolic_continuous", signature = (pdf_expr, var, interval, options=None))]
fn py_symbolic_continuous(
    pdf_expr: &str,
    var: &str,
    interval: PyInterval,
    options: Option<PyBuildOptions>,
) -> PyResult<PyContinuousDistribution> {
    let sym = build_symbolic(pdf_expr, var, interval.inner)?;
    let opts = options.map(|o| o.inner).unwrap_or_default();
    let owned = SymbolicPdfOwned {
        inner: Arc::new(sym),
    };
    let s = ContinuousSampler::from_pdf(owned, opts).map_err(map_build_err)?;
    Ok(PyContinuousDistribution::wrap_symbolic(s))
}

fn build_symbolic(pdf_expr: &str, var: &str, support: simsam::Interval) -> PyResult<SymbolicContinuous> {
    use simsym::prelude::*;
    let v = symbol(var);
    let pdf = parse_symbolic_expr(pdf_expr, v, var)?;
    SymbolicContinuous::with_defaults(pdf, v, support).map_err(map_build_err)
}

fn parse_symbolic_expr(expr: &str, var: simsym::Symbol, var_name: &str) -> PyResult<simsym::Expr> {
    let s = expr.replace(' ', "");
    if let Ok(v) = parse_monomial(&s, var, var_name) {
        return Ok(v);
    }
    Err(pyo3::exceptions::PyValueError::new_err(format!(
        "unsupported symbolic expression '{expr}'; use forms like '3 * x^2' or '2 * x'"
    )))
}

fn parse_monomial(s: &str, var: simsym::Symbol, var_name: &str) -> Result<simsym::Expr, ()> {
    use simsym::prelude::*;
    let s = s.replace("**", "^");
    let parts: Vec<&str> = s.split('*').filter(|p| !p.is_empty()).collect();
    if parts.len() == 2 {
        let coef = parts[0].parse::<i64>().map_err(|_| ())?;
        let power_part = parts[1];
        if power_part == var_name {
            return Ok(rational(coef, 1) * var);
        }
        if let Some(exp) = power_part.strip_prefix(&format!("{var_name}^")) {
            let exp: u32 = exp.parse().map_err(|_| ())?;
            return Ok(rational(coef, 1) * var.pow(exp as i64));
        }
    }
    if let Ok(c) = s.parse::<i64>() {
        return Ok(simsym::expr::const_(rational(c, 1)));
    }
    if s == var_name {
        return Ok(rational(1, 1) * var);
    }
    if s == format!("1-{var_name}^2") || s == "1-x^2" {
        return Ok(simsym::expr::const_(rational(1, 1)) - var.pow(2));
    }
    Err(())
}

pub struct SymbolicPdfOwned {
    inner: Arc<SymbolicContinuous>,
}

impl Clone for SymbolicPdfOwned {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl Pdf for SymbolicPdfOwned {
    fn pdf(&self, x: f64) -> f64 {
        self.inner.pdf(x)
    }
}

impl HasSupport for SymbolicPdfOwned {
    fn support(&self) -> simsam::Interval {
        self.inner.support()
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_symbolic_continuous, m)?)?;
    Ok(())
}
