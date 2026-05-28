use crate::error::BuildError;
use crate::multivar::support::HyperRect;
use crate::multivar::traits::{GradientLogPdfNd, HasSupportNd, LogPdfNd, PdfNd};
use simsym::eval::eval_f64;
use simsym::{Expr, Symbol};

/// Multi-variable symbolic PDF backed by simsym `Expr`.
pub struct SymbolicPdfNd {
    pdf: Expr,
    vars: Vec<Symbol>,
    support: HyperRect,
}

impl SymbolicPdfNd {
    pub fn new(pdf: Expr, vars: Vec<Symbol>, support: HyperRect) -> Result<Self, BuildError> {
        if vars.is_empty() {
            return Err(BuildError::InvalidDimension);
        }
        if vars.len() != support.dim() {
            return Err(BuildError::InvalidSupport(
                "vars length must match support dimension",
            ));
        }
        support.validate()?;
        Ok(Self { pdf, vars, support })
    }

    pub fn vars(&self) -> &[Symbol] {
        &self.vars
    }

    pub fn pdf_expr(&self) -> &Expr {
        &self.pdf
    }
}

impl PdfNd for SymbolicPdfNd {
    fn pdf(&self, x: &[f64]) -> f64 {
        if x.len() != self.vars.len() {
            return f64::NAN;
        }
        // Build env pairs; simsym currently takes &[(Symbol, f64)].
        let mut env: Vec<(Symbol, f64)> = Vec::with_capacity(self.vars.len());
        for (s, &v) in self.vars.iter().zip(x.iter()) {
            env.push((*s, v));
        }
        eval_f64(&self.pdf, &env).unwrap_or(f64::NAN)
    }
}

impl LogPdfNd for SymbolicPdfNd {
    fn log_pdf(&self, x: &[f64]) -> f64 {
        let p = self.pdf(x);
        if p > 0.0 {
            p.ln()
        } else {
            f64::NEG_INFINITY
        }
    }
}

impl GradientLogPdfNd for SymbolicPdfNd {
    fn grad_log_pdf(&self, x: &[f64]) -> Vec<f64> {
        if x.len() != self.vars.len() {
            return vec![f64::NAN; self.vars.len()];
        }
        // d/dvar log(pdf) = (pdf' / pdf)
        let p = self.pdf(x);
        if !(p.is_finite() && p > 0.0) {
            return vec![f64::NEG_INFINITY; self.vars.len()];
        }
        let mut out = Vec::with_capacity(self.vars.len());
        let env: Vec<(Symbol, f64)> = self
            .vars
            .iter()
            .copied()
            .zip(x.iter().copied())
            .collect();
        for &var in self.vars.iter() {
            let dp = self
                .pdf
                .clone()
                .diff(var)
                .eval_f64(&env)
                .unwrap_or(f64::NAN);
            out.push(dp / p);
        }
        out
    }
}

impl HasSupportNd for SymbolicPdfNd {
    fn support(&self) -> &HyperRect {
        &self.support
    }
}

