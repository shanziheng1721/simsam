use crate::continuous::sampler::{BuildOptions, ContinuousSampler};
use crate::continuous::integrate::integrate_fn;
use crate::continuous::traits::{Cdf, HasSupport, Pdf};
use crate::error::BuildError;
use crate::support::Interval;
use simsym::eval::eval_f64;
use simsym::{Expr, Symbol};

/// Continuous distribution defined by a simsym expression for the PDF.
pub struct SymbolicContinuous {
    pdf: Expr,
    var: Symbol,
    support: Interval,
    cdf_expr: Option<Expr>,
    quad_tol: f64,
    norm: f64,
}

impl SymbolicContinuous {
    pub fn new(
        pdf: Expr,
        var: Symbol,
        support: Interval,
        quad_tol: f64,
    ) -> Result<Self, BuildError> {
        support.validate()?;

        let pdf_fn = |x: f64| eval_f64(&pdf, &[(var, x)]).unwrap_or(f64::NAN);
        let norm = integrate_fn(&pdf_fn, support.lo, support.hi, quad_tol);
        if !norm.is_finite() || norm <= 0.0 {
            return Err(BuildError::NormalizationFailed);
        }

        let cdf_expr = pdf.clone().integrate(var).ok();

        Ok(Self {
            pdf,
            var,
            support,
            cdf_expr,
            quad_tol,
            norm,
        })
    }

    pub fn with_defaults(pdf: Expr, var: Symbol, support: Interval) -> Result<Self, BuildError> {
        Self::new(
            pdf,
            var,
            support,
            crate::continuous::integrate::default_quad_tol(),
        )
    }

    pub fn support(&self) -> Interval {
        self.support
    }

    pub fn pdf_expr(&self) -> &Expr {
        &self.pdf
    }

    pub fn var(&self) -> Symbol {
        self.var
    }

    pub fn has_symbolic_cdf(&self) -> bool {
        self.cdf_expr.is_some()
    }

    /// Build a [`ContinuousSampler`] via inverse transform on this PDF.
    pub fn sampler(
        &self,
        opts: BuildOptions,
    ) -> Result<
        ContinuousSampler<crate::continuous::cdf::IntegratedPdf<SymbolicPdfAdapter<'_>>>,
        BuildError,
    > {
        let adapter = SymbolicPdfAdapter { inner: self };
        ContinuousSampler::from_pdf(adapter, opts)
    }
}

impl Pdf for SymbolicContinuous {
    fn pdf(&self, x: f64) -> f64 {
        let raw = eval_f64(&self.pdf, &[(self.var, x)]).unwrap_or(0.0);
        raw / self.norm
    }
}

impl Cdf for SymbolicContinuous {
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.support.lo {
            return 0.0;
        }
        if x >= self.support.hi {
            return 1.0;
        }

        if let Some(ref antiderivative) = self.cdf_expr {
            let fa = eval_f64(antiderivative, &[(self.var, self.support.lo)]).unwrap_or(0.0);
            let fx = eval_f64(antiderivative, &[(self.var, x)]).unwrap_or(fa);
            let fb = eval_f64(antiderivative, &[(self.var, self.support.hi)]).unwrap_or(fa);
            let span = fb - fa;
            if span.abs() >= 1e-300 {
                return ((fx - fa) / span).clamp(0.0, 1.0);
            }
        }

        integrate_fn(&|t| self.pdf(t), self.support.lo, x, self.quad_tol)
    }
}

impl HasSupport for SymbolicContinuous {
    fn support(&self) -> Interval {
        self.support
    }
}

/// Adapts [`SymbolicContinuous`] as a [`Pdf`] for [`ContinuousSampler::from_pdf`].
pub struct SymbolicPdfAdapter<'a> {
    inner: &'a SymbolicContinuous,
}

impl Pdf for SymbolicPdfAdapter<'_> {
    fn pdf(&self, x: f64) -> f64 {
        self.inner.pdf(x)
    }
}

impl HasSupport for SymbolicPdfAdapter<'_> {
    fn support(&self) -> Interval {
        self.inner.support
    }
}
