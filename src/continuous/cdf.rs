use crate::continuous::integrate::{integrate_pdf, normalize_pdf};
use crate::continuous::traits::{Cdf, HasSupport, Pdf};
use crate::error::BuildError;
use crate::support::Interval;

/// Normalized CDF built from a user-provided [`Cdf`], affine-scaled to `[0, 1]` on support.
pub struct AffineCdf<C> {
    inner: C,
    support: Interval,
    fa: f64,
    span: f64,
}

impl<C> AffineCdf<C>
where
    C: Cdf,
{
    pub fn new(inner: C, support: Interval) -> Result<Self, BuildError> {
        support.validate()?;
        let fa = inner.cdf(support.lo);
        let fb = inner.cdf(support.hi);
        let span = fb - fa;
        if !span.is_finite() || span <= 0.0 {
            return Err(BuildError::NormalizationFailed);
        }
        Ok(Self {
            inner,
            support,
            fa,
            span,
        })
    }
}

impl<C> Cdf for AffineCdf<C>
where
    C: Cdf,
{
    fn cdf(&self, x: f64) -> f64 {
        let raw = self.inner.cdf(x);
        (raw - self.fa) / self.span
    }
}

impl<C> AffineCdf<C> {
    pub fn support(&self) -> Interval {
        self.support
    }
}

/// CDF obtained by integrating and normalizing a PDF on support.
pub struct IntegratedPdf<P> {
    pdf: P,
    support: Interval,
    norm: f64,
    quad_tol: f64,
}

impl<P> IntegratedPdf<P>
where
    P: Pdf,
{
    pub fn new(pdf: P, support: Interval, quad_tol: f64) -> Result<Self, BuildError> {
        support.validate()?;
        let norm = normalize_pdf(&pdf, support.lo, support.hi, quad_tol)?;
        Ok(Self {
            pdf,
            support,
            norm,
            quad_tol,
        })
    }
}

impl<P> Cdf for IntegratedPdf<P>
where
    P: Pdf,
{
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.support.lo {
            return 0.0;
        }
        if x >= self.support.hi {
            return 1.0;
        }
        integrate_pdf(&self.pdf, self.support.lo, x, self.quad_tol) / self.norm
    }
}

impl<P> IntegratedPdf<P>
where
    P: Pdf,
{
    pub fn support(&self) -> Interval {
        self.support
    }

    pub fn pdf(&self) -> &P {
        &self.pdf
    }

    pub fn normalized_pdf(&self, x: f64) -> f64 {
        self.pdf.pdf(x) / self.norm
    }
}

impl<P> Pdf for IntegratedPdf<P>
where
    P: Pdf,
{
    fn pdf(&self, x: f64) -> f64 {
        self.normalized_pdf(x)
    }
}

impl<P> HasSupport for IntegratedPdf<P>
where
    P: Pdf,
{
    fn support(&self) -> Interval {
        self.support
    }
}
