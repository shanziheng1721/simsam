use crate::continuous::integrate::{integrate_pdf, normalize_pdf};
use crate::continuous::traits::{Cdf, HasSupport, Pdf};
use crate::error::BuildError;
use crate::support::Interval;

/// Truncate a base PDF to `[lo, hi]` and renormalize.
#[derive(Debug, Clone)]
pub struct Truncated<P> {
    inner: P,
    interval: Interval,
    norm: f64,
    quad_tol: f64,
}

impl<P> Truncated<P>
where
    P: Pdf + HasSupport,
{
    pub fn new(inner: P, interval: Interval, quad_tol: f64) -> Result<Self, BuildError> {
        interval.validate()?;
        let base = inner.support();
        base.validate()?;
        if interval.lo < base.lo || interval.hi > base.hi {
            return Err(BuildError::TruncationOutOfSupport);
        }
        let norm = normalize_pdf(&inner, interval.lo, interval.hi, quad_tol)?;
        Ok(Self {
            inner,
            interval,
            norm,
            quad_tol,
        })
    }
}

impl<P> Pdf for Truncated<P>
where
    P: Pdf,
{
    fn pdf(&self, x: f64) -> f64 {
        if !self.interval.contains(x) {
            return 0.0;
        }
        self.inner.pdf(x) / self.norm
    }
}

impl<P> Cdf for Truncated<P>
where
    P: Pdf,
{
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.interval.lo {
            return 0.0;
        }
        if x >= self.interval.hi {
            return 1.0;
        }
        integrate_pdf(&self.inner, self.interval.lo, x, self.quad_tol) / self.norm
    }
}

impl<P> HasSupport for Truncated<P> {
    fn support(&self) -> Interval {
        self.interval
    }
}
