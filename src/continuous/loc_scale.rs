use crate::continuous::traits::{HasSupport, Pdf};
use crate::error::BuildError;
use crate::support::Interval;

/// Location-scale transform: `Y = loc + scale * X` for base PDF on `base_support`.
#[derive(Debug, Clone)]
pub struct LocScale<P> {
    inner: P,
    loc: f64,
    scale: f64,
    base_support: Interval,
    support: Interval,
}

impl<P> LocScale<P>
where
    P: Pdf + HasSupport,
{
    pub fn new(inner: P, loc: f64, scale: f64) -> Result<Self, BuildError> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(BuildError::InvalidScale);
        }
        let base_support = inner.support();
        base_support.validate()?;
        let support = Interval::new(
            loc + scale * base_support.lo,
            loc + scale * base_support.hi,
        )?;
        Ok(Self {
            inner,
            loc,
            scale,
            base_support,
            support,
        })
    }

    pub fn loc(&self) -> f64 {
        self.loc
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn inner(&self) -> &P {
        &self.inner
    }
}

impl<P> Pdf for LocScale<P>
where
    P: Pdf,
{
    fn pdf(&self, y: f64) -> f64 {
        let x = (y - self.loc) / self.scale;
        if x < self.base_support.lo || x > self.base_support.hi {
            return 0.0;
        }
        self.inner.pdf(x) / self.scale
    }
}

impl<P> HasSupport for LocScale<P> {
    fn support(&self) -> Interval {
        self.support
    }
}
