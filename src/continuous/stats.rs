use crate::continuous::integrate::{default_quad_tol, integrate_fn};
use crate::continuous::sampler::ContinuousSampler;
use crate::continuous::CdfSource;
use crate::error::SampleError;

impl<D> ContinuousSampler<D>
where
    D: CdfSource,
{
    /// Probability density at `x` (zero outside support). Requires an underlying PDF.
    pub fn pdf(&self, x: f64) -> Option<f64> {
        self.pdf_at(x)
    }

    /// Natural logarithm of the PDF; `None` if PDF unavailable or non-positive.
    pub fn logpdf(&self, x: f64) -> Option<f64> {
        self.pdf(x).and_then(|p| {
            if p > 0.0 {
                Some(p.ln())
            } else {
                None
            }
        })
    }

    /// Log of the CDF.
    pub fn logcdf(&self, x: f64) -> f64 {
        let p = self.cdf(x).clamp(0.0, 1.0);
        if p <= 0.0 {
            f64::NEG_INFINITY
        } else {
            p.ln()
        }
    }

    /// Survival function `sf(x) = 1 - cdf(x)`, stable for large `cdf`.
    pub fn sf(&self, x: f64) -> f64 {
        let p = self.cdf(x);
        (1.0 - p).max(0.0)
    }

    /// Log survival function.
    pub fn logsf(&self, x: f64) -> f64 {
        let p = self.cdf(x).clamp(0.0, 1.0);
        if p >= 1.0 {
            f64::NEG_INFINITY
        } else {
            (1.0 - p).ln()
        }
    }

    /// Inverse survival function `isf(q) = ppf(1 - q)`.
    pub fn isf(&self, q: f64) -> Result<f64, SampleError> {
        self.ppf(1.0 - q)
    }

    /// Median `ppf(0.5)`.
    pub fn median(&self) -> Result<f64, SampleError> {
        self.ppf(0.5)
    }

    /// Mean `E[X]` via `expect(|x| x)`.
    pub fn mean(&self) -> Result<f64, SampleError> {
        self.expect(|x| x, default_quad_tol())
    }

    /// Variance `Var[X]`.
    pub fn var(&self) -> Result<f64, SampleError> {
        let m = self.mean()?;
        let m2 = self.expect(|x| x * x, default_quad_tol())?;
        Ok((m2 - m * m).max(0.0))
    }

    /// Standard deviation.
    pub fn std(&self) -> Result<f64, SampleError> {
        Ok(self.var()?.sqrt())
    }

    /// Differential entropy `H = -∫ f log f dx` (nats).
    pub fn entropy(&self) -> Result<f64, SampleError> {
        let support = self.support();
        let tol = default_quad_tol();
        let h = integrate_fn(
            &|x| {
                self.pdf(x)
                    .filter(|&p| p > 0.0)
                    .map(|p| -p * p.ln())
                    .unwrap_or(0.0)
            },
            support.lo,
            support.hi,
            tol,
        );
        if h.is_finite() {
            Ok(h)
        } else {
            Err(SampleError::IntegrationFailed)
        }
    }

    /// `E[func(X)]` on the support.
    pub fn expect(&self, func: impl Fn(f64) -> f64, tol: f64) -> Result<f64, SampleError> {
        if !self.has_pdf() {
            return Err(SampleError::PdfRequired);
        }
        let support = self.support();
        let m = integrate_fn(
            &|x| {
                self.pdf(x)
                    .map(|p| func(x) * p)
                    .unwrap_or(0.0)
            },
            support.lo,
            support.hi,
            tol,
        );
        if m.is_finite() {
            Ok(m)
        } else {
            Err(SampleError::IntegrationFailed)
        }
    }

    /// Equal-tailed confidence interval `(ppf((1-p)/2), ppf(1-(1-p)/2))`.
    pub fn interval(&self, confidence: f64) -> Result<(f64, f64), SampleError> {
        if !(confidence > 0.0 && confidence < 1.0) {
            return Err(SampleError::InvalidConfidence { confidence });
        }
        let tail = 0.5 * (1.0 - confidence);
        let lo = self.ppf(tail)?;
        let hi = self.ppf(1.0 - tail)?;
        Ok((lo, hi))
    }
}
