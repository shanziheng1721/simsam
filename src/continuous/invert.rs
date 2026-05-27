use crate::continuous::traits::{Cdf, Pdf};
use crate::error::SampleError;
use crate::support::Interval;

#[derive(Debug, Clone, Copy)]
pub struct InvertOptions {
    pub tolerance: f64,
    pub max_iterations: u32,
}

impl Default for InvertOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-12,
            max_iterations: 128,
        }
    }
}

/// Percent-point function: find `x` with `cdf(x) = u` on `support`.
pub fn ppf<C>(cdf: &C, support: Interval, u: f64, opts: InvertOptions) -> Result<f64, SampleError>
where
    C: Cdf,
{
    if !(u > 0.0 && u < 1.0) {
        return Err(SampleError::QuantileOutOfRange { u });
    }

    let fa = cdf.cdf(support.lo);
    let fb = cdf.cdf(support.hi);
    let span = fb - fa;
    if !span.is_finite() || span <= 0.0 {
        return Err(SampleError::PpfFailed);
    }
    let target = fa + u * span;

    let mut lo = support.lo;
    let mut hi = support.hi;
    for _ in 0..opts.max_iterations {
        let mid = 0.5 * (lo + hi);
        let fmid = cdf.cdf(mid);
        if (fmid - target).abs() <= opts.tolerance {
            return Ok(mid);
        }
        if fmid < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    Ok(0.5 * (lo + hi))
}

/// Newton refinement when PDF is available.
pub fn ppf_with_pdf<C, P>(
    cdf: &C,
    pdf: &P,
    support: Interval,
    u: f64,
    opts: InvertOptions,
) -> Result<f64, SampleError>
where
    C: Cdf,
    P: Pdf,
{
    let mut x = ppf(cdf, support, u, opts)?;
    let fa = cdf.cdf(support.lo);
    let fb = cdf.cdf(support.hi);
    let span = fb - fa;
    let target = fa + u * span;

    for _ in 0..8 {
        let fx = cdf.cdf(x) - target;
        if fx.abs() <= opts.tolerance {
            return Ok(x);
        }
        let dfx = pdf.pdf(x);
        if dfx.abs() < 1e-300 {
            break;
        }
        let step = fx / dfx;
        x = (x - step).clamp(support.lo, support.hi);
    }
    Ok(x)
}
