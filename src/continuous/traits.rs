use crate::support::Interval;

/// Probability density function on a finite support.
pub trait Pdf {
    fn pdf(&self, x: f64) -> f64;
}

/// Cumulative distribution function, monotone on support with `cdf(lo) ≈ 0`, `cdf(hi) ≈ 1`.
pub trait Cdf {
    fn cdf(&self, x: f64) -> f64;
}

/// Distribution support.
pub trait HasSupport {
    fn support(&self) -> Interval;
}

/// Closure-backed PDF with explicit support.
pub struct PdfFn<F> {
    pub f: F,
    pub support: Interval,
}

impl<F> PdfFn<F>
where
    F: Fn(f64) -> f64,
{
    pub fn new(f: F, support: Interval) -> Self {
        Self { f, support }
    }
}

impl<F> Pdf for PdfFn<F>
where
    F: Fn(f64) -> f64,
{
    fn pdf(&self, x: f64) -> f64 {
        (self.f)(x)
    }
}

impl<F> HasSupport for PdfFn<F> {
    fn support(&self) -> Interval {
        self.support
    }
}

/// Closure-backed CDF with explicit support.
pub struct CdfFn<F> {
    pub f: F,
    pub support: Interval,
}

impl<F> CdfFn<F>
where
    F: Fn(f64) -> f64,
{
    pub fn new(f: F, support: Interval) -> Self {
        Self { f, support }
    }
}

impl<F> Cdf for CdfFn<F>
where
    F: Fn(f64) -> f64,
{
    fn cdf(&self, x: f64) -> f64 {
        (self.f)(x)
    }
}

impl<F> HasSupport for CdfFn<F> {
    fn support(&self) -> Interval {
        self.support
    }
}
