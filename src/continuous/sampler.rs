use crate::continuous::cdf::{AffineCdf, IntegratedPdf};
use crate::continuous::hermite::HermitePpfTable;
use crate::continuous::invert::{ppf, ppf_with_pdf, InvertOptions};
use crate::continuous::traits::{Cdf, CdfFn, HasSupport, Pdf, PdfFn};
use crate::error::{BuildError, SampleError};
use crate::support::Interval;
use rand::{Rng, RngExt};

/// How to invert the CDF when sampling.
#[derive(Debug, Clone, Copy)]
pub enum PpfMethod {
    /// Bisection (+ optional Newton) on each `ppf` call.
    Bisection,
    /// Precomputed Hermite table ([`HermitePpfTable`]); faster for many samples.
    Hermite { grid_size: usize },
}

impl Default for PpfMethod {
    fn default() -> Self {
        Self::Bisection
    }
}

/// Options for building a continuous sampler from PDF or CDF.
#[derive(Debug, Clone, Copy)]
pub struct BuildOptions {
    pub quad_tolerance: f64,
    pub ppf_tolerance: f64,
    pub max_iterations: u32,
    pub use_newton: bool,
    pub ppf_method: PpfMethod,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            quad_tolerance: crate::continuous::integrate::default_quad_tol(),
            ppf_tolerance: 1e-12,
            max_iterations: 128,
            use_newton: true,
            ppf_method: PpfMethod::default(),
        }
    }
}

impl BuildOptions {
    /// SciPy-style fast inversion via a Hermite table (default grid 64).
    pub fn with_hermite(mut self, grid_size: usize) -> Self {
        self.ppf_method = PpfMethod::Hermite { grid_size };
        self
    }
}

/// Internal CDF source for inverse-transform sampling.
pub trait CdfSource: Cdf {
    fn support(&self) -> Interval;
    fn has_pdf(&self) -> bool;
    fn pdf_at(&self, x: f64) -> Option<f64>;
}

impl<P> CdfSource for IntegratedPdf<P>
where
    P: Pdf,
{
    fn support(&self) -> Interval {
        IntegratedPdf::support(self)
    }

    fn has_pdf(&self) -> bool {
        true
    }

    fn pdf_at(&self, x: f64) -> Option<f64> {
        Some(self.normalized_pdf(x))
    }
}

impl<C> CdfSource for AffineCdf<C>
where
    C: Cdf,
{
    fn support(&self) -> Interval {
        AffineCdf::support(self)
    }

    fn has_pdf(&self) -> bool {
        false
    }

    fn pdf_at(&self, _x: f64) -> Option<f64> {
        None
    }
}

struct PdfBridge<'a, D> {
    inner: &'a D,
}

impl<D> Pdf for PdfBridge<'_, D>
where
    D: CdfSource,
{
    fn pdf(&self, x: f64) -> f64 {
        self.inner.pdf_at(x).unwrap_or(0.0)
    }
}

/// Continuous distribution sampler via inverse transform (numerical PPF).
pub struct ContinuousSampler<D> {
    dist: D,
    invert: InvertOptions,
    use_newton: bool,
    hermite: Option<HermitePpfTable>,
}

impl<P> ContinuousSampler<IntegratedPdf<P>>
where
    P: Pdf,
{
    pub fn from_pdf(pdf: P, opts: BuildOptions) -> Result<Self, BuildError>
    where
        P: HasSupport,
    {
        let support = pdf.support();
        support.validate()?;
        let integrated = IntegratedPdf::new(pdf, support, opts.quad_tolerance)?;
        Self::new(integrated, support, opts, opts.use_newton)
    }
}

impl<C> ContinuousSampler<AffineCdf<C>>
where
    C: Cdf,
{
    pub fn from_cdf(cdf: C, opts: BuildOptions) -> Result<Self, BuildError>
    where
        C: HasSupport,
    {
        let support = cdf.support();
        support.validate()?;
        let affine = AffineCdf::new(cdf, support)?;
        Self::new(affine, support, opts, false)
    }
}

impl<D> ContinuousSampler<D>
where
    D: CdfSource,
{
    fn new(
        dist: D,
        support: Interval,
        opts: BuildOptions,
        use_newton: bool,
    ) -> Result<Self, BuildError> {
        let invert = InvertOptions {
            tolerance: opts.ppf_tolerance,
            max_iterations: opts.max_iterations,
        };
        let hermite = match opts.ppf_method {
            PpfMethod::Bisection => None,
            PpfMethod::Hermite { grid_size } => Some(HermitePpfTable::build(
                &dist,
                support,
                grid_size,
                invert,
            )),
        };
        Ok(Self {
            dist,
            invert,
            use_newton,
            hermite,
        })
    }

    /// Rebuild or replace the Hermite PPF table (e.g. after tuning `grid_size`).
    pub fn set_hermite_table(&mut self, grid_size: usize) {
        let support = self.dist.support();
        self.hermite = Some(HermitePpfTable::build(
            &self.dist,
            support,
            grid_size,
            self.invert,
        ));
    }

    /// Use bisection-only PPF (removes Hermite table).
    pub fn clear_hermite_table(&mut self) {
        self.hermite = None;
    }

    pub fn uses_hermite_table(&self) -> bool {
        self.hermite.is_some()
    }
    pub fn support(&self) -> Interval {
        self.dist.support()
    }

    pub fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }

    /// Normalized PDF at `x` when available.
    pub fn pdf_at(&self, x: f64) -> Option<f64> {
        self.dist.pdf_at(x)
    }

    pub fn has_pdf(&self) -> bool {
        self.dist.has_pdf()
    }

    pub fn ppf(&self, u: f64) -> Result<f64, SampleError> {
        if !(u > 0.0 && u < 1.0) {
            return Err(SampleError::QuantileOutOfRange { u });
        }
        if let Some(table) = &self.hermite {
            return Ok(table.eval(u));
        }
        let support = self.dist.support();
        if self.use_newton && self.dist.has_pdf() {
            let bridge = PdfBridge { inner: &self.dist };
            return ppf_with_pdf(&self.dist, &bridge, support, u, self.invert);
        }
        ppf(&self.dist, support, u, self.invert)
    }

    /// Draw one sample using the thread-local RNG ([`rand::rng`]).
    pub fn sample(&self) -> Result<f64, SampleError> {
        self.sample_with_rng(&mut rand::rng())
    }

    /// Draw one sample using the given RNG.
    pub fn sample_with_rng<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<f64, SampleError> {
        let u: f64 = rng.random();
        self.ppf(u)
    }

    /// Draw `n` samples using the thread-local RNG.
    pub fn sample_n(&self, n: usize) -> Result<Vec<f64>, SampleError> {
        self.sample_n_with_rng(&mut rand::rng(), n)
    }

    /// Draw `n` samples using the given RNG.
    pub fn sample_n_with_rng<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        n: usize,
    ) -> Result<Vec<f64>, SampleError> {
        (0..n).map(|_| self.sample_with_rng(rng)).collect()
    }
}

/// Build a sampler from a PDF closure on `[lo, hi]`.
pub fn from_pdf_fn<F>(
    f: F,
    support: Interval,
    opts: BuildOptions,
) -> Result<ContinuousSampler<IntegratedPdf<PdfFn<F>>>, BuildError>
where
    F: Fn(f64) -> f64,
{
    let pdf = PdfFn::new(f, support);
    ContinuousSampler::from_pdf(pdf, opts)
}

/// Build from a histogram ([`HistogramPdf`](crate::continuous::HistogramPdf)).
pub fn from_histogram(
    edges: Vec<f64>,
    counts: Vec<f64>,
    density: bool,
    opts: BuildOptions,
) -> Result<ContinuousSampler<IntegratedPdf<crate::continuous::HistogramPdf>>, BuildError> {
    let pdf = crate::continuous::HistogramPdf::new(edges, counts, density)?;
    ContinuousSampler::from_pdf(pdf, opts)
}

/// Build from a base PDF with `y = loc + scale * x`.
pub fn from_pdf_loc_scale<P>(
    pdf: P,
    loc: f64,
    scale: f64,
    opts: BuildOptions,
) -> Result<ContinuousSampler<IntegratedPdf<crate::continuous::LocScale<P>>>, BuildError>
where
    P: Pdf + HasSupport,
{
    let wrapped = crate::continuous::LocScale::new(pdf, loc, scale)?;
    ContinuousSampler::from_pdf(wrapped, opts)
}

/// Build a sampler from a CDF closure on `[lo, hi]`.
pub fn from_cdf_fn<F>(
    f: F,
    support: Interval,
    opts: BuildOptions,
) -> Result<ContinuousSampler<AffineCdf<CdfFn<F>>>, BuildError>
where
    F: Fn(f64) -> f64,
{
    let cdf = CdfFn::new(f, support);
    ContinuousSampler::from_cdf(cdf, opts)
}
