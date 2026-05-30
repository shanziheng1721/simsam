use crate::continuous::cdf::{AffineCdf, IntegratedPdf};
use crate::continuous::dpdf::NumericalDpdf;
use crate::continuous::hermite::HermitePpfTable;
use crate::continuous::invert::{ppf, ppf_with_pdf, InvertOptions};
use crate::continuous::tdr::{DpdfFn, TdrHat, TdrOptions, TdrTransform};
use crate::continuous::traits::{Cdf, CdfFn, HasSupport, Pdf, PdfFn};
use crate::error::{BuildError, SampleError};
use crate::support::Interval;
use rand::{Rng, RngExt};

/// How to invert the CDF when computing `ppf` (inverse transform sampling path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// TDR configuration used when [`SampleMethod::Tdr`] is selected.
#[derive(Debug, Clone, Copy)]
pub struct TdrBuildConfig {
    pub transform: TdrTransform,
    pub construction_points: usize,
    pub mode: Option<f64>,
    pub center: Option<f64>,
    pub max_trials: usize,
    /// Relative support width for central-difference dPDF: `h = rel * (hi - lo)`.
    pub dpdf_rel_step: f64,
}

impl Default for TdrBuildConfig {
    fn default() -> Self {
        Self {
            transform: TdrTransform::InvSqrt,
            construction_points: 30,
            mode: None,
            center: None,
            max_trials: 100_000,
            dpdf_rel_step: 1e-4,
        }
    }
}

impl From<TdrBuildConfig> for TdrOptions {
    fn from(cfg: TdrBuildConfig) -> Self {
        Self {
            support: Interval {
                lo: -1.0,
                hi: 1.0,
            },
            mode: cfg.mode,
            center: cfg.center,
            transform: cfg.transform,
            construction_points: cfg.construction_points,
            max_trials: cfg.max_trials,
        }
    }
}

/// How to draw samples from a continuous distribution.
#[derive(Debug, Clone, Copy)]
pub enum SampleMethod {
    /// Numerical inverse CDF (bisection or Hermite table).
    Inverse(PpfMethod),
    /// Transformed density rejection; dPDF from numerical diff or explicit provider.
    Tdr(TdrBuildConfig),
}

impl Default for SampleMethod {
    fn default() -> Self {
        Self::Inverse(PpfMethod::default())
    }
}

/// Options for building a continuous sampler from PDF or CDF.
#[derive(Debug, Clone, Copy)]
pub struct BuildOptions {
    pub quad_tolerance: f64,
    pub ppf_tolerance: f64,
    pub max_iterations: u32,
    pub use_newton: bool,
    pub sample_method: SampleMethod,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            quad_tolerance: crate::continuous::integrate::default_quad_tol(),
            ppf_tolerance: 1e-12,
            max_iterations: 128,
            use_newton: true,
            sample_method: SampleMethod::default(),
        }
    }
}

impl BuildOptions {
    /// SciPy-style fast inversion via a Hermite table (default grid 64).
    pub fn with_hermite(mut self, grid_size: usize) -> Self {
        self.sample_method = SampleMethod::Inverse(PpfMethod::Hermite { grid_size });
        self
    }

    /// Sample via TDR with automatic numerical dPDF (default InvSqrt transform).
    pub fn with_tdr(mut self) -> Self {
        self.sample_method = SampleMethod::Tdr(TdrBuildConfig::default());
        self
    }

    /// Sample via TDR with custom configuration.
    pub fn with_tdr_config(mut self, cfg: TdrBuildConfig) -> Self {
        self.sample_method = SampleMethod::Tdr(cfg);
        self
    }

    /// PPF method used for `ppf` / statistics (always inverse transform).
    pub fn ppf_method(&self) -> PpfMethod {
        match self.sample_method {
            SampleMethod::Inverse(m) => m,
            SampleMethod::Tdr(_) => PpfMethod::Bisection,
        }
    }
}

enum SamplingBackend {
    Invert {
        invert: InvertOptions,
        use_newton: bool,
        hermite: Option<HermitePpfTable>,
    },
    Tdr(TdrHat),
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

/// Continuous distribution sampler.
///
/// Sampling uses [`SampleMethod`] (inverse transform or TDR). `ppf` and statistics always
/// use numerical inverse CDF.
pub struct ContinuousSampler<D> {
    dist: D,
    backend: SamplingBackend,
}

impl<P> ContinuousSampler<IntegratedPdf<P>>
where
    P: Pdf + HasSupport,
{
    pub fn from_pdf(pdf: P, opts: BuildOptions) -> Result<Self, BuildError>
    where
        P: HasSupport,
    {
        let support = pdf.support();
        support.validate()?;
        let integrated = IntegratedPdf::new(pdf, support, opts.quad_tolerance)?;
        let backend = build_backend(&integrated, support, opts)?;
        Ok(Self {
            dist: integrated,
            backend,
        })
    }

    pub fn from_pdf_with_dpdf<DpdfT>(
        pdf: P,
        dpdf: DpdfT,
        opts: BuildOptions,
    ) -> Result<ContinuousSampler<IntegratedPdf<P>>, BuildError>
    where
        P: HasSupport,
        DpdfT: crate::continuous::tdr::Dpdf + HasSupport,
    {
        let support = pdf.support();
        support.validate()?;
        let integrated = IntegratedPdf::new(pdf, support, opts.quad_tolerance)?;
        let backend = build_backend_with_dpdf(&integrated, support, opts, &dpdf)?;
        Ok(Self {
            dist: integrated,
            backend,
        })
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
        let backend = build_invert_backend(&affine, support, opts)?;
        Ok(Self {
            dist: affine,
            backend,
        })
    }
}

fn build_invert_backend<D: CdfSource>(
    dist: &D,
    support: Interval,
    opts: BuildOptions,
) -> Result<SamplingBackend, BuildError> {
    if matches!(opts.sample_method, SampleMethod::Tdr(_)) {
        return Err(BuildError::InvalidSupport(
            "TDR sampling requires a PDF (use from_pdf_fn, not from_cdf_fn)",
        ));
    }
    let invert = InvertOptions {
        tolerance: opts.ppf_tolerance,
        max_iterations: opts.max_iterations,
    };
    let hermite = match opts.ppf_method() {
        PpfMethod::Bisection => None,
        PpfMethod::Hermite { grid_size } => Some(HermitePpfTable::build(
            dist,
            support,
            grid_size,
            invert,
        )),
    };
    Ok(SamplingBackend::Invert {
        invert,
        use_newton: opts.use_newton,
        hermite,
    })
}

fn build_backend<P>(
    integrated: &IntegratedPdf<P>,
    support: Interval,
    opts: BuildOptions,
) -> Result<SamplingBackend, BuildError>
where
    P: Pdf,
{
    match opts.sample_method {
        SampleMethod::Tdr(cfg) => {
            let mut tdr_opts: TdrOptions = cfg.into();
            tdr_opts.support = support;
            let nd = NumericalDpdf::new(integrated, cfg.dpdf_rel_step);
            let hat = TdrHat::try_build(integrated, &nd, tdr_opts)?;
            Ok(SamplingBackend::Tdr(hat))
        }
        SampleMethod::Inverse(_) => build_invert_backend(integrated, support, opts),
    }
}

fn build_backend_with_dpdf<P, D>(
    integrated: &IntegratedPdf<P>,
    support: Interval,
    opts: BuildOptions,
    dpdf: &D,
) -> Result<SamplingBackend, BuildError>
where
    P: Pdf,
    D: crate::continuous::tdr::Dpdf + HasSupport,
{
    match opts.sample_method {
        SampleMethod::Tdr(cfg) => {
            let mut tdr_opts: TdrOptions = cfg.into();
            tdr_opts.support = support;
            let hat = TdrHat::try_build(integrated, dpdf, tdr_opts)?;
            Ok(SamplingBackend::Tdr(hat))
        }
        SampleMethod::Inverse(_) => build_invert_backend(integrated, support, opts),
    }
}

impl<D> ContinuousSampler<D>
where
    D: CdfSource,
{
    /// Rebuild or replace the Hermite PPF table (inverse-transform path only).
    pub fn set_hermite_table(&mut self, grid_size: usize) {
        if let SamplingBackend::Invert {
            invert,
            hermite,
            ..
        } = &mut self.backend
        {
            let support = self.dist.support();
            *hermite = Some(HermitePpfTable::build(
                &self.dist,
                support,
                grid_size,
                *invert,
            ));
        }
    }

    /// Use bisection-only PPF (removes Hermite table).
    pub fn clear_hermite_table(&mut self) {
        if let SamplingBackend::Invert { hermite, .. } = &mut self.backend {
            *hermite = None;
        }
    }

    pub fn uses_hermite_table(&self) -> bool {
        matches!(
            self.backend,
            SamplingBackend::Invert {
                hermite: Some(_),
                ..
            }
        )
    }

    pub fn uses_tdr(&self) -> bool {
        matches!(self.backend, SamplingBackend::Tdr(_))
    }

    pub fn support(&self) -> Interval {
        self.dist.support()
    }

    pub fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }

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
        let support = self.dist.support();
        match &self.backend {
            SamplingBackend::Invert {
                invert,
                use_newton,
                hermite,
            } => {
                if let Some(table) = hermite {
                    return Ok(table.eval(u));
                }
                if *use_newton && self.dist.has_pdf() {
                    let bridge = PdfBridge { inner: &self.dist };
                    return ppf_with_pdf(&self.dist, &bridge, support, u, *invert);
                }
                ppf(&self.dist, support, u, *invert)
            }
            SamplingBackend::Tdr(_) => {
                let invert = InvertOptions {
                    tolerance: 1e-12,
                    max_iterations: 128,
                };
                if self.dist.has_pdf() {
                    let bridge = PdfBridge { inner: &self.dist };
                    ppf_with_pdf(&self.dist, &bridge, support, u, invert)
                } else {
                    ppf(&self.dist, support, u, invert)
                }
            }
        }
    }

    pub fn sample(&self) -> Result<f64, SampleError> {
        self.sample_with_rng(&mut rand::rng())
    }

    pub fn sample_with_rng<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<f64, SampleError> {
        match &self.backend {
            SamplingBackend::Invert { .. } => {
                let u: f64 = rng.random();
                self.ppf(u)
            }
            SamplingBackend::Tdr(hat) => {
                if self.dist.has_pdf() {
                    hat.sample_with_rng(&PdfBridge { inner: &self.dist }, rng)
                } else {
                    Err(SampleError::PdfRequired)
                }
            }
        }
    }

    pub fn sample_n(&self, n: usize) -> Result<Vec<f64>, SampleError> {
        self.sample_n_with_rng(&mut rand::rng(), n)
    }

    pub fn sample_n_with_rng<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        n: usize,
    ) -> Result<Vec<f64>, SampleError> {
        (0..n).map(|_| self.sample_with_rng(rng)).collect()
    }
}

/// Build a sampler from a PDF closure on `[lo, hi]` with default options.
pub fn from_pdf_fn<F>(
    f: F,
    support: Interval,
) -> Result<ContinuousSampler<IntegratedPdf<PdfFn<F>>>, BuildError>
where
    F: Fn(f64) -> f64,
{
    from_pdf_fn_with_options(f, support, BuildOptions::default())
}

/// Build a sampler from a PDF closure with custom [`BuildOptions`].
pub fn from_pdf_fn_with_options<F>(
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

/// Build a sampler from PDF and explicit dPDF closures.
pub fn from_pdf_dpdf_fn<F, G>(
    f: F,
    g: G,
    support: Interval,
    opts: BuildOptions,
) -> Result<ContinuousSampler<IntegratedPdf<PdfFn<F>>>, BuildError>
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    let pdf = PdfFn::new(f, support);
    let dpdf = DpdfFn::new(g, support);
    ContinuousSampler::from_pdf_with_dpdf(pdf, dpdf, opts)
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn tdr_auto_dpdf_quadratic_mean() {
        let support = Interval::new(-1.0, 1.0).unwrap();
        let cfg = TdrBuildConfig {
            transform: TdrTransform::InvSqrt,
            ..TdrBuildConfig::default()
        };
        let opts = BuildOptions::default().with_tdr_config(cfg);
        let sampler = from_pdf_fn_with_options(|x| 1.0 - x * x, support, opts).unwrap();
        assert!(sampler.uses_tdr());
        let n = 10_000;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += sampler.sample().unwrap();
        }
        assert!((sum / n as f64).abs() < 0.04);
    }

    #[test]
    fn sampling_methods_triangle_mean() {
        let support = Interval::new(0.0, 1.0).unwrap();
        let expected = 2.0 / 3.0;
        let n = 5_000;

        let bisect = from_pdf_fn(|x| 2.0 * x, support).unwrap();
        let mut s1 = 0.0;
        for _ in 0..n {
            s1 += bisect.sample().unwrap();
        }

        let hermite = from_pdf_fn_with_options(
            |x| 2.0 * x,
            support,
            BuildOptions::default().with_hermite(64),
        )
        .unwrap();
        let mut s2 = 0.0;
        for _ in 0..n {
            s2 += hermite.sample().unwrap();
        }

        let tdr = from_pdf_fn_with_options(
            |x| 2.0 * x,
            support,
            BuildOptions::default().with_tdr_config(TdrBuildConfig {
                transform: TdrTransform::InvSqrt,
                ..TdrBuildConfig::default()
            }),
        )
        .unwrap();
        let mut s3 = 0.0;
        for _ in 0..n {
            s3 += tdr.sample().unwrap();
        }

        assert!((s1 / n as f64 - expected).abs() < 0.05);
        assert!((s2 / n as f64 - expected).abs() < 0.05);
        assert!((s3 / n as f64 - expected).abs() < 0.05);
    }

    #[test]
    fn from_pdf_dpdf_fn_matches_tdr_from_fns() {
        use crate::continuous::tdr::tdr_from_fns;
        use crate::continuous::TdrOptions;

        let support = Interval::new(-1.0, 1.0).unwrap();
        let tdr_opts = TdrOptions {
            support,
            transform: TdrTransform::InvSqrt,
            ..TdrOptions::default()
        };

        let sampler = from_pdf_dpdf_fn(
            |x| 1.0 - x * x,
            |x| -2.0 * x,
            support,
            BuildOptions::default().with_tdr_config(TdrBuildConfig {
                transform: TdrTransform::InvSqrt,
                ..TdrBuildConfig::default()
            }),
        )
        .unwrap();
        let tdr = tdr_from_fns(|x| 1.0 - x * x, |x| -2.0 * x, support, tdr_opts).unwrap();

        let mut rng_a = ChaCha8Rng::seed_from_u64(99);
        let mut rng_b = ChaCha8Rng::seed_from_u64(99);
        let mut sa = 0.0;
        let mut sb = 0.0;
        let n = 2000;
        for _ in 0..n {
            sa += sampler.sample_with_rng(&mut rng_a).unwrap();
            sb += tdr.sample_with_rng(&mut rng_b).unwrap();
        }
        assert!((sa / n as f64).abs() < 0.04);
        assert!((sb / n as f64).abs() < 0.04);
    }
}
