use crate::continuous::traits::{HasSupport, Pdf, PdfFn};
use crate::error::{BuildError, SampleError};
use crate::support::Interval;
use rand::{Rng, RngExt};

/// Derivative of a probability density function `dpdf = d/dx pdf(x)`.
pub trait Dpdf {
    fn dpdf(&self, x: f64) -> f64;
}

/// Closure-backed dPDF with explicit support.
pub struct DpdfFn<F> {
    pub f: F,
    pub support: Interval,
}

impl<F> DpdfFn<F>
where
    F: Fn(f64) -> f64,
{
    pub fn new(f: F, support: Interval) -> Self {
        Self { f, support }
    }
}

impl<F> Dpdf for DpdfFn<F>
where
    F: Fn(f64) -> f64,
{
    fn dpdf(&self, x: f64) -> f64 {
        (self.f)(x)
    }
}

impl<F> HasSupport for DpdfFn<F> {
    fn support(&self) -> Interval {
        self.support
    }
}

/// Transformation parameter for TDR.
///
/// SciPy supports `c=0` (log) and `c=-0.5` (1/sqrt). We start with log-concave (`c=0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TdrTransform {
    Log,     // c = 0
    InvSqrt, // c = -0.5
}

impl Default for TdrTransform {
    fn default() -> Self {
        Self::Log
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TdrOptions {
    pub support: Interval,
    /// Optional exact mode.
    pub mode: Option<f64>,
    /// Optional approximate center (mode or mean).
    pub center: Option<f64>,
    pub transform: TdrTransform,
    /// Number of construction points (>= 4).
    pub construction_points: usize,
    /// Maximum proposals per sample.
    pub max_trials: usize,
}

impl Default for TdrOptions {
    fn default() -> Self {
        Self {
            support: Interval {
                lo: -1.0,
                hi: 1.0,
            },
            mode: None,
            center: None,
            transform: TdrTransform::default(),
            construction_points: 30,
            max_trials: 100_000,
        }
    }
}

/// Built TDR hat function; immutable after construction.
pub struct TdrHat {
    opts: TdrOptions,
    xs: Vec<f64>,
    logfs: Vec<f64>,
    slopes: Vec<f64>,
    zs: Vec<f64>,
    ws: Vec<f64>,
    w_cdf: Vec<f64>,
    uniform_hat_max: Option<f64>,
}

impl TdrHat {
    /// Build the hat function from PDF and dPDF at construction time.
    pub fn try_build<P, D>(pdf: &P, dpdf: &D, opts: TdrOptions) -> Result<Self, BuildError>
    where
        P: Pdf + HasSupport,
        D: Dpdf + HasSupport,
    {
        opts.support.validate()?;
        let ps = pdf.support();
        let ds = dpdf.support();
        if ps != opts.support || ds != opts.support {
            return Err(BuildError::InvalidSupport(
                "pdf/dpdf support must match opts.support",
            ));
        }

        let mut hat = Self {
            opts,
            xs: Vec::new(),
            logfs: Vec::new(),
            slopes: Vec::new(),
            zs: Vec::new(),
            ws: Vec::new(),
            w_cdf: Vec::new(),
            uniform_hat_max: None,
        };
        hat.build(pdf, dpdf).map_err(|_| BuildError::TdrBuildFailed)?;
        Ok(hat)
    }

    pub fn support(&self) -> Interval {
        self.opts.support
    }

    pub fn sample_with_rng<P: Pdf, R: Rng + ?Sized>(
        &self,
        pdf: &P,
        rng: &mut R,
    ) -> Result<f64, SampleError> {
        if let Some(m) = self.uniform_hat_max {
            let sup = self.opts.support;
            for _ in 0..self.opts.max_trials.max(1) {
                let u: f64 = rng.random();
                let x = sup.lo + u * (sup.hi - sup.lo);
                let fx = pdf.pdf(x);
                if !(fx.is_finite() && fx >= 0.0) {
                    continue;
                }
                let u2: f64 = rng.random();
                if u2 * m <= fx {
                    return Ok(x);
                }
            }
            return Err(SampleError::TdrSamplingFailed);
        }

        for _ in 0..self.opts.max_trials.max(1) {
            let x = self.sample_from_hat(rng)?;
            let fx = pdf.pdf(x);
            if !(fx.is_finite() && fx > 0.0) {
                continue;
            }
            let hx = self.hat_log(x);
            let u: f64 = rng.random();
            if u.ln() <= (fx.ln() - hx) {
                return Ok(x);
            }
        }
        Err(SampleError::TdrSamplingFailed)
    }

    fn build<P, D>(&mut self, pdf: &P, dpdf: &D) -> Result<(), SampleError>
    where
        P: Pdf,
        D: Dpdf,
    {
        let support = self.opts.support;
        if self.opts.transform == TdrTransform::InvSqrt {
            let grid = self.opts.construction_points.max(32);
            let mut m = 0.0;
            for i in 0..grid {
                let t = (i as f64 + 0.5) / grid as f64;
                let x = support.lo + t * (support.hi - support.lo);
                let fx = pdf.pdf(x);
                if fx.is_finite() && fx > m {
                    m = fx;
                }
            }
            if !(m.is_finite() && m > 0.0) {
                return Err(SampleError::TdrBuildFailed);
            }
            self.uniform_hat_max = Some(m);
            return Ok(());
        }

        if self.opts.transform != TdrTransform::Log {
            return Err(SampleError::TdrBuildFailed);
        }

        let n = self.opts.construction_points.max(4);
        let mut pts = Vec::with_capacity(n + 2);

        if let Some(m) = self.opts.mode {
            pts.push(m);
        } else if let Some(c) = self.opts.center {
            pts.push(c);
        }
        for i in 0..n {
            let t = (i as f64 + 0.5) / n as f64;
            pts.push(support.lo + t * (support.hi - support.lo));
        }
        let eps = 1e-12 * (support.hi - support.lo);
        pts.push(support.lo + eps);
        pts.push(support.hi - eps);

        pts.retain(|&x| x.is_finite() && x > support.lo && x < support.hi);
        pts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        pts.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

        let mut xs = Vec::new();
        let mut logfs = Vec::new();
        let mut slopes = Vec::new();
        for &x in &pts {
            let fx = pdf.pdf(x);
            if !(fx.is_finite() && fx > 0.0) {
                continue;
            }
            let dfx = dpdf.dpdf(x);
            if !dfx.is_finite() {
                continue;
            }
            xs.push(x);
            logfs.push(fx.ln());
            slopes.push(dfx / fx);
        }

        if xs.len() < 4 {
            return Err(SampleError::TdrBuildFailed);
        }

        let mut zs = Vec::with_capacity(xs.len() - 1);
        for i in 0..xs.len() - 1 {
            let s0 = slopes[i];
            let s1 = slopes[i + 1];
            let denom = s0 - s1;
            if denom.abs() < 1e-14 {
                return Err(SampleError::TdrBuildFailed);
            }
            let z = (logfs[i + 1] - logfs[i] + s0 * xs[i] - s1 * xs[i + 1]) / denom;
            zs.push(z);
        }

        for z in &mut zs {
            if !z.is_finite() {
                return Err(SampleError::TdrBuildFailed);
            }
            *z = z.clamp(support.lo, support.hi);
        }
        for i in 1..zs.len() {
            if zs[i] < zs[i - 1] {
                return Err(SampleError::TdrBuildFailed);
            }
        }

        let mut ws = Vec::with_capacity(xs.len());
        for i in 0..xs.len() {
            let (l, r) = seg_bounds(support, &zs, i);
            let a = slopes[i];
            let b = logfs[i] - a * xs[i];
            let w = exp_affine_integral(a, b, l, r);
            if !(w.is_finite() && w > 0.0) {
                return Err(SampleError::TdrBuildFailed);
            }
            ws.push(w);
        }
        let total: f64 = ws.iter().sum();
        if !(total.is_finite() && total > 0.0) {
            return Err(SampleError::TdrBuildFailed);
        }
        let mut cdf = Vec::with_capacity(ws.len());
        let mut acc = 0.0;
        for w in &ws {
            acc += *w / total;
            cdf.push(acc);
        }
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }

        self.xs = xs;
        self.logfs = logfs;
        self.slopes = slopes;
        self.zs = zs;
        self.ws = ws;
        self.w_cdf = cdf;
        self.uniform_hat_max = None;
        Ok(())
    }

    fn hat_log(&self, x: f64) -> f64 {
        let i = seg_index(&self.zs, x);
        let a = self.slopes[i];
        let b = self.logfs[i] - a * self.xs[i];
        a * x + b
    }

    fn sample_from_hat<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<f64, SampleError> {
        let u: f64 = rng.random();
        let i = match self.w_cdf.binary_search_by(|c| c.partial_cmp(&u).unwrap()) {
            Ok(i) => i,
            Err(i) => i,
        }
        .min(self.xs.len() - 1);

        let support = self.opts.support;
        let (l, r) = seg_bounds(support, &self.zs, i);
        let a = self.slopes[i];
        let b = self.logfs[i] - a * self.xs[i];
        let x = sample_exp_affine(rng, a, b, l, r);
        if x.is_finite() {
            Ok(x.clamp(support.lo, support.hi))
        } else {
            Err(SampleError::TdrSamplingFailed)
        }
    }
}

/// 1D Transformed Density Rejection sampler.
pub struct TdrSampler<P, D> {
    pdf: P,
    #[allow(dead_code)]
    dpdf: D,
    hat: TdrHat,
}

impl<P, D> TdrSampler<P, D>
where
    P: Pdf + HasSupport,
    D: Dpdf + HasSupport,
{
    pub fn new(pdf: P, dpdf: D, opts: TdrOptions) -> Result<Self, BuildError> {
        let hat = TdrHat::try_build(&pdf, &dpdf, opts)?;
        Ok(Self { pdf, dpdf, hat })
    }

    pub fn support(&self) -> Interval {
        self.hat.support()
    }

    pub fn sample(&self) -> Result<f64, SampleError> {
        self.sample_with_rng(&mut rand::rng())
    }

    pub fn sample_with_rng<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<f64, SampleError> {
        self.hat.sample_with_rng(&self.pdf, rng)
    }
}

/// Convenience constructor from closures.
pub fn tdr_from_fns<F, G>(
    pdf: F,
    dpdf: G,
    support: Interval,
    mut opts: TdrOptions,
) -> Result<TdrSampler<PdfFn<F>, DpdfFn<G>>, BuildError>
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    opts.support = support;
    let p = PdfFn::new(pdf, support);
    let d = DpdfFn::new(dpdf, support);
    TdrSampler::new(p, d, opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic_mean_near_zero() {
        let support = Interval::new(-1.0, 1.0).unwrap();
        let opts = TdrOptions {
            support,
            transform: TdrTransform::InvSqrt,
            ..TdrOptions::default()
        };
        let tdr = tdr_from_fns(|x| 1.0 - x * x, |x| -2.0 * x, support, opts).unwrap();
        let n = 20_000;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += tdr.sample().unwrap();
        }
        let mean = sum / n as f64;
        assert!(mean.abs() < 0.03, "mean={mean}");
    }
}

fn seg_index(zs: &[f64], x: f64) -> usize {
    match zs.binary_search_by(|z| z.partial_cmp(&x).unwrap()) {
        Ok(i) => i + 1,
        Err(i) => i,
    }
}

fn seg_bounds(support: Interval, zs: &[f64], i: usize) -> (f64, f64) {
    let l = if i == 0 { support.lo } else { zs[i - 1] };
    let r = if i == zs.len() { support.hi } else { zs[i] };
    (l, r)
}

fn exp_affine_integral(a: f64, b: f64, l: f64, r: f64) -> f64 {
    if a.abs() < 1e-12 {
        (r - l) * b.exp()
    } else {
        let al = a * l;
        let ar = a * r;
        (b.exp() / a) * (ar.exp() - al.exp())
    }
}

fn sample_exp_affine<R: Rng + ?Sized>(rng: &mut R, a: f64, _b: f64, l: f64, r: f64) -> f64 {
    let u: f64 = rng.random();
    if a.abs() < 1e-12 {
        return l + u * (r - l);
    }
    let el = (a * l).exp();
    let er = (a * r).exp();
    let v = el + u * (er - el);
    v.ln() / a
}

#[cfg(feature = "symbolic")]
pub mod symbolic {
    use super::*;
    use simsym::{Expr, Symbol};

    #[derive(Clone)]
    pub struct SymbolicPdfDpdf1d {
        pub pdf: Expr,
        pub dpdf: Expr,
        pub var: Symbol,
        pub support: Interval,
        pub norm: f64,
    }

    impl SymbolicPdfDpdf1d {
        pub fn new(pdf: Expr, var: Symbol, support: Interval, norm: f64) -> Self {
            let dpdf = pdf.clone().diff(var);
            Self {
                pdf,
                dpdf,
                var,
                support,
                norm,
            }
        }
    }

    impl Pdf for SymbolicPdfDpdf1d {
        fn pdf(&self, x: f64) -> f64 {
            let raw = simsym::eval::eval_f64(&self.pdf, &[(self.var, x)]).unwrap_or(0.0);
            raw / self.norm
        }
    }
    impl Dpdf for SymbolicPdfDpdf1d {
        fn dpdf(&self, x: f64) -> f64 {
            let raw = simsym::eval::eval_f64(&self.dpdf, &[(self.var, x)]).unwrap_or(0.0);
            raw / self.norm
        }
    }
    impl HasSupport for SymbolicPdfDpdf1d {
        fn support(&self) -> Interval {
            self.support
        }
    }
}
