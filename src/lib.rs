//! simsam — sample from custom discrete and continuous distributions.
//!
//! Build a distribution from a PDF or CDF (closures, histograms, location-scale transforms,
//! or [simsym](https://docs.rs/simsym) symbolic expressions), then draw samples via inverse
//! transform sampling — similar to [SciPy `rv_continuous`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html).
//!
//! ## Features
//!
//! - `pdf`, `cdf`, `ppf`, `sf`, `isf`, `logpdf`, `logcdf`, `logsf`
//! - `mean`, `var`, `std`, `median`, `entropy`, `expect`, `interval`
//! - Fast sampling via [`HermitePpfTable`](continuous::HermitePpfTable) ([`BuildOptions::with_hermite`])
//! - [`LocScale`](continuous::LocScale), [`Truncated`](continuous::Truncated), [`HistogramPdf`](continuous::HistogramPdf)
//!
//! ## Limitations
//!
//! - Continuous distributions require **finite support** `[lo, hi]` (use [`Truncated`] on a wide interval).
//! - Inverse transform assumes a **unimodal** CDF on that interval.
//!
//! ## Example
//!
//! ```
//! use simsam::{from_pdf_fn, BuildOptions, Interval};
//!
//! let support = Interval::new(0.0, 1.0).unwrap();
//! let sampler = from_pdf_fn(|x| 3.0 * x * x, support, BuildOptions::default()).unwrap();
//! let x = sampler.sample().unwrap();
//! assert!((0.0..=1.0).contains(&x));
//! assert!((sampler.mean().unwrap() - 0.75).abs() < 1e-2);
//! ```

mod continuous;
mod discrete;
mod error;
mod sample;
mod support;

pub use continuous::{
    from_cdf_fn, from_histogram, from_pdf_fn, from_pdf_loc_scale, AffineCdf, BuildOptions, Cdf,
    CdfFn, CdfSource, ContinuousSampler, HasSupport, HermitePpfTable, HistogramPdf, IntegratedPdf,
    InvertOptions, LocScale, Pdf, PdfFn, PpfMethod, SymbolicContinuous, SymbolicPdfAdapter,
    Truncated, default_quad_tol,
};
pub use discrete::{CdfDiscrete, DiscreteSampler, Pmf};
pub use error::{BuildError, SampleError};
pub use support::Interval;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_pdf_mean() {
        let support = Interval::new(0.0, 1.0).unwrap();
        let sampler = from_pdf_fn(|_| 1.0, support, BuildOptions::default()).unwrap();
        let mut sum = 0.0;
        let n = 4000;
        for _ in 0..n {
            sum += sampler.sample().unwrap();
        }
        let mean = sum / n as f64;
        assert!((mean - 0.5).abs() < 0.05, "mean={mean}");
        assert!((sampler.mean().unwrap() - 0.5).abs() < 1e-3);
    }

    #[test]
    fn triangular_ppf() {
        let support = Interval::new(0.0, 1.0).unwrap();
        let sampler = from_pdf_fn(|x| 2.0 * x, support, BuildOptions::default()).unwrap();
        let x = sampler.ppf(0.25).unwrap();
        assert!((x - 0.5).abs() < 1e-3, "ppf(0.25)={x}");
        assert!((sampler.median().unwrap() - 2.0_f64.sqrt().recip()).abs() < 0.02);
    }

    #[test]
    fn cdf_only_quadratic() {
        let support = Interval::new(0.0, 1.0).unwrap();
        let sampler = from_cdf_fn(|x| x * x, support, BuildOptions::default()).unwrap();
        let x = sampler.ppf(0.25).unwrap();
        assert!((x - 0.5).abs() < 1e-3);
        assert!((sampler.sf(0.5) - 0.75).abs() < 1e-3);
    }

    #[test]
    fn discrete_bernoulli() {
        let dist = DiscreteSampler::from_pmf(vec![0.0, 1.0], vec![0.3, 0.7]).unwrap();
        assert!((dist.mean() - 0.7).abs() < 1e-10);
        let mut ones = 0;
        let n = 5000;
        for _ in 0..n {
            if dist.sample().unwrap() > 0.5 {
                ones += 1;
            }
        }
        let p = ones as f64 / n as f64;
        assert!((p - 0.7).abs() < 0.05, "p={p}");
    }

    #[test]
    fn symbolic_triangular() {
        use simsym::prelude::*;

        let x = symbol("x");
        let pdf = rational(2, 1) * x;
        let support = Interval::new(0.0, 1.0).unwrap();
        let sym = SymbolicContinuous::with_defaults(pdf, x, support).unwrap();
        let sampler = sym.sampler(BuildOptions::default()).unwrap();
        assert!((sampler.mean().unwrap() - 2.0 / 3.0).abs() < 1e-2);
    }

    #[test]
    fn hermite_matches_bisection() {
        let support = Interval::new(0.0, 1.0).unwrap();
        let opts = BuildOptions::default().with_hermite(64);
        let fast = from_pdf_fn(|x| 2.0 * x, support, opts).unwrap();
        assert!(fast.uses_hermite_table());
        let u = 0.37;
        let mut slow = from_pdf_fn(|x| 2.0 * x, support, BuildOptions::default()).unwrap();
        slow.clear_hermite_table();
        let xh = fast.ppf(u).unwrap();
        let xb = slow.ppf(u).unwrap();
        assert!((xh - xb).abs() < 0.02);
    }

    #[test]
    fn histogram_and_loc_scale() {
        let edges = vec![0.0, 0.5, 1.0];
        let counts = vec![1.0, 1.0];
        let h = from_histogram(edges, counts, false, BuildOptions::default()).unwrap();
        assert!((h.mean().unwrap() - 0.5).abs() < 1e-2);

        let inner = Interval::new(0.0, 1.0).unwrap();
        let base = from_pdf_fn(|_| 1.0, inner, BuildOptions::default()).unwrap();
        let _ = base.mean().unwrap();

        let scaled = from_pdf_loc_scale(PdfFn::new(|_| 1.0, inner), 10.0, 2.0, BuildOptions::default())
            .unwrap();
        let (lo, hi) = scaled.interval(0.9).unwrap();
        assert!(lo > 9.0 && hi < 12.0);
    }

    #[test]
    fn truncated_uniform() {
        let inner = Interval::new(0.0, 1.0).unwrap();
        let pdf = PdfFn::new(|_| 1.0, inner);
        let trunc =
            Truncated::new(pdf, Interval::new(0.25, 0.75).unwrap(), crate::default_quad_tol())
                .unwrap();
        let s = ContinuousSampler::from_pdf(trunc, BuildOptions::default()).unwrap();
        assert!((s.mean().unwrap() - 0.5).abs() < 1e-2);
    }

    #[test]
    fn rand_distribution_trait() {
        use rand::distr::Distribution as RandDist;

        let support = Interval::new(0.0, 1.0).unwrap();
        let sampler = from_pdf_fn(|_| 1.0, support, BuildOptions::default()).unwrap();
        let mut rng = rand::rng();
        let x: f64 = RandDist::sample(&sampler, &mut rng);
        assert!((0.0..=1.0).contains(&x));
    }

}
