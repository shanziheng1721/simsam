use crate::multivar::support::HyperRect;

/// Joint probability density function on a bounded hyper-rectangle support.
pub trait PdfNd {
    fn pdf(&self, x: &[f64]) -> f64;
}

/// Log-density (up to additive constant). Prefer this for MCMC.
pub trait LogPdfNd {
    fn log_pdf(&self, x: &[f64]) -> f64;
}

/// Gradient of log-density (optional; used by HMC).
pub trait GradientLogPdfNd {
    fn grad_log_pdf(&self, x: &[f64]) -> Vec<f64>;
}

pub trait HasSupportNd {
    fn support(&self) -> &HyperRect;
}

/// Closure-backed ND PDF with explicit support.
pub struct PdfNdFn<F> {
    pub f: F,
    pub support: HyperRect,
}

impl<F> PdfNdFn<F>
where
    F: Fn(&[f64]) -> f64,
{
    pub fn new(f: F, support: HyperRect) -> Self {
        Self { f, support }
    }
}

impl<F> PdfNd for PdfNdFn<F>
where
    F: Fn(&[f64]) -> f64,
{
    fn pdf(&self, x: &[f64]) -> f64 {
        (self.f)(x)
    }
}

impl<F> LogPdfNd for PdfNdFn<F>
where
    F: Fn(&[f64]) -> f64,
{
    fn log_pdf(&self, x: &[f64]) -> f64 {
        let p = (self.f)(x);
        if p > 0.0 {
            p.ln()
        } else {
            f64::NEG_INFINITY
        }
    }
}

impl<F> HasSupportNd for PdfNdFn<F> {
    fn support(&self) -> &HyperRect {
        &self.support
    }
}

