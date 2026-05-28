use crate::error::{BuildError, SampleError};
use crate::multivar::support::HyperRect;
use crate::multivar::traits::{HasSupportNd, PdfNd};
use rand::{Rng, RngExt};

#[derive(Debug, Clone, Copy)]
pub struct CdfMcOptions {
    pub normalization_samples: usize,
    pub cdf_samples: usize,
}

impl Default for CdfMcOptions {
    fn default() -> Self {
        Self {
            normalization_samples: 50_000,
            cdf_samples: 50_000,
        }
    }
}

/// Monte-Carlo estimator for normalization and CDF of an N-D PDF on a bounded support.
///
/// This provides *approximate* CDF values:
/// \(F(x) = \int_{lo}^{min(x,hi)} f(t) dt / \int_{lo}^{hi} f(t) dt\).
pub struct CdfMcEstimator<P> {
    pdf: P,
    support: HyperRect,
    opts: CdfMcOptions,
    z_hat: Option<f64>,
}

impl<P> CdfMcEstimator<P>
where
    P: PdfNd + HasSupportNd,
{
    pub fn new(pdf: P, opts: CdfMcOptions) -> Result<Self, BuildError> {
        let support = pdf.support().clone();
        support.validate()?;
        Ok(Self {
            pdf,
            support,
            opts,
            z_hat: None,
        })
    }

    pub fn support(&self) -> &HyperRect {
        &self.support
    }

    /// Estimate normalization constant \(Z = \int f\) via uniform sampling.
    pub fn estimate_normalization(&mut self) -> Result<f64, SampleError> {
        let z = self.estimate_normalization_with_rng(&mut rand::rng())?;
        Ok(z)
    }

    pub fn estimate_normalization_with_rng<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> Result<f64, SampleError> {
        let n = self.opts.normalization_samples.max(1);
        let dim = self.support.dim();
        let mut x = vec![0.0; dim];
        let mut sum = 0.0;
        for _ in 0..n {
            for i in 0..dim {
                let u: f64 = rng.random();
                x[i] = self.support.lo[i] + u * (self.support.hi[i] - self.support.lo[i]);
            }
            let fx = self.pdf.pdf(&x);
            if fx.is_finite() && fx >= 0.0 {
                sum += fx;
            }
        }
        let vol = volume(&self.support);
        let z_hat = (sum / n as f64) * vol;
        if !z_hat.is_finite() || z_hat <= 0.0 {
            return Err(SampleError::IntegrationFailed);
        }
        self.z_hat = Some(z_hat);
        Ok(z_hat)
    }

    /// Approximate CDF at `x` via Monte Carlo.
    pub fn cdf(&mut self, x: &[f64]) -> Result<f64, SampleError> {
        self.cdf_with_rng(&mut rand::rng(), x)
    }

    pub fn cdf_with_rng<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        x: &[f64],
    ) -> Result<f64, SampleError> {
        if x.len() != self.support.dim() {
            return Err(SampleError::IntegrationFailed);
        }
        let z = match self.z_hat {
            Some(z) => z,
            None => self.estimate_normalization_with_rng(rng)?,
        };

        let n = self.opts.cdf_samples.max(1);
        let dim = self.support.dim();
        let mut t = vec![0.0; dim];
        let mut sum = 0.0;
        for _ in 0..n {
            for i in 0..dim {
                let hi = self.support.hi[i].min(x[i]);
                let lo = self.support.lo[i];
                if hi <= lo {
                    t[i] = lo;
                    continue;
                }
                let u: f64 = rng.random();
                t[i] = lo + u * (hi - lo);
            }
            let ft = self.pdf.pdf(&t);
            if ft.is_finite() && ft >= 0.0 {
                sum += ft;
            }
        }
        let vol = volume_cdf(&self.support, x);
        let num = (sum / n as f64) * vol;
        let p = (num / z).clamp(0.0, 1.0);
        Ok(p)
    }
}

fn volume(rect: &HyperRect) -> f64 {
    rect.lo
        .iter()
        .zip(rect.hi.iter())
        .map(|(lo, hi)| hi - lo)
        .product()
}

fn volume_cdf(rect: &HyperRect, x: &[f64]) -> f64 {
    rect.lo
        .iter()
        .zip(rect.hi.iter())
        .zip(x.iter())
        .map(|((lo, hi), xi)| {
            let top = hi.min(*xi);
            (top - lo).max(0.0)
        })
        .product()
}

