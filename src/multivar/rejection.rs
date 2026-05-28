use crate::error::{BuildError, SampleError};
use crate::multivar::support::HyperRect;
use crate::multivar::traits::{HasSupportNd, PdfNd};
use rand::{Rng, RngExt};

#[derive(Debug, Clone, Copy)]
pub struct RejectionOptions {
    /// Maximum number of proposal draws per sample.
    pub max_trials: usize,
}

impl Default for RejectionOptions {
    fn default() -> Self {
        Self { max_trials: 100_000 }
    }
}

/// Generic N-D rejection sampler on a bounded hyper-rectangle.
///
/// Requires an upper bound `pdf_max` such that `pdf(x) <= pdf_max` on the whole support.
pub struct RejectionSamplerNd<P> {
    pdf: P,
    support: HyperRect,
    pdf_max: f64,
    opts: RejectionOptions,
}

impl<P> RejectionSamplerNd<P>
where
    P: PdfNd + HasSupportNd,
{
    pub fn new(pdf: P, pdf_max: f64, opts: RejectionOptions) -> Result<Self, BuildError> {
        if !pdf_max.is_finite() || pdf_max <= 0.0 {
            return Err(BuildError::InvalidPdfMax);
        }
        let support = pdf.support().clone();
        support.validate()?;
        Ok(Self {
            pdf,
            support,
            pdf_max,
            opts,
        })
    }

    pub fn dim(&self) -> usize {
        self.support.dim()
    }

    pub fn support(&self) -> &HyperRect {
        &self.support
    }

    pub fn pdf_max(&self) -> f64 {
        self.pdf_max
    }

    pub fn sample(&self) -> Result<Vec<f64>, SampleError> {
        self.sample_with_rng(&mut rand::rng())
    }

    pub fn sample_with_rng<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Vec<f64>, SampleError> {
        let dim = self.dim();
        let mut x = vec![0.0; dim];

        for _ in 0..self.opts.max_trials {
            for i in 0..dim {
                let u: f64 = rng.random();
                x[i] = self.support.lo[i] + u * (self.support.hi[i] - self.support.lo[i]);
            }
            let fx = self.pdf.pdf(&x);
            if !fx.is_finite() || fx < 0.0 {
                continue;
            }
            let u: f64 = rng.random();
            if u * self.pdf_max <= fx {
                return Ok(x);
            }
        }
        Err(SampleError::RejectionSamplingFailed)
    }
}

