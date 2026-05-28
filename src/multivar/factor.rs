use crate::error::SampleError;
use rand::Rng;

/// Conditional factorization `x0 ~ f0()`, `x1 ~ f1(x0)`, ..., `x_{n-1} ~ f_{n-1}(x0..x_{n-2})`.
///
/// This is the minimal building block for models where you can sample each conditional
/// distribution directly (possibly via existing 1D samplers).
pub trait ConditionalFactorization {
    fn dim(&self) -> usize;
    fn sample_i<R: Rng + ?Sized>(&self, rng: &mut R, i: usize, prefix: &[f64]) -> f64;
}

pub struct ConditionalFactorSampler<F> {
    inner: F,
}

impl<F> ConditionalFactorSampler<F>
where
    F: ConditionalFactorization,
{
    pub fn new(inner: F) -> Self {
        Self { inner }
    }

    pub fn sample(&self) -> Result<Vec<f64>, SampleError> {
        self.sample_with_rng(&mut rand::rng())
    }

    pub fn sample_with_rng<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Vec<f64>, SampleError> {
        let dim = self.inner.dim();
        if dim == 0 {
            return Err(SampleError::McmcFailed);
        }
        let mut out = Vec::with_capacity(dim);
        for i in 0..dim {
            let v = self.inner.sample_i(rng, i, &out);
            out.push(v);
        }
        Ok(out)
    }
}

