use crate::error::{BuildError, SampleError};
use rand::{Rng, RngExt};

/// Finite discrete distribution sampler via inverse transform.
#[derive(Debug, Clone)]
pub struct DiscreteSampler {
    points: Vec<f64>,
    cdf: Vec<f64>,
}

impl DiscreteSampler {
    /// Build from support points and unnormalized masses (PMF).
    pub fn from_pmf(points: Vec<f64>, pmf: Vec<f64>) -> Result<Self, BuildError> {
        if points.is_empty() || pmf.is_empty() {
            return Err(BuildError::EmptyDiscreteSupport);
        }
        if points.len() != pmf.len() {
            return Err(BuildError::InvalidSupport(
                "points and pmf must have the same length",
            ));
        }
        for &p in &pmf {
            if !p.is_finite() || p < 0.0 {
                return Err(BuildError::NonPositiveMass);
            }
        }
        let sum: f64 = pmf.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(BuildError::NormalizationFailed);
        }
        let mut cdf = Vec::with_capacity(pmf.len());
        let mut acc = 0.0;
        for &p in &pmf {
            acc += p / sum;
            cdf.push(acc);
        }
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }
        Ok(Self { points, cdf })
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn points(&self) -> &[f64] {
        &self.points
    }

    pub fn point(&self, index: usize) -> f64 {
        self.points[index]
    }

    pub fn cdf_at(&self, index: usize) -> f64 {
        self.cdf[index]
    }

    pub fn ppf(&self, u: f64) -> Result<f64, SampleError> {
        if !(u > 0.0 && u < 1.0) {
            return Err(SampleError::QuantileOutOfRange { u });
        }
        match self.cdf.binary_search_by(|c| c.partial_cmp(&u).unwrap()) {
            Ok(i) => Ok(self.points[i]),
            Err(i) => Ok(self.points[i]),
        }
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
