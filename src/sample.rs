use crate::continuous::ContinuousSampler;
use crate::discrete::DiscreteSampler;
use rand::distr::Distribution;
use rand::Rng;

impl<D> Distribution<f64> for ContinuousSampler<D>
where
    D: crate::continuous::CdfSource,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.sample_with_rng(rng).unwrap_or(f64::NAN)
    }
}

impl Distribution<f64> for DiscreteSampler {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.sample_with_rng(rng).unwrap_or(f64::NAN)
    }
}
