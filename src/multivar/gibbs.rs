use crate::error::SampleError;
use crate::multivar::support::HyperRect;
use rand::Rng;

/// Trait for sampling coordinate `i` conditional on other coordinates.
pub trait ConditionalSampler {
    fn dim(&self) -> usize;
    fn support(&self) -> &HyperRect;
    fn sample_coord<R: Rng + ?Sized>(&self, rng: &mut R, i: usize, state: &[f64]) -> f64;
}

#[derive(Debug, Clone, Copy)]
pub struct GibbsOptions {
    pub sweeps_per_sample: usize,
}

impl Default for GibbsOptions {
    fn default() -> Self {
        Self { sweeps_per_sample: 1 }
    }
}

pub struct GibbsSamplerNd<C> {
    cond: C,
    opts: GibbsOptions,
    state: Vec<f64>,
}

impl<C> GibbsSamplerNd<C>
where
    C: ConditionalSampler,
{
    pub fn new(cond: C, opts: GibbsOptions, init: Vec<f64>) -> Result<Self, SampleError> {
        if init.len() != cond.dim() {
            return Err(SampleError::McmcFailed);
        }
        Ok(Self {
            cond,
            opts,
            state: init,
        })
    }

    pub fn state(&self) -> &[f64] {
        &self.state
    }

    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<&[f64], SampleError> {
        let dim = self.cond.dim();
        for _ in 0..self.opts.sweeps_per_sample {
            for i in 0..dim {
                let v = self.cond.sample_coord(rng, i, &self.state);
                self.state[i] = v;
            }
        }
        Ok(&self.state)
    }

    pub fn sample(&mut self) -> Result<Vec<f64>, SampleError> {
        self.sample_with_rng(&mut rand::rng())
    }

    pub fn sample_with_rng<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Vec<f64>, SampleError> {
        self.step(rng)?;
        Ok(self.state.clone())
    }
}

