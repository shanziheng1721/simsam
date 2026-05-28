use crate::error::SampleError;
use crate::multivar::support::HyperRect;
use crate::multivar::traits::{HasSupportNd, LogPdfNd};
use rand::{Rng, RngExt};

#[derive(Debug, Clone, Copy)]
pub struct MhOptions {
    /// Random-walk proposal step size (per coordinate).
    pub step_size: f64,
    /// Maximum attempts to find an initial finite log-density point.
    pub init_trials: usize,
}

impl Default for MhOptions {
    fn default() -> Self {
        Self {
            step_size: 0.25,
            init_trials: 100_000,
        }
    }
}

/// Random-walk Metropolis–Hastings sampler on a bounded hyper-rectangle.
pub struct MetropolisHastingsNd<P> {
    log_pdf: P,
    support: HyperRect,
    opts: MhOptions,
    x: Option<Vec<f64>>,
    logp: f64,
}

impl<P> MetropolisHastingsNd<P>
where
    P: LogPdfNd + HasSupportNd,
{
    pub fn new(log_pdf: P, opts: MhOptions) -> Result<Self, crate::error::BuildError> {
        let support = log_pdf.support().clone();
        support.validate()?;
        Ok(Self {
            log_pdf,
            support,
            opts,
            x: None,
            logp: f64::NEG_INFINITY,
        })
    }

    pub fn support(&self) -> &HyperRect {
        &self.support
    }

    pub fn init(&mut self) -> Result<(), SampleError> {
        self.init_with_rng(&mut rand::rng())
    }

    pub fn init_with_rng<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<(), SampleError> {
        let dim = self.support.dim();
        let mut x = vec![0.0; dim];
        for _ in 0..self.opts.init_trials {
            for i in 0..dim {
                let u: f64 = rng.random();
                x[i] = self.support.lo[i] + u * (self.support.hi[i] - self.support.lo[i]);
            }
            let lp = self.log_pdf.log_pdf(&x);
            if lp.is_finite() {
                self.logp = lp;
                self.x = Some(x);
                return Ok(());
            }
        }
        Err(SampleError::McmcFailed)
    }

    pub fn step(&mut self) -> Result<&[f64], SampleError> {
        self.step_with_rng(&mut rand::rng()).map(|v| v.as_slice())
    }

    pub fn step_with_rng<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<&Vec<f64>, SampleError> {
        if self.x.is_none() {
            return Err(SampleError::McmcNotInitialized);
        }
        let dim = self.support.dim();
        let mut y = self.x.as_ref().unwrap().clone();
        for i in 0..dim {
            y[i] = reflect(
                y[i] + self.opts.step_size * standard_normal(rng),
                self.support.lo[i],
                self.support.hi[i],
            );
        }
        let logp_y = self.log_pdf.log_pdf(&y);
        if !logp_y.is_finite() {
            return Ok(self.x.as_ref().unwrap());
        }

        let log_alpha = logp_y - self.logp;
        let u: f64 = rng.random();
        if u.ln() < log_alpha {
            *self.x.as_mut().unwrap() = y;
            self.logp = logp_y;
        }
        Ok(self.x.as_ref().unwrap())
    }

    pub fn sample(&mut self) -> Result<Vec<f64>, SampleError> {
        self.sample_with_rng(&mut rand::rng())
    }

    pub fn sample_with_rng<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Vec<f64>, SampleError> {
        self.step_with_rng(rng).map(|x| x.clone())
    }
}

fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    // Box-Muller transform
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();
    let r = (-2.0 * u1.max(1e-300).ln()).sqrt();
    let theta = core::f64::consts::TAU * u2;
    r * theta.cos()
}

fn reflect(mut x: f64, lo: f64, hi: f64) -> f64 {
    let w = hi - lo;
    if w <= 0.0 {
        return lo;
    }
    while x < lo || x > hi {
        if x < lo {
            x = lo + (lo - x);
        } else if x > hi {
            x = hi - (x - hi);
        }
    }
    x
}

