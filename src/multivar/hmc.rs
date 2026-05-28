use crate::error::SampleError;
use crate::multivar::support::HyperRect;
use crate::multivar::traits::{GradientLogPdfNd, HasSupportNd, LogPdfNd};
use rand::{Rng, RngExt};

#[derive(Debug, Clone, Copy)]
pub struct HmcOptions {
    pub step_size: f64,
    pub leapfrog_steps: usize,
    pub grad_eps: f64,
    pub init_trials: usize,
}

impl Default for HmcOptions {
    fn default() -> Self {
        Self {
            step_size: 0.05,
            leapfrog_steps: 20,
            grad_eps: 1e-6,
            init_trials: 100_000,
        }
    }
}

pub struct HmcSamplerNd<P> {
    log_pdf: P,
    support: HyperRect,
    opts: HmcOptions,
    x: Option<Vec<f64>>,
    logp: f64,
}

impl<P> HmcSamplerNd<P>
where
    P: LogPdfNd + HasSupportNd,
{
    pub fn new(log_pdf: P, opts: HmcOptions) -> Result<Self, crate::error::BuildError> {
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

    pub fn sample(&mut self) -> Result<Vec<f64>, SampleError> {
        self.sample_with_rng(&mut rand::rng())
    }

    pub fn sample_with_rng<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Vec<f64>, SampleError> {
        if self.x.is_none() {
            return Err(SampleError::McmcNotInitialized);
        }
        let dim = self.support.dim();
        let x = self.x.as_ref().unwrap().clone();
        let mut p = vec![0.0; dim];
        for i in 0..dim {
            p[i] = standard_normal(rng);
        }
        let current_h = hamiltonian(self.logp, &p);

        // Leapfrog
        let mut x_new = x.clone();
        let mut p_new = p.clone();
        let mut g = grad_log_pdf_numeric(&self.log_pdf, &x_new, self.opts.grad_eps);
        for i in 0..dim {
            p_new[i] += 0.5 * self.opts.step_size * g[i];
        }
        for _ in 0..self.opts.leapfrog_steps {
            for i in 0..dim {
                x_new[i] = reflect(x_new[i] + self.opts.step_size * p_new[i], self.support.lo[i], self.support.hi[i]);
            }
            g = grad_log_pdf_numeric(&self.log_pdf, &x_new, self.opts.grad_eps);
            for i in 0..dim {
                p_new[i] += self.opts.step_size * g[i];
            }
        }
        for i in 0..dim {
            p_new[i] -= 0.5 * self.opts.step_size * g[i];
        }
        for i in 0..dim {
            p_new[i] = -p_new[i];
        }

        let logp_new = self.log_pdf.log_pdf(&x_new);
        if !logp_new.is_finite() {
            return Ok(self.x.as_ref().unwrap().clone());
        }
        let new_h = hamiltonian(logp_new, &p_new);
        let log_alpha = current_h - new_h;
        let u: f64 = rng.random();
        if u.ln() < log_alpha {
            self.x = Some(x_new);
            self.logp = logp_new;
        }
        Ok(self.x.as_ref().unwrap().clone())
    }
}

impl<P> HmcSamplerNd<P>
where
    P: LogPdfNd + HasSupportNd + GradientLogPdfNd,
{
    /// Same as `sample_with_rng` but uses analytic/compiled gradient.
    pub fn sample_with_gradient<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Vec<f64>, SampleError> {
        if self.x.is_none() {
            return Err(SampleError::McmcNotInitialized);
        }
        let dim = self.support.dim();
        let mut p = vec![0.0; dim];
        for i in 0..dim {
            p[i] = standard_normal(rng);
        }
        let current_x = self.x.as_ref().unwrap().clone();
        let current_logp = self.logp;
        let current_h = hamiltonian(current_logp, &p);

        let mut x_new = current_x.clone();
        let mut p_new = p.clone();
        let mut g = self.log_pdf.grad_log_pdf(&x_new);
        for i in 0..dim {
            p_new[i] += 0.5 * self.opts.step_size * g[i];
        }
        for _ in 0..self.opts.leapfrog_steps {
            for i in 0..dim {
                x_new[i] = reflect(x_new[i] + self.opts.step_size * p_new[i], self.support.lo[i], self.support.hi[i]);
            }
            g = self.log_pdf.grad_log_pdf(&x_new);
            for i in 0..dim {
                p_new[i] += self.opts.step_size * g[i];
            }
        }
        for i in 0..dim {
            p_new[i] -= 0.5 * self.opts.step_size * g[i];
            p_new[i] = -p_new[i];
        }

        let logp_new = self.log_pdf.log_pdf(&x_new);
        if !logp_new.is_finite() {
            return Ok(self.x.as_ref().unwrap().clone());
        }
        let new_h = hamiltonian(logp_new, &p_new);
        let log_alpha = current_h - new_h;
        let u: f64 = rng.random();
        if u.ln() < log_alpha {
            self.x = Some(x_new);
            self.logp = logp_new;
        }
        Ok(self.x.as_ref().unwrap().clone())
    }
}

fn grad_log_pdf_numeric<P: LogPdfNd>(p: &P, x: &[f64], eps: f64) -> Vec<f64> {
    let mut g = vec![0.0; x.len()];
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    for i in 0..x.len() {
        xp[i] = x[i] + eps;
        xm[i] = x[i] - eps;
        let fp = p.log_pdf(&xp);
        let fm = p.log_pdf(&xm);
        g[i] = (fp - fm) / (2.0 * eps);
        xp[i] = x[i];
        xm[i] = x[i];
    }
    g
}

fn hamiltonian(logp: f64, p: &[f64]) -> f64 {
    let kin: f64 = 0.5 * p.iter().map(|v| v * v).sum::<f64>();
    -logp + kin
}

fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
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

