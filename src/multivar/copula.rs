use crate::error::SampleError;
use rand::{Rng, RngExt};

#[derive(Debug, Clone, PartialEq)]
pub enum CopulaError {
    InvalidCorrelation,
    DimensionMismatch,
}

/// Gaussian copula sampler.
///
/// Given marginal PPFs (univariate) and an SPD correlation matrix `R`,
/// sample `Z ~ N(0, R)`, transform `U_i = Φ(Z_i)`, then `X_i = F_i^{-1}(U_i)`.
pub struct GaussianCopula {
    chol: Vec<Vec<f64>>,
}

impl GaussianCopula {
    pub fn new(corr: Vec<Vec<f64>>) -> Result<Self, CopulaError> {
        let n = corr.len();
        if n == 0 || corr.iter().any(|r| r.len() != n) {
            return Err(CopulaError::InvalidCorrelation);
        }
        let chol = cholesky_spd(&corr).ok_or(CopulaError::InvalidCorrelation)?;
        Ok(Self { chol })
    }

    pub fn dim(&self) -> usize {
        self.chol.len()
    }

    pub fn sample_with_ppfs<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        ppfs: &[&dyn Fn(f64) -> Result<f64, SampleError>],
    ) -> Result<Vec<f64>, SampleError> {
        if ppfs.len() != self.dim() {
            return Err(SampleError::McmcFailed);
        }
        let z = correlated_normals(rng, &self.chol);
        let mut out = Vec::with_capacity(z.len());
        for (i, zi) in z.iter().enumerate() {
            let u = normal_cdf(*zi).clamp(1e-12, 1.0 - 1e-12);
            out.push((ppfs[i])(u)?);
        }
        Ok(out)
    }
}

fn correlated_normals<R: Rng + ?Sized>(rng: &mut R, chol: &[Vec<f64>]) -> Vec<f64> {
    let n = chol.len();
    let mut z0 = vec![0.0; n];
    for i in 0..n {
        z0[i] = standard_normal(rng);
    }
    let mut z = vec![0.0; n];
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..=i {
            acc += chol[i][j] * z0[j];
        }
        z[i] = acc;
    }
    z
}

fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();
    let r = (-2.0 * u1.max(1e-300).ln()).sqrt();
    let theta = core::f64::consts::TAU * u2;
    r * theta.cos()
}

// Acklam inverse-normal is not needed here; only normal CDF.
fn normal_cdf(x: f64) -> f64 {
    // Abramowitz-Stegun erf approximation
    // Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
    0.5 * (1.0 + erf(x / 2f64.sqrt()))
}

fn erf(x: f64) -> f64 {
    // Numerical Recipes approximation (max error ~1.5e-7)
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.5 * x);
    let tau = t
        * (-x * x
            - 1.26551223
            + 1.00002368 * t
            + 0.37409196 * t.powi(2)
            + 0.09678418 * t.powi(3)
            - 0.18628806 * t.powi(4)
            + 0.27886807 * t.powi(5)
            - 1.13520398 * t.powi(6)
            + 1.48851587 * t.powi(7)
            - 0.82215223 * t.powi(8)
            + 0.17087277 * t.powi(9))
            .exp();
    sign * (1.0 - tau)
}

fn cholesky_spd(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return None;
                }
                l[i][j] = sum.sqrt();
            } else {
                if l[j][j] == 0.0 {
                    return None;
                }
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Some(l)
}

