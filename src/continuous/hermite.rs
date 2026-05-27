use crate::continuous::invert::{ppf, InvertOptions};
use crate::continuous::CdfSource;
use crate::support::Interval;

/// Precomputed inverse-CDF table with monotone cubic Hermite interpolation (fast `ppf`).
#[derive(Debug, Clone)]
pub struct HermitePpfTable {
    support: Interval,
    u_knots: Vec<f64>,
    x_knots: Vec<f64>,
    x_slopes: Vec<f64>,
}

impl HermitePpfTable {
    /// Build a table by inverting the CDF at uniform quantiles on `(0, 1)`.
    pub fn build<D>(dist: &D, support: Interval, grid_size: usize, invert: InvertOptions) -> Self
    where
        D: CdfSource,
    {
        let n = grid_size.max(4);
        let mut u_knots = Vec::with_capacity(n);
        let mut x_knots = Vec::with_capacity(n);
        let eps = 1e-12;
        for i in 0..n {
            let t = i as f64 / (n - 1) as f64;
            let u = eps + (1.0 - 2.0 * eps) * t;
            u_knots.push(u);
            let x = ppf(dist, support, u, invert).unwrap_or_else(|_| {
                support.lo + t * support.width()
            });
            x_knots.push(x);
        }
        let x_slopes = monotone_slopes(&u_knots, &x_knots);
        Self {
            support,
            u_knots,
            x_knots,
            x_slopes,
        }
    }

    /// Evaluate approximate `ppf(u)`.
    pub fn eval(&self, u: f64) -> f64 {
        if u <= self.u_knots[0] {
            return self.x_knots[0];
        }
        if u >= *self.u_knots.last().unwrap() {
            return *self.x_knots.last().unwrap();
        }
        let i = self
            .u_knots
            .partition_point(|&uk| uk < u)
            .saturating_sub(1)
            .min(self.u_knots.len() - 2);
        hermite_segment(
            self.u_knots[i],
            self.u_knots[i + 1],
            self.x_knots[i],
            self.x_knots[i + 1],
            self.x_slopes[i],
            self.x_slopes[i + 1],
            u,
        )
        .clamp(self.support.lo, self.support.hi)
    }

    pub fn support(&self) -> Interval {
        self.support
    }
}

/// Fritsch–Carlson monotone slopes for `(u_k, x_k)`.
fn monotone_slopes(u: &[f64], x: &[f64]) -> Vec<f64> {
    let n = u.len();
    let mut m = vec![0.0; n];
    let mut d = vec![0.0; n - 1];
    for i in 0..n - 1 {
        let du = u[i + 1] - u[i];
        d[i] = if du.abs() > 1e-300 {
            (x[i + 1] - x[i]) / du
        } else {
            0.0
        };
    }
    m[0] = d[0];
    m[n - 1] = d[n - 2];
    for i in 1..n - 1 {
        if d[i - 1] * d[i] <= 0.0 {
            m[i] = 0.0;
        } else {
            m[i] = 2.0 / (1.0 / d[i - 1] + 1.0 / d[i]);
        }
    }
    for i in 0..n - 1 {
        if d[i].abs() < 1e-300 {
            continue;
        }
        let a = m[i] / d[i];
        let b = m[i + 1] / d[i];
        let s = a * a + b * b;
        if s > 9.0 {
            let tau = 3.0 / s.sqrt();
            m[i] *= tau;
            m[i + 1] *= tau;
        }
    }
    m
}

fn hermite_segment(u0: f64, u1: f64, x0: f64, x1: f64, m0: f64, m1: f64, u: f64) -> f64 {
    let h = u1 - u0;
    if h.abs() < 1e-300 {
        return x0;
    }
    let t = (u - u0) / h;
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    h00 * x0 + h10 * h * m0 + h01 * x1 + h11 * h * m1
}
