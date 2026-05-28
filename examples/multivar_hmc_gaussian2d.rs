//! 2D sampling with HMC on a bounded hyper-rectangle.
//!
//! Target: unnormalized log-pdf of a standard Gaussian `log f(x) = -0.5 * ||x||^2`
//! truncated to [-4,4]^2.

use simsam::{HasSupportNd, HmcOptions, HmcSamplerNd, HyperRect, LogPdfNd};

struct TruncGaussian2d {
    support: HyperRect,
}

impl HasSupportNd for TruncGaussian2d {
    fn support(&self) -> &HyperRect {
        &self.support
    }
}

impl LogPdfNd for TruncGaussian2d {
    fn log_pdf(&self, x: &[f64]) -> f64 {
        if !self.support.contains(x) {
            return f64::NEG_INFINITY;
        }
        let r2 = x[0] * x[0] + x[1] * x[1];
        -0.5 * r2
    }
}

fn main() {
    let support = HyperRect::new(vec![-4.0, -4.0], vec![4.0, 4.0]).expect("support");
    let target = TruncGaussian2d { support };

    let mut hmc = HmcSamplerNd::new(
        target,
        HmcOptions {
            step_size: 0.08,
            leapfrog_steps: 25,
            ..HmcOptions::default()
        },
    )
    .expect("hmc");
    hmc.init().expect("init");

    let burn_in = 2_000;
    for _ in 0..burn_in {
        let _ = hmc.sample().expect("sample");
    }

    let n = 20_000;
    let mut sx = 0.0;
    let mut sy = 0.0;
    for _ in 0..n {
        let v = hmc.sample().expect("sample");
        sx += v[0];
        sy += v[1];
    }
    println!("mean x ~ 0: {:.4}", sx / n as f64);
    println!("mean y ~ 0: {:.4}", sy / n as f64);
}

