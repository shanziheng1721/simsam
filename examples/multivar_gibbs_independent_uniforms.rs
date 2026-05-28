//! Gibbs sampling when you can sample conditionals directly.
//!
//! Here the target is independent Uniform(0,1) for each coordinate,
//! so the conditional sampler ignores the state.

use rand::{Rng, RngExt};
use simsam::{ConditionalSampler, GibbsOptions, GibbsSamplerNd, HyperRect};

struct IndepUniform2d {
    support: HyperRect,
}

impl ConditionalSampler for IndepUniform2d {
    fn dim(&self) -> usize {
        2
    }

    fn support(&self) -> &HyperRect {
        &self.support
    }

    fn sample_coord<R: Rng + ?Sized>(&self, rng: &mut R, i: usize, _state: &[f64]) -> f64 {
        let u: f64 = rng.random();
        self.support.lo[i] + u * (self.support.hi[i] - self.support.lo[i])
    }
}

fn main() {
    let support = HyperRect::new(vec![0.0, 0.0], vec![1.0, 1.0]).expect("support");
    let cond = IndepUniform2d { support };

    let mut gibbs = GibbsSamplerNd::new(
        cond,
        GibbsOptions { sweeps_per_sample: 1 },
        vec![0.5, 0.5],
    )
    .expect("gibbs");

    let n = 20_000;
    let mut sx = 0.0;
    let mut sy = 0.0;
    for _ in 0..n {
        let v = gibbs.sample().expect("sample");
        sx += v[0];
        sy += v[1];
    }
    println!("mean x ~ 0.5: {:.4}", sx / n as f64);
    println!("mean y ~ 0.5: {:.4}", sy / n as f64);
}

