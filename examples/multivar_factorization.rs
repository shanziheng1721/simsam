//! Sampling via conditional factorization x0 ~ f0(), x1 ~ f1(x0), ...

use rand::{Rng, RngExt};
use simsam::{ConditionalFactorSampler, ConditionalFactorization};

// Simple 2D toy model:
// x ~ Uniform(0,1)
// y | x ~ Uniform(0, x)
struct ToyFactor;

impl ConditionalFactorization for ToyFactor {
    fn dim(&self) -> usize {
        2
    }

    fn sample_i<R: Rng + ?Sized>(&self, rng: &mut R, i: usize, prefix: &[f64]) -> f64 {
        let u: f64 = rng.random();
        match i {
            0 => u, // x in [0,1]
            1 => {
                let x = prefix[0];
                u * x // y in [0,x]
            }
            _ => unreachable!(),
        }
    }
}

fn main() {
    let sampler = ConditionalFactorSampler::new(ToyFactor);
    let n = 50_000;
    let mut sx = 0.0;
    let mut sy = 0.0;
    for _ in 0..n {
        let v = sampler.sample().expect("sample");
        sx += v[0];
        sy += v[1];
    }
    // E[x]=1/2, E[y]=E[x/2]=1/4
    println!("mean x ~ 0.5: {:.4}", sx / n as f64);
    println!("mean y ~ 0.25: {:.4}", sy / n as f64);
}

