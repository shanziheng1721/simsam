//! Parameterized arbitrary PDF example (closure captures parameters).
//!
//! We define a family on [0, 1]:
//!   g(x; lambda) = exp(-lambda * x)   (unnormalized)
//! and show how to sample with a reproducible RNG.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use simsam::{from_pdf_fn, BuildOptions, Interval};

fn main() {
    let support = Interval::new(0.0, 1.0).expect("support");

    let lambda = 4.0_f64;
    let dist = from_pdf_fn(|x| (-lambda * x).exp(), support, BuildOptions::default())
        .expect("build dist");

    // Reproducible draws: supply your own RNG
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let n = 20_000;
    let mut sum = 0.0;
    for _ in 0..n {
        sum += dist.sample_with_rng(&mut rng).expect("sample");
    }
    let mean = sum / n as f64;

    println!("parameterized PDF example: g(x)=exp(-lambda x), lambda={lambda}");
    println!("sample mean ≈ {mean:.6}");
    println!("interval(0.9) = {:?}", dist.interval(0.9).unwrap());
}

