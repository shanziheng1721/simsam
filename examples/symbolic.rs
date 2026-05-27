//! Build a distribution from a simsym symbolic PDF and sample from it.

use simsam::{BuildOptions, Interval, SymbolicContinuous};
use simsym::prelude::*;

fn main() {
    let x = symbol("x");
    let pdf = rational(3, 1) * x.pow(2);
    let support = Interval::new(0.0, 1.0).expect("valid interval");

    let dist = SymbolicContinuous::with_defaults(pdf, x, support).expect("symbolic dist");
    println!("symbolic CDF available: {}", dist.has_symbolic_cdf());

    let sampler = dist.sampler(BuildOptions::default()).expect("sampler");
    let n = 10_000;
    let mut sum = 0.0;
    for _ in 0..n {
        sum += sampler.sample().expect("sample");
    }
    let mean = sum / n as f64;
    println!("sample mean (expected 0.75): {mean:.4}");
}
