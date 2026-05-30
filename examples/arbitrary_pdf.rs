//! Sample from an arbitrary PDF on a bounded interval.
//!
//! This example uses an unnormalized PDF on [0, 1]:
//!   g(x) = exp(-10 * x)
//! simsam will normalize it numerically and sample via inverse transform.

use simsam::{from_pdf_fn, Interval};

fn main() {
    let support = Interval::new(0.0, 1.0).expect("valid interval");
    let dist = from_pdf_fn(|x| (-10.0 * x).exp(), support).expect("build dist");

    let n = 20_000;
    let mut sum = 0.0;
    for _ in 0..n {
        sum += dist.sample().expect("sample");
    }
    let mean = sum / n as f64;

    println!("arbitrary PDF example: g(x)=exp(-10x) on [0,1] (unnormalized)");
    println!("mean ≈ {mean:.6}");
    println!("interval(0.9) = {:?}", dist.interval(0.9).unwrap());
}

