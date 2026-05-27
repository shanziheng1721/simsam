//! Sample from an arbitrary CDF on a bounded interval.
//!
//! Here we define a distribution on [0, 1] with CDF F(x) = x^2.
//! The corresponding PDF is f(x) = 2x, and the analytic ppf is sqrt(u).

use simsam::{from_cdf_fn, BuildOptions, Interval};

fn main() {
    let support = Interval::new(0.0, 1.0).expect("valid interval");
    let dist = from_cdf_fn(|x| x * x, support, BuildOptions::default()).expect("build dist");

    let n = 10_000;
    let mut sum = 0.0;
    for _ in 0..n {
        sum += dist.sample().expect("sample");
    }
    let mean = sum / n as f64;

    println!("arbitrary CDF example: F(x)=x^2 on [0,1]");
    println!("ppf(0.25) = {:.6} (expected 0.5)", dist.ppf(0.25).unwrap());
    println!("sample mean = {:.6} (expected 2/3 ≈ 0.666667)", mean);
}

