//! Sample from a triangular distribution on [0, 1] with PDF f(x) = 2x.

use simsam::{from_pdf_fn, Interval};

fn main() {
    let support = Interval::new(0.0, 1.0).expect("valid interval");
    let sampler = from_pdf_fn(|x| 2.0 * x, support).expect("build sampler");

    let n = 10_000;
    let mut sum = 0.0;
    for _ in 0..n {
        sum += sampler.sample().expect("sample");
    }
    let mean = sum / n as f64;
    println!("triangular sample mean (expected ~0.667): {mean:.4}");
    println!("ppf(0.5) = {}", sampler.ppf(0.5).expect("ppf"));
}
