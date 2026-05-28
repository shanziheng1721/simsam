//! TDR sampling example using explicit PDF and dPDF.
//!
//! Target (unnormalized) density:
//!   f(x) = 1 - x^2 on [-1, 1],  dpdf(x) = -2x
//!
//! This is the same example as SciPy's TDR documentation.

use simsam::{tdr_from_fns, Interval, TdrOptions};
use simsam::TdrTransform;

fn main() {
    let support = Interval::new(-1.0, 1.0).unwrap();
    let opts = TdrOptions {
        support,
        transform: TdrTransform::InvSqrt,
        ..TdrOptions::default()
    };
    let mut tdr = tdr_from_fns(|x| 1.0 - x * x, |x| -2.0 * x, support, opts).unwrap();

    let n = 50_000;
    let mut sum = 0.0;
    for _ in 0..n {
        sum += tdr.sample().unwrap();
    }
    println!("mean ~ 0: {:.4}", sum / n as f64);
}

