//! Symbolic PDF where analytic integration is (likely) unavailable in simsym.
//!
//! We use a PDF proportional to exp(-x^2) on a bounded interval.
//! The antiderivative involves erf, which is typically not implemented as an elementary function,
//! so `simsym` may fail symbolic integration and simsam will fall back to numerical CDF.

use simsam::{BuildOptions, Interval, SymbolicContinuous};
use simsym::prelude::*;

fn main() {
    let x = symbol("x");

    // Unnormalized PDF on [0, 3]: exp(-x^2)
    let pdf = exp(-(Expr::from(x).pow(2)));
    let support = Interval::new(0.0, 3.0).expect("support");

    let sym = SymbolicContinuous::with_defaults(pdf, x, support).expect("symbolic dist");
    println!("symbolic CDF available: {}", sym.has_symbolic_cdf());
    println!("mean (numeric) = {:.6}", sym.sampler(BuildOptions::default()).unwrap().mean().unwrap());

    let dist = sym.sampler(BuildOptions::default()).expect("sampler");
    let n = 20_000;
    let mut sum = 0.0;
    for _ in 0..n {
        sum += dist.sample().expect("sample");
    }
    println!("sample mean ≈ {:.6}", sum / n as f64);
}

