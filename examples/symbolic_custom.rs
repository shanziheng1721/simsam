//! Symbolic PDF sampling (simsym) with an explicit symbolic expression.
//!
//! PDF on [0, 1]:
//!   f(x) = 6x(1-x)  (Beta(2,2))
//! CDF is polynomial: 3x^2 - 2x^3

use simsam::{BuildOptions, Interval, SymbolicContinuous};
use simsym::prelude::*;

fn main() {
    let x = symbol("x");
    let pdf = rational(6, 1) * x * (simsym::Expr::const_(rational(1, 1)) - simsym::Expr::from(x));
    let support = Interval::new(0.0, 1.0).expect("valid interval");

    let sym = SymbolicContinuous::with_defaults(pdf, x, support).expect("symbolic dist");
    println!("symbolic CDF available: {}", sym.has_symbolic_cdf());

    let dist = sym.sampler(BuildOptions::default()).expect("sampler");
    println!("mean (numeric) = {:.6} (expected 0.5)", dist.mean().unwrap());
    println!("median = {:.6}", dist.median().unwrap());

    let n = 20_000;
    let mut sum = 0.0;
    for _ in 0..n {
        sum += dist.sample().expect("sample");
    }
    println!("sample mean ≈ {:.6}", sum / n as f64);
}

