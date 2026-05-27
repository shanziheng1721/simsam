//! Symbolic parameters example (simsym): define PDF with symbolic parameters, bind values later.
//!
//! PDF on [0, 1]:
//!   f(x; a, b) = a*x + b
//! This is not necessarily normalized; simsam will normalize numerically.
//!
//! We bind (a, b) to concrete values by substitution (building a new Expr),
//! then sample from the resulting symbolic PDF.

use simsam::{BuildOptions, Interval, SymbolicContinuous};
use simsym::prelude::*;

fn main() {
    let x = symbol("x");
    let a = symbol("a");
    let b = symbol("b");

    // Build f(x; a,b) = a*x + b
    let pdf_param = Expr::from(a) * Expr::from(x) + Expr::from(b);

    // Bind a,b -> numbers by evaluating into an expression that only depends on x.
    // Here we do it by simply rebuilding the expression using the chosen values.
    // (A future simsym helper like Expr::substitute would make this cleaner.)
    let a_val = 2.0_f64;
    let b_val = 1.0_f64;
    let pdf = Expr::from(a_val as i32) * Expr::from(x) + Expr::from(b_val as i32);

    let support = Interval::new(0.0, 1.0).expect("support");
    let sym = SymbolicContinuous::with_defaults(pdf, x, support).expect("symbolic dist");
    let dist = sym.sampler(BuildOptions::default()).expect("sampler");

    println!("symbolic parameters example");
    println!("template pdf: {pdf_param}");
    println!("bound a={a_val}, b={b_val}");
    println!("mean (numeric) = {:.6}", dist.mean().unwrap());

    let n = 20_000;
    let mut sum = 0.0;
    for _ in 0..n {
        sum += dist.sample().expect("sample");
    }
    println!("sample mean ≈ {:.6}", sum / n as f64);
}

