//! 2D multivariate sampling from a symbolic joint PDF (simsym) via rejection sampling.
//!
//! Run with:
//!   cargo run --example symbolic_multivar_rejection --features symbolic

use simsam::{HyperRect, RejectionOptions, RejectionSamplerNd, SymbolicPdfNd};
use simsym::prelude::*;

fn main() {
    // Unnormalized PDF: f(x,y) = 1 - x^2 - y^2 on [-1,1]^2 (non-negative on unit disk).
    // Rejection sampling requires a global bound; here max is 1 at (0,0).
    let x = symbol("x");
    let y = symbol("y");
    let pdf_expr = simsym::expr::const_(rational(1, 1)) - (x.pow(2) + y.pow(2));

    let support = HyperRect::new(vec![-1.0, -1.0], vec![1.0, 1.0]).expect("support");
    let pdf = SymbolicPdfNd::new(pdf_expr, vec![x, y], support).expect("symbolic pdf");

    let sampler = RejectionSamplerNd::new(pdf, 1.0, RejectionOptions::default()).expect("sampler");

    // Draw a few samples.
    for _ in 0..5 {
        let v = sampler.sample().expect("sample");
        println!("x={:.4}, y={:.4}", v[0], v[1]);
    }
}

