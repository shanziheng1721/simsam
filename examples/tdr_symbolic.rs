//! TDR sampling example using simsym to obtain dPDF automatically.
//!
//! Run with:
//!   cargo run --example tdr_symbolic --features symbolic

use simsam::{Interval, SymbolicPdfDpdf1d, TdrOptions, TdrSampler, TdrTransform};
use simsym::prelude::*;

fn main() {
    let x = symbol("x");
    let support = Interval::new(-1.0, 1.0).unwrap();
    let pdf_expr = simsym::expr::const_(rational(1, 1)) - x.pow(2);
    let pdf = SymbolicPdfDpdf1d::new(pdf_expr, x, support);

    let opts = TdrOptions {
        support,
        transform: TdrTransform::InvSqrt,
        ..TdrOptions::default()
    };
    let mut tdr = TdrSampler::new(pdf.clone(), pdf, opts).unwrap();

    let n = 20_000;
    let mut sum = 0.0;
    for _ in 0..n {
        sum += tdr.sample().unwrap();
    }
    println!("mean ~ 0: {:.4}", sum / n as f64);
}

