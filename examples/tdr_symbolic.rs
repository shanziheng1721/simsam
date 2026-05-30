//! TDR sampling via unified [`BuildOptions::with_tdr`] on a symbolic PDF.
//!
//! Run with:
//!   cargo run --example tdr_symbolic --features symbolic

use simsam::{BuildOptions, Interval, SymbolicContinuous, TdrBuildConfig, TdrTransform};
use simsym::prelude::*;

fn main() {
    let x = symbol("x");
    let support = Interval::new(-1.0, 1.0).unwrap();
    let pdf_expr = simsym::expr::const_(rational(1, 1)) - x.pow(2);
    let sym = SymbolicContinuous::with_defaults(pdf_expr, x, support).unwrap();

    let opts = BuildOptions::default().with_tdr_config(TdrBuildConfig {
        transform: TdrTransform::InvSqrt,
        ..TdrBuildConfig::default()
    });
    let sampler = sym.sampler(opts).unwrap();
    assert!(sampler.uses_tdr());

    let n = 20_000;
    let mut sum = 0.0;
    for _ in 0..n {
        sum += sampler.sample().unwrap();
    }
    println!("mean ~ 0: {:.4}", sum / n as f64);
}
