//! Compare inverse transform, Hermite, and TDR on the same triangular PDF.

use simsam::{from_pdf_fn, from_pdf_fn_with_options, BuildOptions, Interval, TdrBuildConfig, TdrTransform};
use std::time::Instant;

fn main() {
    let support = Interval::new(0.0, 1.0).unwrap();
    let pdf = |x: f64| 2.0 * x;
    let n = 100_000;

    let bisect = from_pdf_fn(pdf, support).unwrap();
    let t0 = Instant::now();
    for _ in 0..n {
        let _ = bisect.sample().unwrap();
    }
    println!("bisection:  {:?} (mean ~ {:.4})", t0.elapsed(), bisect.mean().unwrap());

    let hermite = from_pdf_fn_with_options(pdf, support, BuildOptions::default().with_hermite(128)).unwrap();
    let t1 = Instant::now();
    for _ in 0..n {
        let _ = hermite.sample().unwrap();
    }
    println!("hermite:    {:?} (uses table: {})", t1.elapsed(), hermite.uses_hermite_table());

    let tdr = from_pdf_fn_with_options(
        pdf,
        support,
        BuildOptions::default().with_tdr_config(TdrBuildConfig {
            transform: TdrTransform::InvSqrt,
            ..TdrBuildConfig::default()
        }),
    )
    .unwrap();
    let t2 = Instant::now();
    for _ in 0..n {
        let _ = tdr.sample().unwrap();
    }
    println!("tdr:        {:?} (uses_tdr: {})", t2.elapsed(), tdr.uses_tdr());
}
