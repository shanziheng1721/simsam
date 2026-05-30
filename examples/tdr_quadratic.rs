//! TDR sampling example: unified API with automatic numerical dPDF.
//!
//! Target (unnormalized) density:
//!   f(x) = 1 - x^2 on [-1, 1]
//!
//! Same example as SciPy's TDR documentation.

use simsam::{
    from_pdf_dpdf_fn, from_pdf_fn_with_options, BuildOptions, Interval, TdrBuildConfig,
    TdrTransform,
};

fn main() {
    let support = Interval::new(-1.0, 1.0).unwrap();
    let cfg = TdrBuildConfig {
        transform: TdrTransform::InvSqrt,
        ..TdrBuildConfig::default()
    };

    // Primary path: PDF only, numerical dPDF
    let sampler = from_pdf_fn_with_options(
        |x| 1.0 - x * x,
        support,
        BuildOptions::default().with_tdr_config(cfg),
    )
    .unwrap();

    // Explicit dPDF (optional, higher precision for log-concave TDR)
    let _explicit = from_pdf_dpdf_fn(
        |x| 1.0 - x * x,
        |x| -2.0 * x,
        support,
        BuildOptions::default().with_tdr_config(cfg),
    )
    .unwrap();

    let n = 50_000;
    let mut sum = 0.0;
    for _ in 0..n {
        sum += sampler.sample().unwrap();
    }
    println!("mean ~ 0: {:.4}", sum / n as f64);
}
