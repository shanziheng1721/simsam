//! HMC using a symbolic multivariate log-pdf gradient (simsym).
//!
//! Run with:
//!   cargo run --example multivar_symbolic_hmc --features symbolic

use simsam::{GradientLogPdfNd, HasSupportNd, HmcOptions, HmcSamplerNd, HyperRect, LogPdfNd, SymbolicPdfNd};
use simsym::prelude::*;

fn main() {
    // Unnormalized PDF: exp(-(x^2 + y^2)/2) on [-4,4]^2
    let x = symbol("x");
    let y = symbol("y");
    let pdf_expr = exp(-(x.pow(2) + y.pow(2)) / rational(2, 1));

    let support = HyperRect::new(vec![-4.0, -4.0], vec![4.0, 4.0]).expect("support");
    let pdf = SymbolicPdfNd::new(pdf_expr, vec![x, y], support).expect("symbolic pdf");

    // Use log-pdf & gradient through the implemented traits.
    struct Wrap(SymbolicPdfNd);
    impl HasSupportNd for Wrap {
        fn support(&self) -> &HyperRect { self.0.support() }
    }
    impl LogPdfNd for Wrap {
        fn log_pdf(&self, v: &[f64]) -> f64 { self.0.log_pdf(v) }
    }
    impl GradientLogPdfNd for Wrap {
        fn grad_log_pdf(&self, v: &[f64]) -> Vec<f64> { self.0.grad_log_pdf(v) }
    }

    let mut hmc = HmcSamplerNd::new(
        Wrap(pdf),
        HmcOptions { step_size: 0.08, leapfrog_steps: 25, ..HmcOptions::default() },
    )
    .expect("hmc");
    hmc.init().expect("init");

    // Symbolic gradients are cached now; keep this moderate.
    let burn_in = 300;
    for _ in 0..burn_in {
        let _ = hmc.sample().expect("sample");
    }

    let n = 2_000;
    let mut sx = 0.0;
    let mut sy = 0.0;
    for _ in 0..n {
        let v = hmc
            .sample_with_gradient(&mut rand::rng())
            .expect("sample");
        sx += v[0];
        sy += v[1];
    }
    println!("mean x ~ 0: {:.4}", sx / n as f64);
    println!("mean y ~ 0: {:.4}", sy / n as f64);
}

