//! Approximate multivariate CDF via Monte Carlo.
//!
//! Uniform on [0,1]^2 has CDF F(x,y)=x*y for x,y in [0,1].

use simsam::{CdfMcEstimator, CdfMcOptions, HyperRect, PdfNdFn};

fn main() {
    let support = HyperRect::new(vec![0.0, 0.0], vec![1.0, 1.0]).expect("support");
    let pdf = PdfNdFn::new(|_| 1.0, support);
    let mut est = CdfMcEstimator::new(
        pdf,
        CdfMcOptions {
            normalization_samples: 50_000,
            cdf_samples: 50_000,
        },
    )
    .expect("estimator");

    let x = [0.25, 0.4];
    let p = est.cdf(&x).expect("cdf");
    println!("cdf({:?}) ≈ {:.4} (true 0.1)", x, p);
}

