//! 2D multivariate sampling via rejection sampling on a bounded hyper-rectangle.
//!
//! This is a minimal example: sample uniformly from [0,1]^2 by giving `pdf(x)=1`
//! and a tight bound `pdf_max = 1`.

use simsam::{HyperRect, PdfNdFn, RejectionOptions, RejectionSamplerNd};

fn main() {
    let support = HyperRect::new(vec![0.0, 0.0], vec![1.0, 1.0]).expect("support");
    let pdf = PdfNdFn::new(|_x| 1.0, support);
    let sampler =
        RejectionSamplerNd::new(pdf, 1.0, RejectionOptions::default()).expect("sampler");

    let n = 20_000;
    let mut sx = 0.0;
    let mut sy = 0.0;
    for _ in 0..n {
        let v = sampler.sample().expect("sample");
        sx += v[0];
        sy += v[1];
    }
    println!("mean x ~ 0.5: {:.4}", sx / n as f64);
    println!("mean y ~ 0.5: {:.4}", sy / n as f64);
}

