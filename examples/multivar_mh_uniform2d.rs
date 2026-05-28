//! 2D sampling with Metropolis–Hastings on a bounded hyper-rectangle.
//!
//! Target: uniform on [0,1]^2 (log-pdf constant).

use simsam::{HyperRect, MhOptions, MetropolisHastingsNd, PdfNdFn};

fn main() {
    let support = HyperRect::new(vec![0.0, 0.0], vec![1.0, 1.0]).expect("support");
    let log_pdf = PdfNdFn::new(|_| 1.0, support);

    let mut mh = MetropolisHastingsNd::new(
        log_pdf,
        MhOptions {
            step_size: 0.35,
            ..MhOptions::default()
        },
    )
    .expect("mh");
    mh.init().expect("init");

    let burn_in = 1_000;
    for _ in 0..burn_in {
        let _ = mh.sample().expect("sample");
    }

    let n = 20_000;
    let mut sx = 0.0;
    let mut sy = 0.0;
    for _ in 0..n {
        let v = mh.sample().expect("sample");
        sx += v[0];
        sy += v[1];
    }
    println!("mean x ~ 0.5: {:.4}", sx / n as f64);
    println!("mean y ~ 0.5: {:.4}", sy / n as f64);
}

