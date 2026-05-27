//! Location-scale and truncation examples.
//!
//! 1) Build Y = loc + scale * X where X ~ Uniform(0,1)
//! 2) Truncate the base uniform to [0.25, 0.75]

use simsam::{from_pdf_loc_scale, BuildOptions, ContinuousSampler, Interval, PdfFn, Truncated};

fn main() {
    let base = Interval::new(0.0, 1.0).expect("base interval");
    let base_pdf = PdfFn::new(|_| 1.0, base);

    // Location-scale: Y = 10 + 2X on [10, 12]
    let scaled = from_pdf_loc_scale(base_pdf, 10.0, 2.0, BuildOptions::default()).expect("scaled");
    println!("loc/scale support = {:?}", scaled.support());
    println!("loc/scale mean = {:.6} (expected 11)", scaled.mean().unwrap());

    // Truncation: Uniform(0,1) truncated to [0.25, 0.75] is Uniform(0.25,0.75)
    let base_pdf2 = PdfFn::new(|_| 1.0, base);
    let trunc = Truncated::new(base_pdf2, Interval::new(0.25, 0.75).unwrap(), simsam::default_quad_tol())
        .expect("truncated pdf");
    let trunc_dist = ContinuousSampler::from_pdf(trunc, BuildOptions::default()).expect("trunc sampler");
    println!("trunc support = {:?}", trunc_dist.support());
    println!("trunc mean = {:.6} (expected 0.5)", trunc_dist.mean().unwrap());
}

