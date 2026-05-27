//! Build a distribution from a histogram (rv_histogram-like) and sample from it.
//!
//! We define 3 bins on [0, 3] with different masses, then sample and print the mean.

use simsam::{from_histogram, BuildOptions};

fn main() {
    let edges = vec![0.0, 1.0, 2.0, 3.0];
    let counts = vec![1.0, 2.0, 1.0];
    let dist = from_histogram(edges, counts, false, BuildOptions::default()).expect("histogram dist");

    let n = 20_000;
    let mut sum = 0.0;
    for _ in 0..n {
        sum += dist.sample().expect("sample");
    }
    let mean = sum / n as f64;

    println!("histogram example");
    println!("mean via integration = {:.6}", dist.mean().unwrap());
    println!("mean via sampling    = {:.6}", mean);
    println!("interval(0.9)        = {:?}", dist.interval(0.9).unwrap());
}

