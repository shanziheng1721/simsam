//! Fast sampling using a Hermite PPF table (setup once, sample many times).

use simsam::{from_pdf_fn, BuildOptions, Interval};

fn main() {
    let support = Interval::new(0.0, 1.0).expect("support");

    // Triangular on [0,1]: f(x)=2x, F(x)=x^2
    let opts = BuildOptions::default().with_hermite(128);
    let fast = from_pdf_fn(|x| 2.0 * x, support, opts).expect("fast sampler");
    println!("uses hermite table: {}", fast.uses_hermite_table());

    // Compare a couple of quantiles
    for &u in &[0.01, 0.1, 0.5, 0.9, 0.99] {
        let x = fast.ppf(u).unwrap();
        println!("ppf({u:.2}) ≈ {x:.8} (expected ≈ sqrt(u)={:.8})", u.sqrt());
    }

    let n = 200_000;
    let mut sum = 0.0;
    for _ in 0..n {
        sum += fast.sample().unwrap();
    }
    println!("sample mean ≈ {:.6} (expected 2/3 ≈ 0.666667)", sum / n as f64);
}

