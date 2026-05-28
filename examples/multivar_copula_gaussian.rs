//! Gaussian copula: correlated samples from arbitrary 1D marginals.

use simsam::{from_cdf_fn, BuildOptions, GaussianCopula, Interval};

fn main() {
    // Correlation matrix with rho = 0.7
    let rho = 0.7;
    let corr = vec![vec![1.0, rho], vec![rho, 1.0]];
    let cop = GaussianCopula::new(corr).expect("copula");

    // Marginals:
    // X ~ Uniform(0,1): CDF(x)=x
    // Y ~ Triangular on [0,1] with CDF(x)=x^2  (PDF 2x)
    let support = Interval::new(0.0, 1.0).unwrap();
    let x = from_cdf_fn(|t| t, support, BuildOptions::default()).unwrap();
    let y = from_cdf_fn(|t| t * t, support, BuildOptions::default()).unwrap();

    let mut rng = rand::rng();
    let n = 30_000;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for _ in 0..n {
        let v = cop
            .sample_with_ppfs(&mut rng, &[&|u| x.ppf(u), &|u| y.ppf(u)])
            .expect("sample");
        let (a, b) = (v[0], v[1]);
        sx += a;
        sy += b;
        sxx += a * a;
        syy += b * b;
        sxy += a * b;
    }
    let mx = sx / n as f64;
    let my = sy / n as f64;
    let vx = sxx / n as f64 - mx * mx;
    let vy = syy / n as f64 - my * my;
    let cov = sxy / n as f64 - mx * my;
    let corr = cov / (vx * vy).sqrt();

    println!("mean X ~ 0.5: {:.4}", mx);
    println!("mean Y ~ 0.666: {:.4}", my);
    println!("corr(X,Y) ~ {:.3}", corr);
}

