use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use simsam::{from_pdf_fn, from_pdf_fn_with_options, BuildOptions, Interval};
use std::hint::black_box;

fn bench_ppf(c: &mut Criterion) {
    let support = Interval::new(0.0, 1.0).unwrap();

    // Triangular distribution: pdf(x) = 2x on [0,1]
    let slow = from_pdf_fn(|x| 2.0 * x, support).unwrap();
    let fast = from_pdf_fn_with_options(
        |x| 2.0 * x,
        support,
        BuildOptions::default().with_hermite(128),
    )
    .unwrap();

    let mut group = c.benchmark_group("ppf");
    for &u in &[0.01, 0.1, 0.25, 0.5, 0.9, 0.99] {
        group.bench_with_input(BenchmarkId::new("bisection", u), &u, |b, &u| {
            b.iter(|| slow.ppf(black_box(u)).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("hermite128", u), &u, |b, &u| {
            b.iter(|| fast.ppf(black_box(u)).unwrap())
        });
    }
    group.finish();
}

fn bench_sampling(c: &mut Criterion) {
    let support = Interval::new(0.0, 1.0).unwrap();
    let slow = from_pdf_fn(|x| 2.0 * x, support).unwrap();
    let fast = from_pdf_fn_with_options(
        |x| 2.0 * x,
        support,
        BuildOptions::default().with_hermite(128),
    )
    .unwrap();

    let mut group = c.benchmark_group("sample_n");
    for &n in &[256_usize, 4096, 65536] {
        group.bench_with_input(BenchmarkId::new("bisection", n), &n, |b, &n| {
            b.iter(|| {
                let mut rng = ChaCha8Rng::seed_from_u64(123);
                let xs = slow.sample_n_with_rng(&mut rng, black_box(n)).unwrap();
                black_box(xs)
            })
        });
        group.bench_with_input(BenchmarkId::new("hermite128", n), &n, |b, &n| {
            b.iter(|| {
                let mut rng = ChaCha8Rng::seed_from_u64(123);
                let xs = fast.sample_n_with_rng(&mut rng, black_box(n)).unwrap();
                black_box(xs)
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_ppf, bench_sampling);
criterion_main!(benches);

