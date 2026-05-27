# simsam

Sim(ple)sam(ple) — a Rust library for sampling from custom discrete and continuous distributions.

Define distributions by PDF or CDF (closures, histograms, location-scale transforms, truncation, or [simsym](https://docs.rs/simsym) symbolic expressions), then draw samples via inverse transform sampling — similar to [SciPy `rv_continuous`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html).

## Features

### Sampling

- `sample()` / `sample_with_rng()` — inverse transform (default: bisection + Newton)
- [`BuildOptions::with_hermite(n)`](https://docs.rs/simsam) — fast PPF table (SciPy [`stats.sampling`](https://docs.scipy.org/doc/scipy/tutorial/stats/sampling.html)-style numerical inversion)
- `rand::distr::Distribution` integration

### Distribution API (SciPy-like)

| Method | Description |
|--------|-------------|
| `pdf`, `logpdf` | Density (when PDF is available) |
| `cdf`, `logcdf` | Cumulative distribution |
| `sf`, `logsf`, `isf` | Survival functions |
| `ppf` | Percent point function (inverse CDF) |
| `mean`, `var`, `std`, `median` | Summary statistics |
| `entropy`, `expect`, `interval` | Entropy, E[f(X)], confidence interval |

### Constructors

| simsam | SciPy analogue |
|--------|----------------|
| `from_pdf_fn` / `from_cdf_fn` | Subclass `rv_continuous` |
| `from_histogram` | `rv_histogram` |
| `from_pdf_loc_scale` | `loc` / `scale` parameters |
| `Truncated` | Truncate to sub-interval |
| `SymbolicContinuous` | Custom `_pdf` + symbolic CAS (via **simsym**) |
| `DiscreteSampler::from_pmf` | `rv_discrete(values=...)` |

## Examples

### PDF + statistics

```rust
use simsam::{from_pdf_fn, BuildOptions, Interval};

let support = Interval::new(0.0, 1.0).unwrap();
let dist = from_pdf_fn(|x| 3.0 * x * x, support, BuildOptions::default()).unwrap();
let x = dist.sample().unwrap();
let m = dist.mean().unwrap();   // 0.75
let v = dist.var().unwrap();
let (lo, hi) = dist.interval(0.9).unwrap();
```

### Fast Hermite sampling

```rust
let opts = BuildOptions::default().with_hermite(128);
let dist = from_pdf_fn(|x| 2.0 * x, support, opts).unwrap();
for _ in 0..10_000 {
    let _ = dist.sample().unwrap();
}
```

### Histogram

```rust
use simsam::{from_histogram, BuildOptions};

let edges = vec![0.0, 1.0, 2.0];
let counts = vec![1.0, 3.0];
let dist = from_histogram(edges, counts, false, BuildOptions::default()).unwrap();
```

### Location-scale

```rust
use simsam::{from_pdf_loc_scale, PdfFn, BuildOptions, Interval};

let base = Interval::new(0.0, 1.0).unwrap();
let dist = from_pdf_loc_scale(PdfFn::new(|_| 1.0, base), 10.0, 2.0, BuildOptions::default()).unwrap();
// Y = 10 + 2 * Uniform(0, 1) on [10, 12]
```

### Symbolic (simsym)

```rust
use simsam::{BuildOptions, Interval, SymbolicContinuous};
use simsym::prelude::*;

let x = symbol("x");
let pdf = rational(2, 1) * x;
let sym = SymbolicContinuous::with_defaults(pdf, x, Interval::new(0.0, 1.0).unwrap()).unwrap();
let dist = sym.sampler(BuildOptions::default()).unwrap();
```

## Limitations

- **Finite support** required; use a wide interval + [`Truncated`](https://docs.rs/simsam) for partial ranges.
- **Unimodal** CDF assumed for inverse transform on continuous distributions.
- No built-in catalog of named distributions (use `rand_distr` / `statrs`), multivariate laws, KDE, or `fit(data)`.

## License

BSD-3-Clause — see [LICENSE](LICENSE).
