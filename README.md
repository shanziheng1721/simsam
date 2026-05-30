# simsam

Sim(ple)sam(ple) — a Rust library for sampling from custom discrete and continuous distributions.

Define distributions by PDF or CDF (closures, histograms, location-scale transforms, truncation, or [simsym](https://docs.rs/simsym) symbolic expressions), then draw samples via inverse transform sampling — similar to [SciPy `rv_continuous`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html).

## Features

### Sampling

- `sample()` / `sample_with_rng()` — inverse transform (default: bisection + Newton)
- [`BuildOptions::with_hermite(n)`](https://docs.rs/simsam) — fast PPF table (SciPy [`stats.sampling`](https://docs.scipy.org/doc/scipy/tutorial/stats/sampling.html)-style numerical inversion)
- [`BuildOptions::with_tdr()`](https://docs.rs/simsam) — transformed density rejection (TDR); automatic numerical dPDF when only PDF is given
- `from_pdf_dpdf_fn` — explicit dPDF for TDR (log-concave hat construction)
- `rand::distr::Distribution` integration
- Multivariate sampling: rejection sampling, Metropolis–Hastings (MH), Gibbs, HMC
- Multivariate CDF approximation via Monte Carlo
- Gaussian copula for correlated samples from arbitrary 1D marginals

#### Choosing a 1D sampler

| Goal | Method |
|------|--------|
| Default / need accurate `ppf` | `BuildOptions::default()` (inverse transform) |
| Many samples, smooth unimodal CDF | `BuildOptions::default().with_hermite(128)` |
| Have PDF (+ optional dPDF), rejection-friendly density | `BuildOptions::default().with_tdr()` |

Note: `ppf`, `mean`, and other statistics always use numerical inverse CDF even when TDR is selected for `sample()`.

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
| `from_pdf_dpdf_fn` | TDR with explicit dPDF |
| `from_histogram` | `rv_histogram` |
| `from_pdf_loc_scale` | `loc` / `scale` parameters |
| `Truncated` | Truncate to sub-interval |
| `SymbolicContinuous` | Custom `_pdf` + symbolic CAS (via **simsym**) |
| `DiscreteSampler::from_pmf` | `rv_discrete(values=...)` |

### Cargo features

- **Default**: no symbolic dependency
- **`symbolic`**: enables `SymbolicContinuous` and `SymbolicPdfNd`

## Examples

### PDF + statistics

```rust
use simsam::{from_pdf_fn, Interval};

let support = Interval::new(0.0, 1.0).unwrap();
let dist = from_pdf_fn(|x| 3.0 * x * x, support).unwrap();
let x = dist.sample().unwrap();
let m = dist.mean().unwrap();   // 0.75
let v = dist.var().unwrap();
let (lo, hi) = dist.interval(0.9).unwrap();
```

### Fast Hermite sampling

```rust
use simsam::{from_pdf_fn_with_options, BuildOptions, Interval};

let opts = BuildOptions::default().with_hermite(128);
let dist = from_pdf_fn_with_options(|x| 2.0 * x, support, opts).unwrap();
for _ in 0..10_000 {
    let _ = dist.sample().unwrap();
}
```

### TDR sampling

```rust
use simsam::{from_pdf_fn_with_options, BuildOptions, Interval, TdrBuildConfig, TdrTransform};

let support = Interval::new(-1.0, 1.0).unwrap();
let dist = from_pdf_fn_with_options(
    |x| 1.0 - x * x,
    support,
    BuildOptions::default().with_tdr_config(TdrBuildConfig {
        transform: TdrTransform::InvSqrt,
        ..TdrBuildConfig::default()
    }),
)
.unwrap();
```

```bash
cargo run --example tdr_quadratic
cargo run --example sampling_methods
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

Run (symbolic examples are gated):

```bash
cargo run --example symbolic --features symbolic
```

### Multivariate: rejection sampling (2D uniform)

```bash
cargo run --example multivar_rejection_uniform2d
```

### Multivariate: Metropolis–Hastings (2D uniform)

```bash
cargo run --example multivar_mh_uniform2d
```

### Multivariate: HMC (2D truncated Gaussian; numeric gradient)

```bash
cargo run --example multivar_hmc_gaussian2d
```

### Multivariate: Gibbs (independent uniforms via conditionals)

```bash
cargo run --example multivar_gibbs_independent_uniforms
```

### Multivariate: CDF approximation (Monte Carlo)

```bash
cargo run --example multivar_cdf_mc_uniform2d
```

### Copula: Gaussian copula with custom marginals

```bash
cargo run --example multivar_copula_gaussian
```

### Conditional factorization sampler

```bash
cargo run --example multivar_factorization
```

### Multivariate symbolic HMC (symbolic gradient)

```bash
cargo run --example multivar_symbolic_hmc --features symbolic
```

## Limitations

- **Finite support** required; use a wide interval + [`Truncated`](https://docs.rs/simsam) for partial ranges.
- **Unimodal** CDF assumed for inverse transform on continuous distributions.
- No built-in catalog of named distributions (use `rand_distr` / `statrs`), multivariate laws, KDE, or `fit(data)`.

## License

BSD-3-Clause — see [LICENSE](LICENSE).
