"""Sample from custom discrete and continuous distributions (Rust core via PyO3)."""

from simsam._simsam import (
    BuildError,
    BuildOptions,
    CdfMcEstimator,
    ConditionalFactorizationSampler,
    ContinuousDistribution,
    DiscreteDistribution,
    GaussianCopula,
    GibbsSampler,
    HmcSampler,
    HyperRect,
    Interval,
    MetropolisHastings,
    RejectionSampler,
    SampleError,
    cdf_mc_estimator,
    conditional_factorization_sampler,
    from_cdf,
    from_cdf_with_options,
    from_histogram,
    from_pdf,
    from_pdf_dpdf,
    from_pdf_loc_scale,
    from_pdf_with_options,
    from_pmf,
    gaussian_copula,
    gibbs_sampler,
    hmc_sampler,
    metropolis_hastings,
    rejection_sampler,
    truncated,
)

__all__ = [
    "BuildError",
    "BuildOptions",
    "CdfMcEstimator",
    "ConditionalFactorizationSampler",
    "ContinuousDistribution",
    "DiscreteDistribution",
    "GaussianCopula",
    "GibbsSampler",
    "HmcSampler",
    "HyperRect",
    "Interval",
    "MetropolisHastings",
    "RejectionSampler",
    "SampleError",
    "cdf_mc_estimator",
    "conditional_factorization_sampler",
    "from_cdf",
    "from_cdf_with_options",
    "from_histogram",
    "from_pdf",
    "from_pdf_dpdf",
    "from_pdf_loc_scale",
    "from_pdf_with_options",
    "from_pmf",
    "gaussian_copula",
    "gibbs_sampler",
    "hmc_sampler",
    "metropolis_hastings",
    "rejection_sampler",
    "truncated",
]

try:
    from simsam._simsam import symbolic_continuous

    __all__ += ["symbolic_continuous"]
except ImportError:
    pass

__version__ = "0.1.0"
