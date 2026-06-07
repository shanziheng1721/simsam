import simsam


def test_interval():
    iv = simsam.Interval(0.0, 1.0)
    assert iv.lo == 0.0
    assert iv.hi == 1.0


def test_triangular_pdf_mean():
    dist = simsam.from_pdf(lambda x: 2.0 * x, simsam.Interval(0.0, 1.0))
    assert abs(dist.mean() - 2.0 / 3.0) < 1e-2
    samples = dist.sample_n(2000, seed=42)
    assert len(samples) == 2000
    assert all(0.0 <= x <= 1.0 for x in samples)


def test_cdf_only():
    dist = simsam.from_cdf(lambda x: x * x, simsam.Interval(0.0, 1.0))
    assert abs(dist.ppf(0.5) - 0.707106781) < 1e-3


def test_hermite_options():
    dist = simsam.from_pdf_with_options(
        lambda x: 2.0 * x,
        simsam.Interval(0.0, 1.0),
        simsam.BuildOptions.hermite(64),
    )
    assert dist.uses_hermite_table()


def test_discrete():
    dist = simsam.from_pmf([0.0, 1.0], [1.0, 3.0])
    assert abs(dist.mean() - 0.75) < 1e-12


def test_rejection_2d():
    sampler = simsam.rejection_sampler(
        lambda x, y: 1.0,
        simsam.HyperRect([0.0, 0.0], [1.0, 1.0]),
        1.0,
    )
    v = sampler.sample()
    assert len(v) == 2


def test_mh_smoke():
    mh = simsam.metropolis_hastings(
        lambda x, y: 1.0,
        simsam.HyperRect([0.0, 0.0], [1.0, 1.0]),
        burn_in=100,
        adapt_steps=200,
    )
    mh.init()
    samples = mh.sample_n(500, seed=7)
    assert len(samples) == 500


def test_copula():
    cop = simsam.gaussian_copula([[1.0, 0.0], [0.0, 1.0]])
    u = simsam.from_cdf(lambda x: x, simsam.Interval(0.0, 1.0))
    v = cop.sample([lambda q: u.ppf(q), lambda q: u.ppf(q)], seed=1)
    assert len(v) == 2
