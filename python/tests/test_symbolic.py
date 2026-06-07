import pytest

simsam = pytest.importorskip("simsam")

pytestmark = pytest.mark.skipif(
    not hasattr(simsam, "symbolic_continuous"),
    reason="simsam built without symbolic feature",
)


def test_symbolic_triangular():
    dist = simsam.symbolic_continuous("3 * x^2", "x", simsam.Interval(0.0, 1.0))
    assert abs(dist.mean() - 0.75) < 1e-2
