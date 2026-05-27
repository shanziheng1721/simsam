/// Probability mass function on a finite support.
pub trait Pmf {
    fn pmf(&self, index: usize) -> f64;
    fn len(&self) -> usize;
}

/// Discrete cumulative distribution at support index `k` (inclusive).
pub trait CdfDiscrete {
    fn cdf_at(&self, index: usize) -> f64;
    fn len(&self) -> usize;
}
