use crate::discrete::DiscreteSampler;

impl DiscreteSampler {
    /// Normalized probability mass at support index `i`.
    pub fn pmf_at(&self, index: usize) -> f64 {
        let prev = if index == 0 {
            0.0
        } else {
            self.cdf_at(index - 1)
        };
        self.cdf_at(index) - prev
    }

    /// Probability mass at `x` (exact support match only).
    pub fn pmf(&self, x: f64) -> f64 {
        (0..self.len())
            .find(|&i| (self.point(i) - x).abs() < 1e-12)
            .map(|i| self.pmf_at(i))
            .unwrap_or(0.0)
    }

    /// `E[X]`.
    pub fn mean(&self) -> f64 {
        (0..self.len())
            .map(|i| self.point(i) * self.pmf_at(i))
            .sum()
    }

    /// `Var[X]`.
    pub fn var(&self) -> f64 {
        let m = self.mean();
        (0..self.len())
            .map(|i| (self.point(i) - m).powi(2) * self.pmf_at(i))
            .sum()
    }

    pub fn std(&self) -> f64 {
        self.var().sqrt()
    }
}
