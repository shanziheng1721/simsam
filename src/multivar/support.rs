use crate::error::BuildError;

/// Axis-aligned bounded support in R^n: `lo[i] <= x[i] <= hi[i]`.
#[derive(Debug, Clone, PartialEq)]
pub struct HyperRect {
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
}

impl HyperRect {
    pub fn new(lo: Vec<f64>, hi: Vec<f64>) -> Result<Self, BuildError> {
        let rect = Self { lo, hi };
        rect.validate()?;
        Ok(rect)
    }

    pub fn dim(&self) -> usize {
        self.lo.len()
    }

    pub fn validate(&self) -> Result<(), BuildError> {
        if self.lo.is_empty() || self.hi.is_empty() {
            return Err(BuildError::InvalidSupport("hyper-rect cannot be empty"));
        }
        if self.lo.len() != self.hi.len() {
            return Err(BuildError::InvalidSupport(
                "hyper-rect lo/hi must have same dimension",
            ));
        }
        for i in 0..self.lo.len() {
            let lo = self.lo[i];
            let hi = self.hi[i];
            if !lo.is_finite() || !hi.is_finite() {
                return Err(BuildError::NonFiniteValue("hyper-rect bounds"));
            }
            if lo >= hi {
                return Err(BuildError::InvalidSupport("each lo must be < hi"));
            }
        }
        Ok(())
    }

    pub fn contains(&self, x: &[f64]) -> bool {
        if x.len() != self.dim() {
            return false;
        }
        for i in 0..x.len() {
            if x[i] < self.lo[i] || x[i] > self.hi[i] {
                return false;
            }
        }
        true
    }
}

