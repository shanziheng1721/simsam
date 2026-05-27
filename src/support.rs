use crate::error::BuildError;

/// Finite support interval `[lo, hi]` with `lo < hi`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    pub fn new(lo: f64, hi: f64) -> Result<Self, BuildError> {
        let interval = Self { lo, hi };
        interval.validate()?;
        Ok(interval)
    }

    pub fn validate(&self) -> Result<(), BuildError> {
        if !self.lo.is_finite() || !self.hi.is_finite() {
            return Err(BuildError::NonFiniteValue("support bounds"));
        }
        if self.lo >= self.hi {
            return Err(BuildError::InvalidSupport("lo must be strictly less than hi"));
        }
        Ok(())
    }

    pub fn contains(&self, x: f64) -> bool {
        x >= self.lo && x <= self.hi
    }

    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }
}
