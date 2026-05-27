use crate::continuous::traits::{HasSupport, Pdf};
use crate::error::BuildError;
use crate::support::Interval;

/// Piecewise-constant PDF from a histogram (SciPy [`rv_histogram`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_histogram.html)).
#[derive(Debug, Clone)]
pub struct HistogramPdf {
    edges: Vec<f64>,
    heights: Vec<f64>,
    support: Interval,
}

impl HistogramPdf {
    /// `edges` has length `bins + 1`; `counts` has length `bins`.
    ///
    /// If `density` is true, heights are `count / width`; otherwise `count / (width * total)`.
    pub fn new(edges: Vec<f64>, counts: Vec<f64>, density: bool) -> Result<Self, BuildError> {
        if edges.len() < 2 || counts.len() + 1 != edges.len() {
            return Err(BuildError::InvalidHistogram);
        }
        for w in edges.windows(2) {
            if w[1] <= w[0] || !w[0].is_finite() || !w[1].is_finite() {
                return Err(BuildError::InvalidHistogram);
            }
        }
        for &c in &counts {
            if c < 0.0 || !c.is_finite() {
                return Err(BuildError::InvalidHistogram);
            }
        }
        let total: f64 = counts.iter().sum();
        if total <= 0.0 {
            return Err(BuildError::NormalizationFailed);
        }
        let mut heights = Vec::with_capacity(counts.len());
        for (i, &c) in counts.iter().enumerate() {
            let width = edges[i + 1] - edges[i];
            let h = if density {
                c / width
            } else {
                c / (width * total)
            };
            heights.push(h);
        }
        let support = Interval::new(edges[0], *edges.last().unwrap())?;
        Ok(Self {
            edges,
            heights,
            support,
        })
    }

    pub fn edges(&self) -> &[f64] {
        &self.edges
    }

    pub fn heights(&self) -> &[f64] {
        &self.heights
    }
}

impl Pdf for HistogramPdf {
    fn pdf(&self, x: f64) -> f64 {
        if x < self.support.lo || x >= self.support.hi {
            return 0.0;
        }
        for i in 0..self.heights.len() {
            if x >= self.edges[i] && x < self.edges[i + 1] {
                return self.heights[i];
            }
        }
        if x == self.support.hi {
            *self.heights.last().unwrap_or(&0.0)
        } else {
            0.0
        }
    }
}

impl HasSupport for HistogramPdf {
    fn support(&self) -> Interval {
        self.support
    }
}
