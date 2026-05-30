use crate::continuous::cdf::IntegratedPdf;
use crate::continuous::tdr::Dpdf;
use crate::continuous::traits::{HasSupport, Pdf};
use crate::support::Interval;

/// Central difference `(f(x+h) - f(x-h)) / (2h)` with one-sided fallback near boundaries.
pub fn central_diff(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    let h = h.abs().max(f64::EPSILON);
    let fp = f(x + h);
    let fm = f(x - h);
    if fp.is_finite() && fm.is_finite() {
        (fp - fm) / (2.0 * h)
    } else {
        f64::NAN
    }
}

/// Numerical derivative of a normalized PDF on a finite support interval.
pub struct NumericalDpdf<'a, P> {
    pdf: &'a IntegratedPdf<P>,
    rel_step: f64,
}

impl<'a, P> NumericalDpdf<'a, P>
where
    P: Pdf,
{
    pub fn new(pdf: &'a IntegratedPdf<P>, rel_step: f64) -> Self {
        Self { pdf, rel_step }
    }

    fn step(&self) -> f64 {
        let support = self.pdf.support();
        let width = support.hi - support.lo;
        (self.rel_step * width).max(f64::EPSILON.sqrt())
    }
}

impl<P> Dpdf for NumericalDpdf<'_, P>
where
    P: Pdf,
{
    fn dpdf(&self, x: f64) -> f64 {
        let support = self.pdf.support();
        let h = self.step();
        let f = |t: f64| self.pdf.normalized_pdf(t);

        if x - h >= support.lo && x + h <= support.hi {
            central_diff(f, x, h)
        } else if x + h <= support.hi {
            // forward difference
            let fp = f(x + h);
            let fx = f(x);
            if fp.is_finite() && fx.is_finite() {
                (fp - fx) / h
            } else {
                f64::NAN
            }
        } else if x - h >= support.lo {
            // backward difference
            let fx = f(x);
            let fm = f(x - h);
            if fx.is_finite() && fm.is_finite() {
                (fx - fm) / h
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        }
    }
}

impl<P> HasSupport for NumericalDpdf<'_, P>
where
    P: Pdf,
{
    fn support(&self) -> Interval {
        self.pdf.support()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::continuous::traits::PdfFn;
    use crate::support::Interval;

    #[test]
    fn numerical_dpdf_quadratic() {
        let support = Interval::new(-1.0, 1.0).unwrap();
        let raw = PdfFn::new(|x| 1.0 - x * x, support);
        let integrated = IntegratedPdf::new(raw, support, 1e-10).unwrap();
        let nd = NumericalDpdf::new(&integrated, 1e-4);
        for &x in &[-0.5, 0.0, 0.5] {
            let h = nd.step();
            let approx = nd.dpdf(x);
            let exact = central_diff(|t| integrated.normalized_pdf(t), x, h);
            assert!(
                (approx - exact).abs() < 1e-3,
                "x={x} approx={approx} exact={exact}"
            );
        }
    }
}
