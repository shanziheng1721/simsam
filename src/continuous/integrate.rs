use crate::continuous::traits::Pdf;

const DEFAULT_QUAD_TOL: f64 = 1e-10;
const MAX_DEPTH: u32 = 48;

/// Adaptive Simpson quadrature of `f` on `[a, b]`.
pub fn integrate_fn<F>(f: &F, a: f64, b: f64, tol: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    if a == b {
        return 0.0;
    }
    if b < a {
        return -integrate_fn(f, b, a, tol);
    }
    let mut stack = vec![(a, b, simpson(f, a, b))];
    let mut total = 0.0;
    while let Some((a, b, s0)) = stack.pop() {
        let m = 0.5 * (a + b);
        let s_left = simpson(f, a, m);
        let s_right = simpson(f, m, b);
        let s1 = s_left + s_right;
        if (s1 - s0).abs() <= 15.0 * tol || stack.len() > MAX_DEPTH as usize {
            total += s1 + (s1 - s0) / 15.0;
        } else {
            stack.push((m, b, s_right));
            stack.push((a, m, s_left));
        }
    }
    total
}

fn simpson<F>(f: &F, a: f64, b: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    let m = 0.5 * (a + b);
    let h = b - a;
    (h / 6.0) * (f(a) + 4.0 * f(m) + f(b))
}

/// ∫_a^x pdf(t) dt
pub fn integrate_pdf<P>(pdf: &P, a: f64, x: f64, tol: f64) -> f64
where
    P: Pdf,
{
    integrate_fn(&|t| pdf.pdf(t), a, x, tol)
}

/// Normalizing constant ∫_lo^hi pdf.
pub fn normalize_pdf<P>(pdf: &P, lo: f64, hi: f64, tol: f64) -> Result<f64, crate::error::BuildError>
where
    P: Pdf,
{
    let z = integrate_pdf(pdf, lo, hi, tol);
    if !z.is_finite() || z <= 0.0 {
        return Err(crate::error::BuildError::NormalizationFailed);
    }
    Ok(z)
}

pub fn default_quad_tol() -> f64 {
    DEFAULT_QUAD_TOL
}
