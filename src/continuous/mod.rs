mod cdf;
mod dpdf;
mod hermite;
mod histogram;
mod integrate;
mod invert;
mod loc_scale;
mod sampler;
mod stats;
mod tdr;
#[cfg(feature = "symbolic")]
mod symbolic;
mod traits;
mod truncated;

pub use cdf::{AffineCdf, IntegratedPdf};
pub use integrate::default_quad_tol;
pub use hermite::HermitePpfTable;
pub use histogram::HistogramPdf;
pub use invert::InvertOptions;
pub use loc_scale::LocScale;
pub use sampler::{
    from_cdf_fn, from_histogram, from_pdf_dpdf_fn, from_pdf_fn, from_pdf_fn_with_options,
    from_pdf_loc_scale, BuildOptions, ContinuousSampler, CdfSource, PpfMethod, SampleMethod,
    TdrBuildConfig,
};
pub use tdr::{tdr_from_fns, Dpdf, DpdfFn, TdrHat, TdrOptions, TdrSampler, TdrTransform};
#[cfg(feature = "symbolic")]
pub use tdr::symbolic::SymbolicPdfDpdf1d;
#[cfg(feature = "symbolic")]
pub use symbolic::{SymbolicContinuous, SymbolicPdfAdapter};
pub use traits::{Cdf, CdfFn, HasSupport, Pdf, PdfFn};
pub use truncated::Truncated;
