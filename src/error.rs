use core::fmt;

/// Error while constructing a distribution or sampler.
#[derive(Debug, Clone, PartialEq)]
pub enum BuildError {
    InvalidSupport(&'static str),
    EmptyDiscreteSupport,
    NonPositiveMass,
    NonFiniteValue(&'static str),
    NormalizationFailed,
    SymbolicIntegrationFailed,
    SimsymEval(simsym::EvalError),
    InvalidScale,
    InvalidHistogram,
    TruncationOutOfSupport,
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSupport(msg) => write!(f, "invalid support interval: {msg}"),
            Self::EmptyDiscreteSupport => write!(f, "discrete distribution has no support points"),
            Self::NonPositiveMass => write!(f, "probability mass must be positive"),
            Self::NonFiniteValue(ctx) => write!(f, "non-finite value in {ctx}"),
            Self::NormalizationFailed => write!(f, "failed to normalize PDF on support"),
            Self::SymbolicIntegrationFailed => write!(f, "symbolic integration failed"),
            Self::SimsymEval(e) => write!(f, "simsym evaluation error: {e}"),
            Self::InvalidScale => write!(f, "scale must be finite and positive"),
            Self::InvalidHistogram => write!(f, "invalid histogram edges or counts"),
            Self::TruncationOutOfSupport => {
                write!(f, "truncation interval must lie inside base support")
            }
        }
    }
}

impl std::error::Error for BuildError {}

impl From<simsym::EvalError> for BuildError {
    fn from(e: simsym::EvalError) -> Self {
        Self::SimsymEval(e)
    }
}

/// Error during sampling (e.g. invalid quantile).
#[derive(Debug, Clone, PartialEq)]
pub enum SampleError {
    QuantileOutOfRange { u: f64 },
    PpfFailed,
    PdfRequired,
    IntegrationFailed,
    InvalidConfidence { confidence: f64 },
}

impl fmt::Display for SampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QuantileOutOfRange { u } => write!(f, "quantile u={u} not in (0, 1)"),
            Self::PpfFailed => write!(f, "failed to invert CDF"),
            Self::PdfRequired => write!(f, "operation requires an underlying PDF"),
            Self::IntegrationFailed => write!(f, "numerical integration failed"),
            Self::InvalidConfidence { confidence } => {
                write!(f, "confidence {confidence} must be in (0, 1)")
            }
        }
    }
}

impl std::error::Error for SampleError {}
