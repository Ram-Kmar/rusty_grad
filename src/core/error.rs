use thiserror::Error;

#[derive(Error, Debug)]
pub enum TensorError {
    #[error("Shape mismatch: expected {expected:?} but found {found:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        found: Vec<usize>,
    },

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Invalid shape provided: {0}")]
    InvalidShape(String),

    #[error("Device mismatch between tensors")]
    DeviceMismatch,

    #[error("Backend mismatch between tensors")]
    BackendMismatch,
}

pub type Result<T> = std::result::Result<T, TensorError>;
