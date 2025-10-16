#[derive(Debug)]
pub enum TensorError {
    ShapeMismatch,
    InvalidOperation,
    InvalidShape(String),
    DeviceMismatch,
    BackendMismatch,
}

pub type Result<T> = std::result::Result<T, TensorError>;
