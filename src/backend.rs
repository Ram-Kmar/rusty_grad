use crate::device::Device;
use crate::error::Result;
use crate::storage::Storage;
use crate::tensor::Tensor;
use crate::traits::TensorFloat;

//~~~~~~~~~ TRAIT DEFINITION (MODIFIED) ~~~~~~~~~//

/// A trait that defines the computational operations for a specific backend (e.g., CPU, CUDA).
///
/// This trait is now generic over the element type `T` to align with the new Tensor struct.
pub trait Backend<T: TensorFloat> {
    /// The storage type that this backend operates on. It must match the element type `T`.
    type Storage: Storage<Elem = T>;

    // --- Tensor Creation --- //

    fn zeros(shape: &[usize], device: Device) -> Result<Tensor<T>>;
    fn ones(shape: &[usize], device: Device) -> Result<Tensor<T>>;
    fn from_cpu_data(data: &[T], shape: &[usize], device: Device) -> Result<Tensor<T>>;

    // --- Binary Operations --- //

    fn add(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>;
    fn sub(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>;
    fn mul(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>;
    fn div(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>;
    fn matmul(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>;

    // --- Unary Operations --- //

    fn relu(input: &Tensor<T>) -> Result<Tensor<T>>;
    fn exp(input: &Tensor<T>) -> Result<Tensor<T>>;
    fn log(input: &Tensor<T>) -> Result<Tensor<T>>;

    // --- Movement Operations --- //

    fn transpose(input: &Tensor<T>) -> Result<Tensor<T>>;
}
