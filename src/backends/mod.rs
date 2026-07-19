pub mod cpu;
// pub mod cuda;

use crate::core::device::Device;
use crate::core::error::Result;
use crate::core::storage::Storage;
use crate::tensor::TensorData;
use crate::core::traits::TensorFloat;

//~~~~~~~~~ TRAIT DEFINITION (MODIFIED) ~~~~~~~~~//

/// A trait that defines the computational operations for a specific backend (e.g., CPU, CUDA).
///
/// This trait is now generic over the element type `T` to align with the new TensorData struct.
pub trait Backend<T: TensorFloat> {
    /// The storage type that this backend operates on. It must match the element type `T`.
    type Storage: Storage<Elem = T>;

    // --- TensorData Creation --- //

    fn zeros(shape: &[usize], device: Device) -> Result<TensorData<T>>;
    fn ones(shape: &[usize], device: Device) -> Result<TensorData<T>>;
    fn from_cpu_data(data: &[T], shape: &[usize], device: Device) -> Result<TensorData<T>>;
    fn random_uniform(
        shape: &[usize],
        device: Device,
        min: f64,
        max: f64,
    ) -> Result<TensorData<T>>;

    // --- Binary Operations --- //

    fn add(a: &TensorData<T>, b: &TensorData<T>) -> Result<TensorData<T>>;
    fn sub(a: &TensorData<T>, b: &TensorData<T>) -> Result<TensorData<T>>;
    fn mul(a: &TensorData<T>, b: &TensorData<T>) -> Result<TensorData<T>>;
    fn div(a: &TensorData<T>, b: &TensorData<T>) -> Result<TensorData<T>>;
    fn matmul(a: &TensorData<T>, b: &TensorData<T>) -> Result<TensorData<T>>;

    // --- Unary Operations --- //

    fn relu(input: &TensorData<T>) -> Result<TensorData<T>>;
    fn exp(input: &TensorData<T>) -> Result<TensorData<T>>;
    fn log(input: &TensorData<T>) -> Result<TensorData<T>>;

    // --- Movement Operations --- //

    fn transpose(input: &TensorData<T>) -> Result<TensorData<T>>;
}
