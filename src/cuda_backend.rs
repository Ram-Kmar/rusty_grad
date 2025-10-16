use crate::backend::Backend;
use crate::device::Device;
use crate::error::Result;
use crate::storage::Storage;
use crate::tensor::Tensor;
use crate::traits::TensorFloat;

//~~~~~~~~~ CUDA BACKEND (SKELETON) ~~~~~~~~~//

/// A backend for computations on a CUDA-enabled GPU.
/// NOTE: This is a skeleton and requires a proper CUDA library (like cudarc) to be implemented.
pub struct CudaBackend;

/// A placeholder for CUDA-specific storage. This would wrap a `CudaSlice` from a library like `cudarc`.
#[derive(Debug, Clone)]
pub struct CudaStorage<T: TensorFloat> {
    // In a real implementation, this would be something like:
    // data: cudarc::driver::CudaSlice<T>,
    _phantom: std::marker::PhantomData<T>,
}

// Placeholder implementation of the Storage trait for our skeleton CudaStorage.
impl<T: TensorFloat> Storage for CudaStorage<T> {
    type Elem = T;
    fn device(&self) -> Device {
        todo!()
    }
    fn from_cpu(device: Device, data: &[Self::Elem]) -> Result<Self> {
        todo!()
    }
    fn to_cpu_vec(&self) -> Result<Vec<Self::Elem>> {
        todo!()
    }
    fn as_slice(&self) -> &[Self::Elem] {
        todo!()
    }
}

impl Backend for CudaBackend {
    type Storage = CudaStorage<f32>; // For now, we'll specify f32.

    fn zeros(shape: &[usize], device: Device) -> Result<Tensor<Self::Storage>> {
        todo!()
    }

    fn ones(shape: &[usize], device: Device) -> Result<Tensor<Self::Storage>> {
        todo!()
    }

    fn from_cpu_data<T: TensorFloat>(
        data: &[T],
        shape: &[usize],
        device: Device,
    ) -> Result<Tensor<Self::Storage>> {
        todo!()
    }

    fn add(a: &Tensor<Self::Storage>, b: &Tensor<Self::Storage>) -> Result<Tensor<Self::Storage>> {
        todo!()
    }

    fn sub(a: &Tensor<Self::Storage>, b: &Tensor<Self::Storage>) -> Result<Tensor<Self::Storage>> {
        todo!()
    }

    fn mul(a: &Tensor<Self::Storage>, b: &Tensor<Self::Storage>) -> Result<Tensor<Self::Storage>> {
        todo!()
    }

    fn div(a: &Tensor<Self::Storage>, b: &Tensor<Self::Storage>) -> Result<Tensor<Self::Storage>> {
        todo!()
    }

    fn matmul(
        a: &Tensor<Self::Storage>,
        b: &Tensor<Self::Storage>,
    ) -> Result<Tensor<Self::Storage>> {
        todo!()
    }

    fn relu(input: &Tensor<Self::Storage>) -> Result<Tensor<Self::Storage>> {
        todo!()
    }

    fn exp(input: &Tensor<Self::Storage>) -> Result<Tensor<Self::Storage>> {
        todo!()
    }

    fn log(input: &Tensor<Self::Storage>) -> Result<Tensor<Self::Storage>> {
        todo!()
    }

    fn transpose(input: &Tensor<Self::Storage>) -> Result<Tensor<Self::Storage>> {
        todo!()
    }
}
