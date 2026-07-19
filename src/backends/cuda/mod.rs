pub mod bindings;

use crate::backends::Backend;
use crate::core::device::Device;
use crate::core::error::Result;
use crate::core::storage::Storage;
use crate::tensor::Tensor;
use crate::core::traits::TensorFloat;

//~~~~~~~~~ CUDA BACKEND (SKELETON) ~~~~~~~~~//

/// A backend for computations on a CUDA-enabled GPU.
/// NOTE: This is a skeleton and requires a proper CUDA library (like cudarc) to be implemented.

/// A placeholder for CUDA-specific storage. This would wrap a `CudaSlice` from a library like `cudarc`.
#[derive(Debug, Clone)]
pub struct CudaStorage<T: TensorFloat> {
    pub data: Vec<T>,
    pub device: Device,
}

impl<T: TensorFloat> Storage for CudaStorage<T> {
    fn as_any(&self) -> &dyn std::any::Any { todo!() }
    fn get_data(&self) -> &Vec<Self::Elem> { todo!() }
    fn update_data(&mut self, _a: Vec<Self::Elem>) { todo!() }
    fn new(_size: usize) -> Self { todo!() }
    fn zeros(_size: usize) -> Self { todo!() }
    fn len(&self) -> usize { todo!() }
    fn fill_ones(&mut self) { todo!() }
    fn from_data(_data: Vec<Self::Elem>) -> Self { todo!() }
    fn ones(_size: usize) -> Self { todo!() }
    fn fill_data(&mut self, _data: Vec<Self::Elem>) { todo!() }
    fn add_grad(&mut self, _data: Vec<Self::Elem>) { todo!() }
    fn get_dim_slice(self, _dim: usize, _gap: usize) -> Vec<Self::Elem> { todo!() }
    type Elem = T;
    fn device(&self) -> Device { todo!() }
}

pub struct CudaBackend;

impl<T: TensorFloat> Backend<T> for CudaBackend {
    type Storage = CudaStorage<T>;

    fn zeros(_shape: &[usize], _device: Device) -> Result<Tensor<T>> { todo!() }
    fn random_uniform(_shape: &[usize], _device: Device, _min: f64, _max: f64) -> Result<Tensor<T>> { todo!() }
    fn from_cpu_data(_data: &[T], _shape: &[usize], _device: Device) -> Result<Tensor<T>> { todo!() }
    fn ones(_shape: &[usize], _device: Device) -> Result<Tensor<T>> { todo!() }
    fn add(_a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>> { todo!() }
    fn sub(_a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>> { todo!() }
    fn mul(_a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>> { todo!() }
    fn div(_a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>> { todo!() }
    fn matmul(_a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>> { todo!() }
    fn relu(_input: &Tensor<T>) -> Result<Tensor<T>> { todo!() }
    fn exp(_input: &Tensor<T>) -> Result<Tensor<T>> { todo!() }
    fn log(_input: &Tensor<T>) -> Result<Tensor<T>> { todo!() }
    fn transpose(_input: &Tensor<T>) -> Result<Tensor<T>> { todo!() }
}
