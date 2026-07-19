use crate::core::device::Device;
use crate::core::error::{Result, TensorError};
use crate::nn::initializers::*;
use crate::core::traits::TensorFloat;
use std::any::Any;

/// A trait representing a buffer of data on a specific device.
///
/// This trait provides a hardware-agnostic interface for memory buffers
/// that can store tensor data.
// --- Storage Trait ---
pub trait Storage: 'static + Any {
    type Elem: TensorFloat;

    fn as_any(&self) -> &dyn Any;

    fn get_data(&self) -> &Vec<Self::Elem>;
    fn update_data(&mut self, a: Vec<Self::Elem>);

    fn device(&self) -> Device;

    fn new(size: usize) -> Self
    where
        Self: Sized;
    fn zeros(size: usize) -> Self
    where
        Self: Sized;
    fn len(&self) -> usize;
    fn fill_ones(&mut self);

    fn from_data(data: Vec<Self::Elem>) -> Self
    where
        Self: Sized;
    fn ones(size: usize) -> Self
    where
        Self: Sized;
    fn fill_data(&mut self, data: Vec<Self::Elem>);
    fn add_grad(&mut self, data: Vec<Self::Elem>);
    fn get_dim_slice(self, dim: usize, gap: usize) -> Vec<Self::Elem>;
}

// --- CpuStorage Implementation ---
#[derive(Debug, Clone)]
pub struct CpuStorage<T: TensorFloat> {
    pub data: Vec<T>,
}
