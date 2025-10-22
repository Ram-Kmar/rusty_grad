use crate::device::Device;
use crate::error::{Result, TensorError};
use crate::initializers::*;
use crate::traits::TensorFloat;
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

    // fn get_mut_data(self) -> Self;

    fn device(&self) -> Device;

    fn new(size: usize) -> Self
    where
        Self: Sized;
    fn zeros(size: usize) -> Self
    where
        Self: Sized;
    fn len(&self) -> usize;
    fn fill_ones(&mut self);
    // fn add(&Vec<Self::Elem>,&Vec<Self::Elem>) -> Self
    //     where
    //         Self:Sized;
    //
    // fn add(&self, b: &CpuStorage<Self::Elem>) -> Self
    // where
    //     Self: Sized;
    fn from_data(data: Vec<Self::Elem>) -> Self
    where
        Self: Sized;
    fn ones(size: usize) -> Self
    where
        Self: Sized;
    fn fill_data(&mut self, data: Vec<Self::Elem>);
    fn add_grad(&mut self, data: Vec<Self::Elem>);
    fn get_dim_slice(self, dim: usize, gap: usize) -> Vec<Self::Elem>;
    // fn fill_ones(&self) -> Self
    // where
    //     Self: Sized;
}

// --- CpuStorage Implementation ---
#[derive(Debug, Clone)]
pub struct CpuStorage<T: TensorFloat> {
    pub data: Vec<T>,
}

// #[derive(Debug, Clone)]
// pub struct CudaStorage<T: TensorFloat> {
//     pub data: Vec<T>,
// }
//
// impl<T: TensorFloat> Storage for CudaStorage<T> {
//     type Elem = T;
//
//     fn device(&self) -> Device {
//         Device::Cpu
//     }
//
//     fn new(size: usize) -> Self {
//         let my_gpu_buffer =
//
//
//     }
// }

// // --- Generic sum function using associated type ---
// fn sum<S: Storage>(a: &S::Elem, b: &S::Elem) -> S::Elem {
//     *a + *b
// }
