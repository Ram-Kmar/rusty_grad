use crate::tensor::Tensor;
use std::fmt::Debug;
use crate::core::traits::TensorFloat;

pub trait Module<T: TensorFloat>: Debug {
    fn forward(&self, input: Tensor<T>) -> Tensor<T>;
    fn backward(&mut self, previous_grad: Tensor<T>) -> Tensor<T>;
}
