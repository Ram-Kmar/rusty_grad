use crate::tensor::TensorHandle;
use std::fmt::Debug;
use crate::traits::TensorFloat;

pub trait Module<T: TensorFloat>: Debug {
    fn forward(&self, input: TensorHandle<T>) -> TensorHandle<T>;
    fn backward(&mut self, previous_grad: TensorHandle<T>) -> TensorHandle<T>;
}
