use crate::tensor::Tensor;

use crate::core::traits::TensorFloat;

impl<T: TensorFloat> Tensor<T> {
    pub fn print(&self){
        for i in &self.data{
            println!("this is a from the sample.rs {}",i);
        }
    }
}
