use crate::tensor::Tensor;
use crate::nn::module::Module;
use crate::core::traits::TensorFloat;

#[derive(Debug)]
pub struct Embedding<T: TensorFloat>{
    pub shape:Vec<usize>,
    pub data: Tensor<T>, 
}

impl<T: TensorFloat> Embedding<T>{

    pub fn new(rows:usize, columns:usize) -> Self{
        Self{
            shape: vec![rows,columns],
            data: Tensor::new(vec![rows, columns],true,"Embedding".to_string()),
        }
    }
}

impl<T: TensorFloat> Module<T> for Embedding<T>{

     fn forward(&self, _input:Tensor<T>) -> Tensor<T>{
        self.data.clone()
    }
    fn backward(&mut self,previous_grad:Tensor<T>) -> Tensor<T>{
        //TensorData::default()
        unimplemented!();
    }
}
