use crate::tensor::TensorHandle;
use crate::module::Module;
use crate::traits::TensorFloat;

#[derive(Debug)]
pub struct Embedding<T: TensorFloat>{
    pub shape:Vec<usize>,
    pub data: TensorHandle<T>, 
}

impl<T: TensorFloat> Embedding<T>{

    pub fn new(rows:usize, columns:usize) -> Self{
        Self{
            shape: vec![rows,columns],
            data: TensorHandle::new(vec![rows, columns],true,"Embedding".to_string()),
        }
    }
}

impl<T: TensorFloat> Module<T> for Embedding<T>{

     fn forward(&self, _input:TensorHandle<T>) -> TensorHandle<T>{
        self.data.clone()
    }
    fn backward(&mut self,previous_grad:TensorHandle<T>) -> TensorHandle<T>{
        //Tensor::default()
        unimplemented!();
    }
}
