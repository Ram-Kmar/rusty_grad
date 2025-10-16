use crate::tensor::Tensor;


impl Tensor{
    pub fn print(&self){
        for i in &self.data{
            println!("this is a from the sample.rs {}",i);
        }
    }
}
