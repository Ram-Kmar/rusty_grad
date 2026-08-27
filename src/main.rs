use rusty_grad::core::device::Device;
use rusty_grad::nn::optimizer::SGD;
use rusty_grad::tensor::{TensorData, Tensor};
use rusty_grad::tensor::ops::unary::TensorUrnaryOps;
use rusty_grad::core::traits::TensorFloat;


fn main() {
    let x = Tensor::<f32>::new(vec![5,5],true, Device::Cpu);
    let y = Tensor::<f32>::new(vec![5,5],true, Device::Cpu);
    let z = Tensor::matmul(x.0.clone(), y.0.clone()).relu();
    let irloss = z.sum();
    let loss = irloss.neg();
    loss.backward();
    println!("This is z.data {}", z);
}
