use rusty_grad::core::device::Device;
use rusty_grad::nn::optimizer::SGD;
use rusty_grad::tensor::{TensorData, Tensor};
use rusty_grad::tensor::ops::unary::TensorUrnaryOps;
use rusty_grad::core::traits::TensorFloat;

pub fn give_t<T: TensorFloat>(input: f64) -> T {
    T::from(input).unwrap()
}

fn main() {
    let x = Tensor::<f32>::new(vec![5,5],true, Device::Cpu);
    let y = Tensor::<f32>::new(vec![5,5],true, Device::Cpu);
    let z = Tensor::matmul(x.0.clone(), y.0.clone()).relu();
    println!("This is z.data {}", z);

    // for i in 0..10 {
    //     let ir = Tensor::matmul(x.0.clone(), w1.0.clone());
    //     ir.sigmoid();
    //     println!("This is ir{}", ir);
    //     let ir2 = Tensor::matmul(ir.0.clone(), w2.0.clone());
    //     ir2.sigmoid();
    //     let result = Tensor::matmul(ir2.0.clone(), w3.0.clone());
    //     let epsilon = give_t(0.000000001);
    //     let ir1 = Tensor::SV_add(result.clone(), epsilon,"cpu".to_string());
    //     let irlog = ir1.log();
    //     let irmul = Tensor::matmul(result.0.clone(), irlog.0.clone());
    //     let irloss = irmul.sum();
    //     let loss = irloss.neg();
    //     loss.backward();
    //     sgd.update(&loss);
    //     // loss = -np.sum(y_true * np.log(y_pred + epsilon))
    // }
}
