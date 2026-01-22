mod backend;
mod cpu_backend;
mod cpubinaryops;
// mod cuda_bindings;
mod cpuurnaryops;
mod device;
mod error;
mod gemm;
// mod hrm;
mod initializers;
mod tensorbackprop;
mod tensorurnaryops;
// mod linear;
mod optimizer;
mod shared;
mod storage;
mod tensor;
mod tensorbinaryops;
mod traits;

use crate::device::Device;
// use crate::cpuurnaryops::sigmoid;
use crate::error::Result;
use crate::optimizer::SGD;
use crate::tensor::{Tensor, TensorHandle};
// use crate::storage::{CpuStorage, Storage};
// use crate::tensorbinaryops::TensorBinaryOps;
use crate::tensorurnaryops::TensorUrnaryOps;
use crate::traits::TensorFloat;
// use std::any::type_name;
use std::time::Instant;

pub fn give_t<T: TensorFloat>(input: f64) -> T {
    T::from(input).unwrap()
}

fn main() {
    let x = Tensor::<f32>::new(vec![1, 4], true, Device::Cpu);
    let y = Tensor::<f32>::new(vec![1, 1], true, Device::Cpu);
    let w1 = Tensor::<f32>::new(vec![4, 4], true, Device::Cpu);
    let w2 = Tensor::<f32>::new(vec![4, 20], true, Device::Cpu);
    let w3 = Tensor::<f32>::new(vec![20, 4], true, Device::Cpu);
    let sgd = SGD {
        learning_rate: 0.01,
    };
    for i in 0..10 {
        let ir = TensorHandle::matmul(x.0.clone(), w1.0.clone());
        ir.sigmoid();
        let ir2 = TensorHandle::matmul(ir.0.clone(), w2.0.clone());
        ir2.sigmoid();
        let result = TensorHandle::matmul(ir2.0.clone(), w3.0.clone());
        let epsilon = give_t(0.000000001);
        let ir1 = TensorHandle::SV_add(result.clone(), epsilon,"cpu".to_string());
        let irlog = ir1.log();
        let irmul = TensorHandle::matmul(result.0.clone(), irlog.0.clone());
        let irloss = irmul.sum();
        let loss = irloss.neg();
        loss.backward();
        sgd.update(&loss);
        // loss = -np.sum(y_true * np.log(y_pred + epsilon))
    }
}
