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

use crate::backend::Backend;
use crate::cpuurnaryops::sigmoid;
// use crate::cpu_backend::CpuBackend;
use crate::device::Device;
use crate::error::Result;
use crate::optimizer::Sgd;
use crate::storage::{CpuStorage, Storage};
use crate::tensor::{Tensor, TensorHandle};
use crate::tensorbinaryops::TensorBinaryOps;
use crate::tensorurnaryops::TensorUrnaryOps;
use crate::traits::TensorFloat;
use std::any::type_name;
use std::ops::Add;

pub struct test {
    data: Vec<f32>,
}

impl Add for &test {
    type Output = test;
    fn add(self, rhs: &test) -> test {
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(x, y)| x + y)
            .collect();
        test { data: data }
    }
}

fn print_type_of<T>(_: &T) {
    println!("Type: {}", std::any::type_name::<T>());
}
// this is c [-0.017901637, -0.005195707, 0.055670634, -0.03256449, -0.07355864, -0.010251187, -0.26222235, -0.06694054, -0.39142877, 0.1030898, 0.23791271, -0.37167054, -0.03707202, -0.105934605, -0.5205066, 0.20631973, 0.13277031, -0.16509311, 0.2910568, -0.17109655, 0.04870092, -0.23087135, 0.012311934, -0.13111834, -0.21692447, 0.005659366, 0.032579985, 0.106928736, 0.17722386, 0.06564638, -0.11930749, -0.3270426, -0.12863702, 0.054531086, -0.2780119, 0.12978739, 0.15382196, 0.2804397, 0.24856412, -0.1643827, -0.12115116, 0.13952221, -0.1493939, 0.11313491, 0.19249041, 0.18636212, 0.20277607, -0.38971522, 0.023805052, -0.41601458, 0.039821062, -0.25299174, 0.00050481356, -0.10433393, -0.32950246, 0.10366926, -0.11522423, -0.08426422, -0.12450547, -0.0208868, -0.17963248, 0.1111751, -0.07798356, 0.18599501, 0.25871453, -0.08117168, 0.19588502, 0.39376694, 0.2556423, -0.026143987, -0.13967389, -0.18560469, -0.026698947, -0.090261154, -0.1308265, 0.042640295, -0.108588815, 0.036003053, -0.15939684, 0.0013237856, 0.0013020709, 0.114763215, -0.013872549, 0.07167619, 0.020026743, 0.09492733, -0.121126376, -0.30235147, -0.32557738, -0.13292545, -0.24834925, -0.120372444, -0.08235924, 0.17588772, -0.06578336, 0.010410626, -0.07279524, 0.43843877, -0.13419758, -0.007751433]
// fn convert_t<T:TensorFloat>(input: Vec<f32>){
//     input
// }
pub fn give_t<T: TensorFloat>(input: f64) -> T {
    let epsilon = T::from(0.000000001).unwrap();
    epsilon
}

fn main() {
    let x = Tensor::<f32>::new(vec![1, 4], true, Device::Cpu);
    let y = Tensor::<f32>::new(vec![1, 1], true, Device::Cpu);
    let w1 = Tensor::<f32>::new(vec![4, 4], true, Device::Cpu);
    let w2 = Tensor::<f32>::new(vec![4, 20], true, Device::Cpu);
    let w3 = Tensor::<f32>::new(vec![20, 4], true, Device::Cpu);
    let sgd = Sgd {
        lr: 0.01,
        momentum: 0.0,
        weight_decay: 0.0,
        nesterov: false,
    };
    for i in 0..10 {
        let ir = TensorHandle::matmul(x.0.clone(), w1.0.clone());
        ir.sigmoid();
        let ir2 = TensorHandle::matmul(ir.0.clone(), w2.0.clone());
        ir2.sigmoid();
        let result = TensorHandle::matmul(ir2.0.clone(), w3.0.clone());
        let epsilon = give_t(0.000000001);
        let ir1 = TensorHandle::SV_add(result.clone(), epsilon);
        let irlog = ir1.log();
        let irmul = TensorHandle::matmul(result.0.clone(), irlog.0.clone());
        let irloss = irmul.sum();
        let loss = irloss.neg();
        loss.backward();
        sgd.update(&loss);
        // loss = -np.sum(y_true * np.log(y_pred + epsilon))
    }

    let b = Tensor::<f32>::new(vec![4, 3], true, Device::Cpu);
    let sgd = Sgd {
        lr: 0.01,
        momentum: 0.0,
        weight_decay: 0.0,
        nesterov: false,
    };

    println!(
        "A = torch.tensor( {:?}, requires_grad = True)",
        a.clone().data.borrow().get_data()
    );
    println!(
        "B = torch.tensor( {:?}, requires_grad = True)",
        b.clone().data.borrow().get_data()
    );
    let c = TensorHandle::matmul(a.0.clone(), b.0.clone());
    println!(
        "C = torch.tensor( {:?}, requires_grad = True)",
        c.clone().data.borrow().get_data()
    );
    let e = d.sigmoid();
    println!(
        "D = torch.tensor( {:?}, requires_grad = True)",
        d.clone().data.borrow().get_data()
    );
    // let c = TensorHandle::matmul(a.0.clone(), b.0.clone());
    e.backward();

    println!(
        "this is a_grad{:?}",
        a.clone().grad.as_ref().unwrap().clone().borrow().get_data()
    );
    println!(
        "this is b_grad{:?}",
        b.clone().grad.as_ref().unwrap().clone().borrow().get_data()
    );
    println!(
        "this is d_grad{:?}",
        d.clone().grad.as_ref().unwrap().clone().borrow().get_data()
    );
    println!(
        "this is e_grad{:?}",
        e.clone().grad.as_ref().unwrap().clone().borrow().get_data()
    );
    sgd.update(&e);
    println!(
        "A = torch.tensor( {:?}, requires_grad = True)",
        a.clone().data.borrow().get_data()
    );
    println!(
        "B = torch.tensor( {:?}, requires_grad = True)",
        b.clone().data.borrow().get_data()
    );
    println!(
        "C = torch.tensor( {:?}, requires_grad = True)",
        c.clone().data.borrow().get_data()
    );
    println!(
        "D = torch.tensor( {:?}, requires_grad = True)",
        d.clone().data.borrow().get_data()
    );

    // let a1 = a.clone().data.clone();
    // let b1 = b.clone().data.clone();
    // let a_vec = a1.get_data();
    // let b_vec = b1.get_data();
    // let a_transpose = cpubinaryops::transpose(&a_vec, a.shape[0], a.shape[1]);
    // let c = cpubinaryops::matmul(&a_vec, &b_vec, a.shape[0],a.shape[1], b.shape[1]);
    // let c_grad:Vec<f32>= vec![1.0;a.shape[0]* b.shape[1]];
    // let b_grad = cpubinaryops::matmul(&a_transpose, &c_grad, a.shape[1], a.shape[0], b.shape[1]);
    // println!("a_transpose = torch.tensor({:?}) ",&a_transpose);
    // println!("c_grad = torch.tensor({:?}) ",&c_grad);
    // println!("b_grad {:?} ",&b_grad);
    // println!("this is c {:?}",c);
    // let d = test {
    //     data: vec![10.0, 10.0],
    // };
    // let e = test {
    //     data: vec![10.0, 10.0],
    // println!("This is a {:?}", &a.clone().data.get_data());
    // println!("This is b {:?}", &b.clone().data.get_data());
    // println!("This is a {:?}", &a.is_child);
    // let c = &a + &b;
    //
    // let cmat = TensorHandle::matmul(a.0.clone(), b.0.clone());
    // println!(
    //     "this is cmat_grad{:?}",
    //     cmat.clone()
    //         .grad
    //         .as_ref()
    //         .unwrap()
    //         .clone()
    //         .borrow()
    //         .get_data()
    // );
    // println!(
    //     "this is the operation that maked c{}",
    //     a.operation.as_ref().unwrap()[0]
    // );
    // if "add" == c.operation.as_ref().unwrap()[0] {
    //     println!("this is worked");
    // }
    // cmat.backward();
    // c.backward();
    // let c = &a.clone() - &b.clone();
    // let c1 = &a.clone() / &b.clone();
    // let c2 = &a.clone() * &b.clone();
    // let c3 = &a.clone() + &b.clone();
    // let cmat = TensorHandle::matmul(a.0.clone(), b.0.clone());
    // println!("A = torch.tensor( {:?}, requires_grad = True)", a.clone().data.get_data());
    // println!("B = torch.tensor( {:?}, requires_grad = True)", b.clone().data.get_data());
    // println!("C = torch.tensor( {:?}, requires_grad = True)", cmat.clone().data.get_data());
    // println!(
    //     "this is a_grad{:?}",
    //     a.clone().grad.as_ref().unwrap().clone().borrow().get_data()
    // );
    // println!(
    //     "this is b_grad{:?}",
    //     b.clone().grad.as_ref().unwrap().clone().borrow().get_data()
    // );
    // println!(
    //     "this is cmat_grad{:?}",
    //     cmat.clone()
    //         .grad
    //         .as_ref()
    //         .unwrap()
    //         .clone()
    //         .borrow()
    //         .get_data()
    // );

    // println!("this is the size of cmat {:?}", cmat.shape);
    // println!("this is the size of a {:?}", a.shape);
    // println!("this is the size of b {:?}", b.shape);

    // println!("This is c1 {:?}", c1.0.data.get_data());
    // println!("This is c2 {:?}", c2.0.data.get_data());
    // println!("This is c3 {:?}", c3.0.data.get_data());
    // println!("This is cmat {:?}", cmat.0.data.get_data());
    // let c = CpuStorage::matmul(a.as_ref().unwrap().0.clone(), b.as_ref().unwrap().0.clone());
    // let d = CpuStorage::add(a.as_ref().unwrap().0.clone(), b.as_ref().unwrap().0.clone());
    // println!("This is c {:?}", c);
    // println!("This is d {:?}", d);
    //
    // let a_storage = a
    //     .as_ref()
    //     .unwrap()
    //     .0
    //     .clone()
    //     .data
    //     .as_any()
    //     .downcast_ref::<CpuStorage<f32>>()
    //     .expect("this is happening in Test")
    //     .clone();
    // let b_storage = b
    //     .as_ref()
    //     .unwrap()
    //     .0
    //     .clone()
    //     .data
    //     .as_any()
    //     .downcast_ref::<CpuStorage<f32>>()
    //     .expect("this is happening in Test")
    //     .clone();
    //
    // let c_storage = c
    //     .as_ref()
    //     .unwrap()
    //     .0
    //     .clone()
    //     .data
    //     .as_any()
    //     .downcast_ref::<CpuStorage<f32>>()
    //     .expect("this is happening in Test")
    //     .clone();
    // let d_storage = d
    //     .as_ref()
    //     .unwrap()
    //     .0
    //     .clone()
    //     .data
    //     .as_any()
    //     .downcast_ref::<CpuStorage<f32>>()
    //     .expect("this is happening in Test")
    //     .clone();
    // println!("this is Storage a_storage{:?}", a_storage.data);
    // println!("this is Storage b_storage{:?}", b_storage.data);
    // println!("this is Storage c_storage{:?}", c_storage.data);
    // println!("this is Storage d{:?}", d_storage.data);
    // println!("this is Storage c_storage{:?}", c.data);

    // Test::<f32>();
    // let x = 10;
    // let ptr: *const i32 = &x as *const i32;
    //
    // println!("Pointer address: {:p}", ptr);

    // let mut c_data = vec![T::default(); m * n];

    // print_type_of(&a);
    // let d = CpuBackend::mul(&a, &b);

    // println!("This is c {:?}", c);
}
// println!("--- Calling CUDA vector addition from Rust ---");
//
// const N: usize = 10;
// let mut a = Vec::new();
// let mut b = Vec::new();
// for i in 0..N {
//     a.push(i as i32);
//     b.push((N - i) as i32);
// }
//
// let mut c = vec![0; N];
//
// println!("Vector a: {:?}", a);
// println!("Vector b: {:?}", b);
//
// // Calling the CUDA wrapper function.
// // This is unsafe because we are calling external C code.
// unsafe {
//     cuda_bindings::add_vectors_wrapper(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), N as i32);
// }
//
// println!("Result of CUDA addition (Vector c): {:?}", c);
//
// // Verification
// let mut success = true;
// for i in 0..N {
//     if c[i] != N as i32 {
//         println!(
//             "Verification failed at index {}: expected {}, got {}",
//             i, N, c[i]
//         );
//         success = false;
//         break;
//     }
// }
//
// if success {
//     println!("Verification successful!");
// }
//
