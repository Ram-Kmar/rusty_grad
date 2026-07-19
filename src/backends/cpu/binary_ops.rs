// use crate::core::device::Device;
// use crate::core::error::{Result, TensorError};
use crate::math::gemm::Gemm;
// use crate::core::shared::{Shared, new_shared};
// use crate::core::storage::{CpuStorage, Storage};
// use crate::tensor::{TensorData, Tensor};
use crate::core::traits::TensorFloat;
// use std::cell::RefCell;
// use std::ops::{Add, Div, Mul, Sub};

pub fn add<T: TensorFloat>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}
pub fn sub<T: TensorFloat>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    
    a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
}
pub fn mul<T: TensorFloat>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
}
pub fn div<T: TensorFloat>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    
    a.iter().zip(b.iter()).map(|(&x, &y)| x / y).collect()
}
pub fn matmul<T: TensorFloat>(a: &Vec<T>, b: &Vec<T>, m: usize, k: usize, n: usize) -> Vec<T> {
    let mut c_data = vec![T::default(); m * n];

    unsafe {
        T::gemm(
            m,                   // m
            k,                   // k
            n,                   // n
            T::one(),            // alpha
            a.as_ptr(),          // a
            k as isize,          // rsa
            1,                   // csa
            b.as_ptr(),          // b
            n as isize,          // rsb
            1,                   // csb
            T::zero(),           // beta
            c_data.as_mut_ptr(), // c
            n as isize,          // rsc
            1,                   // csc
        );
    }
    c_data
}
//
pub fn transpose<T: TensorFloat>(input: &Vec<T>, m: usize, n: usize) -> Vec<T> {
    // print!("input from transpose{:?}",input);
    let mut transposed_data = vec![T::default(); m * n];
    for i in 0..n {
        for j in 0..m {
            let original_index = j * n + i;
            let transposed_index = i * m + j;
            transposed_data[transposed_index] = input[original_index];
        }
    }
    // print!("transposed_data from transpose{:?}",&transposed_data);
    transposed_data
}
pub fn sv_add<T: TensorFloat>(input: &Vec<T>, scalar: T) -> Vec<T> {
    let size = input.len();
    let mut data = vec![T::from(0.0).unwrap(); size];
    let mut k = 0;
    for i in input.iter() {
        data[k] = *i + scalar;
        k += 1;
    }
    data
}
pub fn sv_mul<T: TensorFloat>(input: &Vec<T>, scalar: T) -> Vec<T> {
    let size = input.len();
    let mut data = vec![T::from(0.0).unwrap(); size];
    let mut k = 0;
    for i in input.iter() {
        data[k] = *i * scalar;
        k += 1;
    }
    data
}
pub fn sv_sub<T: TensorFloat>(input: &Vec<T>, scalar: T) -> Vec<T> {
    let size = input.len();
    let mut data = vec![T::from(0.0).unwrap(); size];
    let mut k = 0;
    for i in input.iter() {
        data[k] = *i - scalar;
        k += 1;
    }
    data
}

pub fn add_derivate<T: TensorFloat>(parent_grad: &Vec<T>, grad_update: &Vec<T>) -> Vec<T> {
    
    parent_grad
        .iter()
        .zip(grad_update.iter())
        .map(|(&x, &y)| x + y)
        .collect()
}

pub fn sub_derivate<T: TensorFloat>(parent_grad: &Vec<T>, grad_update: &Vec<T>) -> Vec<T> {
    
    parent_grad
        .iter()
        .zip(grad_update.iter())
        .map(|(&x, &y)| x + y)
        .collect()
}

pub fn mul_derivate<T: TensorFloat>(parent_grad: &Vec<T>, grad_update: &Vec<T>) -> Vec<T> {
    
    parent_grad
        .iter()
        .zip(grad_update.iter())
        .map(|(&x, &y)| x + y)
        .collect()
}

pub fn div_derivate<T: TensorFloat>(parent_grad: &Vec<T>, grad_update: &Vec<T>) -> Vec<T> {
    
    parent_grad
        .iter()
        .zip(grad_update.iter())
        .map(|(&x, &y)| x + y)
        .collect()
}
pub fn sv_mul_derivate<T: TensorFloat>(_parent_grad: &Vec<T>, _grad_update: &Vec<T>) -> Vec<T> {
    // TODO: Fix this. Requires scalar value to be known.
    // let size = input.len();
    // let mut data = vec![T::from(scalar).unwrap(); size];
    // let mut k = 0;
    // for i in input.iter() {
    //     data[k] = *i * scalar;
    //     k = k + 1;
    // }
    // data
    panic!("sv_mul_derivate not implemented");
}

pub fn matmul_derivate<T: TensorFloat>(
    lhs: &Vec<T>,
    rhs: &Vec<T>,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<T> {
    let mut data = vec![T::default(); m * n];
    unsafe {
        T::gemm(
            m,                 // m
            k,                 // k
            n,                 // n
            T::one(),          // alpha
            lhs.as_ptr(),      // a
            k as isize,        // rsa
            1,                 // csa
            rhs.as_ptr(),      // b
            n as isize,        // rsb
            1,                 // csb
            T::zero(),         // beta
            data.as_mut_ptr(), // c
            n as isize,        // rsc
            1,                 // csc
        );
    }
    // println!("this is from matmul_derivate {:?}",&data);
    data
}
