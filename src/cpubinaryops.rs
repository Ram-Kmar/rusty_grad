use crate::device::Device;
use crate::error::{Result, TensorError};
use crate::gemm::Gemm;
use crate::shared::{Shared, new_shared};
use crate::storage::{CpuStorage, Storage};
use crate::tensor::{Tensor, TensorHandle};
use crate::traits::TensorFloat;
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Sub};

pub fn add<T: TensorFloat>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    let data = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
    data
}
pub fn sub<T: TensorFloat>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    let data = a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect();
    data
}
pub fn mul<T: TensorFloat>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    let data = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect();
    data
}
pub fn div<T: TensorFloat>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    let data = a.iter().zip(b.iter()).map(|(&x, &y)| x / y).collect();
    data
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
pub fn SV_add<T: TensorFloat>(input: &Vec<T>, scalar: T) -> Vec<T> {
    let size = input.len();
    let mut data = vec![T::from(0.0).unwrap(); size];
    let mut k = 0;
    for i in input.iter() {
        data[k] = *i + scalar;
        k = k + 1;
    }
    data
}
pub fn SV_mul<T: TensorFloat>(input: &Vec<T>, scalar: T) -> Vec<T> {
    let size = input.len();
    let mut data = vec![T::from(0.0).unwrap(); size];
    let mut k = 0;
    for i in input.iter() {
        data[k] = *i * scalar;
        k = k + 1;
    }
    data
}
pub fn SV_sub<T: TensorFloat>(input: &Vec<T>, scalar: T) -> Vec<T> {
    let size = input.len();
    let mut data = vec![T::from(0.0).unwrap(); size];
    let mut k = 0;
    for i in input.iter() {
        data[k] = *i - scalar;
        k = k + 1;
    }
    data
}

pub fn add_derivate<T: TensorFloat>(parent_grad: &Vec<T>, grad_update: &Vec<T>) -> Vec<T> {
    let data = parent_grad
        .iter()
        .zip(grad_update.iter())
        .map(|(&x, &y)| x + y)
        .collect();
    data
}

pub fn sub_derivate<T: TensorFloat>(parent_grad: &Vec<T>, grad_update: &Vec<T>) -> Vec<T> {
    let data = parent_grad
        .iter()
        .zip(grad_update.iter())
        .map(|(&x, &y)| x + y)
        .collect();
    data
}

pub fn mul_derivate<T: TensorFloat>(parent_grad: &Vec<T>, grad_update: &Vec<T>) -> Vec<T> {
    let data = parent_grad
        .iter()
        .zip(grad_update.iter())
        .map(|(&x, &y)| x + y)
        .collect();
    data
}

pub fn div_derivate<T: TensorFloat>(parent_grad: &Vec<T>, grad_update: &Vec<T>) -> Vec<T> {
    let data = parent_grad
        .iter()
        .zip(grad_update.iter())
        .map(|(&x, &y)| x + y)
        .collect();
    data
}
pub fn SV_mul_derivate<T: TensorFloat>(parent_grad: &Vec<T>: &Vec<T>, grad_update: &Vec<T>) -> Vec<T> {
    let size = input.len();
    let mut data = vec![T::from(scalar).unwrap(); size];
    // let mut k = 0;
    // for i in input.iter() {
    //     data[k] = *i * scalar;
    //     k = k + 1;
    // }
    data
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
