use crate::core::storage::Storage;
use crate::core::traits::TensorFloat;
use std::iter::Sum;

pub fn relu<T: TensorFloat>(input: &Vec<T>) -> Vec<T> {
    input
        .iter()
        .map(|&x| if x > T::zero() { x } else { T::zero() })
        .collect()
}

pub fn sigmoid<T: TensorFloat>(input: &Vec<T>) -> Vec<T> {
    input
        .iter()
        .map(|&x| T::one() / (T::one() + (-x).exp()))
        .collect()
}

pub fn tanh<T: TensorFloat>(input: &Vec<T>) -> Vec<T> {
    input.iter().map(|&x| x.tanh()).collect()
}

pub fn exp<T: TensorFloat>(input: &Vec<T>) -> Vec<T> {
    input.iter().map(|&x| x.exp()).collect()
}

pub fn log<T: TensorFloat>(input: &Vec<T>) -> Vec<T> {
    input.iter().map(|&x| x.ln()).collect()
}

pub fn neg<T: TensorFloat>(input: &Vec<T>) -> Vec<T> {
    input.iter().map(|&x| -x).collect()
}

pub fn abs<T: TensorFloat>(input: &Vec<T>) -> Vec<T> {
    input.iter().map(|&x| x.abs()).collect()
}

pub fn square<T: TensorFloat>(input: &Vec<T>) -> Vec<T> {
    input.iter().map(|&x| x * x).collect()
}

pub fn sqrt<T: TensorFloat>(input: &Vec<T>) -> Vec<T> {
    input.iter().map(|&x| x.sqrt()).collect()
}

pub fn mean<T: TensorFloat>(input: &Vec<T>) -> Vec<T> {
    let sum: T = input.iter().fold(T::zero(), |acc, &x| acc + x);
    let mean = sum / T::from(input.len() as f64).unwrap();
    vec![mean]
}
pub fn sum<T: TensorFloat + Sum>(input: &Vec<T>) -> Vec<T> {
    let sum: T = input.iter().copied().sum();
    vec![sum]
}
// feature needs to work on
// pub fn sum<T: TensorFloat>(
//     input: Shared<dyn Storage<Elem = T>>,
//     access_dim: usize,
//     gap: usize,
// ) -> Vec<T> {
//     let sum = vec![T::zero(); access_dim];
//     let data: Vec<T> = Vec::new();
//     for i in 0..gap {
//         sum[i] = input.get_dim_slice(access_dim, gap);
//         data.push(sum.iter().fold(0, |acc, x| acc + x));
//     }
//     // let sum = input.iter().fold(T::zero(), |acc, &x| acc + x);
//     // sum
//     data
// }
pub fn reciprocal<T: TensorFloat>(input: &Vec<T>) -> Vec<T> {
    input.iter().map(|&x| T::one() / x).collect()
}
pub fn power<T: TensorFloat>(input: &Vec<T>, power: T) -> Vec<T> {
    input.iter().map(|&x| x.powf(power)).collect()
}

pub fn relu_derivative<T: TensorFloat>(data: &Vec<T>, child_grad: &Vec<T>) -> Vec<T> {
    data.iter()
        .zip(child_grad.iter())
        .map(|(&x, &g)| if x > T::zero() { g } else { T::zero() })
        .collect()
}

pub fn sigmoid_derivative<T: TensorFloat>(data: &Vec<T>, child_grad: &Vec<T>) -> Vec<T> {
    data.iter()
        .zip(child_grad.iter())
        .map(|(&x, &g)| g * (x * (T::one() - x)))
        .collect()
}

pub fn tanh_derivative<T: TensorFloat>(data: &Vec<T>, child_grad: &Vec<T>) -> Vec<T> {
    data.iter()
        .zip(child_grad.iter())
        .map(|(&x, &g)| g * (T::one() - (x * x)))
        .collect()
}

pub fn exp_derivative<T: TensorFloat>(data: &Vec<T>, child_grad: &Vec<T>) -> Vec<T> {
    data.iter()
        .zip(child_grad.iter())
        .map(|(&x, &g)| g * x.exp())
        .collect()
}

pub fn log_derivative<T: TensorFloat>(data: &Vec<T>, child_grad: &Vec<T>) -> Vec<T> {
    data.iter()
        .zip(child_grad.iter())
        .map(|(&x, &g)| g / x)
        .collect()
}

pub fn neg_derivative<T: TensorFloat>(child_grad: &Vec<T>) -> Vec<T> {
    child_grad.iter().map(|&g| -g).collect()
}

pub fn abs_derivative<T: TensorFloat>(data: &Vec<T>, child_grad: &Vec<T>) -> Vec<T> {
    data.iter()
        .zip(child_grad.iter())
        .map(|(&x, &g)| {
            if x > T::zero() {
                g
            } else if x < T::zero() {
                -g
            } else {
                T::zero()
            }
        })
        .collect()
}

pub fn square_derivative<T: TensorFloat>(data: &Vec<T>, child_grad: &Vec<T>) -> Vec<T> {
    data.iter()
        .zip(child_grad.iter())
        .map(|(&x, &g)| g * (x + x))
        .collect()
}

pub fn sqrt_derivative<T: TensorFloat>(data: &Vec<T>, child_grad: &Vec<T>) -> Vec<T> {
    data.iter()
        .zip(child_grad.iter())
        .map(|(&x, &g)| g / (x + x))
        .collect()
}

pub fn mean_derivative<T: TensorFloat>(data: &Vec<T>, child_grad: &Vec<T>) -> Vec<T> {
    let n = T::from(data.len() as f64).unwrap();
    child_grad.iter().map(|&g| g / n).collect()
}

pub fn reciprocal_derivative<T: TensorFloat>(data: &Vec<T>, child_grad: &Vec<T>) -> Vec<T> {
    data.iter()
        .zip(child_grad.iter())
        .map(|(&x, &g)| -g / (x * x))
        .collect()
}
