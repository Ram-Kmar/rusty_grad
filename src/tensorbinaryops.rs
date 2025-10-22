// use crate::cpu_backend;
use crate::device::Device;
use crate::shared::{Shared, new_shared};
use crate::storage::{CpuStorage, Storage};
use crate::tensor::{Tensor, TensorHandle};
use crate::traits::TensorFloat;
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;
// pub trait TensorBinaryOps<T: TensorFloat> {
//     // fn relu(&self) -> TensorHandle<T>;
//     fn SV_mul(a: &T, b: &Vec<T>) -> CpuStorage<T>;
//     fn SV_add(a: &T, b: &Vec<T>) -> CpuStorage<T>;
//     fn SV_sub(a: &T, b: &Vec<T>) -> CpuStorage<T>;
//     // fn sigmoid(&self) -> TensorHandle<T>;
//     // fn tanh(&self) -> TensorHandle<T>;
//     // fn exp(&self) -> TensorHandle<T>;
//     // fn log(&self) -> TensorHandle<T>;
//     // fn neg(&self) -> TensorHandle<T>;
//     // fn abs(&self) -> TensorHandle<T>;
//     // fn square(&self) -> TensorHandle<T>;
//     // fn sqrt(&self) -> TensorHandle<T>;
//     // fn mean(&self) -> TensorHandle<T>;
//     // fn power(&self, power: T) -> TensorHandle<T>;
// }
impl<T: TensorFloat> TensorHandle<T> {
    pub fn SV_add(input: TensorHandle<T>, scalar: T, which_store: String) -> TensorHandle<T> {
        let data;
        if which_store == "grad" {
            data = CpuStorage::SV_add(input.grad.clone().unwrap(), scalar);
        } else {
            data = CpuStorage::SV_add(input.data.clone(), scalar);
        }
        let grad_require = input.grad_require; // is the value is move or copy
        TensorHandle(new_shared(Tensor {
            shape: vec![input.shape[0], input.shape[1]],
            data: Shared::new(RefCell::new(data)),
            grad: if grad_require {
                Some(Shared::new(RefCell::new(CpuStorage::ones(
                    input.shape[0] * input.shape[1],
                ))))
            } else {
                None
            },
            grad_require,
            operation: Some(vec!["SV_add".to_string()]),
            is_child: true,
            parent: Some(vec![input.0.clone()]),
        }))
    }
    pub fn SV_sub(input: TensorHandle<T>, scalar: T, which_store: String) -> TensorHandle<T> {
        let data;
        if which_store == "grad" {
            data = CpuStorage::SV_add(input.grad.clone().unwrap(), scalar);
        } else {
            data = CpuStorage::SV_add(input.data.clone(), scalar);
        }
        let data = CpuStorage::SV_sub(input.data.clone(), scalar);
        let grad_require = input.grad_require; // is the value is move or copy
        TensorHandle(new_shared(Tensor {
            shape: vec![input.shape[0], input.shape[1]],
            data: Shared::new(RefCell::new(data)),
            grad: if grad_require {
                Some(Shared::new(RefCell::new(CpuStorage::ones(
                    input.shape[0] * input.shape[1],
                ))))
            } else {
                None
            },
            grad_require,
            operation: Some(vec!["SV_sub".to_string()]),
            is_child: true,
            parent: Some(vec![input.0.clone()]),
        }))
    }
    pub fn SV_mul(input: TensorHandle<T>, scalar: T, which_store: String) -> TensorHandle<T> {
        let data;
        if which_store == "grad" {
            data = CpuStorage::SV_add(input.grad.clone().unwrap(), scalar);
        } else {
            data = CpuStorage::SV_add(input.data.clone(), scalar);
        }
        let grad_require = input.grad_require; // is the value is move or copy
        TensorHandle(new_shared(Tensor {
            shape: vec![input.shape[0], input.shape[1]],
            data: Shared::new(RefCell::new(data)),
            grad: if grad_require {
                Some(Shared::new(RefCell::new(CpuStorage::ones(
                    input.shape[0] * input.shape[1],
                ))))
            } else {
                None
            },
            grad_require,
            operation: Some(vec!["SV_mul".to_string()]),
            is_child: true,
            // right now this for eplison mul only,if want to make general you need to add scalar
            // parent as well
            parent: Some(vec![input.0.clone()]),
        }))
    }
}

impl<T: TensorFloat> Add for &TensorHandle<T> {
    type Output = TensorHandle<T>;
    fn add(self, rhs: &TensorHandle<T>) -> TensorHandle<T> {
        let data = CpuStorage::add(self.data.clone(), rhs.data.clone());
        let grad_require = self.grad_require || rhs.grad_require;
        TensorHandle(new_shared(Tensor {
            shape: self.shape.clone(),
            data: Shared::new(RefCell::new(data)),
            grad: if grad_require {
                Some(Shared::new(RefCell::new(CpuStorage::zeros(
                    self.shape.iter().product(),
                ))))
            } else {
                None
            },
            grad_require,
            operation: Some(vec!["add".to_string()]),
            is_child: true,
            parent: Some(vec![self.0.clone(), rhs.0.clone()]),
        }))
    }
}

impl<T: TensorFloat> Sub for &TensorHandle<T> {
    type Output = TensorHandle<T>;
    fn sub(self, rhs: &TensorHandle<T>) -> TensorHandle<T> {
        let data = CpuStorage::sub(self.data.clone(), rhs.data.clone());
        let grad_require = self.grad_require || rhs.grad_require;
        TensorHandle(new_shared(Tensor {
            shape: self.shape.clone(),
            data: Shared::new(RefCell::new(data)),
            grad: if grad_require {
                Some(Shared::new(RefCell::new(CpuStorage::zeros(
                    self.shape.iter().product(),
                ))))
            } else {
                None
            },
            grad_require,
            operation: Some(vec!["sub".to_string()]),
            is_child: true,
            parent: Some(vec![self.0.clone(), rhs.0.clone()]),
        }))
    }
}
impl<T: TensorFloat> Mul for &TensorHandle<T> {
    type Output = TensorHandle<T>;
    fn mul(self, rhs: &TensorHandle<T>) -> TensorHandle<T> {
        let data = CpuStorage::mul(self.data.clone(), rhs.data.clone());
        let grad_require = self.grad_require || rhs.grad_require;
        TensorHandle(new_shared(Tensor {
            shape: self.shape.clone(),
            data: Shared::new(RefCell::new(data)),
            grad: if grad_require {
                Some(Shared::new(RefCell::new(CpuStorage::zeros(
                    self.shape.iter().product(),
                ))))
            } else {
                None
            },
            grad_require,
            operation: Some(vec!["mul".to_string()]),
            is_child: true,
            parent: Some(vec![self.0.clone(), rhs.0.clone()]),
        }))
    }
}
impl<T: TensorFloat> Div for &TensorHandle<T> {
    type Output = TensorHandle<T>;
    fn div(self, rhs: &TensorHandle<T>) -> TensorHandle<T> {
        let data = CpuStorage::div(self.data.clone(), rhs.data.clone());
        let grad_require = self.grad_require || rhs.grad_require;
        TensorHandle(new_shared(Tensor {
            shape: self.shape.clone(),
            data: Shared::new(RefCell::new(data)),
            grad: if grad_require {
                Some(Shared::new(RefCell::new(CpuStorage::zeros(
                    self.shape.iter().product(),
                ))))
            } else {
                None
            },
            grad_require,
            operation: Some(vec!["div".to_string()]),
            is_child: true,
            parent: Some(vec![self.0.clone(), rhs.0.clone()]),
        }))
    }
}
impl<T: TensorFloat> TensorHandle<T> {
    pub fn matmul(a: Shared<Tensor<T>>, b: Shared<Tensor<T>>) -> TensorHandle<T> {
        // if a.shape.len() != 2 || b.shape.len() != 2 {
        //     return Err(TensorError::ShapeMismatch); // matmul is for 2D tensors
        // }
        // let m = a.shape[0];
        // let k = a.shape[1];
        // let n = b.shape[1];

        // if k != b.shape[0] {
        //     return Err(TensorError::ShapeMismatch);
        // }
        let data = CpuStorage::matmul(
            a.data.clone(),
            b.data.clone(),
            a.shape[0],
            a.shape[1],
            b.shape[1],
        );
        let grad_require = a.grad_require || b.grad_require;
        TensorHandle(new_shared(Tensor {
            shape: vec![a.shape[0], b.shape[1]],
            data: Shared::new(RefCell::new(data)),
            grad: if grad_require {
                Some(Shared::new(RefCell::new(CpuStorage::ones(
                    a.shape[0] * b.shape[1],
                ))))
            } else {
                None
            },
            grad_require,
            operation: Some(vec!["matmul".to_string()]),
            is_child: true,
            parent: Some(vec![a.clone(), b.clone()]),
        }))
    }
}
//     fn add(){
//         a.data.sub();
//     }
// }
//
// impl<T: TensorFloat> Mul for TensorHandle<T>{
//     fn add(){
//         a.data.mul();
//     }
// }
// impl<T: TensorFloat> Div for TensorHandle<T>{
//     fn add(){
//         a.data.div();
//     }
// }
