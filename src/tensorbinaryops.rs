// use crate::cpu_backend;
use crate::shared::{Shared, new_shared};
use crate::storage::{CpuStorage, Storage};
use crate::tensor::{Tensor, TensorHandle};
use crate::traits::TensorFloat;
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;

impl<T: TensorFloat> Add for &TensorHandle<T> {
    type Output = TensorHandle<T>;
    fn add(self, rhs: &TensorHandle<T>) -> TensorHandle<T> {
        let data = CpuStorage::add(self.data.clone(), rhs.data.clone());
        let grad_require = self.grad_require || rhs.grad_require;
        TensorHandle(new_shared(Tensor {
            shape: self.shape.clone(),
            data: Shared::new(data),
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
            data: Shared::new(data),
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
impl<T: TensorFloat> Mul for &TensorHandle<T> {
    type Output = TensorHandle<T>;
    fn mul(self, rhs: &TensorHandle<T>) -> TensorHandle<T> {
        let data = CpuStorage::mul(self.data.clone(), rhs.data.clone());
        let grad_require = self.grad_require || rhs.grad_require;
        TensorHandle(new_shared(Tensor {
            shape: self.shape.clone(),
            data: Shared::new(data),
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
impl<T: TensorFloat> Div for &TensorHandle<T> {
    type Output = TensorHandle<T>;
    fn div(self, rhs: &TensorHandle<T>) -> TensorHandle<T> {
        let data = CpuStorage::div(self.data.clone(), rhs.data.clone());
        let grad_require = self.grad_require || rhs.grad_require;
        TensorHandle(new_shared(Tensor {
            shape: self.shape.clone(),
            data: Shared::new(data),
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
            data: Shared::new(data),
            grad: if grad_require {
                Some(Shared::new(RefCell::new(CpuStorage::ones(
                    a.shape[0]*b.shape[1],
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
