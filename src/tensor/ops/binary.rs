// use crate::backends::cpu;
use crate::backends::Backend;
use crate::backends::cpu::CpuBackend;
use crate::backends::cpu::binary_ops;
use crate::core::device::Device;
use crate::core::error::TensorError;
use crate::core::shared::{new_shared, Shared};
use crate::core::storage::{CpuStorage, Storage};
use crate::tensor::{Tensor, TensorHandle};
use crate::core::traits::TensorFloat;
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;

impl<T: TensorFloat> TensorHandle<T> {
    pub fn SV_add(input: TensorHandle<T>, scalar: T, which_store: String) -> TensorHandle<T> {
        let storage = if which_store == "grad" {
            input
                .grad
                .as_ref()
                .unwrap()
                .borrow()
                .as_any()
                .downcast_ref::<CpuStorage<T>>()
                .ok_or(TensorError::BackendMismatch)
                .unwrap()
                .clone()
        } else {
            input
                .data
                .borrow()
                .as_any()
                .downcast_ref::<CpuStorage<T>>()
                .ok_or(TensorError::BackendMismatch)
                .unwrap()
                .clone()
        };

        let data = crate::backends::cpu::binary_ops::SV_add(storage.get_data(), scalar);
        let new_storage = CpuStorage::from_data(data);

        let grad_require = input.grad_require;
        TensorHandle(new_shared(Tensor {
            shape: vec![input.shape[0], input.shape[1]],
            data: Shared::new(RefCell::new(new_storage)),
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
        let storage = if which_store == "grad" {
            input
                .grad
                .as_ref()
                .unwrap()
                .borrow()
                .as_any()
                .downcast_ref::<CpuStorage<T>>()
                .ok_or(TensorError::BackendMismatch)
                .unwrap()
                .clone()
        } else {
            input
                .data
                .borrow()
                .as_any()
                .downcast_ref::<CpuStorage<T>>()
                .ok_or(TensorError::BackendMismatch)
                .unwrap()
                .clone()
        };

        let data = crate::backends::cpu::binary_ops::SV_sub(storage.get_data(), scalar);
        let new_storage = CpuStorage::from_data(data);

        let grad_require = input.grad_require;
        TensorHandle(new_shared(Tensor {
            shape: vec![input.shape[0], input.shape[1]],
            data: Shared::new(RefCell::new(new_storage)),
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
        let storage = if which_store == "grad" {
            input
                .grad
                .as_ref()
                .unwrap()
                .borrow()
                .as_any()
                .downcast_ref::<CpuStorage<T>>()
                .ok_or(TensorError::BackendMismatch)
                .unwrap()
                .clone()
        } else {
            input
                .data
                .borrow()
                .as_any()
                .downcast_ref::<CpuStorage<T>>()
                .ok_or(TensorError::BackendMismatch)
                .unwrap()
                .clone()
        };

        let data = crate::backends::cpu::binary_ops::SV_mul(storage.get_data(), scalar);
        let new_storage = CpuStorage::from_data(data);

        let grad_require = input.grad_require;
        TensorHandle(new_shared(Tensor {
            shape: vec![input.shape[0], input.shape[1]],
            data: Shared::new(RefCell::new(new_storage)),
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
            parent: Some(vec![input.0.clone()]),
        }))
    }
}

impl<T: TensorFloat> Add for &TensorHandle<T> {
    type Output = TensorHandle<T>;
    fn add(self, rhs: &TensorHandle<T>) -> TensorHandle<T> {
        let tensor = CpuBackend::add(&self.0, &rhs.0).expect("Add failed");
        TensorHandle(new_shared(tensor))
    }
}

impl<T: TensorFloat> Sub for &TensorHandle<T> {
    type Output = TensorHandle<T>;
    fn sub(self, rhs: &TensorHandle<T>) -> TensorHandle<T> {
        let tensor = CpuBackend::sub(&self.0, &rhs.0).expect("Sub failed");
        TensorHandle(new_shared(tensor))
    }
}

impl<T: TensorFloat> Mul for &TensorHandle<T> {
    type Output = TensorHandle<T>;
    fn mul(self, rhs: &TensorHandle<T>) -> TensorHandle<T> {
        let tensor = CpuBackend::mul(&self.0, &rhs.0).expect("Mul failed");
        TensorHandle(new_shared(tensor))
    }
}

impl<T: TensorFloat> Div for &TensorHandle<T> {
    type Output = TensorHandle<T>;
    fn div(self, rhs: &TensorHandle<T>) -> TensorHandle<T> {
        let tensor = CpuBackend::div(&self.0, &rhs.0).expect("Div failed");
        TensorHandle(new_shared(tensor))
    }
}

impl<T: TensorFloat> TensorHandle<T> {
    pub fn matmul(a: Shared<Tensor<T>>, b: Shared<Tensor<T>>) -> TensorHandle<T> {
        let tensor = CpuBackend::matmul(&a, &b).expect("Matmul failed");
        TensorHandle(new_shared(tensor))
    }
}
