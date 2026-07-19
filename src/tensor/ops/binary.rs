// use crate::backends::cpu;
use crate::backends::Backend;
use crate::backends::cpu::CpuBackend;
use crate::core::error::TensorError;
use crate::core::shared::{new_shared, Shared};
use crate::core::storage::{CpuStorage, Storage};
use crate::tensor::{TensorData, Tensor};
use crate::core::traits::TensorFloat;
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Sub};

impl<T: TensorFloat> Tensor<T> {
    pub fn sv_add(input: Tensor<T>, scalar: T, which_store: String) -> Tensor<T> {
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

        let data = crate::backends::cpu::binary_ops::sv_add(storage.get_data(), scalar);
        let new_storage = CpuStorage::from_data(data);

        let grad_require = input.grad_require;
        Tensor(new_shared(TensorData {
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
            operation: Some(vec!["sv_add".to_string()]),
            is_child: true,
            parent: Some(vec![input.0.clone()]),
        }))
    }

    pub fn sv_sub(input: Tensor<T>, scalar: T, which_store: String) -> Tensor<T> {
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

        let data = crate::backends::cpu::binary_ops::sv_sub(storage.get_data(), scalar);
        let new_storage = CpuStorage::from_data(data);

        let grad_require = input.grad_require;
        Tensor(new_shared(TensorData {
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
            operation: Some(vec!["sv_sub".to_string()]),
            is_child: true,
            parent: Some(vec![input.0.clone()]),
        }))
    }

    pub fn sv_mul(input: Tensor<T>, scalar: T, which_store: String) -> Tensor<T> {
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

        let data = crate::backends::cpu::binary_ops::sv_mul(storage.get_data(), scalar);
        let new_storage = CpuStorage::from_data(data);

        let grad_require = input.grad_require;
        Tensor(new_shared(TensorData {
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
            operation: Some(vec!["sv_mul".to_string()]),
            is_child: true,
            parent: Some(vec![input.0.clone()]),
        }))
    }
}

impl<T: TensorFloat> Add for &Tensor<T> {
    type Output = Tensor<T>;
    fn add(self, rhs: &Tensor<T>) -> Tensor<T> {
        let tensor = CpuBackend::add(&self.0, &rhs.0).unwrap_or_else(|e| panic!("TensorData addition failed: {:?}", e));
        Tensor(new_shared(tensor))
    }
}

impl<T: TensorFloat> Sub for &Tensor<T> {
    type Output = Tensor<T>;
    fn sub(self, rhs: &Tensor<T>) -> Tensor<T> {
        let tensor = CpuBackend::sub(&self.0, &rhs.0).unwrap_or_else(|e| panic!("TensorData subtraction failed: {:?}", e));
        Tensor(new_shared(tensor))
    }
}

impl<T: TensorFloat> Mul for &Tensor<T> {
    type Output = Tensor<T>;
    fn mul(self, rhs: &Tensor<T>) -> Tensor<T> {
        let tensor = CpuBackend::mul(&self.0, &rhs.0).unwrap_or_else(|e| panic!("TensorData multiplication failed: {:?}", e));
        Tensor(new_shared(tensor))
    }
}

impl<T: TensorFloat> Div for &Tensor<T> {
    type Output = Tensor<T>;
    fn div(self, rhs: &Tensor<T>) -> Tensor<T> {
        let tensor = CpuBackend::div(&self.0, &rhs.0).unwrap_or_else(|e| panic!("TensorData division failed: {:?}", e));
        Tensor(new_shared(tensor))
    }
}

impl<T: TensorFloat> Tensor<T> {
    pub fn matmul(a: Shared<TensorData<T>>, b: Shared<TensorData<T>>) -> Tensor<T> {
        let tensor = CpuBackend::matmul(&a, &b).unwrap_or_else(|e| panic!("TensorData matmul failed: {:?}", e));
        Tensor(new_shared(tensor))
    }
}
