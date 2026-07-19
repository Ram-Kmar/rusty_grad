// use crate::backends::cpu;
use crate::backends::Backend;
use crate::backends::cpu::CpuBackend;
use crate::core::shared::{new_shared, Shared};
use crate::core::storage::{CpuStorage, Storage};
use crate::tensor::{TensorData, Tensor};
use crate::core::traits::TensorFloat;
use std::cell::RefCell;

pub trait TensorUrnaryOps<T: TensorFloat> {
    fn relu(&self) -> Tensor<T>;
    fn sigmoid(&self) -> Tensor<T>;
    fn tanh(&self) -> Tensor<T>;
    fn exp(&self) -> Tensor<T>;
    fn log(&self) -> Tensor<T>;
    fn neg(&self) -> Tensor<T>;
    fn abs(&self) -> Tensor<T>;
    fn sum(&self) -> Tensor<T>;
    fn square(&self) -> Tensor<T>;
    fn sqrt(&self) -> Tensor<T>;
    fn mean(&self) -> Tensor<T>;
    fn power(&self, power: T) -> Tensor<T>;
    // fn sum(&self, access_dim: usize) -> Tensor<T>;
}

impl<T: TensorFloat> TensorUrnaryOps<T> for Tensor<T> {
    fn relu(&self) -> Tensor<T> {
        let tensor = CpuBackend::relu(&self.0).expect("Relu failed");
        Tensor(new_shared(tensor))
    }

    fn sigmoid(&self) -> Tensor<T> {
        let storage = self
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .expect("Backend mismatch")
            .clone();
        let data = crate::backends::cpu::unary_ops::sigmoid(storage.get_data());
        let new_storage = CpuStorage::from_data(data);

        let grad_require = self.grad_require;
        Tensor(new_shared(TensorData {
            shape: self.shape.clone(),
            data: Shared::new(RefCell::new(new_storage)),
            grad: if grad_require {
                Some(Shared::new(RefCell::new(CpuStorage::zeros(
                    self.shape.iter().product(),
                ))))
            } else {
                None
            },
            grad_require,
            operation: Some(vec!["sigmoid".to_string()]),
            is_child: true,
            parent: Some(vec![self.0.clone()]),
        }))
    }

    fn tanh(&self) -> Tensor<T> {
        let storage = self
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .expect("Backend mismatch")
            .clone();
        let data = crate::backends::cpu::unary_ops::tanh(storage.get_data());
        let new_storage = CpuStorage::from_data(data);

        let grad_require = self.grad_require;
        Tensor(new_shared(TensorData {
            shape: self.shape.clone(),
            data: Shared::new(RefCell::new(new_storage)),
            grad: if grad_require {
                Some(Shared::new(RefCell::new(CpuStorage::zeros(
                    self.shape.iter().product(),
                ))))
            } else {
                None
            },
            grad_require,
            operation: Some(vec!["tanh".to_string()]),
            is_child: true,
            parent: Some(vec![self.0.clone()]),
        }))
    }

    fn exp(&self) -> Tensor<T> {
        let tensor = CpuBackend::exp(&self.0).expect("Exp failed");
        Tensor(new_shared(tensor))
    }

    fn sum(&self) -> Tensor<T> {
        let storage = self
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .expect("Backend mismatch")
            .clone();
        let data = crate::backends::cpu::unary_ops::sum(storage.get_data());
        let new_storage = CpuStorage::from_data(data);

        let grad_require = self.grad_require;
        Tensor(new_shared(TensorData {
            shape: vec![1], // Sum reduces to a scalar (vector of length 1)
            data: Shared::new(RefCell::new(new_storage)),
            grad: if grad_require {
                Some(Shared::new(RefCell::new(CpuStorage::zeros(1))))
            } else {
                None
            },
            grad_require,
            operation: Some(vec!["sum".to_string()]), // Fixed operation name
            is_child: true,
            parent: Some(vec![self.0.clone()]),
        }))
    }

    fn log(&self) -> Tensor<T> {
        let tensor = CpuBackend::log(&self.0).expect("Log failed");
        Tensor(new_shared(tensor))
    }

    fn neg(&self) -> Tensor<T> {
        let storage = self
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .expect("Backend mismatch")
            .clone();
        let data = crate::backends::cpu::unary_ops::neg(storage.get_data());
        let new_storage = CpuStorage::from_data(data);

        let grad_require = self.grad_require;
        Tensor(new_shared(TensorData {
            shape: self.shape.clone(),
            data: Shared::new(RefCell::new(new_storage)),
            grad: if grad_require {
                Some(Shared::new(RefCell::new(CpuStorage::zeros(
                    self.shape.iter().product(),
                ))))
            } else {
                None
            },
            grad_require,
            operation: Some(vec!["neg".to_string()]),
            is_child: true,
            parent: Some(vec![self.0.clone()]),
        }))
    }

    fn abs(&self) -> Tensor<T> {
        let storage = self
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .expect("Backend mismatch")
            .clone();
        let data = crate::backends::cpu::unary_ops::abs(storage.get_data());
        let new_storage = CpuStorage::from_data(data);

        let grad_require = self.grad_require;
        Tensor(new_shared(TensorData {
            shape: self.shape.clone(),
            data: Shared::new(RefCell::new(new_storage)),
            grad: if grad_require {
                Some(Shared::new(RefCell::new(CpuStorage::zeros(
                    self.shape.iter().product(),
                ))))
            } else {
                None
            },
            grad_require,
            operation: Some(vec!["abs".to_string()]),
            is_child: true,
            parent: Some(vec![self.0.clone()]),
        }))
    }

    fn square(&self) -> Tensor<T> {
        let storage = self
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .expect("Backend mismatch")
            .clone();
        let data = crate::backends::cpu::unary_ops::square(storage.get_data());
        let new_storage = CpuStorage::from_data(data);

        let grad_require = self.grad_require;
        Tensor(new_shared(TensorData {
            shape: self.shape.clone(),
            data: Shared::new(RefCell::new(new_storage)),
            grad: if grad_require {
                Some(Shared::new(RefCell::new(CpuStorage::zeros(
                    self.shape.iter().product(),
                ))))
            } else {
                None
            },
            grad_require,
            operation: Some(vec!["square".to_string()]),
            is_child: true,
            parent: Some(vec![self.0.clone()]),
        }))
    }

    fn sqrt(&self) -> Tensor<T> {
        let storage = self
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .expect("Backend mismatch")
            .clone();
        let data = crate::backends::cpu::unary_ops::sqrt(storage.get_data());
        let new_storage = CpuStorage::from_data(data);

        let grad_require = self.grad_require;
        Tensor(new_shared(TensorData {
            shape: self.shape.clone(),
            data: Shared::new(RefCell::new(new_storage)),
            grad: if grad_require {
                Some(Shared::new(RefCell::new(CpuStorage::zeros(
                    self.shape.iter().product(),
                ))))
            } else {
                None
            },
            grad_require,
            operation: Some(vec!["sqrt".to_string()]),
            is_child: true,
            parent: Some(vec![self.0.clone()]),
        }))
    }

    fn mean(&self) -> Tensor<T> {
        let storage = self
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .expect("Backend mismatch")
            .clone();
        let data = crate::backends::cpu::unary_ops::mean(storage.get_data());
        let new_storage = CpuStorage::from_data(data);

        let grad_require = self.grad_require;
        Tensor(new_shared(TensorData {
            shape: vec![1], // Mean reduces to scalar
            data: Shared::new(RefCell::new(new_storage)),
            grad: if grad_require {
                Some(Shared::new(RefCell::new(CpuStorage::zeros(1))))
            } else {
                None
            },
            grad_require,
            operation: Some(vec!["mean".to_string()]),
            is_child: true,
            parent: Some(vec![self.0.clone()]),
        }))
    }

    fn power(&self, power: T) -> Tensor<T> {
        let storage = self
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .expect("Backend mismatch")
            .clone();
        let data = crate::backends::cpu::unary_ops::power(storage.get_data(), power);
        let new_storage = CpuStorage::from_data(data);

        let grad_require = self.grad_require;
        Tensor(new_shared(TensorData {
            shape: self.shape.clone(),
            data: Shared::new(RefCell::new(new_storage)),
            grad: if grad_require {
                Some(Shared::new(RefCell::new(CpuStorage::zeros(
                    self.shape.iter().product(),
                ))))
            } else {
                None
            },
            grad_require,
            operation: Some(vec!["power".to_string()]),
            is_child: true,
            parent: Some(vec![self.0.clone()]),
        }))
    }
}
