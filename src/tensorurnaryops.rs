use crate::cpu_backend;
use crate::shared::{Shared, new_shared};
use crate::storage::{CpuStorage, Storage};
use crate::tensor::{Tensor, TensorHandle};
use crate::traits::TensorFloat;
use std::cell::RefCell;

pub trait TensorUrnaryOps<T: TensorFloat> {
    fn relu(&self) -> TensorHandle<T>;
    fn sigmoid(&self) -> TensorHandle<T>;
    fn tanh(&self) -> TensorHandle<T>;
    fn exp(&self) -> TensorHandle<T>;
    fn log(&self) -> TensorHandle<T>;
    fn neg(&self) -> TensorHandle<T>;
    fn abs(&self) -> TensorHandle<T>;
    fn square(&self) -> TensorHandle<T>;
    fn sqrt(&self) -> TensorHandle<T>;
    fn mean(&self) -> TensorHandle<T>;
    fn power(&self, power: T) -> TensorHandle<T>;
    fn sum(&self, access_dim: usize) -> TensorHandle<T>;
}

impl<T: TensorFloat> TensorUrnaryOps<T> for TensorHandle<T> {
    fn relu(&self) -> TensorHandle<T> {
        let data = CpuStorage::relu(self.data.clone());
        let grad_require = self.grad_require;
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
            operation: Some(vec!["relu".to_string()]),
            is_child: true,
            parent: Some(vec![self.0.clone()]),
        }))
    }

    fn sigmoid(&self) -> TensorHandle<T> {
        let data = CpuStorage::sigmoid(self.data.clone());
        let grad_require = self.grad_require;
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
            operation: Some(vec!["sigmoid".to_string()]),
            is_child: true,
            parent: Some(vec![self.0.clone()]),
        }))
    }

    fn tanh(&self) -> TensorHandle<T> {
        let data = CpuStorage::tanh(self.data.clone());
        let grad_require = self.grad_require;
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
            operation: Some(vec!["tanh".to_string()]),
            is_child: true,
            parent: Some(vec![self.0.clone()]),
        }))
    }

    fn exp(&self) -> TensorHandle<T> {
        let data = CpuStorage::exp(self.data.clone());
        let grad_require = self.grad_require;
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
            operation: Some(vec!["exp".to_string()]),
            is_child: true,
            parent: Some(vec![self.0.clone()]),
        }))
    }

    fn log(&self) -> TensorHandle<T> {
        let data = CpuStorage::log(self.data.clone());
        let grad_require = self.grad_require;
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
            operation: Some(vec!["log".to_string()]),
            is_child: true,
            parent: Some(vec![self.0.clone()]),
        }))
    }

    fn neg(&self) -> TensorHandle<T> {
        let data = CpuStorage::neg(self.data.clone());
        let grad_require = self.grad_require;
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
            operation: Some(vec!["neg".to_string()]),
            is_child: true,
            parent: Some(vec![self.0.clone()]),
        }))
    }

    fn abs(&self) -> TensorHandle<T> {
        let data = CpuStorage::abs(self.data.clone());
        let grad_require = self.grad_require;
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
            operation: Some(vec!["abs".to_string()]),
            is_child: true,
            parent: Some(vec![self.0.clone()]),
        }))
    }

    fn square(&self) -> TensorHandle<T> {
        let data = CpuStorage::square(self.data.clone());
        let grad_require = self.grad_require;
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
            operation: Some(vec!["square".to_string()]),
            is_child: true,
            parent: Some(vec![self.0.clone()]),
        }))
    }

    fn sqrt(&self) -> TensorHandle<T> {
        let data = CpuStorage::sqrt(self.data.clone());
        let grad_require = self.grad_require;
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
            operation: Some(vec!["sqrt".to_string()]),
            is_child: true,
            parent: Some(vec![self.0.clone()]),
        }))
    }
    fn sum(&self, access_dim: usize) -> TensorHandle<T> {
        let access_dim = self.shape[access_dim];
        let data = CpuStorage::sum(self.data.clone(), self.shape.clone(), access_dim);
        let grad_require = self.grad_require;
        TensorHandle(new_shared(Tensor {
            shape: vec![1],
            data: Shared::new(data),
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

    fn mean(&self) -> TensorHandle<T> {
        let data = CpuStorage::mean(self.data.clone());
        let grad_require = self.grad_require;
        TensorHandle(new_shared(Tensor {
            shape: vec![1],
            data: Shared::new(data),
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
    fn power(&self, power: T) -> TensorHandle<T> {
        let data = CpuStorage::power(self.data.clone(), power);
        let grad_require = self.grad_require;
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
            operation: Some(vec!["power".to_string()]),
            is_child: true,
            parent: Some(vec![self.0.clone()]),
        }))
    }
}
