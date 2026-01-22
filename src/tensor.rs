use crate::backend::Backend;
use crate::cpu_backend::CpuBackend;
use crate::device::{self, Device};
use crate::error::Result;
use crate::storage::{CpuStorage, Storage};
// use crate::tensorbackprop::mul_backward;
use crate::traits::TensorFloat;
use std::cell::RefCell;
// use std::collections::HashSet;
use std::fmt::{self, Debug};
use std::ops::Deref;

use crate::shared::{new_shared, Shared};

// --- Tensor Definition ---

// NOTE: Tensor is now generic over the element type `T`, not the storage type `S`.
#[derive(Clone)]
pub struct Tensor<T: TensorFloat> {
    pub shape: Vec<usize>,
    // NOTE: The `data` field is now a trait object.
    // It can hold a CpuStorage, CudaStorage, or any other type that implements the Storage trait.
    pub data: Shared<RefCell<dyn Storage<Elem = T>>>,
    pub grad: Option<Shared<RefCell<dyn Storage<Elem = T>>>>,
    pub grad_require: bool,
    pub operation: Option<Vec<String>>,
    pub is_child: bool,
    // NOTE: Parents are also Tensors of the same element type.
    pub parent: Option<Vec<Shared<Tensor<T>>>>,
}

// --- TensorHandle Definition ---
// The handle is also generic over `T` now.
#[derive(Clone)]
pub struct TensorHandle<T: TensorFloat>(pub Shared<Tensor<T>>);

impl<T: TensorFloat> Deref for TensorHandle<T> {
    type Target = Shared<Tensor<T>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// --- Implementations ---

impl<T: TensorFloat> Tensor<T> {
    /// A constructor to create a new tensor on a specific device.
    pub fn new(shape: Vec<usize>, grad_require: bool, device: Device) -> TensorHandle<T> {
        // Dispatch to backend for data creation
        let mut tensor = match device {
            Device::Cpu => {
                CpuBackend::random_uniform(&shape, device, -0.5, 0.5).expect("Failed to create tensor")
            }
            Device::Cuda => todo!("Cuda backend not yet fully integrated"),
        };

        tensor.grad_require = grad_require;

        if grad_require {
            let grad_tensor = match device {
                Device::Cpu => CpuBackend::zeros(&shape, device).expect("Failed to create grad tensor"),
                Device::Cuda => todo!("Cuda backend not yet fully integrated"),
            };
            // Extract storage from the grad tensor
            tensor.grad = Some(grad_tensor.data);
        }

        TensorHandle(new_shared(tensor))
    }

    pub fn zeros(shape: Vec<usize>, grad_require: bool, device: Device) -> TensorHandle<T> {
        let mut tensor = match device {
            Device::Cpu => CpuBackend::zeros(&shape, device).expect("Failed to create zeros tensor"),
            Device::Cuda => todo!("Cuda backend not yet fully integrated"),
        };

        tensor.grad_require = grad_require;

        if grad_require {
            let grad_tensor = match device {
                Device::Cpu => CpuBackend::zeros(&shape, device).expect("Failed to create grad tensor"),
                Device::Cuda => todo!("Cuda backend not yet fully integrated"),
            };
            tensor.grad = Some(grad_tensor.data);
        }

        TensorHandle(new_shared(tensor))
    }

    pub fn ones(shape: Vec<usize>, grad_require: bool, device: Device) -> TensorHandle<T> {
        let mut tensor = match device {
            Device::Cpu => CpuBackend::ones(&shape, device).expect("Failed to create ones tensor"),
            Device::Cuda => todo!("Cuda backend not yet fully integrated"),
        };

        tensor.grad_require = grad_require;

        if grad_require {
            let grad_tensor = match device {
                Device::Cpu => CpuBackend::zeros(&shape, device).expect("Failed to create grad tensor"),
                Device::Cuda => todo!("Cuda backend not yet fully integrated"),
            };
            tensor.grad = Some(grad_tensor.data);
        }

        TensorHandle(new_shared(tensor))
    }

    pub fn from_data(
        data: Vec<T>,
        shape: Vec<usize>,
        grad_require: bool,
        device: Device,
    ) -> TensorHandle<T> {
        let mut tensor = match device {
            Device::Cpu => {
                CpuBackend::from_cpu_data(&data, &shape, device).expect("Failed to create tensor from data")
            }
            Device::Cuda => todo!("Cuda backend not yet fully integrated"),
        };

        tensor.grad_require = grad_require;

        if grad_require {
            let grad_tensor = match device {
                Device::Cpu => CpuBackend::zeros(&shape, device).expect("Failed to create grad tensor"),
                Device::Cuda => todo!("Cuda backend not yet fully integrated"),
            };
            tensor.grad = Some(grad_tensor.data);
        }

        TensorHandle(new_shared(tensor))
    }
}
// --- Debug and Display impls ---
//

impl<T: TensorFloat> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("device", &self.data.borrow().device())
            .field("grad_require", &self.grad_require)
            .field("operation", &self.operation)
            .finish()
    }
}

impl<T: TensorFloat> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, device={:?})",
            self.shape,
            self.data.borrow().device()
        )
    }
}

impl<T: TensorFloat> fmt::Debug for TensorHandle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("TensorHandle").field(&self.0).finish()
    }
}

impl<T: TensorFloat> fmt::Display for TensorHandle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T: TensorFloat> TensorHandle<T> {
    pub fn new(shape: Vec<usize>, device: Device, grad_require: bool) -> TensorHandle<T> {
        Tensor::new(shape, grad_require, device)
    }
}

