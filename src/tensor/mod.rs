pub mod backprop;
pub mod ops;

use crate::backends::Backend;
use crate::backends::cpu::CpuBackend;
use crate::core::device::Device;
use crate::core::storage::Storage;
// use crate::tensor::backprop::mul_backward;
use crate::core::traits::TensorFloat;
use std::cell::RefCell;
// use std::collections::HashSet;
use std::fmt::{self};
use std::ops::Deref;

use crate::core::shared::{new_shared, Shared};

// --- TensorData Definition ---

// NOTE: TensorData is now generic over the element type `T`, not the storage type `S`.
#[derive(Clone)]
pub struct TensorData<T: TensorFloat> {
    pub shape: Vec<usize>,
    // NOTE: The `data` field is now a trait object.
    // It can hold a CpuStorage, CudaStorage, or any other type that implements the Storage trait.
    pub data: Shared<RefCell<dyn Storage<Elem = T>>>,
    pub grad: Option<Shared<RefCell<dyn Storage<Elem = T>>>>,
    pub grad_require: bool,
    pub operation: Option<Vec<String>>,
    pub is_child: bool,
    // NOTE: Parents are also Tensors of the same element type.
    pub parent: Option<Vec<Shared<TensorData<T>>>>,
}

// --- Tensor Definition ---
// The handle is also generic over `T` now.
#[derive(Clone)]
pub struct Tensor<T: TensorFloat>(pub Shared<TensorData<T>>);

impl<T: TensorFloat> Deref for Tensor<T> {
    type Target = Shared<TensorData<T>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// --- Implementations ---

impl<T: TensorFloat> Tensor<T> {
    /// A constructor to create a new tensor on a specific device.
    pub fn new(shape: Vec<usize>, grad_require: bool, device: Device) -> Tensor<T> {
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

        Tensor(new_shared(tensor))
    }

    pub fn zeros(shape: Vec<usize>, grad_require: bool, device: Device) -> Tensor<T> {
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

        Tensor(new_shared(tensor))
    }

    pub fn ones(shape: Vec<usize>, grad_require: bool, device: Device) -> Tensor<T> {
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

        Tensor(new_shared(tensor))
    }

    pub fn from_data(
        data: Vec<T>,
        shape: Vec<usize>,
        grad_require: bool,
        device: Device,
    ) -> Tensor<T> {
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

        Tensor(new_shared(tensor))
    }
}
// --- Debug and Display impls ---
//

impl<T: TensorFloat> fmt::Debug for TensorData<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorData")
            .field("shape", &self.shape)
            .field("device", &self.data.borrow().device())
            .field("grad_require", &self.grad_require)
            .field("operation", &self.operation)
            .finish()
    }
}

impl<T: TensorFloat> fmt::Display for TensorData<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TensorData(shape={:?}, device={:?}, data={:?})",
            self.shape,
            self.data.borrow().device(),
            self.data.borrow()
        )
    }
}

impl<T: TensorFloat> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Tensor").field(&self.0).finish()
    }
}

impl<T: TensorFloat> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}


