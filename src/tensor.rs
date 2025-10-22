use crate::device::{self, Device};
use crate::error::Result;
use crate::storage::{CpuStorage, Storage};
// use crate::tensorbackprop::mul_backward;
use crate::traits::TensorFloat;
use std::cell::RefCell;
// use std::collections::HashSet;
use std::fmt::{self, Debug};
use std::ops::Deref;

use crate::shared::{Shared, new_shared};

// --- Tensor Definition ---

// NOTE: Tensor is now generic over the element type `T`, not the storage type `S`.
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
        let size = shape.iter().product();

        // TODO: This should be dispatched via a backend, but for now, we hardcode CPU
        // to demonstrate the structure.
        let data = match device {
            Device::Cpu => Shared::new(RefCell::new(CpuStorage::new(size))),
            Device::Cuda => Shared::new(RefCell::new(CpuStorage::new(size))),
        };
        let grad = if grad_require {
            let grad_storage: Shared<RefCell<dyn Storage<Elem = T>>> = match device {
                Device::Cpu => Shared::new(RefCell::new(CpuStorage::zeros(size))),
                Device::Cuda => Shared::new(RefCell::new(CpuStorage::zeros(size))),
            };
            Some(grad_storage)
        } else {
            None
        };
        // return Err(crate::error::TensorError::ShapeMismatch);
        // println!("this is the vector {:?}", data.get_data());

        TensorHandle(new_shared(Self {
            shape,
            data,
            grad,
            grad_require,
            operation: None,
            is_child: false,
            parent: None,
        }))
    }

    pub fn zeros(shape: Vec<usize>, grad_require: bool, device: Device) -> TensorHandle<T> {
        let size = shape.iter().product();

        let data = match device {
            Device::Cpu => Shared::new(RefCell::new(CpuStorage::new(size))),
            Device::Cuda => Shared::new(RefCell::new(CpuStorage::new(size))),
        };
        let grad = if grad_require {
            let grad_storage: Shared<RefCell<dyn Storage<Elem = T>>> = match device {
                Device::Cpu => Shared::new(RefCell::new(CpuStorage::zeros(size))),
                Device::Cuda => Shared::new(RefCell::new(CpuStorage::zeros(size))),
            };
            Some(grad_storage)
        } else {
            None
        };

        TensorHandle(new_shared(Self {
            shape,
            data,
            grad,
            grad_require,
            operation: None,
            is_child: false,
            parent: None,
        }))
    }
    pub fn ones(shape: Vec<usize>, grad_require: bool, device: Device) -> TensorHandle<T> {
        let size = shape.iter().product();

        let data = match device {
            Device::Cpu => Shared::new(RefCell::new(CpuStorage::new(size))),
            Device::Cuda => Shared::new(RefCell::new(CpuStorage::new(size))),
        };
        let grad = if grad_require {
            let grad_storage: Shared<RefCell<dyn Storage<Elem = T>>> = match device {
                Device::Cpu => Shared::new(RefCell::new(CpuStorage::zeros(size))),
                Device::Cuda => Shared::new(RefCell::new(CpuStorage::zeros(size))),
            };
            // let grad = if grad_require {
            //     let grad_storage: Shared<dyn Storage<Elem = T>> = match device {
            //         Device::Cpu => Shared::new(CpuStorage::zeros(size)),
            //         Device::Cuda => Shared::new(CpuStorage::zeros(size)),
            //     };
            Some(grad_storage)
        } else {
            None
        };

        TensorHandle(new_shared(Self {
            shape,
            data,
            grad,
            grad_require,
            operation: None,
            is_child: false,
            parent: None,
        }))
    }
    pub fn from_data(
        data: Vec<T>,
        shape: Vec<usize>,
        grad_require: bool,
        device: Device,
    ) -> TensorHandle<T> {
        let size = shape.iter().product();
        let data = match device {
            Device::Cpu => Shared::new(RefCell::new(CpuStorage::new(size))),
            Device::Cuda => Shared::new(RefCell::new(CpuStorage::new(size))),
        };
        let grad = if grad_require {
            let grad_storage: Shared<RefCell<dyn Storage<Elem = T>>> = match device {
                Device::Cpu => Shared::new(RefCell::new(CpuStorage::zeros(size))),
                Device::Cuda => Shared::new(RefCell::new(CpuStorage::zeros(size))),
            };
            Some(grad_storage)
        } else {
            None
        };
        // println!("this is the vector {:?}", data.get_data());
        TensorHandle(new_shared(Self {
            shape,
            data,
            grad_require,
            grad,
            operation: None,
            is_child: false,
            parent: None,
        }))
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

