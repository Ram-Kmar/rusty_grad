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
    pub data: Shared<dyn Storage<Elem = T>>,
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
        let data: Shared<dyn Storage<Elem = T>> = match device {
            Device::Cpu => Shared::new(CpuStorage::new(size)),
            Device::Cuda => Shared::new(CpuStorage::new(size)),
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

        let data: Shared<dyn Storage<Elem = T>> = match device {
            Device::Cpu => Shared::new(CpuStorage::zeros(size)),
            Device::Cuda => Shared::new(CpuStorage::zeros(size)),
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

        let data: Shared<dyn Storage<Elem = T>> = match device {
            Device::Cpu => Shared::new(CpuStorage::zeros(size)),
            Device::Cuda => Shared::new(CpuStorage::zeros(size)),
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
        let data: Shared<dyn Storage<Elem = T>> = match device {
            Device::Cpu => Shared::new(CpuStorage::from_data(data)),
            Device::Cuda => Shared::new(CpuStorage::zeros(size)),
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
            .field("device", &self.data.device())
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
            self.data.device()
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

// impl<T: TensorFloat> Default for TensorHandle<T> {
//     fn default() -> Self {
//         Tensor::new()
//     }
// }
impl<T: TensorFloat> TensorHandle<T> {
    // pub fn backward(&self) {
    //     let sorted_nodes = self.build_topological_sort();
    //     self.grad.as_ref().unwrap().borrow_mut().fill(); // fill method is not defined on either on tensor struct or cpustorage struct
    //
    //     for node in sorted_nodes.iter().rev() {
    //
    //
    //     }
    // }
    // fn build_topological_sort(&self) -> Vec<TensorHandle<T>> {
    //     let mut sorted = Vec::new();
    //     let mut visited = HashSet::new();
    //
    //     fn visit<T: TensorFloat>(
    //         node: &TensorHandle<T>,
    //         sorted: &mut Vec<TensorHandle<T>>,
    //         visited: &mut HashSet<*const Tensor<T>>,
    //     ) {
    //         let node_ptr = Shared::as_ptr(&node.0);
    //         if visited.contains(&node_ptr) {
    //             return;
    //         }
    //         visited.insert(node_ptr);
    //
    //         if let Some(parents) = &node.parent {
    //             for parent in parents {
    //                 visit(&TensorHandle(parent.clone()), sorted, visited);
    //             }
    //         }
    //         sorted.push(node.clone());
    //     }
    //
    //     visit(self, &mut sorted, &mut visited);
    //     sorted
    // }
}
