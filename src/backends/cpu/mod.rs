pub mod binary_ops;
pub mod ternary_ops;
pub mod unary_ops;

use crate::backends::Backend;
use crate::core::device::Device;
use crate::core::error::{Result, TensorError};
use crate::nn::initializers::uniform_;
use crate::core::shared::{new_shared, Shared};
use crate::core::storage::{CpuStorage, Storage};
use crate::tensor::TensorData;
use crate::core::traits::TensorFloat;
use std::any::Any;
use std::cell::RefCell;
use std::marker::PhantomData;

//~~~~~~~~~ CPU BACKEND ~~~~~~~~~//

/// A backend for computations on the CPU.
#[derive(Debug, Clone, Copy)]
pub struct CpuBackend<T>(PhantomData<T>);

impl<T: TensorFloat> Backend<T> for CpuBackend<T> {
    type Storage = CpuStorage<T>;

    fn zeros(shape: &[usize], _device: Device) -> Result<TensorData<T>> {
        let size = shape.iter().product();
        let storage = CpuStorage::zeros(size);
        Ok(TensorData {
            shape: shape.to_vec(),
            data: new_shared(RefCell::new(storage)),
            grad: None,
            grad_require: false,
            operation: None,
            is_child: false,
            parent: None,
        })
    }

    fn ones(shape: &[usize], _device: Device) -> Result<TensorData<T>> {
        let size = shape.iter().product();
        let storage = CpuStorage::ones(size);
        Ok(TensorData {
            shape: shape.to_vec(),
            data: new_shared(RefCell::new(storage)),
            grad: None,
            grad_require: false,
            operation: None,
            is_child: false,
            parent: None,
        })
    }

    fn from_cpu_data(data: &[T], shape: &[usize], _device: Device) -> Result<TensorData<T>> {
        let size = shape.iter().product();
        if data.len() != size {
            return Err(TensorError::ShapeMismatch {
                expected: vec![size],
                found: vec![data.len()],
            });
        }
        let storage = CpuStorage::from_data(data.to_vec());
        Ok(TensorData {
            shape: shape.to_vec(),
            data: new_shared(RefCell::new(storage)),
            grad: None,
            grad_require: false,
            operation: None,
            is_child: false,
            parent: None,
        })
    }

    fn random_uniform(
        shape: &[usize],
        _device: Device,
        min: f64,
        max: f64,
    ) -> Result<TensorData<T>> {
        let size = shape.iter().product();
        let data = uniform_::<T>(size, min as f32, max as f32);
        let storage = CpuStorage::from_data(data);
        Ok(TensorData {
            shape: shape.to_vec(),
            data: new_shared(RefCell::new(storage)),
            grad: None,
            grad_require: false,
            operation: None,
            is_child: false,
            parent: None,
        })
    }

    fn add(a: &TensorData<T>, b: &TensorData<T>) -> Result<TensorData<T>> {
        let a_storage = a
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .ok_or(TensorError::BackendMismatch)?
            .clone();
        let b_storage = b
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .ok_or(TensorError::BackendMismatch)?
            .clone();

        let new_data = crate::backends::cpu::binary_ops::add(a_storage.get_data(), b_storage.get_data());
        let new_storage = CpuStorage::from_data(new_data);

        Ok(TensorData {
            shape: a.shape.clone(), // Assuming shapes are broadcastable/checked elsewhere
            data: new_shared(RefCell::new(new_storage)),
            grad: None,
            grad_require: a.grad_require || b.grad_require,
            operation: Some(vec!["add".to_string()]),
            is_child: true,
            parent: Some(vec![
                new_shared(a.clone()), // Use explicit clone if TensorData doesn't implement Copy (it shouldn't)
                new_shared(b.clone()),
            ]),
        })
    }

    fn sub(a: &TensorData<T>, b: &TensorData<T>) -> Result<TensorData<T>> {
        let a_storage = a
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .ok_or(TensorError::BackendMismatch)?
            .clone();
        let b_storage = b
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .ok_or(TensorError::BackendMismatch)?
            .clone();

        let new_data = crate::backends::cpu::binary_ops::sub(a_storage.get_data(), b_storage.get_data());
        let new_storage = CpuStorage::from_data(new_data);

        Ok(TensorData {
            shape: a.shape.clone(),
            data: new_shared(RefCell::new(new_storage)),
            grad: None,
            grad_require: a.grad_require || b.grad_require,
            operation: Some(vec!["sub".to_string()]),
            is_child: true,
            parent: Some(vec![new_shared(a.clone()), new_shared(b.clone())]),
        })
    }

    fn mul(a: &TensorData<T>, b: &TensorData<T>) -> Result<TensorData<T>> {
        let a_storage = a
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .ok_or(TensorError::BackendMismatch)?
            .clone();
        let b_storage = b
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .ok_or(TensorError::BackendMismatch)?
            .clone();

        let new_data = crate::backends::cpu::binary_ops::mul(a_storage.get_data(), b_storage.get_data());
        let new_storage = CpuStorage::from_data(new_data);

        Ok(TensorData {
            shape: a.shape.clone(),
            data: new_shared(RefCell::new(new_storage)),
            grad: None,
            grad_require: a.grad_require || b.grad_require,
            operation: Some(vec!["mul".to_string()]),
            is_child: true,
            parent: Some(vec![new_shared(a.clone()), new_shared(b.clone())]),
        })
    }

    fn div(a: &TensorData<T>, b: &TensorData<T>) -> Result<TensorData<T>> {
        let a_storage = a
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .ok_or(TensorError::BackendMismatch)?
            .clone();
        let b_storage = b
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .ok_or(TensorError::BackendMismatch)?
            .clone();

        let new_data = crate::backends::cpu::binary_ops::div(a_storage.get_data(), b_storage.get_data());
        let new_storage = CpuStorage::from_data(new_data);

        Ok(TensorData {
            shape: a.shape.clone(),
            data: new_shared(RefCell::new(new_storage)),
            grad: None,
            grad_require: a.grad_require || b.grad_require,
            operation: Some(vec!["div".to_string()]),
            is_child: true,
            parent: Some(vec![new_shared(a.clone()), new_shared(b.clone())]),
        })
    }

    fn matmul(a: &TensorData<T>, b: &TensorData<T>) -> Result<TensorData<T>> {
        let a_storage = a
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .ok_or(TensorError::BackendMismatch)?
            .clone();
        let b_storage = b
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .ok_or(TensorError::BackendMismatch)?
            .clone();

        // Assuming 2D tensors for now, similar to original logic
        let m = a.shape[0];
        let k = a.shape[1];
        let n = b.shape[1];

        let new_data = crate::backends::cpu::binary_ops::matmul(a_storage.get_data(), b_storage.get_data(), m, k, n);
        let new_storage = CpuStorage::from_data(new_data);

        Ok(TensorData {
            shape: vec![m, n],
            data: new_shared(RefCell::new(new_storage)),
            grad: None,
            grad_require: a.grad_require || b.grad_require,
            operation: Some(vec!["matmul".to_string()]),
            is_child: true,
            parent: Some(vec![new_shared(a.clone()), new_shared(b.clone())]),
        })
    }

    fn relu(input: &TensorData<T>) -> Result<TensorData<T>> {
        let storage = input
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .ok_or(TensorError::BackendMismatch)?
            .clone();

        let new_data = crate::backends::cpu::unary_ops::relu(storage.get_data());
        let new_storage = CpuStorage::from_data(new_data);

        Ok(TensorData {
            shape: input.shape.clone(),
            data: new_shared(RefCell::new(new_storage)),
            grad: None,
            grad_require: input.grad_require,
            operation: Some(vec!["relu".to_string()]),
            is_child: true,
            parent: Some(vec![new_shared(input.clone())]),
        })
    }

    fn exp(input: &TensorData<T>) -> Result<TensorData<T>> {
        let storage = input
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .ok_or(TensorError::BackendMismatch)?
            .clone();

        let new_data = crate::backends::cpu::unary_ops::exp(storage.get_data());
        let new_storage = CpuStorage::from_data(new_data);

        Ok(TensorData {
            shape: input.shape.clone(),
            data: new_shared(RefCell::new(new_storage)),
            grad: None,
            grad_require: input.grad_require,
            operation: Some(vec!["exp".to_string()]),
            is_child: true,
            parent: Some(vec![new_shared(input.clone())]),
        })
    }

    fn log(input: &TensorData<T>) -> Result<TensorData<T>> {
        let storage = input
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .ok_or(TensorError::BackendMismatch)?
            .clone();

        let new_data = crate::backends::cpu::unary_ops::log(storage.get_data());
        let new_storage = CpuStorage::from_data(new_data);

        Ok(TensorData {
            shape: input.shape.clone(),
            data: new_shared(RefCell::new(new_storage)),
            grad: None,
            grad_require: input.grad_require,
            operation: Some(vec!["log".to_string()]),
            is_child: true,
            parent: Some(vec![new_shared(input.clone())]),
        })
    }

    fn transpose(input: &TensorData<T>) -> Result<TensorData<T>> {
        let storage = input
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .ok_or(TensorError::BackendMismatch)?
            .clone();

        let m = input.shape[0];
        let n = input.shape[1];

        let new_data = crate::backends::cpu::binary_ops::transpose(storage.get_data(), m, n);
        let new_storage = CpuStorage::from_data(new_data);

        Ok(TensorData {
            shape: vec![n, m], // Swap dims
            data: new_shared(RefCell::new(new_storage)),
            grad: None,
            grad_require: input.grad_require,
            operation: Some(vec!["transpose".to_string()]),
            is_child: true,
            parent: Some(vec![new_shared(input.clone())]),
        })
    }
}

// Keep the explicit derivatives for now, but they might need to move to a different module or be integrated into the Backend trait later.
// For now, making them standalone functions or static methods on a helper struct is better than hanging them on CpuStorage.
pub struct CpuBackprop<T>(PhantomData<T>);

impl<T: TensorFloat> CpuBackprop<T> {
    pub fn derivate_add(
        parent_grad: Shared<RefCell<dyn Storage<Elem = T>>>,
        updated_grad: Shared<RefCell<dyn Storage<Elem = T>>>,
    ) {
        let parent_data = parent_grad.borrow();
        let updated_data = updated_grad.borrow();
        let data = crate::backends::cpu::binary_ops::add_derivate(parent_data.get_data(), updated_data.get_data());
        drop(parent_data);
        drop(updated_data);
        let a = parent_grad.clone();
        let mut b = a.borrow_mut();
        b.add_grad(data);
    }

    pub fn matmul_derivate(
        aparent_data: Shared<TensorData<T>>,
        parent: Shared<TensorData<T>>,
        child: Shared<TensorData<T>>,
        swap: bool,
    ) {
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let input1 = temp1.get_data();
        let input2 = crate::backends::cpu::binary_ops::transpose(
            aparent_data.data.borrow().get_data(),
            aparent_data.shape[0],
            aparent_data.shape[1],
        );

        let data = if swap {
            crate::backends::cpu::binary_ops::matmul(
                input1,
                &input2,
                child.shape[0],
                child.shape[1],
                aparent_data.shape[0],
            )
        } else {
            crate::backends::cpu::binary_ops::matmul(
                &input2,
                input1,
                aparent_data.shape[1],
                aparent_data.shape[0],
                child.shape[1],
            )
        };
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }

    pub fn relu_derivative(parent: Shared<TensorData<T>>, child: Shared<TensorData<T>>) {
        let temp0 = parent.data.clone();
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let child_grad = temp1.get_data();
        let data = crate::backends::cpu::unary_ops::relu_derivative(temp0.borrow().get_data(), child_grad);
        drop(temp0);
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }

    pub fn neg_derivative(parent: Shared<TensorData<T>>, child: Shared<TensorData<T>>) {
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let child_grad = temp1.get_data();
        let data = crate::backends::cpu::unary_ops::neg_derivative(child_grad);
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }

    pub fn sigmodi_derivative(parent: Shared<TensorData<T>>, child: Shared<TensorData<T>>) {
        let temp0 = parent.data.clone();
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let child_grad = temp1.get_data();
        let data = crate::backends::cpu::unary_ops::sigmoid_derivative(temp0.borrow().get_data(), child_grad);
        drop(temp0);
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }

    pub fn tanh_derivative(parent: Shared<TensorData<T>>, child: Shared<TensorData<T>>) {
        let temp0 = parent.data.clone();
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let child_grad = temp1.get_data();
        let data = crate::backends::cpu::unary_ops::tanh_derivative(temp0.borrow().get_data(), child_grad);
        drop(temp0);
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }

    pub fn exp_derivative(parent: Shared<TensorData<T>>, child: Shared<TensorData<T>>) {
        let temp0 = parent.data.clone();
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let child_grad = temp1.get_data();
        let data = crate::backends::cpu::unary_ops::exp_derivative(temp0.borrow().get_data(), child_grad);
        drop(temp0);
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }

    pub fn log_derivative(parent: Shared<TensorData<T>>, child: Shared<TensorData<T>>) {
        let temp0 = parent.data.clone();
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let child_grad = temp1.get_data();
        let data = crate::backends::cpu::unary_ops::log_derivative(temp0.borrow().get_data(), child_grad);
        drop(temp0);
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }

    pub fn abs_derivative(parent: Shared<TensorData<T>>, child: Shared<TensorData<T>>) {
        let temp0 = parent.data.clone();
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let child_grad = temp1.get_data();
        let data = crate::backends::cpu::unary_ops::abs_derivative(temp0.borrow().get_data(), child_grad);
        drop(temp0);
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }

    pub fn square_derivative(parent: Shared<TensorData<T>>, child: Shared<TensorData<T>>) {
        let temp0 = parent.data.clone();
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let child_grad = temp1.get_data();
        let data = crate::backends::cpu::unary_ops::square_derivative(temp0.borrow().get_data(), child_grad);
        drop(temp0);
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }

    pub fn sqrt_derivative(parent: Shared<TensorData<T>>, child: Shared<TensorData<T>>) {
        let temp0 = parent.data.clone();
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let child_grad = temp1.get_data();
        let data = crate::backends::cpu::unary_ops::sqrt_derivative(temp0.borrow().get_data(), child_grad);
        drop(temp0);
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }

    pub fn mean_derivative(parent: Shared<TensorData<T>>, child: Shared<TensorData<T>>) {
        let temp0 = parent.data.clone();
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let child_grad = temp1.get_data();
        let data = crate::backends::cpu::unary_ops::mean_derivative(temp0.borrow().get_data(), child_grad);
        drop(temp0);
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }

    pub fn sv_mul_derivate(
        _aparent_data: Shared<TensorData<T>>,
        parent: Shared<TensorData<T>>,
        child: Shared<TensorData<T>>,
    ) {
        let temp0 = parent.data.clone();
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let child_grad = temp1.get_data();
        let data = crate::backends::cpu::binary_ops::sv_mul_derivate(temp0.borrow().get_data(), child_grad);
        drop(temp0);
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }

    pub fn sv_add_derivate(parent: Shared<TensorData<T>>, child: Shared<TensorData<T>>) {
        let temp0 = parent.data.clone();
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let child_grad = temp1.get_data();
        let data = crate::backends::cpu::binary_ops::sv_mul_derivate(temp0.borrow().get_data(), child_grad);
        drop(temp0);
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }
}

// Storage impl
impl<T: TensorFloat> Storage for CpuStorage<T> {
    type Elem = T;

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn get_data(&self) -> &Vec<Self::Elem> {
        &self.data
    }

    fn device(&self) -> Device {
        Device::Cpu
    }

    fn new(size: usize) -> Self {
        let data = uniform_::<Self::Elem>(size, -0.5, 0.5);
        CpuStorage { data }
    }

    fn zeros(size: usize) -> Self {
        CpuStorage {
            data: vec![Self::Elem::default(); size],
        }
    }
    fn len(&self) -> usize {
        self.data.len()
    }
    fn update_data(&mut self, a: Vec<T>) {
        self.data = a
    }
    fn from_data(data: Vec<Self::Elem>) -> Self {
        Self { data }
    }
    fn ones(size: usize) -> Self {
        CpuStorage {
            data: vec![Self::Elem::from(1.0).unwrap(); size],
        }
    }
    fn fill_ones(&mut self) {
        self.data = vec![Self::Elem::from(1.0).unwrap(); self.data.len()]
    }
    fn fill_data(&mut self, data: Vec<Self::Elem>) {
        self.data = data;
    }
    fn add_grad(&mut self, data: Vec<Self::Elem>) {
        let data = self
            .data
            .iter()
            .zip(data.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        self.data = data;
    }
    fn get_dim_slice(self, dim: usize, gap: usize) -> Vec<T> {
        let mut data: Vec<T> = vec![Self::Elem::from(1.0).unwrap(); dim];
        for i in 0..dim {
            data[i] = self.data[i * gap];
        }
        data
    }
}
