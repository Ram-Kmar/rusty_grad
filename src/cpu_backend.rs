use crate::backend::Backend;
use crate::device::Device;
use crate::error::{Result, TensorError};
use crate::initializers::uniform_;
use crate::shared::{new_shared, Shared};
use crate::storage::{CpuStorage, Storage};
use crate::tensor::Tensor;
use crate::traits::TensorFloat;
use crate::{cpubinaryops, cpuurnaryops};
use std::any::Any;
use std::cell::RefCell;
use std::marker::PhantomData;

//~~~~~~~~~ CPU BACKEND ~~~~~~~~~//

/// A backend for computations on the CPU.
#[derive(Debug, Clone, Copy)]
pub struct CpuBackend<T>(PhantomData<T>);

impl<T: TensorFloat> Backend<T> for CpuBackend<T> {
    type Storage = CpuStorage<T>;

    fn zeros(shape: &[usize], device: Device) -> Result<Tensor<T>> {
        let size = shape.iter().product();
        let storage = CpuStorage::zeros(size);
        Ok(Tensor {
            shape: shape.to_vec(),
            data: new_shared(RefCell::new(storage)),
            grad: None,
            grad_require: false,
            operation: None,
            is_child: false,
            parent: None,
        })
    }

    fn ones(shape: &[usize], device: Device) -> Result<Tensor<T>> {
        let size = shape.iter().product();
        let storage = CpuStorage::ones(size);
        Ok(Tensor {
            shape: shape.to_vec(),
            data: new_shared(RefCell::new(storage)),
            grad: None,
            grad_require: false,
            operation: None,
            is_child: false,
            parent: None,
        })
    }

    fn from_cpu_data(data: &[T], shape: &[usize], device: Device) -> Result<Tensor<T>> {
        let size = shape.iter().product();
        if data.len() != size {
            return Err(TensorError::ShapeMismatch);
        }
        let storage = CpuStorage::from_data(data.to_vec());
        Ok(Tensor {
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
        device: Device,
        min: f64,
        max: f64,
    ) -> Result<Tensor<T>> {
        let size = shape.iter().product();
        let data = uniform_::<T>(size, min as f32, max as f32);
        let storage = CpuStorage::from_data(data);
        Ok(Tensor {
            shape: shape.to_vec(),
            data: new_shared(RefCell::new(storage)),
            grad: None,
            grad_require: false,
            operation: None,
            is_child: false,
            parent: None,
        })
    }

    fn add(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
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

        let new_data = cpubinaryops::add(a_storage.get_data(), b_storage.get_data());
        let new_storage = CpuStorage::from_data(new_data);

        Ok(Tensor {
            shape: a.shape.clone(), // Assuming shapes are broadcastable/checked elsewhere
            data: new_shared(RefCell::new(new_storage)),
            grad: None,
            grad_require: a.grad_require || b.grad_require,
            operation: Some(vec!["add".to_string()]),
            is_child: true,
            parent: Some(vec![
                new_shared(a.clone()), // Use explicit clone if Tensor doesn't implement Copy (it shouldn't)
                new_shared(b.clone()),
            ]),
        })
    }

    fn sub(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
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

        let new_data = cpubinaryops::sub(a_storage.get_data(), b_storage.get_data());
        let new_storage = CpuStorage::from_data(new_data);

        Ok(Tensor {
            shape: a.shape.clone(),
            data: new_shared(RefCell::new(new_storage)),
            grad: None,
            grad_require: a.grad_require || b.grad_require,
            operation: Some(vec!["sub".to_string()]),
            is_child: true,
            parent: Some(vec![new_shared(a.clone()), new_shared(b.clone())]),
        })
    }

    fn mul(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
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

        let new_data = cpubinaryops::mul(a_storage.get_data(), b_storage.get_data());
        let new_storage = CpuStorage::from_data(new_data);

        Ok(Tensor {
            shape: a.shape.clone(),
            data: new_shared(RefCell::new(new_storage)),
            grad: None,
            grad_require: a.grad_require || b.grad_require,
            operation: Some(vec!["mul".to_string()]),
            is_child: true,
            parent: Some(vec![new_shared(a.clone()), new_shared(b.clone())]),
        })
    }

    fn div(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
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

        let new_data = cpubinaryops::div(a_storage.get_data(), b_storage.get_data());
        let new_storage = CpuStorage::from_data(new_data);

        Ok(Tensor {
            shape: a.shape.clone(),
            data: new_shared(RefCell::new(new_storage)),
            grad: None,
            grad_require: a.grad_require || b.grad_require,
            operation: Some(vec!["div".to_string()]),
            is_child: true,
            parent: Some(vec![new_shared(a.clone()), new_shared(b.clone())]),
        })
    }

    fn matmul(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
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

        let new_data = cpubinaryops::matmul(a_storage.get_data(), b_storage.get_data(), m, k, n);
        let new_storage = CpuStorage::from_data(new_data);

        Ok(Tensor {
            shape: vec![m, n],
            data: new_shared(RefCell::new(new_storage)),
            grad: None,
            grad_require: a.grad_require || b.grad_require,
            operation: Some(vec!["matmul".to_string()]),
            is_child: true,
            parent: Some(vec![new_shared(a.clone()), new_shared(b.clone())]),
        })
    }

    fn relu(input: &Tensor<T>) -> Result<Tensor<T>> {
        let storage = input
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .ok_or(TensorError::BackendMismatch)?
            .clone();

        let new_data = cpuurnaryops::relu(storage.get_data());
        let new_storage = CpuStorage::from_data(new_data);

        Ok(Tensor {
            shape: input.shape.clone(),
            data: new_shared(RefCell::new(new_storage)),
            grad: None,
            grad_require: input.grad_require,
            operation: Some(vec!["relu".to_string()]),
            is_child: true,
            parent: Some(vec![new_shared(input.clone())]),
        })
    }

    fn exp(input: &Tensor<T>) -> Result<Tensor<T>> {
        let storage = input
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .ok_or(TensorError::BackendMismatch)?
            .clone();

        let new_data = cpuurnaryops::exp(storage.get_data());
        let new_storage = CpuStorage::from_data(new_data);

        Ok(Tensor {
            shape: input.shape.clone(),
            data: new_shared(RefCell::new(new_storage)),
            grad: None,
            grad_require: input.grad_require,
            operation: Some(vec!["exp".to_string()]),
            is_child: true,
            parent: Some(vec![new_shared(input.clone())]),
        })
    }

    fn log(input: &Tensor<T>) -> Result<Tensor<T>> {
        let storage = input
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .ok_or(TensorError::BackendMismatch)?
            .clone();

        let new_data = cpuurnaryops::log(storage.get_data());
        let new_storage = CpuStorage::from_data(new_data);

        Ok(Tensor {
            shape: input.shape.clone(),
            data: new_shared(RefCell::new(new_storage)),
            grad: None,
            grad_require: input.grad_require,
            operation: Some(vec!["log".to_string()]),
            is_child: true,
            parent: Some(vec![new_shared(input.clone())]),
        })
    }

    fn transpose(input: &Tensor<T>) -> Result<Tensor<T>> {
        let storage = input
            .data
            .borrow()
            .as_any()
            .downcast_ref::<CpuStorage<T>>()
            .ok_or(TensorError::BackendMismatch)?
            .clone();

        let m = input.shape[0];
        let n = input.shape[1];

        let new_data = cpubinaryops::transpose(storage.get_data(), m, n);
        let new_storage = CpuStorage::from_data(new_data);

        Ok(Tensor {
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
        let data = cpubinaryops::add_derivate(parent_data.get_data(), updated_data.get_data());
        drop(parent_data);
        drop(updated_data);
        let a = parent_grad.clone();
        let mut b = a.borrow_mut();
        b.add_grad(data);
    }

    pub fn matmul_derivate(
        aparent_data: Shared<Tensor<T>>,
        parent: Shared<Tensor<T>>,
        child: Shared<Tensor<T>>,
        swap: bool,
    ) {
        let mut data: Vec<T> = Vec::new();
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let input1 = temp1.get_data();
        let input2 = cpubinaryops::transpose(
            aparent_data.data.borrow().get_data(),
            aparent_data.shape[0],
            aparent_data.shape[1],
        );

        if swap == true {
            data = cpubinaryops::matmul(
                input1,
                &input2,
                child.shape[0],
                child.shape[1],
                aparent_data.shape[0],
            );
        } else {
            data = cpubinaryops::matmul(
                &input2,
                input1,
                aparent_data.shape[1],
                aparent_data.shape[0],
                child.shape[1],
            );
        }
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }

    pub fn relu_derivative(parent: Shared<Tensor<T>>, child: Shared<Tensor<T>>) {
        let mut data: Vec<T> = Vec::new();
        let temp0 = parent.data.clone();
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let child_grad = temp1.get_data();
        data = cpuurnaryops::relu_derivative(temp0.borrow().get_data(), child_grad);
        drop(temp0);
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }

    pub fn neg_derivative(parent: Shared<Tensor<T>>, child: Shared<Tensor<T>>) {
        let mut data: Vec<T> = Vec::new();
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let child_grad = temp1.get_data();
        data = cpuurnaryops::neg_derivative(child_grad);
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }

    pub fn sigmodi_derivative(parent: Shared<Tensor<T>>, child: Shared<Tensor<T>>) {
        let mut data: Vec<T> = Vec::new();
        let temp0 = parent.data.clone();
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let child_grad = temp1.get_data();
        data = cpuurnaryops::sigmoid_derivative(temp0.borrow().get_data(), child_grad);
        drop(temp0);
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }

    pub fn tanh_derivative(parent: Shared<Tensor<T>>, child: Shared<Tensor<T>>) {
        let mut data: Vec<T> = Vec::new();
        let temp0 = parent.data.clone();
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let child_grad = temp1.get_data();
        data = cpuurnaryops::tanh_derivative(temp0.borrow().get_data(), child_grad);
        drop(temp0);
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }

    pub fn exp_derivative(parent: Shared<Tensor<T>>, child: Shared<Tensor<T>>) {
        let mut data: Vec<T> = Vec::new();
        let temp0 = parent.data.clone();
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let child_grad = temp1.get_data();
        data = cpuurnaryops::exp_derivative(temp0.borrow().get_data(), child_grad);
        drop(temp0);
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }

    pub fn log_derivative(parent: Shared<Tensor<T>>, child: Shared<Tensor<T>>) {
        let mut data: Vec<T> = Vec::new();
        let temp0 = parent.data.clone();
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let child_grad = temp1.get_data();
        data = cpuurnaryops::log_derivative(temp0.borrow().get_data(), child_grad);
        drop(temp0);
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }

    pub fn abs_derivative(parent: Shared<Tensor<T>>, child: Shared<Tensor<T>>) {
        let mut data: Vec<T> = Vec::new();
        let temp0 = parent.data.clone();
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let child_grad = temp1.get_data();
        data = cpuurnaryops::abs_derivative(temp0.borrow().get_data(), child_grad);
        drop(temp0);
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }

    pub fn square_derivative(parent: Shared<Tensor<T>>, child: Shared<Tensor<T>>) {
        let mut data: Vec<T> = Vec::new();
        let temp0 = parent.data.clone();
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let child_grad = temp1.get_data();
        data = cpuurnaryops::square_derivative(temp0.borrow().get_data(), child_grad);
        drop(temp0);
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }

    pub fn sqrt_derivative(parent: Shared<Tensor<T>>, child: Shared<Tensor<T>>) {
        let mut data: Vec<T> = Vec::new();
        let temp0 = parent.data.clone();
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let child_grad = temp1.get_data();
        data = cpuurnaryops::sqrt_derivative(temp0.borrow().get_data(), child_grad);
        drop(temp0);
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }

    pub fn mean_derivative(parent: Shared<Tensor<T>>, child: Shared<Tensor<T>>) {
        let mut data: Vec<T> = Vec::new();
        let temp0 = parent.data.clone();
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let child_grad = temp1.get_data();
        data = cpuurnaryops::mean_derivative(temp0.borrow().get_data(), child_grad);
        drop(temp0);
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }

    pub fn SV_mul_derivate(
        aparent_data: Shared<Tensor<T>>,
        parent: Shared<Tensor<T>>,
        child: Shared<Tensor<T>>,
    ) {
        let mut data: Vec<T> = Vec::new();
        let temp0 = parent.data.clone();
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let child_grad = temp1.get_data();
        data = cpubinaryops::SV_mul_derivate(temp0.borrow().get_data(), child_grad);
        drop(temp0);
        drop(temp1);
        let temp1 = parent.grad.as_ref().unwrap().clone();
        let mut parent_grad = temp1.borrow_mut();
        parent_grad.add_grad(data);
    }

    pub fn SV_add_derivate(parent: Shared<Tensor<T>>, child: Shared<Tensor<T>>) {
        let mut data: Vec<T> = Vec::new();
        let temp0 = parent.data.clone();
        let temp1 = child.grad.as_ref().unwrap().clone();
        let temp1 = temp1.borrow();
        let child_grad = temp1.get_data();
        data = cpubinaryops::SV_mul_derivate(temp0.borrow().get_data(), child_grad);
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
