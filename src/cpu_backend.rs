use crate::backend::Backend;
use crate::device::Device;
use crate::error::{Result, TensorError};
use crate::gemm::Gemm;
use crate::initializers::uniform_;
use crate::shared::{Shared, new_shared};
use crate::storage::{CpuStorage, Storage};
use crate::tensor::{Tensor, TensorHandle};
use crate::traits::TensorFloat;
use crate::{cpubinaryops, cpuurnaryops, shared};
use std::any::Any;
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Sub};

//~~~~~~~~~ CPU BACKEND ~~~~~~~~~//

/// A backend for computations on the CPU.
// pub struct CpuBackend<T>;
impl<T: TensorFloat> CpuStorage<T> {
    pub fn add(
        a: Shared<RefCell<dyn Storage<Elem = T>>>,
        b: Shared<RefCell<dyn Storage<Elem = T>>>,
    ) -> CpuStorage<T> {
        let new_storage = CpuStorage {
            data: cpubinaryops::add(a.borrow().get_data(), b.borrow().get_data()),
        };
        new_storage
    }
    pub fn sub(
        a: Shared<RefCell<dyn Storage<Elem = T>>>,
        b: Shared<RefCell<dyn Storage<Elem = T>>>,
    ) -> CpuStorage<T> {
        let new_storage = CpuStorage {
            data: cpubinaryops::sub(a.borrow().get_data(), b.borrow().get_data()),
        };
        new_storage
    }
    pub fn mul(
        a: Shared<RefCell<dyn Storage<Elem = T>>>,
        b: Shared<RefCell<dyn Storage<Elem = T>>>,
    ) -> CpuStorage<T> {
        let new_storage = CpuStorage {
            data: cpubinaryops::mul(a.borrow().get_data(), b.borrow().get_data()),
        };
        new_storage
    }
    pub fn div(
        a: Shared<RefCell<dyn Storage<Elem = T>>>,
        b: Shared<RefCell<dyn Storage<Elem = T>>>,
    ) -> CpuStorage<T> {
        let new_storage = CpuStorage {
            data: cpubinaryops::div(a.borrow().get_data(), b.borrow().get_data()),
        };
        new_storage
    }
    pub fn relu(a: Shared<RefCell<dyn Storage<Elem = T>>>) -> CpuStorage<T> {
        let new_storage = CpuStorage {
            data: cpuurnaryops::relu(a.borrow().get_data()),
        };
        new_storage
    }
    pub fn sigmoid(a: Shared<RefCell<dyn Storage<Elem = T>>>) -> CpuStorage<T> {
        let new_storage = CpuStorage {
            data: cpuurnaryops::sigmoid(a.borrow().get_data()),
        };
        new_storage
    }
    pub fn tanh(a: Shared<RefCell<dyn Storage<Elem = T>>>) -> CpuStorage<T> {
        let new_storage = CpuStorage {
            data: cpuurnaryops::tanh(a.borrow().get_data()),
        };
        new_storage
    }
    pub fn exp(a: Shared<RefCell<dyn Storage<Elem = T>>>) -> CpuStorage<T> {
        let new_storage = CpuStorage {
            data: cpuurnaryops::exp(a.borrow().get_data()),
        };
        new_storage
    }
    pub fn neg(a: Shared<RefCell<dyn Storage<Elem = T>>>) -> CpuStorage<T> {
        let new_storage = CpuStorage {
            data: cpuurnaryops::neg(a.borrow().get_data()),
        };
        new_storage
    }
    pub fn abs(a: Shared<RefCell<dyn Storage<Elem = T>>>) -> CpuStorage<T> {
        let new_storage = CpuStorage {
            data: cpuurnaryops::abs(a.borrow().get_data()),
        };
        new_storage
    }
    pub fn log(a: Shared<RefCell<dyn Storage<Elem = T>>>) -> CpuStorage<T> {
        let new_storage = CpuStorage {
            data: cpuurnaryops::log(a.borrow().get_data()),
        };
        new_storage
    }
    pub fn sqrt(a: Shared<RefCell<dyn Storage<Elem = T>>>) -> CpuStorage<T> {
        let new_storage = CpuStorage {
            data: cpuurnaryops::sqrt(a.borrow().get_data()),
        };
        new_storage
    }
    pub fn SV_add(input: Shared<RefCell<dyn Storage<Elem = T>>>, scalar: T) -> CpuStorage<T> {
        let new_storage = CpuStorage {
            data: cpubinaryops::SV_add(input.borrow().get_data(), scalar),
        };
        new_storage
    }
    pub fn SV_mul(input: Shared<RefCell<dyn Storage<Elem = T>>>, scalar: T) -> CpuStorage<T> {
        let new_storage = CpuStorage {
            data: cpubinaryops::SV_add(input.borrow().get_data(), scalar),
        };
        new_storage
    }
    pub fn SV_sub(input: Shared<RefCell<dyn Storage<Elem = T>>>, scalar: T) -> CpuStorage<T> {
        let new_storage = CpuStorage {
            data: cpubinaryops::SV_add(input.borrow().get_data(), scalar),
        };
        new_storage
    }
    pub fn sum(input: Shared<RefCell<dyn Storage<Elem = T>>>) -> CpuStorage<T> {
        let new_storage = CpuStorage {
            data: cpuurnaryops::sum(input.borrow().get_data()),
        };
        new_storage
    }
    // feature needs to work on
    // pub fn sum(
    //     a: Shared<dyn Storage<Elem = T>>,
    //     shape: Vec<usize>,
    //     access_dim: usize,
    // ) -> CpuStorage<T> {
    //     let gap = shape.iter().product();
    //     let gap = gap / access_dim;
    //     let data = cpuurnaryops::sum(a.clone(), access_dim, gap);
    //     data.iter().sum();
    //     let new_storage = CpuStorage {
    //         data: cpuurnaryops::sum(a.get_data()),
    //     };
    //     new_storage
    // }

    pub fn power(a: Shared<RefCell<dyn Storage<Elem = T>>>, power: T) -> CpuStorage<T> {
        let new_storage = CpuStorage {
            data: cpuurnaryops::power(a.borrow().get_data(), power),
        };
        new_storage
    }
    pub fn square(a: Shared<RefCell<dyn Storage<Elem = T>>>) -> CpuStorage<T> {
        let new_storage = CpuStorage {
            data: cpuurnaryops::square(a.borrow().get_data()),
        };
        new_storage
    }
    pub fn mean(a: Shared<RefCell<dyn Storage<Elem = T>>>) -> CpuStorage<T> {
        let new_storage = CpuStorage {
            data: cpuurnaryops::mean(a.borrow().get_data()),
        };
        new_storage
    }
    pub fn matmul(
        a: Shared<RefCell<dyn Storage<Elem = T>>>,
        b: Shared<RefCell<dyn Storage<Elem = T>>>,
        m: usize,
        k: usize,
        n: usize,
    ) -> CpuStorage<T> {
        let new_storage = CpuStorage {
            data: cpubinaryops::matmul(a.borrow().get_data(), b.borrow().get_data(), m, k, n),
        };
        new_storage
    }
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
            // println!("this is input2{:?}",&input2);
            // println!("this is input1{:?}",input1);
            data = cpubinaryops::matmul(
                &input2,
                input1,
                aparent_data.shape[1],
                aparent_data.shape[0],
                child.shape[1],
            );
            // println!("this is from matmul_derivate1{:?}",data);
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
    pub fn SV_mul_derivate(parent: Shared<Tensor<T>>, child: Shared<Tensor<T>>) {
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
impl<T: TensorFloat> Storage for CpuStorage<T> {
    type Elem = T;

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn get_data(&self) -> &Vec<Self::Elem> {
        &self.data
    }
    // fn get_mut_data(self) -> Self {
    //     self.data = vec![Self::Elem::default(); self.data.len()]
    // }

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
pub fn SV_add_derivate(parent: Shared<Tensor<T>>, child: Shared<Tensor<T>>) {
    let mut data: Vec<T> = Vec::new();
    let temp0 = parent.data.clone();
    let temp1 = child.grad.as_ref().unwrap().clone();
    let temp1 = temp1.borrow();
    let child_grad = temp1.get_data();
    data = cpuurnaryops::SV_add_derivate(temp0.borrow().get_data(), child_grad);
    drop(temp0);
    drop(temp1);
    let temp1 = parent.grad.as_ref().unwrap().clone();
    let mut parent_grad = temp1.borrow_mut();
    parent_grad.add_grad(data);
}

// impl<T: TensorFloat> Sub for TensorHandle<T> {
//     type Output = Result<TensorHandle<T>>;
//     fn sub(a: Shared<Tensor<T>>, b: Shared<Tensor<T>>) -> Result<TensorHandle<T>> {
//         cpubinaryops::sub(a, b)
//     }
// }
// impl<T: TensorFloat> Mul for TensorHandle<T> {
//     type Output = Result<TensorHandle<T>>;
//     fn mul(a: Shared<Tensor<T>>, b: Shared<Tensor<T>>) -> Result<TensorHandle<T>> {
//         cpubinaryops::mul(a, b)
//     }
// }
// impl<T: TensorFloat> Div for TensorHandle<T> {
//     type Output = Result<TensorHandle<T>>;
//     fn div(a: Shared<Tensor<T>>, b: Shared<Tensor<T>>) -> Result<TensorHandle<T>> {
//         cpubinaryops::div(a, b)
//     }
// }
// impl<T: TensorFloat> TensorHandle<T> {
//     pub fn matmul(a: Shared<Tensor<T>>, b: Shared<Tensor<T>>) -> Result<TensorHandle<T>> {
//         cpubinaryops::matmul(a, b)
//     }
//
//     pub fn transpose(input: Shared<Tensor<T>>) -> Result<TensorHandle<T>> {
//         cpubinaryops::transpose(input)
//     }

//
// fn relu(input: &Tensor<f>) -> Result<Tensor<f32w2>) -> Result<Tensor<f32>> {
//     let storage = input
//         .data
//         .as_any()
//         .downcast_ref::<Self::Storage>()
//         .ok_or(TensorError::BackendMismatch)?;
//     let new_data = storage.data.iter().map(|&x| x.exp()).collect();
//     let new_storage = CpuStorage { data: new_data };
//     let grad_require = input.grad_require;
//     Ok(Tensor {
//         shape: input.shape.clone(),
//         data: Box::new(new_storage),
//         grad: if grad_require {
//             Some(Box::new(CpuStorage::zeros(
//                 Device::Cpu,
//                 input.shape.iter().product(),
//             )?))
//         } else {
//             None
//         },
//         grad_require,
//         operation: Some(vec!["exp".to_string()]),
//         is_child: true,
//         parent: Some(vec![new_shared(input.clone())]),
//     })
// }
//
// fn log(input: &Tensor<f32>) -> Result<Tensor<f32>> {
//     let storage = input
//         .data
//         .as_any()
//         .downcast_ref::<Self::Storage>()
//         .ok_or(TensorError::BackendMismatch)?;
//     let new_data = storage.data.iter().map(|&x| x.ln()).collect();
//     let new_storage = CpuStorage { data: new_data };
//     let grad_require = input.grad_require;
//     Ok(Tensor {
//         shape: input.shape.clone(),
//         data: Box::new(new_storage),
//         grad: if grad_require {
//             Some(Box::new(CpuStorage::zeros(
//                 Device::Cpu,
//                 input.shape.iter().product(),
//             )?))
//         } else {
//             None
//         },
//         grad_require,
//         operation: Some(vec!["log".to_string()]),
//         is_child: true,
//         parent: Some(vec![new_shared(input.clone())]),
//     })
// }
//
// fn transpose(input: &Tensor<f32>) -> Result<Tensor<f32>> {
//     todo!()
// }
// }
