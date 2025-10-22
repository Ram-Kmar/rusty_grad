use crate::cpu_backend;
use crate::cpubinaryops;
use crate::storage::{CpuStorage, Storage};
use crate::{tensor::TensorHandle, tensorbinaryops::TensorBinaryOps, traits::TensorFloat};
pub struct Sgd {
    pub lr: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub nesterov: bool,
}

impl Sgd {
    pub fn sgd<T: TensorFloat>(weights: &Vec<T>, grads: &Vec<T>, lr: T) -> Vec<T> {
        weights
            .iter()
            .zip(grads.iter())
            .map(|(&w, &g)| w - lr * g)
            .collect()
    }

    pub fn update<T: TensorFloat>(&self, initial_point: &TensorHandle<T>) {
        let sorted = initial_point.build_topological_sort();
        for node in sorted.iter().rev() {
            // println!("this is first element{:?}", node.data.borrow().get_data());
            let lr = T::from(self.lr).unwrap();
            let data = TensorHandle::SV_mul(node.clone(), lr, "grad".to_string());
            let data =
                cpubinaryops::sub(node.data.borrow().get_data(), data.data.borrow().get_data());
            // println!("this is data{:?}", data);
            node.data.borrow_mut().update_data(data);
        }
    }
}
