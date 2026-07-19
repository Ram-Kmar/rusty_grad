use crate::backends::cpu;
// use crate::core::storage::{CpuStorage, Storage};
use crate::{tensor::TensorHandle, core::traits::TensorFloat};
use crate::backends::cpu::binary_ops;

pub struct SGD<T: TensorFloat> {
    pub learning_rate: T,
}

impl<T: TensorFloat> SGD<T> {
    pub fn sgd(weights: &Vec<T>, grads: &Vec<T>, lr: T) -> Vec<T> {
        weights
            .iter()
            .zip(grads.iter())
            .map(|(&w, &g)| w - lr * g)
            .collect()
    }

    pub fn update(&self, initial_point: &TensorHandle<T>) {
        let sorted = initial_point.build_topological_sort();
        for node in sorted.iter().rev() {
            // println!("this is first element{:?}", node.data.borrow().get_data());
            let lr = self.learning_rate;
            let data = TensorHandle::SV_mul(node.clone(), lr, "grad".to_string());
            let data =
                crate::backends::cpu::binary_ops::sub(node.data.borrow().get_data(), data.data.borrow().get_data());
            // println!("this is data{:?}", data);
            node.data.borrow_mut().update_data(data);
        }
    }
}
