// use crate::optimizer::sgd;
use crate::cpu_backend::CpuBackprop;
use crate::shared::{Shared, new_shared};
// use crate::storage::CpuStorage; // No longer needed for derivatives
use crate::tensor::{Tensor, TensorHandle};
use crate::traits::TensorFloat;
use std::clone;
use std::collections::HashSet;

impl<T: TensorFloat> TensorHandle<T> {
    pub fn backward(&self) {
        let sorted = self.build_topological_sort();
        self.grad.as_ref().unwrap().clone().borrow_mut().fill_ones();
        for node in sorted.iter().rev() {
            // println!("number of node it passed");
            // println!("node{}", node);
            if node.is_child == true {
                match node
                    .operation
                    .as_ref()
                    .expect("panic is happening in operation unwrap")[0]
                    .as_str()
                {
                    "add" => {
                        CpuBackprop::derivate_add(
                            node.parent
                                .as_ref()
                                .expect("panic is happening in parent0 unwrap")[0]
                                .clone()
                                .grad
                                .as_ref()
                                .expect("panic is happening in parent[0] grad unwrap")
                                .clone(),
                            node.grad
                                .as_ref()
                                .expect("panic is happening in updated_grad unwrap")
                                .clone(),
                        );
                        CpuBackprop::derivate_add(
                            node.parent
                                .as_ref()
                                .expect("panic is happening in parent[1] unwrap")[1]
                                .clone()
                                .grad
                                .as_ref()
                                .expect("panic is happening in parent[1] grad unwrap")
                                .clone(),
                            node.grad
                                .as_ref()
                                .expect("panic is happening in updated_grad unwrap")
                                .clone(),
                        );
                    }
                    "matmul" => {
                        let another_parent_data = node.parent.as_ref().unwrap()[1].clone();
                        let parent = node.parent.as_ref().unwrap()[0].clone();
                        let child = node.0.clone();
                        CpuBackprop::matmul_derivate(another_parent_data, parent, child, true);
                        let another_parent_data = node.parent.as_ref().unwrap()[0].clone();
                        let parent = node.parent.as_ref().unwrap()[1].clone();
                        let child = node.0.clone();
                        CpuBackprop::matmul_derivate(another_parent_data, parent, child, false);
                    }
                    "neg" => {
                        let parent = node.parent.as_ref().unwrap()[0].clone();
                        let child = node.0.clone();
                        CpuBackprop::neg_derivative(parent, child);
                    }
                    "relu" => {
                        let parent = node.parent.as_ref().unwrap()[0].clone();
                        let child = node.0.clone();
                        CpuBackprop::relu_derivative(parent, child);
                    }
                    "square" => {
                        let parent = node.parent.as_ref().unwrap()[0].clone();
                        let child = node.0.clone();
                        CpuBackprop::square_derivative(parent, child);
                    }
                    "sqrt" => {
                        let parent = node.parent.as_ref().unwrap()[0].clone();
                        let child = node.0.clone();
                        CpuBackprop::sqrt_derivative(parent, child);
                    }
                    "tanh" => {
                        let parent = node.parent.as_ref().unwrap()[0].clone();
                        let child = node.0.clone();
                        CpuBackprop::tanh_derivative(parent, child);
                    }
                    "exp" => {
                        let parent = node.parent.as_ref().unwrap()[0].clone();
                        let child = node.0.clone();
                        CpuBackprop::exp_derivative(parent, child);
                    }
                    "sigmoid" => {
                        let parent = node.parent.as_ref().unwrap()[0].clone();
                        let child = node.0.clone();
                        CpuBackprop::sigmodi_derivative(parent, child);
                    }
                    "abs" => {
                        let parent = node.parent.as_ref().unwrap()[0].clone();
                        let child = node.0.clone();
                        CpuBackprop::abs_derivative(parent, child);
                    }
                    "mean" => {
                        let parent = node.parent.as_ref().unwrap()[0].clone();
                        let child = node.0.clone();
                        CpuBackprop::mean_derivative(parent, child);
                    }
                    "SV_mul" => {
                        let another_parent_data = node.parent.as_ref().unwrap()[0].clone();
                        let parent = node.parent.as_ref().unwrap()[0].clone();
                        let child = node.0.clone();
                        CpuBackprop::SV_mul_derivate(another_parent_data, parent, child);
                    }
                    "SV_add" => {
                        let parent = node.parent.as_ref().unwrap()[0].clone();
                        let child = node.0.clone();
                        CpuBackprop::SV_add_derivate(parent, child);
                    }
                    _ => {}
                }
            }
        }
    }
    pub fn build_topological_sort(&self) -> Vec<TensorHandle<T>> {
        let mut sorted = Vec::new();
        let mut visited = HashSet::new();

        fn visit<T: TensorFloat>(
            node: &TensorHandle<T>,
            sorted: &mut Vec<TensorHandle<T>>,
            visited: &mut HashSet<*const Tensor<T>>,
        ) {
            let node_ptr = Shared::as_ptr(&node.0);
            if visited.contains(&node_ptr) {
                return;
            }
            visited.insert(node_ptr);

            if let Some(parents) = &node.parent {
                for parent in parents {
                    visit(&TensorHandle(parent.clone()), sorted, visited);
                }
            }
            sorted.push(node.clone());
        }

        visit(self, &mut sorted, &mut visited);
        sorted
    }
}

// pub fn sub_backward<T: TensorFloat>(grad_output: &[T], _a: &Tensor<T>, _b: &Tensor<T>) -> (Vec<T>, Vec<T>) {
//     // dL/da = dL/dc * dc/da = dL/dc * 1
//     // dL/db = dL/dc * dc/db = dL/dc * -1
//     let grad_a = grad_output.to_vec();
//     let grad_b = grad_output.iter().map(|&g| -g).collect();
//     (grad_a, grad_b)
// }
//
// pub fn div_backward<T: TensorFloat>(grad_output: &[T], a: &Tensor<T>, b: &Tensor<T>) -> (Vec<T>, Vec<T>) {
//     // dL/da = dL/dc * dc/da = dL/dc * (1/b)
//     // dL/db = dL/dc * dc/db = dL/dc * (-a / b^2)
//     let grad_a = b
//         .data
//         .iter()
//         .zip(grad_output.iter())
//         .map(|(b_val, grad_val)| *grad_val / *b_val)
//         .collect();
//     let grad_b = a
//         .data
//         .iter()
//         .zip(b.data.iter())
//         .zip(grad_output.iter())
//         .map(|((a_val, b_val), grad_val)| -*a_val * *grad_val / (*b_val * *b_val))
//         .collect();
//     (grad_a, grad_b)
// }
//
// pub fn relu_backward<T: TensorFloat>(grad_output: &[T], input: &Tensor<T>) -> Vec<T> {
//     // dL/dx = dL/dy * dy/dx
//     // dy/dx = 1 if x > 0, 0 otherwise
//     input
//         .data
//         .iter()
//         .zip(grad_output.iter())
//         .map(|(x, grad)| if *x > T::from(0.0).unwrap() { *grad } else { T::from(0.0).unwrap() })
//         .collect()
// }
//
// pub fn sigmoid_backward<T: TensorFloat>(grad_output: &[T], input: &Tensor<T>) -> Vec<T> {
//     // dL/dx = dL/dy * dy/dx
//     // dy/dx = sigmoid(x) * (1 - sigmoid(x))
//     let s = input
//         .data
//         .iter()
//         .map(|&x| T::from(1.0).unwrap() / (T::from(1.0).unwrap() + (-x).exp()))
//         .collect::<Vec<T>>();
//     s.iter()
//         .zip(grad_output.iter())
//         .map(|(s_val, grad)| *s_val * (T::from(1.0).unwrap() - *s_val) * *grad)
//         .collect()
// }
//
// pub fn tanh_backward<T: TensorFloat>(grad_output: &[T], input: &Tensor<T>) -> Vec<T> {
//     // dL/dx = dL/dy * dy/dx
//     // dy/dx = 1 - tanh^2(x)
//     let t = input.data.iter().map(|&x| x.tanh()).collect::<Vec<T>>();
//     t.iter()
//         .zip(grad_output.iter())
//         .map(|(t_val, grad)| (T::from(1.0).unwrap() - *t_val * *t_val) * *grad)
//         .collect()
// }
//
// pub fn exp_backward<T: TensorFloat>(grad_output: &[T], input: &Tensor<T>) -> Vec<T> {
//     // dL/dx = dL/dy * dy/dx
//     // dy/dx = exp(x)
//     input
//         .data
//         .iter()
//         .zip(grad_output.iter())
//         .map(|(x, grad)| x.exp() * *grad)
//         .collect()
// }
//
// pub fn log_backward<T: TensorFloat>(grad_output: &[T], input: &Tensor<T>) -> Vec<T> {
//     // dL/dx = dL/dy * dy/dx
//     // dy/dx = 1/x
//     input
//         .data
//         .iter()
//         .zip(grad_output.iter())
//         .map(|(x, grad)| *grad / *x)
//         .collect()
// }
//
// pub fn neg_backward<T: TensorFloat>(grad_output: &[T], _input: &Tensor<T>) -> Vec<T> {
//     // dL/dx = dL/dy * dy/dx
//     // dy/dx = -1
//     grad_output.iter().map(|&g| -g).collect()
// }
//
// pub fn abs_backward<T: TensorFloat>(grad_output: &[T], input: &Tensor<T>) -> Vec<T> {
//     // dL/dx = dL/dy * dy/dx
//     // dy/dx = 1 if x > 0, -1 if x < 0, 0 if x = 0
//     input
//         .data
//         .iter()
//         .zip(grad_output.iter())
//         .map(|(x, grad)| {
//             if *x > T::from(0.0).unwrap() {
//                 *grad
//             } else if *x < T::from(0.0).unwrap() {
//                 -*grad
//             } else {
//                 T::from(0.0).unwrap()
//             }
//         })
//         .collect()
// }
//
// pub fn square_backward<T: TensorFloat>(grad_output: &[T], input: &Tensor<T>) -> Vec<T> {
//     // dL/dx = dL/dy * dy/dx
//     // dy/dx = 2x
//     input
//         .data
//         .iter()
//         .zip(grad_output.iter())
//         .map(|(x, grad)| T::from(2.0).unwrap() * *x * *grad)
//         .collect()
// }
//
// pub fn sqrt_backward<T: TensorFloat>(grad_output: &[T], input: &Tensor<T>) -> Vec<T> {
//     // dL/dx = dL/dy * dy/dx
//     // dy/dx = 1 / (2 * sqrt(x))
//     input
//         .data
//         .iter()
//         .zip(grad_output.iter())
//         .map(|(x, grad)| *grad / (T::from(2.0).unwrap() * x.sqrt()))
//         .collect()
// }
//
// pub fn transpose_backward<T: TensorFloat>(grad_output: &[T], input: &Tensor<T>) -> Vec<T> {
//     // The backward of a transpose is a transpose
//     let mut grad_input = vec![T::from(0.0).unwrap(); grad_output.len()];
//     let rows = input.shape[0];
//     let cols = input.shape[1];
//     for i in 0..rows {
//         for j in 0..cols {
//             grad_input[i * cols + j] = grad_output[j * rows + i];
//         }
//     }
//     grad_input
// }
//
// // pub fn mean_backward<T: TensorFloat>(grad_output: &[T], a: &Tensor<T>) -> Vec<T> {
// //     let n = T::from(a.data.len()).unwrap();
// //     let grad_a = grad_output.iter().map(|g| *g / n).collect();
// //     grad_a
// // }
//
// pub fn mean_backward<T: TensorFloat>(grad_output: &[T], a: &Tensor<T>) -> Vec<T> {
//     let n = T::from(a.data.len()).unwrap();
//     let grad_val = grad_output[0] / n;
//     vec![grad_val; a.data.len()]
// }
