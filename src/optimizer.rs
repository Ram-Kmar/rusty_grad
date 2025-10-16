use crate::traits::TensorFloat;

pub fn sgd<T: TensorFloat>(weights: &Vec<T>, grads: &Vec<T>, lr: T) -> Vec<T> {
    weights
        .iter()
        .zip(grads.iter())
        .map(|(&w, &g)| w - lr * g)
        .collect()
}
