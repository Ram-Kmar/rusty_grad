use crate::tensor::Tensor;
use crate::core::traits::TensorFloat;

pub trait Loss<T: TensorFloat> {
    fn forward(&self, prediction: &Tensor<T>, target: &Tensor<T>) -> Tensor<T>;
}

pub struct CrossEntropyLoss;

impl<T: TensorFloat> Loss<T> for CrossEntropyLoss {
    fn forward(&self, prediction: &Tensor<T>, target: &Tensor<T>) -> Tensor<T> {
        // Softmax + NLLLoss
        let log_softmax = prediction.log_softmax();
        let nll_loss = log_softmax.nll_loss(target);
        nll_loss
    }
}

impl<T: TensorFloat> Tensor<T> {
    pub fn log_softmax(&self) -> Tensor<T> {
        let max = self.data.iter().fold(T::neg_infinity(), |a, &b| a.max(b));
        let exps = self.exp();
        let sum_exps: T = exps.data.iter().copied().sum();
        let log_sum_exps = sum_exps.ln();
        self.try_sub( &Tensor::from_data(vec![log_sum_exps; self.data.len()], self.shape.clone(), false)).unwrap()
    }

    pub fn nll_loss(&self, target: &Tensor<T>) -> Tensor<T> {
        let mut loss = T::zero();
        for i in 0..self.shape[0] {
            loss = loss - self.data[i * self.shape[1] + target.data[i].to_usize().unwrap()];
        }
        Tensor::from_data(vec![loss / T::from(self.shape[0]).unwrap()], vec![1], true)
    }
}
