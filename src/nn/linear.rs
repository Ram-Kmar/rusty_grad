// use crate::nn::module::Module;
use crate::core::device::Device;
use crate::core::shared::new_shared;
use crate::tensor::{TensorData, Tensor};
use crate::core::traits::TensorFloat;

/// Represents a linear transformation layer.
#[derive(Debug)]
pub struct Linear<T: TensorFloat> {
    /// The number of input features for the layer.
    pub input_size: usize,
    /// The number of output features for the layer.
    pub output_size: usize,
    /// A boolean flag that determines whether the layer includes a bias term.
    pub bais_require: bool,
    /// The learnable weights of the layer, stored as a `TensorData`.
    pub weights: Tensor<T>,
    /// The gradient of the loss with respect to the `weights` tensor.
    pub weights_grad: Tensor<T>,
    /// An optional `TensorData` representing the bias term. It is `None` if `bais_require` is `false`.
    pub bais: Option<Tensor<T>>,
    /// The gradient of the loss with respect to the `bais` tensor.
    pub bais_grad: Tensor<T>,
}

// impl<T: TensorFloat> Default for Linear<T> {
//     fn default() -> Self {
//         Self {
//             input_size: 0,
//             output_size: 0,
//             bais_require: false,
//             weights: Tensor::(),
//             weights_grad: Tensor::new(),
//             bais: None,
//             bais_grad: Tensor::new(),
//         }
//     }
// }

impl<T: TensorFloat> Linear<T> {
    /// Creates a new `Linear` layer.
    ///
    /// # Arguments
    ///
    /// * `input_size` - The number of input features.
    /// * `output_size` - The number of output features.
    /// * `bais_require` - Whether to include a bias term.
    pub fn new(input_size: usize, output_size: usize, bais_require: bool, device: Device) -> Self {
        let shape = vec![input_size, output_size];
        if bais_require == true {
            Self {
                input_size: input_size,
                output_size: output_size,
                bais_require: true,
                weights: Tensor::new(shape, true, device),
                weights_grad: Tensor::new(shape, true, device),
                bais: Some(Tensor::new(shape, true, device)),
                bais_require: Tensor::new(shape, true, device),
            }
        } else {
            Self {
                input_size: input_size,
                output_size: output_size,
                bais_require: false,
                weights: Tensor::new(shape.clone(), true, "Linear".to_string()),
                bais: None,
                ..Self::default()
            }
        }
    }
}
// impl<T: TensorFloat> Module<T> for Linear<T> {
//     /// Performs the forward pass of the linear layer.
//     ///
//     /// # Arguments
//     ///
//     /// * `input` - The input tensor.
//     fn forward(&self, input: Tensor<T>) -> Tensor<T> {
//         let mut output = input * self.weights.clone();
//         if self.bais_require {
//             output = output + self.bais.as_ref().unwrap().clone();
//         }
//         output
//     }
//
//     /// Performs the backward pass of the linear layer.
//     ///
//     /// # Arguments
//     ///
//     /// * `previous_grad` - The gradient from the next layer.
//     fn backward(&mut self, previous_grad: Tensor<T>) -> Tensor<T> {
//         Tensor(new_shared(TensorData::init_zeros(
//             self.weights.shape.clone(),
//             false,
//         )))
//     }
// }
