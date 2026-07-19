use crate::nn::linear::Linear;
use crate::nn::module::Module;
use std::marker::PhantomData;
use crate::nn::embedding::Embedding;
use std::rc::Rc;
use crate::core::traits::TensorFloat;
use crate::tensor::TensorHandle;

// Define the mode enum with no associated data
#[derive(Debug)]
pub enum GradMode {
    Tape,
    Precomputed,
}

// Provide a default mode (PyTorch style)
impl Default for GradMode {
    fn default() -> Self {
        GradMode::Tape
    }
}

pub struct Model<T: TensorFloat + 'static> {
    _marker: PhantomData<T>,
    // pub graph: Vec<String>,
    // pub layers: Vec<Rc<dyn Module<T>>>,
    pub grad_mode: GradMode, // ✅ fixed
}

impl<T: TensorFloat + 'static> Model<T> {

    pub fn new() -> Self {
        Self{
            // graph: Vec::new(),
            // layers: Vec::new(),
            grad_mode: GradMode::default(),
        }
    }

    pub fn new_with_gradmode(grad_mode:GradMode)-> Self {

        match grad_mode {
            GradMode::Precomputed => {
                Self {
                    // graph: Vec::new(),
                    // layers: Vec::new(),
                    grad_mode: GradMode::Precomputed,
                }
            }
            GradMode::Tape => {
                Self {
                    // graph: Vec::new(),
                    // layers: Vec::new(),
                    grad_mode: GradMode::Tape,
                }
            }
        }
    }

    pub fn linear<T>(&self, input_size: usize, output_size: usize, bias_require: bool) -> <Linear<T>{
        // self.graph.push("Linear".to_string());
        // let layer = Rc::new(Linear::new(input_size, output_size, bias_required));
        // self.layers.push(layer.clone() as Rc<dyn Module<T>>);
        // layer
        Linear::<T>::new(input_size,output_size,bais_require)

    }

    pub fn print_layers(&self) {
        for name in &self.graph {
            println!("This is the layer: {:?}", name);
        }
    }
    pub fn embedding(&self, rows: usize, d_size: usize) -> Embedding<T> {
        // self.graph.push("Embedding".to_string());
        // let layer = Rc::new(Embedding::new(rows,d_size));
        // self.layers.push(layer.clone() as Rc<dyn Module<T>>);
        // layer
        Embedding::new(row,colums)
    }
    pub fn backward(&self, loss: &TensorHandle<T>) {
        loss.backward();
    }
}

// impl<T: Float> Module<T> for Model<T> {
//     fn forward(&mut self,input: Tensor) -> Tensor {
//         for layer in self.layers.iter_mut() {
//             input = layer.forward(input);
//         }
//         input
//     }
//     // fn model.train(){
//     //     for layer in self.layers.iter_mut(){

//     //     }

//     // }
// }ers.iter_mut(){

//     //     }

//     // }
// }
