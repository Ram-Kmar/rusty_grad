use ndarray::{Array, Array2, IxDyn};
use rusty_grad::core::device::Device;
use rusty_grad::tensor::{Tensor, TensorHandle};
use rusty_grad::assert_tensor_close;

#[test]
fn test_tensor_add() {
    let shape = vec![2, 2];
    let data_a = vec![1.0_f32, 2.0, 3.0, 4.0];
    let data_b = vec![5.0_f32, 6.0, 7.0, 8.0];
    
    // RustyGrad Tensors
    let tensor_a = Tensor::from_data(data_a.clone(), shape.clone(), false, Device::Cpu);
    let tensor_b = Tensor::from_data(data_b.clone(), shape.clone(), false, Device::Cpu);
    let tensor_c = &tensor_a + &tensor_b;
    
    // ndarray Reference
    let nd_a = Array::from_shape_vec(IxDyn(&shape), data_a).unwrap();
    let nd_b = Array::from_shape_vec(IxDyn(&shape), data_b).unwrap();
    let nd_c = nd_a + nd_b;
    
    let expected_data = nd_c.into_raw_vec_and_offset().0;
    let expected_shape = shape.clone();
    
    // We can mock an object with .shape() and .iter() for the macro
    struct Expected {
        shape: Vec<usize>,
        data: Vec<f32>,
    }
    impl Expected {
        fn shape(&self) -> &Vec<usize> { &self.shape }
        fn iter(&self) -> std::slice::Iter<'_, f32> { self.data.iter() }
    }
    
    let expected = Expected { shape: expected_shape, data: expected_data };
    
    assert_tensor_close!(tensor_c, expected);
}

#[test]
fn test_tensor_matmul() {
    let shape_a = vec![2, 3];
    let data_a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    
    let shape_b = vec![3, 2];
    let data_b = vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    
    // RustyGrad Tensors
    let tensor_a = Tensor::from_data(data_a.clone(), shape_a.clone(), false, Device::Cpu);
    let tensor_b = Tensor::from_data(data_b.clone(), shape_b.clone(), false, Device::Cpu);
    let tensor_c = TensorHandle::matmul(tensor_a.0.clone(), tensor_b.0.clone());
    
    // ndarray Reference
    let nd_a = Array2::from_shape_vec((2, 3), data_a).unwrap();
    let nd_b = Array2::from_shape_vec((3, 2), data_b).unwrap();
    let nd_c = nd_a.dot(&nd_b);
    
    let expected_data = nd_c.into_raw_vec_and_offset().0;
    let expected_shape = vec![2, 2];
    
    struct Expected {
        shape: Vec<usize>,
        data: Vec<f32>,
    }
    impl Expected {
        fn shape(&self) -> &Vec<usize> { &self.shape }
        fn iter(&self) -> std::slice::Iter<'_, f32> { self.data.iter() }
    }
    
    let expected = Expected { shape: expected_shape, data: expected_data };
    
    assert_tensor_close!(tensor_c, expected);
}

#[test]
fn test_tensor_sub() {
    // 1. Setup Data
    let shape = vec![2, 2];
    let data_a = vec![5.0_f32, 6.0, 7.0, 8.0];
    let data_b = vec![1.0_f32, 2.0, 3.0, 4.0];

    // 2. RustyGrad Operation
    let tensor_a = Tensor::from_data(data_a.clone(), shape.clone(), false, Device::Cpu);
    let tensor_b = Tensor::from_data(data_b.clone(), shape.clone(), false, Device::Cpu);
    let tensor_c = &tensor_a - &tensor_b; // Subtraction here!

    // 3. ndarray Reference Operation
    let nd_a = Array::from_shape_vec(IxDyn(&shape), data_a).unwrap();
    let nd_b = Array::from_shape_vec(IxDyn(&shape), data_b).unwrap();
    let nd_c = nd_a - nd_b; // Subtraction here!
    
    // 4. Extract expected data
    let expected_data = nd_c.into_raw_vec_and_offset().0;
    
    // 5. Mock the expected struct for the macro
    struct Expected {
        shape: Vec<usize>,
        data: Vec<f32>,
    }
    impl Expected {
        fn shape(&self) -> &Vec<usize> { &self.shape }
        fn iter(&self) -> std::slice::Iter<'_, f32> { self.data.iter() }
    }
    let expected = Expected { shape: shape.clone(), data: expected_data };
    
    // 6. Assert!
    assert_tensor_close!(tensor_c, expected);
}
