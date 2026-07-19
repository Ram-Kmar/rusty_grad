use crate::math::gemm::Gemm;
use num_traits::Float;
use rand::distributions::uniform::SampleUniform;
use std::fmt::Debug;
use std::iter::Sum;

//A TensorFloat is any type that implements all of these traits.
pub trait TensorFloat: Float + Debug + SampleUniform + Sum<Self> + 'static + Default + Gemm{}

//For every type T that satisfies these bounds,automatically implement TensorFloat.
impl<T: Float + Debug + SampleUniform + Sum + 'static + Default + Gemm> TensorFloat for T {}
