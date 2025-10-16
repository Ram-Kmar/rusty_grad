use crate::gemm::Gemm;
use num_traits::Float;
use rand::distributions::uniform::SampleUniform;
use std::fmt::Debug;
use std::iter::Sum;

pub trait TensorFloat:
    Float + Debug + SampleUniform + Sum<Self> + 'static + Default + Gemm
{
}

impl<T: Float + Debug + SampleUniform + Sum + 'static + Default + Gemm> TensorFloat for T {}
