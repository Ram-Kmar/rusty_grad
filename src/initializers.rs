use num_traits::Float;
use rand::distributions::{uniform::SampleUniform, Distribution, Uniform};
// use rand::thread_rng;
use rand::{rngs::StdRng,SeedableRng};

/// Initializes a mutable slice with values drawn from a Uniform distribution.
///
/// Values are sampled from the range [low, high).
///
/// # Arguments
/// * `data` - A mutable slice of T to be filled.
/// * `low` - The lower bound of the uniform distribution (inclusive).
/// * `high` - The upper bound of the uniform distribution (exclusive).
pub fn uniform_<T: Float + SampleUniform>(size: usize, low: f32, high: f32) -> Vec<T> {
    let mut rng = StdRng::seed_from_u64(42);
    let uniform_dist = Uniform::new(low, high);
    (0..size)
        .map(|_| T::from(uniform_dist.sample(&mut rng)).expect("program crash here"))
        .collect()
}
