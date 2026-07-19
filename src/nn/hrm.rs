use crate::tensor::{TensorData, Tensor};
use crate::core::traits::TensorFloat;

pub struct HRM_Innercarry<T: TensorFloat> {
    pub z_H: Tensor<T>,
    pub z_L: Tensor<T>,
}

pub struct HRM_carry<T: TensorFloat> {
    pub inner_carry: HRM_Innercarry<T>,
    pub steps: Tensor<T>,
    pub halted: Tensor<T>,
    pub current_data: Tensor<T>,
}

pub struct HRM_config {
    pub batch_size: i32,
    pub seq_len: i32,
    pub puzzle_emb_ndim: i32,
    pub num_puzzle_identifiers: i32,
    pub vocab_size: i32,

    pub H_cycles: i32,
    pub L_cycles: i32,

    pub H_layers: i32,
    pub L_layers: i32,

   //Transformer_config
    pub hidden_size: i32,
    pub expansion: f32,
    pub num_heads: i32,
    pub pos_encodings: String,

    pub rms_norm_eps: f32,
    pub rope_theta: f32,

    pub halt_max_steps: i32,
    pub halt_exploration_prob: f32,

    pub forward_dtype: String,
}

pub struct HRM_Block<T: TensorFloat> {
    pub self_attn: Tensor<T>, // Attention
    pub mlp: Tensor<T>,       //mlp
    pub norm_eps: f32,
}

impl<T: TensorFloat> HRM_Block<T> {
    pub fn forward(&self) -> Tensor<T> {
        todo!()
    }
}
