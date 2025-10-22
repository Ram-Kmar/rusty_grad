use crate::tensor::{Tensor, TensorHandle};
use crate::traits::TensorFloat;

pub struct HRM_Innercarry<T: TensorFloat> {
    pub z_H: TensorHandle<T>,
    pub z_L: TensorHandle<T>,
}

pub struct HRM_carry<T: TensorFloat> {
    pub inner_carry: HRM_Innercarry,
    pub steps: TensorHandle<T>,
    pub halted: TensorHandle<T>,
    pub current_data: TensorHandle<T>,
}

pub struct HRM_config {
    pub batch_size:i32 ,
    pub seq_len: i32,
    pub puzzle_emb_ndim: i32,
    pub num_puzzle_identifiers:i32,
    pub vocab_size: i32,

    H_cycles:i32,
    L_cycles:i32,

    H_layers: i32,
    L_layers: i32,

   //Transformer_config
    hidden_size: i32,
    expansion: f32,
    num_heads: i32,
    pos_encodings:String,

    rms_norm_eps: f32 = 1e-5,
    rope_theta: f32 = 10000.0,

    halt_max_steps : i32,
    halt_exploration_prob: f32,

    forward_dtype: String = "bfloat16",

}

pub struct HRM_Block {
    pub self_attn: TensorHandle<T>, // Attention
    pub mlp: TensorHandle<T>,       //mlp
    pub norm_eps: f32,
}

impl HRM_Block {
    pub fn forward(&self) -> TensorHandle<T> {
        let hidden_states = rms_norm();
        let hidden_states = rms_norm();
        hidden_states
    }
}
