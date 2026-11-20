//! Moonshot AI Kimi Model Component for the Magnetar local AI Runtime.
//!
//! Reads the Kimi K2 model's MLA/MLP configuration and implements a portable
//! reference for its single-group expert routing. It does not yet build graphs,
//! load weights, execute inference, or export a Component Model interface.

/// The architecture family this Component will serve.
pub const ARCHITECTURE_FAMILY: &str = "kimi";

use serde::Deserialize;
use std::error::Error;
use std::fmt;

const MAX_LAYERS: usize = 4096;
const MAX_EXPERTS: usize = 65_536;

/// The dimensions of Kimi K2's compressed multi-head latent attention.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LatentGeometry {
    pub hidden_size: usize,
    pub attention_heads: usize,
    pub query_rank: Option<usize>,
    pub kv_rank: usize,
    pub query_key_no_rope_width: usize,
    pub query_key_rope_width: usize,
    pub value_width: usize,
}

/// Routing and layer information from a Kimi K2 configuration. Successful
/// parsing alone does not mean that Magnetar can execute the checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct KimiK2Plan {
    attention: LatentGeometry,
    expert_layers: Vec<bool>,
    experts: usize,
    top_k: usize,
    normalize_top_k: bool,
    routed_scaling_factor: f64,
}

#[derive(Deserialize)]
struct RawConfig {
    model_type: String,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    q_lora_rank: Option<usize>,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    first_k_dense_replace: usize,
    moe_layer_freq: usize,
    n_routed_experts: usize,
    num_experts_per_tok: usize,
    n_group: usize,
    topk_group: usize,
    topk_method: String,
    scoring_func: String,
    norm_topk_prob: bool,
    routed_scaling_factor: f64,
}

impl KimiK2Plan {
    /// Latent attention dimensions from the checkpoint.
    pub fn attention(&self) -> &LatentGeometry {
        &self.attention
    }

    /// Whether each model layer has routed experts, in layer order.
    pub fn expert_layers(&self) -> &[bool] {
        &self.expert_layers
    }

    /// The number of logits and correction biases required by the router.
    pub fn expert_count(&self) -> usize {
        self.experts
    }

    /// The number of experts selected per token.
    pub fn experts_per_token(&self) -> usize {
        self.top_k
    }

    /// Parse the explicitly supported single-group Kimi K2 checkpoint profile.
    /// Unrecognized scoring/group variants fail rather than being approximated.
    pub fn from_huggingface_config(bytes: &[u8]) -> Result<Self, KimiError> {
        let raw: RawConfig = serde_json::from_slice(bytes)
            .map_err(|error| KimiError(format!("invalid Kimi K2 config: {error}")))?;
        if raw.model_type != "kimi_k2"
            || raw.scoring_func != "sigmoid"
            || raw.topk_method != "noaux_tc"
            || raw.n_group != 1
            || raw.topk_group != 1
        {
            return Err(KimiError(
                "only single-group Kimi K2 sigmoid/noaux_tc routing is supported".into(),
            ));
        }
        if !(1..=MAX_LAYERS).contains(&raw.num_hidden_layers)
            || raw.first_k_dense_replace > raw.num_hidden_layers
            || raw.moe_layer_freq == 0
            || raw.hidden_size == 0
            || raw.num_attention_heads == 0
            || raw.q_lora_rank == Some(0)
            || raw.kv_lora_rank == 0
            || raw.qk_nope_head_dim == 0
            || raw.qk_rope_head_dim == 0
            || !raw.qk_rope_head_dim.is_multiple_of(2)
            || raw.v_head_dim == 0
        {
            return Err(KimiError(
                "invalid layer or latent attention dimensions".into(),
            ));
        }
        if !(2..=MAX_EXPERTS).contains(&raw.n_routed_experts)
            || raw.num_experts_per_tok == 0
            || raw.num_experts_per_tok > raw.n_routed_experts
            || !raw.routed_scaling_factor.is_finite()
            || raw.routed_scaling_factor <= 0.0
        {
            return Err(KimiError(
                "invalid expert routing dimensions or scale".into(),
            ));
        }
        let expert_layers = (0..raw.num_hidden_layers)
            .map(|index| {
                index >= raw.first_k_dense_replace && index.is_multiple_of(raw.moe_layer_freq)
            })
            .collect();
        Ok(Self {
            attention: LatentGeometry {
                hidden_size: raw.hidden_size,
                attention_heads: raw.num_attention_heads,
                query_rank: raw.q_lora_rank,
                kv_rank: raw.kv_lora_rank,
                query_key_no_rope_width: raw.qk_nope_head_dim,
                query_key_rope_width: raw.qk_rope_head_dim,
                value_width: raw.v_head_dim,
            },
            expert_layers,
            experts: raw.n_routed_experts,
            top_k: raw.num_experts_per_tok,
            normalize_top_k: raw.norm_topk_prob,
            routed_scaling_factor: raw.routed_scaling_factor,
        })
    }

    /// Compute one token's Kimi K2 expert-selection reference from router
    /// logits and correction biases. Weights use the original sigmoid scores,
    /// not the selection-biased scores. Ties use ascending expert index; the
    /// upstream PyTorch `topk` makes no guarantee about equal-score ordering.
    pub fn route(&self, logits: &[f64], bias: &[f64]) -> Result<Vec<ExpertWeight>, KimiError> {
        if logits.len() != self.experts || bias.len() != self.experts {
            return Err(KimiError("expected one logit and bias per expert".into()));
        }
        if logits.iter().chain(bias).any(|value| !value.is_finite()) {
            return Err(KimiError("router logits and biases must be finite".into()));
        }
        let scores: Vec<_> = logits.iter().map(|&value| sigmoid(value)).collect();
        let mut choices: Vec<_> = scores
            .iter()
            .zip(bias)
            .enumerate()
            .map(|(index, (score, correction))| (index, score + correction))
            .collect();
        choices.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut selected: Vec<_> = choices
            .into_iter()
            .take(self.top_k)
            .map(|(expert, _)| ExpertWeight {
                expert,
                weight: scores[expert],
            })
            .collect();
        // Kimi normalizes only when more than one expert is selected; this
        // differs from the GLM-4.7 implementation for top-1.
        if self.top_k > 1 && self.normalize_top_k {
            let denominator = selected.iter().map(|item| item.weight).sum::<f64>() + 1e-20;
            for item in &mut selected {
                item.weight /= denominator;
            }
        }
        for item in &mut selected {
            item.weight *= self.routed_scaling_factor;
        }
        Ok(selected)
    }
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

/// An expert index and its corresponding unrounded weight.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpertWeight {
    pub expert: usize,
    pub weight: f64,
}

/// A malformed or unsupported Kimi K2 configuration or routing request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KimiError(String);

impl fmt::Display for KimiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for KimiError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_its_architecture_family() {
        assert_eq!(ARCHITECTURE_FAMILY, "kimi");
    }
}
