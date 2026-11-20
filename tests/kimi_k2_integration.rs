use magnetar_component_kimi::KimiK2Plan;

fn published() -> serde_json::Value {
    serde_json::from_slice(include_bytes!("fixtures/kimi-k2-instruct.config.json")).unwrap()
}

fn parse(config: &serde_json::Value) -> KimiK2Plan {
    KimiK2Plan::from_huggingface_config(&serde_json::to_vec(config).unwrap()).unwrap()
}

fn logit(probability: f64) -> f64 {
    (probability / (1.0 - probability)).ln()
}

#[test]
fn published_kimi_k2_config_preserves_latent_geometry_and_layer_schedule() {
    let plan = parse(&published());
    assert_eq!(plan.attention().hidden_size, 7168);
    assert_eq!(plan.attention().attention_heads, 64);
    assert_eq!(plan.attention().query_rank, Some(1536));
    assert_eq!(plan.attention().kv_rank, 512);
    assert_eq!(plan.attention().query_key_no_rope_width, 128);
    assert_eq!(plan.attention().query_key_rope_width, 64);
    assert_eq!(plan.attention().value_width, 128);
    assert_eq!(plan.expert_layers().len(), 61);
    assert!(!plan.expert_layers()[0]);
    assert!(plan.expert_layers()[1..].iter().all(|&is_moe| is_moe));
    assert_eq!(plan.expert_count(), 384);
    assert_eq!(plan.experts_per_token(), 8);
}

#[test]
fn reference_router_uses_selection_bias_but_unbiased_sigmoid_for_weights() {
    let plan = parse(&published());
    let mut logits = vec![logit(0.01); plan.expert_count()];
    for (index, value) in logits.iter_mut().take(9).enumerate() {
        *value = logit((index + 1) as f64 / 10.0);
    }
    let mut bias = vec![0.0; plan.expert_count()];
    bias[0] = 1.0;
    let selected = plan.route(&logits, &bias).unwrap();
    assert_eq!(
        selected.iter().map(|x| x.expert).collect::<Vec<_>>(),
        vec![0, 8, 7, 6, 5, 4, 3, 2]
    );
    let expected = [0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3];
    let sum: f64 = expected.iter().sum();
    for (actual, probability) in selected.iter().zip(expected) {
        assert!((actual.weight - probability / sum * 2.827).abs() < 1e-12);
    }
}

#[test]
fn top_one_does_not_normalize_and_layer_frequency_uses_absolute_indices() {
    let mut config = published();
    config["num_experts_per_tok"] = 1.into();
    config["num_hidden_layers"] = 6.into();
    config["first_k_dense_replace"] = 2.into();
    config["moe_layer_freq"] = 2.into();
    let plan = parse(&config);
    assert_eq!(
        plan.expert_layers(),
        &[false, false, true, false, true, false]
    );
    let logits = vec![logit(0.2); plan.expert_count()];
    let mut bias = vec![0.0; plan.expert_count()];
    bias[3] = 1.0;
    let selected = plan.route(&logits, &bias).unwrap();
    assert_eq!(selected[0].expert, 3);
    assert!((selected[0].weight - 0.2 * 2.827).abs() < 1e-12);
}

#[test]
fn rejects_unsupported_groups_bad_dimensions_and_nonfinite_inputs() {
    for (field, invalid) in [
        ("model_type", serde_json::json!("deepseek_v3")),
        ("scoring_func", serde_json::json!("softmax")),
        ("topk_method", serde_json::json!("other")),
        ("n_group", serde_json::json!(2)),
        ("topk_group", serde_json::json!(0)),
        ("num_experts_per_tok", serde_json::json!(385)),
        ("moe_layer_freq", serde_json::json!(0)),
        ("qk_rope_head_dim", serde_json::json!(63)),
        ("kv_lora_rank", serde_json::json!(0)),
    ] {
        let mut config = published();
        config[field] = invalid;
        assert!(
            KimiK2Plan::from_huggingface_config(&serde_json::to_vec(&config).unwrap()).is_err(),
            "field {field}"
        );
    }
    let plan = parse(&published());
    assert!(plan.route(&[], &[]).is_err());
    assert!(
        plan.route(
            &vec![f64::NAN; plan.expert_count()],
            &vec![0.0; plan.expert_count()]
        )
        .is_err()
    );
    assert!(
        plan.route(
            &vec![0.0; plan.expert_count()],
            &vec![f64::INFINITY; plan.expert_count()]
        )
        .is_err()
    );
}
