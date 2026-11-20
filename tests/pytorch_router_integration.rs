use magnetar_component_kimi::KimiK2Plan;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    top_k: usize,
    normalize: bool,
    logits: Vec<f64>,
    bias: Vec<f64>,
    expected: Vec<Expected>,
}

#[derive(Deserialize)]
struct Expected {
    expert: usize,
    weight: f64,
}

#[test]
fn synthetic_inputs_match_pinned_upstream_pytorch_kimi_router() {
    let fixture: Fixture =
        serde_json::from_slice(include_bytes!("fixtures/kimi-k2-router-pytorch.json")).unwrap();
    assert_eq!(fixture.cases.len(), 5);
    for case in fixture.cases {
        let mut config: serde_json::Value =
            serde_json::from_slice(include_bytes!("fixtures/kimi-k2-instruct.config.json"))
                .unwrap();
        config["n_routed_experts"] = case.logits.len().into();
        config["num_experts_per_tok"] = case.top_k.into();
        config["norm_topk_prob"] = case.normalize.into();
        let plan =
            KimiK2Plan::from_huggingface_config(&serde_json::to_vec(&config).unwrap()).unwrap();
        let mut actual = plan.route(&case.logits, &case.bias).unwrap();
        actual.sort_by_key(|item| item.expert);
        assert_eq!(actual.len(), case.expected.len(), "{}", case.name);
        for (actual, expected) in actual.iter().zip(&case.expected) {
            assert_eq!(actual.expert, expected.expert, "{}", case.name);
            assert!(
                (actual.weight - expected.weight).abs() < 1e-6,
                "{}: expert {} weight {} != {}",
                case.name,
                actual.expert,
                actual.weight,
                expected.weight
            );
        }
    }
}
