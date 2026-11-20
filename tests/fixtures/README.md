# Kimi K2 checkpoint configuration

`kimi-k2-instruct.config.json` is the unmodified configuration from
[`moonshotai/Kimi-K2-Instruct`](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/ab29892994e1d5398c0016a995123972df193461/config.json)
at revision `ab29892994e1d5398c0016a995123972df193461`, SHA-256
`8c13ae1049df55f29b3bdcae69a562433f243ff70dac251d819ecad8dbdf7439`.
No weights are included. The upstream checkpoint uses a Modified MIT License;
the upstream repository's license is reproduced in `LICENSE-KIMI` and model
consumers must review its terms independently.

`kimi-k2-router-pytorch.json` contains synthetic inputs and outputs generated
with PyTorch 2.15.0 from the upstream `MoEGate` class in
[`modeling_deepseek.py`](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/ab29892994e1d5398c0016a995123972df193461/modeling_deepseek.py)
at that revision, SHA-256
`ce753fc9076ce13ce9557e680e2f5bd68ae8929a8225772968a9a22a1d17b7d4`.
`generate_router_reference.py` verifies the source hash and executes that
class with an identity projection and synthetic logits and biases. Cases cover
the published 384-expert shape, biased selection, normalization, top-1, and
the small-score normalization epsilon. No checkpoint weights or full-model
inference are involved. Rust computes in f64 while the reference uses f32;
the comparison allows a 1e-6 absolute difference and excludes score ties.
