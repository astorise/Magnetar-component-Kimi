"""Generate synthetic router cases with the pinned Kimi K2 upstream PyTorch code.

Usage: python generate_router_reference.py /path/to/modeling_deepseek.py
The input must be the checkpoint's source at the documented revision and hash.
An identity projection supplies synthetic logits; no checkpoint weights are used.
"""

import ast
import hashlib
import json
import math
import pathlib
import sys
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

SOURCE_SHA256 = "ce753fc9076ce13ce9557e680e2f5bd68ae8929a8225772968a9a22a1d17b7d4"
source = pathlib.Path(sys.argv[1]).read_bytes()
assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256, "unexpected upstream source"
tree = ast.parse(source)
router = next(node for node in tree.body if isinstance(node, ast.ClassDef)
              and node.name == "MoEGate")
namespace = {"torch": torch, "nn": nn, "F": F, "math": math}
exec(compile(ast.Module(body=[router], type_ignores=[]), "verified-kimi-router", "exec"), namespace)
Router = namespace["MoEGate"]


def make_case(name, logits, bias, top_k, normalize):
    logits = torch.tensor(logits, dtype=torch.float32)
    bias = torch.tensor(bias, dtype=torch.float32)
    count = logits.numel()
    config = SimpleNamespace(num_experts_per_tok=top_k, n_routed_experts=count,
        routed_scaling_factor=2.827, scoring_func="sigmoid", seq_aux=False,
        topk_method="noaux_tc", n_group=1, topk_group=1, norm_topk_prob=normalize,
        hidden_size=count)
    model = Router(config).eval()
    with torch.no_grad():
        model.weight.copy_(torch.eye(count))
        model.e_score_correction_bias.copy_(bias)
        indices, weights = model(logits.reshape(1, 1, -1))
    pairs = sorted(zip(indices[0].tolist(), weights[0].tolist()))
    return {"name": name, "top_k": top_k, "normalize": normalize,
            "logits": logits.tolist(), "bias": bias.tolist(),
            "expected": [{"expert": expert, "weight": weight} for expert, weight in pairs]}


cases = [
    make_case("published_dimensions", [((i * 37) % 389 - 194) / 25 for i in range(384)],
              [((i * 11) % 29 - 14) / 100 for i in range(384)], 8, True),
    make_case("bias_changes_selection", [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
              [1.4, 1.1, 0.8, 0.4, 0.1, 0.0, -0.1, -0.2], 3, True),
    make_case("top_one_not_normalized", [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
              [1.4, 1.1, 0.8, 0.4, 0.1, 0.0, -0.1, -0.2], 1, True),
    make_case("multiple_not_normalized", [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
              [1.4, 1.1, 0.8, 0.4, 0.1, 0.0, -0.1, -0.2], 3, False),
    make_case("tiny_scores_epsilon", [-50.0, -51.0, -52.0, -53.0, -54.0, -55.0, -56.0, -57.0],
              [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], 2, True),
]
path = pathlib.Path(__file__).with_name("kimi-k2-router-pytorch.json")
path.write_text(json.dumps({"source_sha256": SOURCE_SHA256, "cases": cases}, indent=2) + "\n")
