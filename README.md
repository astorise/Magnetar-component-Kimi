# magnetar-component-kimi

## Purpose

A WebAssembly Component implementing the Moonshot AI Kimi model families for the
[Magnetar](https://github.com/astorise/Magnetar) local AI Runtime. It will
supply prefill/decode graph semantics for Kimi-compatible Model Artifacts
through the Runtime's Model Component contract, without ever selecting a
Provider, Device, or receiving native Runtime handles -- per Magnetar's
architectural invariants (Components request Capabilities, not Providers
or Devices).

## Status

**Empty template.** This crate is a scaffold: no Kimi graph logic, no WIT
bindings and no Component Model exports exist here yet, and the Runtime has
no Kimi implementation of its own either. It is pinned into Magnetar as a
placeholder so the work has a home; it is not yet scheduled against an
OpenSpec change of its own.

## What this family needs

What sets this family apart from the generic decoder block
`components/qwen` and `components/llama` already implement, each to be
checked against the exact pinned checkpoint's `config.json` before it is
implemented:

- Kimi K2 follows the DeepSeek-V3 architecture: Multi-head Latent Attention (a compressed latent KV cache instead of per-head keys and values) and a Mixture-of-Experts MLP with many routed experts plus a shared expert. It shares those operator needs with `components/deepseek` (MLA, data-dependent expert routing on the Device, expert placement across Devices).
- Kimi Linear is a hybrid of linear-attention layers (Kimi Delta Attention, with a recurrent state) and MLA layers; the roadmap's MLA/KDA qualification item tracks it.
- At full scale these checkpoints exceed a single consumer GPU, so they depend on quantized weights and multi-Device expert placement (`execute-direct-quantized-weights`, `enforce-local-multigpu-and-expert-placement`).

## Governing contract

None dedicated yet. A real implementation will target the same
`model-component-graph-contract` WIT interface (`magnetar:model-component-graph`)
`components/qwen` and `components/llama` implement, extended additively
where this family needs Operators or state the contract does not model yet.
Family-specific requirements will be scoped in the main repository's
`openspec/` (the `qualify-additional-model-and-state-families` change tracks
qualifying new families).

## Relationship to magnetar-runtime

This Component will be loaded and validated by the Runtime's Component
Engine (Wasmtime-backed on native targets, browser-hosted on `wasm32`) and
never run with direct hardware, filesystem, or network access. It is
pinned into the main [Magnetar](https://github.com/astorise/Magnetar)
repository as a git submodule at `components/kimi`.
