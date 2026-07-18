# Upstream parity and production contracts

This document is normative for deciding whether a change is required for upstream compatibility and whether an intentional JAX extension is production-ready. A review claim, benchmark label, or historical implementation detail does not override these contracts.

## What upstream parity means

Released-source parity covers the model behavior that gives released weights and configurations their meaning:

- parameter names, shapes, tying, bias topology, plus-one normalization storage, and conversion semantics;
- released coefficient equations, initialization, residual structure, attention coordinates, masking, and default numerical boundaries;
- logits, gradients, and short optimizer trajectories within the tolerances and evidence limits documented in [JAX and original PyTorch interoperability](jax-torch.md#parity-gates).

The released source is authoritative for released checkpoint semantics when its implementation is more specific than the paper. The paper remains the architectural reference, and intentional JAX extensions have their own documented contracts and regression gates.

Upstream parity does not require reproducing a CUDA kernel's internal decomposition, importing the original training stack, or preserving historical distributed-runtime choices. An implementation may use FFTs, scans, XLA collectives, rematerialization, or another JAX-native decomposition when it preserves the required mathematics and performs appropriately on the intended workload.

## Distributed execution boundary

The released training system's FairScale/FSDP1 and model-parallel plumbing are historical implementation evidence, not a runtime parity target for this JAX library. JAX arrays, explicit shardings, and current compiler collectives are the relevant implementation surface. Future distributed work should retain the useful principles of explicit parameter, optimizer-state, activation, and batch sharding, but it does not promise literal compatibility with PyTorch FSDP1 or FSDP2 APIs, layouts, checkpoint directories, or execution order.

Original model-parallel checkpoint shards remain a conversion concern: consolidation must reconstruct the released world-size-one tensor meanings exactly. That does not require recreating the original distributed trainer.

## Packed training is a required production capability

Packed execution is supported, not experimental. A batch row may contain multiple unrelated documents, and no information from one contiguous document run may affect another. The isolation contract covers every stateful or contextual path:

- TimestepNorm statistics reset at each run boundary;
- CEMA recurrence state resets at each run boundary;
- attention is block-diagonal by contiguous run and uses run-local chunk boundaries;
- default RoPE positions restart for each run;
- padding (`segment_ids == 0`) contributes neither state nor loss;
- language-model targets crossing a run boundary are excluded from the loss.

Positive `segment_ids` identify valid tokens. Boundaries are defined by contiguous runs, so a numeric ID may be reused later in the same row without reconnecting the two runs. `MegalodonModel.supports_segment_reset` and `MegalodonForCausalLM.supports_segment_reset` advertise this complete model-level contract to downstream packers (for example, input pipelines built on Grain, JAX's data-loading library). Packed metadata is a training-time contract and is intentionally incompatible with an incoming inference cache.

Correct outputs are necessary but not sufficient. The default packed CEMA path must provide useful accelerator throughput for representative downstream training; a sequential reference or fallback that preserves isolation but makes packing impractical does not complete the feature. The current production default is the parallel associative affine scan because it preserves exact run resets and has outperformed the alternatives tested at the downstream envelope. The sequential scan remains an explicit fallback and an independent implementation for correctness comparisons, not the preferred production path.

The associative path's live tensors scale with `sequence_length * batch_size * model_dim * cema_ndim`, so memory must be measured rather than hand-waved. A static byte estimate is useful diagnostic context but is not by itself an execution verdict: XLA fusion, autodiff, rematerialization, sharding, model activations, and allocator behavior determine the compiled program's actual peak. Conversely, describing the sequential forward carry as O(1) extra memory does not imply O(1) compiled backward temporary memory.

Changes to packed CEMA must pass both kinds of evidence:

1. Isolation regressions compare packed documents against independent execution, including repeated numeric IDs, padding, gradients, and both associative and sequential implementations.
2. Accelerator benchmarks cover full forward/backward at the downstream envelope of `model_dim=1024`, `sequence_length=2048`, and at least `(batch_size=8, cema_ndim=16)` and `(batch_size=4, cema_ndim=32)`, plus the canonical full-model packed matrix. Reports include compile time, synchronized runtime, compiler temporary memory, peak device memory when available, and finiteness/correctness gates.

A blockwise or hierarchical recurrence remains a valid future optimization if measured evidence shows a useful memory-throughput tradeoff. It is not required merely because it resembles a proposed decomposition, and it must not replace the production path when it makes the intended workload materially slower without resolving an observed constraint.

## Precision policies

Full FP32 parameter storage and computation is the reference policy. FP32 parameter storage with BF16 compute is the higher-update-fidelity mixed policy. Compact BF16 storage for ordinary embedding and projection parameters is also an intentional supported JAX policy; CEMA, normalization, Q/K affine, residual-scale parameters, accumulation, and default softmax paths remain FP32.

Compact BF16 storage means ordinary parameters and their applied updates are quantized to BF16. It does not claim or synthesize hidden FP32 master weights. Users who require FP32 master-parameter update fidelity select `param_dtype=jnp.float32` with `compute_dtype=jnp.bfloat16`. Supporting both policies is deliberate and is not contradicted by the released trainer having used FP32 master shards with BF16 compute.

Native checkpoints are unambiguous across these policies: the configuration fingerprint includes every dtype choice, the dtype policy identifier describes the expected tree, and the tensor manifest validates each stored leaf. Original-upstream export may cast ordinary transport tensors explicitly while retaining precision-sensitive values in FP32.

## Benchmark claims

The canonical configuration in `benchmarks/benchmark_model_paths.py` is a stable reduced-topology cross-revision matrix: model dimension 1024, 12 layers, one attention head, vocabulary 16,000, and tied output. Its default training matrix exercises batches 1/2/4 and lengths 2,048/4,096, including packed forward and forward/backward paths. It is not the paper-7B topology and does not establish that a complete 7B model trains on one device.

Claims about paper-7B feasibility must use the paper topology under an explicit, reported sharding and optimizer-state policy. A per-layer `D=4096, N=16, L=4096` CEMA benchmark is useful component evidence, but neither that component run nor the reduced canonical model alone proves end-to-end 7B feasibility.

## Review rule

Evaluate the current implementation, tests, documentation, and measured target workload before changing a compatibility boundary. Treat stale prose as documentation to fix, not as authority to remove a supported capability. Distinguish mathematical parity, checkpoint compatibility, optional policy, performance limitation, and distributed-system scope; they require different evidence and must not be collapsed into one demand for literal implementation parity.
