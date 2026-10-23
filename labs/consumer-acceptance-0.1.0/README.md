# VeriNeural P2 consumer acceptance

This workspace package depends only on the public `verineural` crate API.
There are no custom backend rules or architecture-specific kernels in the core.

| Consumer | Structured Rust contract / goal | Lean assurance target | GPU capability |
|---|---|---|---|
| Nested fork/fan-in | Two reusable `PositiveJoin` modules; independently declared bound of left branch by the merged output | `architecture_property_nested_branch_bound` and module claims | All selected outputs lower to NVPTX; actual machine equivalence remains unproved |
| Recurrent (H=2) | Named `state`/`input`, `next_state`/`output`, required nonnegative transition claim | Bounded `scan_fixed` invariant, **not** all horizons | Core Add/ReLU supported by GPU compiler; no indefinite recurrence |
| Tensor-heavy | MatMul → externally defined `PointwiseSpec` → positive mask/select → reduction → ReLU | `architecture_property_tensor_heavy_output`; separate certified positive atomic pointwise claim | `AllowUnsupported` yields explicit unsupported status; `AllOutputs` fails |

This is an attention-like tensor-heavy fixture, **not** a softmax implementation or a provided Attention component. Every arithmetic ideal-Int proof is separate from concrete FP32 and CUDA JIT/SASS claims.

From the repository devShell:

```sh
task p2-consumers
task verify-p2-consumers
task check
task clippy
```

The first task exercises deterministic canonical IR, module boundaries, named bindings, rollback, positive/negative obligations and GPU plan statuses. The second task runs slower Lean kernel/axiom auditing, including a mathematically false claim that must be rejected. Both run `prepare` first and can format Rust sources; no automatic commit or push occurs.

A negative proof result is success **only** for a deliberately invalid specification. Returning a generated Lean source alone does not count as a certificate. The P2 suite deliberately does not claim FP32/CUDA semantic equivalence or an all-time recurrent invariant.
