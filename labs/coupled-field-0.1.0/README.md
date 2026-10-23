# CoupledField 0.1.0

Independent Stage 9 VeriNeural consumer. It is not a built-in network,
training pipeline, or verified CUDA/SASS implementation.

## Model and API

The architecture is declared in src/lib.rs with the public ModelBuilder API:

- Three named nonnegative input fields: x, y, z, each [1,4].
- Two named nonnegative, shared parameter fields: p, q, each [4].
- Seven nested semantic modules (four Boundary responses, two Coupled units,
  one CoupledField) using graph operations mul, bias_add, add and relu.
- Three model outputs: final, left and right. They refer to existing graph
  nodes; output registration adds no computation. Only final is executed by
  the legacy end-to-end Stage 9 runner.
- One explicit relational proof obligation, coupled_branch_lower_bound,
  relating left and final. It is not inferred solely from ReLU of final.

The branch outputs are consumed by Model::plan for shared multi-output NVPTX
planning. The runnable GPU evidence path still exercises the single final
output with baseline/fused CUDA Graph, avoiding any silent expansion of the
execution guarantees.

## Local acceptance (Nix devShell, workspace root)

    git pull --ff-only
    task check
    task test
    task clippy
    task lab-stage9
    task verify-lab-stage9

The lab runs the same independent checked i64 oracle for the concrete
[4,10,8,12] values, kernel-checks both architecture_global_nonnegative
and architecture_property_coupled_branch_lower_bound via Lean, checks
baseline/fused CUDA Graph results, validates LLVM/PTX translation evidence,
and writes an 18-file integrity manifest.

The structural contracts are now inferred instead of handwritten. Therefore
all earlier generated Semantic IR, DOT, PNG, Lean, PTX and SHA256SUMS files
must be rebuilt by task lab-stage9 and validated by task verify-lab-stage9.
Do not update the manifest or any proof artifact manually.

Lean establishes properties of the ideal integer semantics under declared
source assumptions. GPU bitwise agreement is a finite concrete probe.
CUDA JIT/SASS and general FP32 equivalence are not formally established.
