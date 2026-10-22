# XOR laboratory 0.1.0

This is a separate VeriNeural consumer, not a built-in XOR primitive or a training framework.

A four-row Boolean truth table is evaluated by a fixed-weight 2 -> 2 -> 1
ReLU MLP, composed using the public NeuralModule and NeuralGraph API:

    Input [4,2]
      -> MatMul([2,2]) -> BiasAdd([2]) -> ReLU
      -> MatMul([2,1]) -> BiasAdd([1]) -> ReLU
      -> Output [4,1]

For each input row, the hidden pre-activations are (x+y, x+y-1).
The output is ReLU(ReLU(x+y) - 2*ReLU(x+y-1)), which equals Boolean XOR
for x,y in {0,1}. All concrete inputs, weights and intermediates are tiny
integers exactly representable in binary32.

## Run (from the workspace root, inside nix develop)

    task test-lab-xor
    task lab-xor
    task verify-lab-xor

The CPU-only tests check composition, static proof/GPU plan generation and
rejection of malformed shape and nonfinite parameters. The run command
uses VeriNeural's auto compiler to generate a Lean architecture proof and
independent nonnegativity goal, audit axioms, lower to LLVM NVPTX/PTX, then
execute the same graph using NVIDIA CUDA Graph and compare all four FP32
outputs bitwise with an independent Boolean XOR oracle. It also exports
the Semantic IR plus the DOT and PNG diagram.

Artifacts are generated under target/verineural/xor-lab. The offline verify
command checks the persisted IR/diagram, Lean/axiom and NVPTX translation
bundle, explicit input/parameter/output bits and the SHA-256 manifests.
It does not execute CUDA again. Pin the printed manifest hash separately
if provenance across machines matters.

## Guarantee boundary

Lean proves only the requested ideal-Int elementwise nonnegativity theorem,
not the complete XOR truth table. Four concrete truth-table cases are
independently checked against NVIDIA GPU output. CUDA driver, JIT, SASS,
strict machine FP32 equivalence and training/convergence are not proven.
There is no optimizer, backpropagation or weight training in this lab.
