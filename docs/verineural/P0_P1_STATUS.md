# P0/P1 implementation status (dev)

This file records the scope of this implementation, not a claim that VN-P01–VN-P12 are complete.

Implemented API slices:
- **P01/P03**: `MathContract::with_claim` declares independent ideal-Int `Nonnegative` and `EqualsInput` goals. Lean emits distinct theorems. An unsupported nonnegative proof fails closed; an invalid equality has an independently stated theorem that Lean must reject. `node_*_semantics` remains only definitional equality.
- **P02**: versioned `PrimitiveSpec` expressions built from existing typed operations, lowered to ordinary IR and hence supported Lean/GPU paths; opaque callbacks and new atomic math are not accepted.
- **P04**: module bodies cannot capture undeclared external nodes, return an output they do not own, or introduce hidden inputs. Nested module bindings and rollback are checked.
- **P05**: `ProofGoal` supports independently named nonnegativity and equality goals with separate checks. Global nonnegativity still uses the existing restricted module-composition proof.
- **P06**: `verify_certified` reruns Lean `#print axioms`, checks the exact theorem identities, and records every transitive axiom dependency. Only foundational `propext` is allowlisted as explicit Lean TCB; `sorryAx`, user axioms, `Classical.choice` and unreviewed axioms fail closed. Stage 5 issue/replay compare the saved checker version, axiom policy and actual dependencies. Other evidence paths still need integration.
- **P07**: atomic named multi-output module boundary, stable `output_named` lookup, and `LeanProofEngine::generate_outputs` generating one ideal-Int Lean artifact for multiple chosen outputs (including fork/fan-in consumers). Each reachable named input/output port has explicit shape obligations; supported output domain and user claims generate separate theorems. Unsupported input claims are rejected at construction instead of silently ignored. This is *not* yet universal global theorem composition across multi-output boundaries.

- **P09**: `NeuralGraph::scan_fixed` builds a finite, atomic sequence of consumer-defined `state/input → next_state/output` multi-output transitions. Shape/domain feedback conditions are checked before unrolling; later failed steps roll back all instances. The returned `FixedScan::proof_outputs` requests separate Lean obligations for every next-state/readout via `generate_outputs`. No inductive all-horizons invariant, state alias/memory proof, or general control-flow IR is claimed.

Open requirements:
- **P02/P08/P10**: genuinely new atomic tensor primitives, reductions, indexing, comparison, symbolic dimensions, generic dtypes, views/strides, and complete GPU/Lean lowering.
- **P05/P07/P09**: fully general composition/reuse of arbitrary properties, proof-checked state transitions and inductive scan; explicit recurrent IR not yet available.
- **P06**: transitive axiom audit integration into every certified artifact, pinned executable/library provenance and reproducible audit evidence.
- **P11/P12**: general nonlinear/branch/recurrent FP32 error composition, backend capability proof obligations for every optimization and full machine-code correspondence.

Do not label these incomplete areas certified. Local Rust, Lean and CUDA build/test commands were not executed by the agent per AGENTS.md.
