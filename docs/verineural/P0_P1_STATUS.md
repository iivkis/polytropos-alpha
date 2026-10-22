# P0/P1 implementation status (dev)

This file records the scope of this implementation, not a claim that VN-P01–VN-P12 are complete.

Implemented API slices:
- **P01/P03**: `MathContract::with_claim` declares independent ideal-Int `Nonnegative` and `EqualsInput` goals. Lean emits distinct theorems. An unsupported nonnegative proof fails closed; an invalid equality has an independently stated theorem that Lean must reject. `node_*_semantics` remains only definitional equality.
- **P02**: versioned `PrimitiveSpec` expressions built from existing typed operations, lowered to ordinary IR and hence supported Lean/GPU paths; opaque callbacks and new atomic math are not accepted.
- **P04**: module bodies cannot capture undeclared external nodes, return an output they do not own, or introduce hidden inputs. Nested module bindings and rollback are checked.
- **P05**: `ProofGoal` supports independently named nonnegativity/equality. Ideal-Int `Nonnegative` module and multi-port claims now use recursive proofs from graph operation lemmas and explicit source hypotheses even when the separate FP32 domain is `Any`; unsupported source assumptions fail closed. `generate_fixed_scan_nonnegative` composes checked local claims into a distinct theorem about every state over a requested finite horizon. Generic arbitrary predicate/theorem-definition reuse is still open.
- **P06**: `verify_certified` reruns Lean `#print axioms`, checks the exact theorem identities, and records every transitive axiom dependency. Only foundational `propext` is allowlisted as explicit Lean TCB; `sorryAx`, user axioms, `Classical.choice` and unreviewed axioms fail closed. Stage 5 issue/replay compare the saved checker version, axiom policy and actual dependencies. Other evidence paths still need integration.
- **P07**: atomic named multi-output module boundary, stable `output_named` lookup, and `LeanProofEngine::generate_outputs` generating one ideal-Int Lean artifact for multiple chosen outputs (including fork/fan-in consumers). Each reachable named input/output port has explicit shape obligations; supported output domain and user claims generate separate theorems. Unsupported input claims are rejected at construction instead of silently ignored. This is *not* yet universal global theorem composition across multi-output boundaries.

- **P09**: `scan_fixed` builds atomic consumer-defined state/input → next_state/output transitions. `generate_fixed_scan_nonnegative` checks the actual ordered multi-module bindings, requires a consumer-owned `Nonnegative` claim for each next-state, proves the initial state under explicit source assumptions, and generates a separate global Lean theorem that *all* states from t=0 through the concrete horizon are nonnegative. The `task proof-scan-invariant` consumer example asks Lean to verify the local/global theorems and audits axioms. This does not prove arbitrary-horizon induction, IEEE binary32 finiteness, or machine execution.

Open requirements:
- **P02/P08/P10**: genuinely new atomic tensor primitives, reductions, indexing, comparison, symbolic dimensions, generic dtypes, views/strides, and complete GPU/Lean lowering.
- **P05/P07/P09**: general user predicate DSL, independently cached module-definition proofs and reusable theorem dependencies, full state/control IR and arbitrary-horizon inductive invariants. The new bounded ideal-Int invariant applies only to the concretely unrolled scan and its source hypotheses.
- **P06**: transitive axiom audit integration into every certified artifact, pinned executable/library provenance and reproducible audit evidence.
- **P11/P12**: general nonlinear/branch/recurrent FP32 error composition, backend capability proof obligations for every optimization and full machine-code correspondence.

Do not label these incomplete areas certified. The generated bounded scan theorem must pass the external Lean checker and axiom audit; code generation alone is not a proof. Local Rust, Lean and CUDA build/test commands were not executed by the agent per AGENTS.md.
