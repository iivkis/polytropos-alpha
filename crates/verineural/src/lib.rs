//! VeriNeural: an importable Rust API and Semantic Neural IR.
//!
//! Stage 3 adds Lean 4 proof generation and checker integration for a bounded
//! ideal-integer fragment (Add, Mul, Relu, Reshape). Shape, pointwise semantics
//! and compositional nonnegativity are verified under explicit input hypotheses.
//! Stage 4 adds experimental LLVM NVPTX + CUDA Driver execution on NVIDIA GPUs.
//! Stage 5 emits an integrity-checked evidence bundle with separate assurance levels.
//! Stage 6 adds MatMul, row-wise bias and composable Dense/Residual FFN blocks.
//! Stage 7 begins controllable CUDA launches and experimental local Relu fusion,
//! with ideal-Int Lean rewrite lemmas, not certified FP32/PTX equivalence.
//! Stage 8 adds exact-integer range proofs plus concrete dyadic binary32
//! RNE traces with Lean-composed error bounds for Add/Mul/MatMul/BiasAdd/Relu/
//! Reshape, including explicit separate multiply/add MatMul reductions. Reusable
//! range/error contracts summarize arbitrary bounded dyadic inputs and module
//! boundaries. Generated whole-module theorems compose each supported
//! ModuleInvocation elementwise, including direct nested-module theorem reuse.
//! Architecture-level range theorems compose the reachable top-level module DAG,
//! including fork/fan-in boundaries, under the same formal RNE target relation.
//! Multi-output reusable proof definitions use one normalized right-nested
//! product for all output tensors and a single shared conjunction theorem,
//! with per-port, per-instance applications and independently checked premises.
//! This ideal-Int profile currently covers explicit Nonnegative claims only.
//!
//! An explicit global ideal-Int nonnegativity request groups compatible
//! top-level module invocations by normalized definition, proves each shared
//! definition once, checks concrete invocation premises separately, and builds
//! the architecture theorem from those applications. Nested module bodies are
//! normalized inside their enclosing definition in this profile; a general
//! typed theorem-dependency graph remains future work. This property is
//! separate from FP32/GPU guarantees.
//! Stage 8 ReLU threshold decisions use conditional per-node error margins;
//! uncertain pre-activations remain indeterminate.
//! Strict FP32 NVPTX lowering uses explicit rounded PTX operations and
//! rejects unsupported contraction at the emitted PTX level. The LLVM/PTX/GPU
//! machine correspondence remains trusted rather than formally certified.
//! Per-kernel translation plans fail closed on inconsistent LLVM/PTX lowering;
//! structural validation is distinct from semantic proof or JIT/SASS assurance.

mod architecture;
mod auto_build;
mod auto_bundle;
mod canonical;
mod certificate;
mod certified;
mod consumer;
mod cuda;
mod error;
mod gpu;
mod ir;
mod model;
mod model_execution;
mod modules;
mod numeric;
mod primitive;
mod proof;
mod special;
mod subnormal;
mod symbolic;
mod tensor;
mod translation;

pub use architecture::{ArchitectureMetadata, ArchitectureSnapshot, snapshot_architecture};
pub use auto_build::{
    ArchitectureAutoCompiler, ArchitectureProofTask, AutoBuildError, AutoBuildPlan,
    AutoBuildReport, ConnectivityEvidence, GpuRequirement, OutputBuildEvidence, PlannedGpu,
    check_architecture_connectivity,
};
pub use certificate::{CertificateEngine, CertificateError, EvidenceStatus, Stage5Report};
pub use certified::{CertifiedBuildConfig, certified_example_main};
pub use consumer::{BindingError, ComponentBinder, ObligationError};
pub use error::NeuralError;
pub use gpu::{
    Fp32PtxAudit, GpuAssuranceBoundary, GpuAssuranceStatus, GpuBenchmark, GpuCapabilityPlan,
    GpuEngine, GpuError, GpuExecution, GpuKernelCapability, GpuLaunchConfig, GpuMemoryContract,
    GpuMemoryPattern, GpuOptimizationCapability, GpuOptimizationKind, GpuProgram, ReluFusion,
    audit_fp32_ptx,
};
pub use ir::{
    ArgMaxTieBreak, ContractClaim, DType, Domain, FP32Policy, FixedScan, FmaMode, GpuSemantics,
    MathContract, ModuleInstanceRef, ModuleInvocation, MultiModuleInvocation, MultiOutputContract,
    MultiOutputModule, NamedPortContract, NeuralGraph, NeuralModule, Node, NodeId, NodeStorage,
    Operation, OperationSemantics, PortContract, Predicate, ReductionOrder, RoundingMode,
    SubnormalMode, ValueRef,
};
pub use model::{Model, ModelBuilder, ModelError, SourceRole};
pub use model_execution::{ModelExecution, ModelExecutionError};
pub use modules::{Dense, ResidualAdd, ResidualFfn};
pub use numeric::{
    ExactFp32Plan, NumericError, RneConditionEvidence, RneErrorContract, RneFp32Plan, RneFp32Trace,
    RneModuleErrorContract, RneRangePlan, RneSelectEvidence, RneSelectPlan, RneStateRolloutPlan,
    RneStateStepContract, RneThresholdEvidence, RneThresholdSide,
};
pub use primitive::{PointwiseExpr, PointwiseSpec, PrimitiveExpr, PrimitiveSpec};
pub use proof::{
    CheckedLean, LeanAxiomAudit, LeanProgram, LeanProofEngine, ProofAssumption, ProofError,
    ProofGoal, ProofGoalKind,
};
pub use special::{
    Fp32SpecialExpected, special_add_boundary, special_mul_bias_relu_boundary,
    special_mul_boundary, special_relu_boundary,
};
pub use subnormal::{subnormal_add_exact_bits, subnormal_mul_rne_bits, subnormal_relu_bits};
pub use symbolic::{
    Dimension, DimensionBindings, DimensionConstraint, ParametricModule, ParametricSignature,
    SymbolicPort, SymbolicShape,
};
pub use tensor::{Shape, Tensor, TensorShape};
pub use translation::{
    TranslationError, TranslationKernel, TranslationKind, TranslationPlan, TranslationReport,
};

#[cfg(test)]
mod tests;
