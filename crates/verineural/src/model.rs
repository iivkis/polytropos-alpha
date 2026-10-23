//! Named, architecture-neutral façade over the single Semantic Neural IR.
//! Role labels are metadata, not new IR operations or runtime mutability.
use crate::{
    ArchitectureAutoCompiler, ArchitectureMetadata, ArchitectureProofTask, AutoBuildError,
    AutoBuildPlan, AutoBuildReport, CertifiedBuildConfig, Domain, FP32Policy, GpuRequirement, NeuralError, NeuralGraph, NodeId, Operation,
    ProofGoal, Shape, Tensor,
};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceRole {
    Input,
    Parameter,
}

impl SourceRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Parameter => "parameter",
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ModelError {
    Invalid(String),
    Neural(NeuralError),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "invalid model: {message}"),
            Self::Neural(error) => write!(f, "{error}"),
        }
    }
}

impl Error for ModelError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Neural(error) => Some(error),
            _ => None,
        }
    }
}

impl From<NeuralError> for ModelError {
    fn from(value: NeuralError) -> Self {
        Self::Neural(value)
    }
}

fn valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn valid_identity(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"-._".contains(&c))
}

/// Consumer model declaration; all tensor operations use NeuralGraph.
pub struct ModelBuilder {
    id: String,
    version: String,
    graph: NeuralGraph,
    sources: BTreeMap<String, (NodeId, SourceRole)>,
    outputs: BTreeMap<String, NodeId>,
    goals: BTreeMap<String, ProofGoal>,
}

impl ModelBuilder {
    pub fn new(id: impl Into<String>, version: impl Into<String>) -> Self {
        Self::with_policy(id, version, FP32Policy::strict())
    }

    pub fn with_policy(
        id: impl Into<String>,
        version: impl Into<String>,
        policy: FP32Policy,
    ) -> Self {
        Self {
            id: id.into(),
            version: version.into(),
            graph: NeuralGraph::new(policy),
            sources: BTreeMap::new(),
            outputs: BTreeMap::new(),
            goals: BTreeMap::new(),
        }
    }

    fn insert_source(
        &mut self,
        name: impl Into<String>,
        dims: impl Into<Vec<usize>>,
        domain: Domain,
        role: SourceRole,
    ) -> Result<NodeId, ModelError> {
        let name = name.into();
        if !valid_name(&name)
            || self.sources.contains_key(&name)
            || self.outputs.contains_key(&name)
        {
            return Err(ModelError::Invalid(format!(
                "bad or duplicate source name: {name}"
            )));
        }
        let shape = Shape::new(dims)?;
        let node = self.graph.input(shape, domain);
        self.sources.insert(name, (node, role));
        Ok(node)
    }

    /// Input domains are hypotheses checked against actual bound values.
    pub fn input(
        &mut self,
        name: impl Into<String>,
        dims: impl Into<Vec<usize>>,
        domain: Domain,
    ) -> Result<NodeId, ModelError> {
        self.insert_source(name, dims, domain, SourceRole::Input)
    }

    /// Parameters are named IR sources; updates and optimizers are not implied.
    pub fn parameter(
        &mut self,
        name: impl Into<String>,
        dims: impl Into<Vec<usize>>,
        domain: Domain,
    ) -> Result<NodeId, ModelError> {
        self.insert_source(name, dims, domain, SourceRole::Parameter)
    }

    /// Atomic, capture-checked block; output contract inferred from the IR.
    /// Independent mathematical properties must use explicit proof goals.
    pub fn block<F>(
        &mut self,
        name: &'static str,
        inputs: &[NodeId],
        build: F,
    ) -> Result<NodeId, ModelError>
    where
        F: FnOnce(&mut NeuralGraph, &[NodeId]) -> Result<NodeId, NeuralError>,
    {
        if !valid_name(name) {
            return Err(ModelError::Invalid(format!("invalid block name: {name}")));
        }
        Ok(self
            .graph
            .compose_inferred(name, "model-block-v1", inputs, build)?)
    }

    pub fn output(&mut self, name: impl Into<String>, id: NodeId) -> Result<(), ModelError> {
        let name = name.into();
        if !valid_name(&name)
            || self.sources.contains_key(&name)
            || self.outputs.contains_key(&name)
        {
            return Err(ModelError::Invalid(format!(
                "bad or duplicate output name: {name}"
            )));
        }
        self.graph.node(id)?;
        self.outputs.insert(name, id);
        Ok(())
    }

    /// Registers an obligation, not a proven conclusion.
    pub fn proof_goal(&mut self, goal: ProofGoal) -> Result<(), ModelError> {
        if !valid_name(&goal.name) || self.goals.contains_key(&goal.name) {
            return Err(ModelError::Invalid(format!(
                "bad or duplicate goal: {}",
                goal.name
            )));
        }
        self.graph.node(goal.output)?;
        self.goals.insert(goal.name.clone(), goal);
        Ok(())
    }

    pub fn graph_mut(&mut self) -> &mut NeuralGraph {
        &mut self.graph
    }

    pub fn finish(self) -> Result<Model, ModelError> {
        if !valid_identity(&self.id) || !valid_identity(&self.version) {
            return Err(ModelError::Invalid(
                "id/version must use ASCII alphanumeric, - . _".into(),
            ));
        }
        if self.outputs.is_empty() {
            return Err(ModelError::Invalid(
                "at least one output is required".into(),
            ));
        }
        let registered: BTreeSet<_> = self.sources.values().map(|(id, _)| id.index()).collect();
        if self.graph.nodes().iter().any(|node| {
            matches!(node.operation, Operation::Input) && !registered.contains(&node.id.index())
        }) {
            return Err(ModelError::Invalid(
                "all graph source inputs must have a registered model name".into(),
            ));
        }
        let roots: Vec<_> = self.outputs.values().copied().collect();
        let scope = self.graph.dependency_scope(&roots)?;
        for goal in self.goals.values() {
            if !scope.contains(&goal.output.index()) {
                return Err(ModelError::Invalid(format!(
                    "disconnected proof goal: {}",
                    goal.name
                )));
            }
        }
        Ok(Model {
            id: self.id,
            version: self.version,
            graph: self.graph,
            sources: self.sources,
            outputs: self.outputs,
            goals: self.goals,
        })
    }
}

/// Finished model, with the original graph and deterministic named roots.
pub struct Model {
    id: String,
    version: String,
    graph: NeuralGraph,
    sources: BTreeMap<String, (NodeId, SourceRole)>,
    outputs: BTreeMap<String, NodeId>,
    goals: BTreeMap<String, ProofGoal>,
}

impl Model {
    pub fn graph(&self) -> &NeuralGraph {
        &self.graph
    }

    pub fn output(&self, name: &str) -> Option<NodeId> {
        self.outputs.get(name).copied()
    }

    pub fn source(&self, name: &str) -> Option<(NodeId, SourceRole)> {
        self.sources.get(name).copied()
    }

    pub fn outputs(&self) -> &BTreeMap<String, NodeId> {
        &self.outputs
    }

    pub fn proof_goals(&self) -> &BTreeMap<String, ProofGoal> {
        &self.goals
    }

    /// Supply source roles to the existing snapshot exporter. Current exporter
    /// supports only input/parameter; state is not yet a distinct IR role.
    pub fn metadata(&self, global_property: impl Into<String>) -> ArchitectureMetadata {
        ArchitectureMetadata {
            id: self.id.clone(),
            version: self.version.clone(),
            input_roles: self
                .sources
                .values()
                .map(|(node, role)| (node.index(), role.as_str().to_owned()))
                .collect(),
            global_property: global_property.into(),
        }
    }

    /// Match every named source exactly once, checking shape and declared domain.
    /// Returns ordinary NodeId/Tensor bindings for existing execution APIs.
    pub fn bind<K: Into<String>>(
        &self,
        values: impl IntoIterator<Item = (K, Tensor<f32>)>,
    ) -> Result<Vec<(NodeId, Tensor<f32>)>, ModelError> {
        let mut supplied = BTreeMap::new();
        let mut seen = BTreeSet::new();
        for (name, tensor) in values {
            let name = name.into();
            if !self.sources.contains_key(&name) || !seen.insert(name.clone()) {
                return Err(ModelError::Invalid(format!(
                    "unknown or repeated binding: {name}"
                )));
            }
            supplied.insert(name, tensor);
        }
        let mut ordered = Vec::with_capacity(self.sources.len());
        for (name, (id, _)) in &self.sources {
            let tensor = supplied
                .remove(name)
                .ok_or_else(|| ModelError::Invalid(format!("missing source binding: {name}")))?;
            let node = self.graph.node(*id)?;
            if tensor.shape() != &node.shape {
                return Err(ModelError::Neural(NeuralError::ShapeMismatch {
                    expected: node.shape.clone(),
                    actual: tensor.shape().clone(),
                }));
            }
            tensor.validate_domain(node.domain)?;
            ordered.push((*id, tensor));
        }
        Ok(ordered)
    }

    /// Stable identity binds id/version, named sources/roles, outputs and
    /// obligations. SHA256SUMS also binds the full canonical IR and artifacts.
    fn certified_identity(&self) -> String {
        let mut identity = format!(
            "verineural-certified-model-v1\nid={}\nversion={}\n",
            self.id, self.version
        );
        for (name, (node, role)) in &self.sources {
            let source = self.graph.node(*node).expect("finished model source exists");
            identity.push_str(&format!(
                "source={name};node={};role={};shape={:?};domain={:?}\n",
                node.index(), role.as_str(), source.shape.dims(), source.domain
            ));
        }
        for (name, output) in &self.outputs {
            identity.push_str(&format!("output={name};node={}\n", output.index()));
        }
        for (name, goal) in &self.goals {
            identity.push_str(&format!(
                "goal={name};node={};kind={:?}\n", goal.output.index(), goal.kind
            ));
        }
        identity
    }

    /// Fail closed: certify every declared goal with Lean kernel + axiom audit,
    /// require GPU lowering for every output, then atomically publish evidence.
    /// The result explicitly does not certify CUDA/IEEE-754 machine semantics.
    pub fn certify(
        &self,
        config: CertifiedBuildConfig,
    ) -> Result<AutoBuildReport, AutoBuildError> {
        self.certify_with(&ArchitectureAutoCompiler::default(), config)
    }

    pub fn certify_with(
        &self,
        compiler: &ArchitectureAutoCompiler,
        config: CertifiedBuildConfig,
    ) -> Result<AutoBuildReport, AutoBuildError> {
        let outputs: Vec<_> = self.outputs.values().copied().collect();
        let tasks: Vec<_> = self.goals.values().cloned().map(ArchitectureProofTask::Goal).collect();
        let root = config.resolved_root();
        let identity = self.certified_identity();
        crate::certified::preflight_root(&root, &identity)?;
        let report = compiler.build_identified(&self.graph, &outputs, &tasks, &root, &identity)?;
        crate::certified::publish_root(&root, &identity)?;
        Ok(report)
    }

    /// Verify the complete current-source bundle without CUDA or GPU execution.
    /// To establish authenticity, pass a digest pinned outside the bundle.
    pub fn verify_certified(
        &self,
        config: CertifiedBuildConfig,
        pinned_sha256: Option<&str>,
    ) -> Result<AutoBuildReport, AutoBuildError> {
        self.verify_certified_with(&ArchitectureAutoCompiler::default(), config, pinned_sha256)
    }

    pub fn verify_certified_with(
        &self,
        compiler: &ArchitectureAutoCompiler,
        config: CertifiedBuildConfig,
        pinned_sha256: Option<&str>,
    ) -> Result<AutoBuildReport, AutoBuildError> {
        let outputs: Vec<_> = self.outputs.values().copied().collect();
        let tasks: Vec<_> = self.goals.values().cloned().map(ArchitectureProofTask::Goal).collect();
        let root = config.resolved_root();
        let identity = self.certified_identity();
        let report = compiler.verify_identified(
            &self.graph, &outputs, &tasks, &root, pinned_sha256, &identity
        )?;
        crate::certified::verify_root(&root, &identity)?;
        Ok(report)
    }

    /// Reuse ArchitectureAutoCompiler; planned Lean goals are not certified.
    pub fn plan(
        &self,
        compiler: &ArchitectureAutoCompiler,
        gpu_requirement: GpuRequirement,
    ) -> Result<AutoBuildPlan, AutoBuildError> {
        let outputs: Vec<_> = self.outputs.values().copied().collect();
        let tasks: Vec<_> = self
            .goals
            .values()
            .cloned()
            .map(ArchitectureProofTask::Goal)
            .collect();
        compiler.plan(&self.graph, &outputs, &tasks, gpu_requirement)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_dense_block_infers_shape_roles_and_bindings() {
        let mut builder = ModelBuilder::new("dense", "0.1.0");
        let x = builder.input("x", [4, 2], Domain::Finite).unwrap();
        let w = builder
            .parameter("weights", [2, 3], Domain::Finite)
            .unwrap();
        let b = builder.parameter("bias", [3], Domain::Finite).unwrap();
        let y = builder
            .block("dense_layer", &[x, w, b], |g, p| {
                let product = g.matmul(p[0], p[1])?;
                let shifted = g.bias_add(product, p[2])?;
                g.relu(shifted)
            })
            .unwrap();
        builder.output("prediction", y).unwrap();
        builder
            .proof_goal(ProofGoal::nonnegative("positive", y, vec![]))
            .unwrap();
        let model = builder.finish().unwrap();
        assert_eq!(model.graph().node(y).unwrap().shape.dims(), &[4, 3]);
        // A finite-input MatMul may overflow FP32; ReLU cannot promise finite output.
        assert_eq!(model.graph().node(y).unwrap().domain, Domain::Any);
        assert_eq!(model.graph().modules().len(), 1);
        assert_eq!(
            model
                .metadata("architecture_global_nonnegative")
                .input_roles[&w.index()],
            "parameter"
        );
        let bound = model
            .bind([
                (
                    "bias",
                    Tensor::new(Shape::new(vec![3]).unwrap(), vec![0.0; 3]).unwrap(),
                ),
                (
                    "weights",
                    Tensor::new(Shape::new(vec![2, 3]).unwrap(), vec![1.0; 6]).unwrap(),
                ),
                (
                    "x",
                    Tensor::new(Shape::new(vec![4, 2]).unwrap(), vec![1.0; 8]).unwrap(),
                ),
            ])
            .unwrap();
        assert_eq!(bound.len(), 3);
    }

    #[test]
    fn certificate_identity_tracks_named_contracts_not_absolute_paths() {
        let mut builder = ModelBuilder::new("ident", "0.1.0");
        let x = builder.input("input", [2], Domain::NonNegativeFinite).unwrap();
        let y = builder.block("nonnegative", &[x], |g, p| g.relu(p[0])).unwrap();
        builder.output("value", y).unwrap();
        builder.proof_goal(ProofGoal::nonnegative("nonnegative_goal", y, vec![])).unwrap();
        let model = builder.finish().unwrap();
        let declaration = model.certified_identity();
        assert!(declaration.contains("id=ident\nversion=0.1.0"));
        assert!(declaration.contains("source=input;node=0;role=input"));
        assert!(declaration.contains("goal=nonnegative_goal;node=1"));
        assert!(!declaration.contains("/tmp/"));
    }

    #[test]
    fn certified_plan_rejects_unsupported_gpu_operations() {
        let mut builder = ModelBuilder::new("unsupported", "1");
        let x = builder.input("x", [2], Domain::Finite).unwrap();
        let y = builder.graph_mut().sqrt(x).unwrap();
        builder.output("y", y).unwrap();
        let model = builder.finish().unwrap();
        assert!(matches!(
            model.plan(&ArchitectureAutoCompiler::default(), GpuRequirement::AllOutputs),
            Err(AutoBuildError::Gpu(_)) | Err(AutoBuildError::Proof(_))
        ));
    }

    #[test]
    fn hidden_capture_is_rejected_without_partial_commit() {
        let mut builder = ModelBuilder::new("capture", "1");
        let x = builder.input("x", [2], Domain::Finite).unwrap();
        let hidden = builder.input("hidden", [2], Domain::Finite).unwrap();
        assert!(matches!(
            builder.block("bad", &[x], |g, p| g.add(p[0], hidden)),
            Err(ModelError::Neural(NeuralError::ContractViolation { .. }))
        ));
        assert_eq!(builder.graph_mut().nodes().len(), 2);
        let y = builder.block("good", &[x], |g, p| g.relu(p[0])).unwrap();
        builder.output("out", y).unwrap();
        assert!(builder.finish().is_ok());
    }

    #[test]
    fn graph_mut_cannot_sneak_unnamed_sources_past_binding_checks() {
        let mut builder = ModelBuilder::new("source_registry", "1");
        builder.input("x", [1], Domain::Finite).unwrap();
        let rogue = builder
            .graph_mut()
            .input(Shape::new(vec![1]).unwrap(), Domain::Finite);
        builder.output("out", rogue).unwrap();
        assert!(matches!(builder.finish(), Err(ModelError::Invalid(_))));
    }

    #[test]
    fn rejects_duplicate_source_and_missing_or_invalid_bindings() {
        let mut builder = ModelBuilder::new("checked", "1");
        let x = builder.input("x", [1], Domain::NonNegativeFinite).unwrap();
        assert!(matches!(
            builder.parameter("x", [1], Domain::Finite),
            Err(ModelError::Invalid(_))
        ));
        builder.output("out", x).unwrap();
        let model = builder.finish().unwrap();
        assert!(matches!(
            model.bind([] as [(&str, Tensor<f32>); 0]),
            Err(ModelError::Invalid(_))
        ));
        assert!(matches!(
            model.bind([(
                "x",
                Tensor::new(Shape::new(vec![1]).unwrap(), vec![-1.0]).unwrap()
            )]),
            Err(ModelError::Neural(NeuralError::ValueOutsideDomain { .. }))
        ));
    }
}
