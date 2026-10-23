//! Small consumer-facing binding/obligation helpers. The Semantic IR,
//! module contracts and Lean proof engine remain the only sources of truth.
use std::error::Error;
use std::fmt;
use std::path::Path;

use crate::{
    CheckedLean, FixedScan, LeanAxiomAudit, LeanProgram, LeanProofEngine, MathContract,
    MultiOutputModule, NeuralError, NeuralGraph, NeuralModule, NodeId, PortContract, Predicate,
    ProofError, ProofGoal, ProofGoalKind, ValueRef,
};

/// A structured error at a caller-owned component path and named port.
/// The underlying shape, domain or contract error is preserved, not translated
/// into an allegedly verified mathematical claim.
#[derive(Debug)]
pub struct BindingError {
    pub component_path: String,
    pub port: String,
    pub constraint: String,
    pub source: Option<NeuralError>,
}

impl fmt::Display for BindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "component {} port {}: {}",
            self.component_path, self.port, self.constraint
        )?;
        if let Some(source) = &self.source {
            write!(f, ": {source}")?;
        }
        Ok(())
    }
}

impl Error for BindingError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_ref().map(|source| source as &dyn Error)
    }
}

/// A proof failure is associated with the explicitly requested theorem.
/// A generated theorem is not a checked theorem.
#[derive(Debug)]
pub struct ObligationError {
    pub theorem: String,
    pub source: ProofError,
}

impl fmt::Display for ObligationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "proof obligation {}: {}", self.theorem, self.source)
    }
}

impl Error for ObligationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}

/// Lightweight facade over the existing atomic module insertion/rollback.
pub struct ComponentBinder<'a> {
    graph: &'a mut NeuralGraph,
    path: String,
}

impl NeuralGraph {
    /// Label subsequent module calls without duplicating the graph or contracts.
    pub fn component(&mut self, path: impl Into<String>) -> ComponentBinder<'_> {
        ComponentBinder {
            graph: self,
            path: path.into(),
        }
    }
}

impl ComponentBinder<'_> {
    fn failed(
        &self,
        port: impl Into<String>,
        constraint: impl Into<String>,
        source: Option<NeuralError>,
    ) -> BindingError {
        BindingError {
            component_path: self.path.clone(),
            port: port.into(),
            constraint: constraint.into(),
            source,
        }
    }

    fn check_input(
        &self,
        id: NodeId,
        name: &str,
        required: &PortContract,
    ) -> Result<(), BindingError> {
        let actual = self.graph.node(id).map_err(|error| {
            self.failed(name, "input must belong to the current graph", Some(error))
        })?;
        if actual.shape != required.shape {
            return Err(self.failed(
                name,
                "input shape",
                Some(NeuralError::ShapeMismatch {
                    expected: required.shape.clone(),
                    actual: actual.shape.clone(),
                }),
            ));
        }
        if !actual.domain.satisfies(required.domain) {
            return Err(self.failed(
                name,
                "input value domain",
                Some(NeuralError::DomainNotEstablished {
                    required: required.domain,
                    actual: actual.domain,
                }),
            ));
        }
        Ok(())
    }

    /// Invoke one existing single-output module with positional ports.
    /// Port names in diagnostics are stable input_0, input_1, etc.
    pub fn single<M: NeuralModule>(
        &mut self,
        module: &M,
        inputs: &[NodeId],
    ) -> Result<NodeId, BindingError> {
        let contract: &MathContract = module.contract();
        if inputs.len() != contract.requires.len() {
            return Err(self.failed(
                "inputs",
                format!("expected {} inputs, received {}", contract.requires.len(), inputs.len()),
                None,
            ));
        }
        for (index, (&id, required)) in inputs.iter().zip(&contract.requires).enumerate() {
            self.check_input(id, &format!("input_{index}"), required)?;
        }
        self.graph.apply_module(module, inputs).map_err(|source| {
            self.failed("output", format!("{} postcondition or body", module.name()), Some(source))
        })
    }

    /// Bind named input ports in arbitrary caller order. Reject missing,
    /// duplicate and unexpected bindings before mutating the graph.
    pub fn multi<M: MultiOutputModule>(
        &mut self,
        module: &M,
        bindings: &[(&str, NodeId)],
    ) -> Result<Vec<NodeId>, BindingError> {
        let ports = &module.contract().inputs;
        let mut ordered = Vec::with_capacity(ports.len());
        for required in ports {
            let mut found = None;
            for &(name, id) in bindings {
                if name == required.name && found.replace(id).is_some() {
                    return Err(self.failed(name, "duplicate port binding", None));
                }
            }
            let id = found.ok_or_else(|| {
                self.failed(required.name, "missing required port binding", None)
            })?;
            self.check_input(id, required.name, &required.port)?;
            ordered.push(id);
        }
        for &(name, _) in bindings {
            if !ports.iter().any(|port| port.name == name) {
                return Err(self.failed(name, "unexpected port binding", None));
            }
        }
        self.graph.apply_multi_module(module, &ordered).map_err(|source| {
            self.failed("outputs", format!("{} output postconditions or body", module.name()), Some(source))
        })
    }

    /// Resolve results by their declared names instead of array positions.
    pub fn multi_named<M: MultiOutputModule>(
        &mut self,
        module: &M,
        bindings: &[(&str, NodeId)],
    ) -> Result<std::collections::BTreeMap<String, NodeId>, BindingError> {
        let outputs = self.multi(module, bindings)?;
        Ok(module
            .contract()
            .outputs
            .iter()
            .zip(outputs)
            .map(|(port, node)| (port.name.to_owned(), node))
            .collect())
    }

    /// A finite scan, not a proof for an unbounded recurrent system.
    pub fn scan<M: MultiOutputModule>(
        &mut self,
        transition: &M,
        initial_state: NodeId,
        inputs: &[NodeId],
    ) -> Result<FixedScan, BindingError> {
        let ports = &transition.contract().inputs;
        if ports.len() == 2 && ports[0].name == "state" && ports[1].name == "input" {
            self.check_input(initial_state, "state", &ports[0].port)?;
            for (step, &input) in inputs.iter().enumerate() {
                self.check_input(input, &format!("input[{step}]"), &ports[1].port)?;
            }
        }
        self.graph.scan_fixed(transition, initial_state, inputs).map_err(|source| {
            self.failed("next_state", "bounded state-transition contract", Some(source))
        })
    }
}

impl ProofGoal {
    /// Make the source assumptions explicit; they are never inferred as facts.
    pub fn nonnegative(
        name: impl Into<String>,
        output: NodeId,
        assumptions: Vec<NodeId>,
    ) -> Self {
        Self {
            name: name.into(),
            output,
            kind: ProofGoalKind::Nonnegative { assumptions },
        }
    }

    pub fn bounded_above(
        name: impl Into<String>,
        lower: NodeId,
        output: NodeId,
    ) -> Self {
        Self {
            name: name.into(),
            output,
            kind: ProofGoalKind::Predicate(Predicate::pointwise_le(
                ValueRef::Node(lower),
                ValueRef::Output,
            )),
        }
    }

    /// Return an independently generated theorem, without claiming it checked.
    pub fn generate_source(
        &self,
        graph: &NeuralGraph,
        checker: &LeanProofEngine,
    ) -> Result<LeanProgram, ObligationError> {
        checker.generate_goal(graph, self).map_err(|source| ObligationError {
            theorem: format!("architecture_property_{}", self.name),
            source,
        })
    }

    /// Check the Lean file and its transitive axiom dependencies.
    pub fn certify(
        &self,
        graph: &NeuralGraph,
        checker: &LeanProofEngine,
        path: &Path,
    ) -> Result<(CheckedLean, LeanAxiomAudit), ObligationError> {
        let program = self.generate_source(graph, checker)?;
        checker.verify_certified(&program, path).map_err(|source| ObligationError {
            theorem: format!("architecture_property_{}", self.name),
            source,
        })
    }
}
