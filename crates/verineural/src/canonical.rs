//! Canonical, versioned serialization of Semantic Neural IR.
//!
//! This module is intentionally boring. The format is explicit and does not
//! depend on Rust `Debug`, so cosmetic struct changes do not silently rewrite
//! semantic fingerprints.

use std::fmt::Write as _;

use crate::{
    ContractClaim, FmaMode, GpuSemantics, ModuleInstanceRef, NeuralGraph, Operation, PortContract,
    Predicate, ReductionOrder, RoundingMode, Shape, SubnormalMode, ValueRef,
};

const FORMAT_VERSION: &str = "verineural-semantic-ir-v3";

fn escaped(value: &str) -> String {
    let mut out = String::new();
    for byte in value.as_bytes() {
        match *byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' => {
                out.push(char::from(*byte));
            }
            other => {
                write!(out, "%{other:02X}").expect("writing to String");
            }
        }
    }
    out
}

fn shape(shape: &Shape) -> String {
    if shape.dims().is_empty() {
        "scalar".to_owned()
    } else {
        shape
            .dims()
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("x")
    }
}

fn operation(operation: &Operation) -> String {
    match operation {
        Operation::Reshape(output) => format!("reshape:{}", shape(output)),
        Operation::Pointwise(spec) => format!(
            "pointwise:{}@{}:arity={}:{}",
            escaped(spec.id()),
            escaped(spec.version()),
            spec.arity(),
            spec.expression().canonical()
        ),
        Operation::ReduceSum { axis } => format!("reduce_sum_axis:{axis}"),
        Operation::ReduceMax { axis } => format!("reduce_max_axis:{axis}"),
        Operation::BroadcastTo(output) => format!("broadcast_to:{}", shape(output)),
        Operation::Permute { axes } => format!(
            "permute_axes:{}",
            axes.iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        ),
        Operation::ReverseAxis { axis } => format!("reverse_axis:{axis}"),
        Operation::Concat { axis } => format!("concat_axis:{axis}"),
        Operation::ArgMax { axis, tie_break } => {
            format!("argmax_axis:{axis}:tie={}", tie_break.name())
        }
        Operation::Take { axis } => format!("take_axis:{axis}"),
        other => other.name().to_owned(),
    }
}

fn policy(graph: &NeuralGraph) -> String {
    let policy = graph.policy();
    let rounding = match policy.rounding {
        RoundingMode::NearestTiesToEven => "nearest_ties_to_even",
    };
    let fma = match policy.fma {
        FmaMode::SeparateOperations => "separate_operations",
    };
    let subnormals = match policy.subnormals {
        SubnormalMode::Preserve => "preserve",
    };
    let reduction = match policy.reduction {
        ReductionOrder::LeftToRight => "left_to_right",
    };
    format!("rounding={rounding},fma={fma},subnormals={subnormals},reduction={reduction}")
}

fn nodes(ids: impl IntoIterator<Item = usize>) -> String {
    let text = ids
        .into_iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(",");
    if text.is_empty() {
        "-".to_owned()
    } else {
        text
    }
}

fn port(port: &PortContract) -> String {
    format!(
        "{}@{}@{}",
        shape(&port.shape),
        port.dtype.name(),
        port.domain.name()
    )
}

fn value_ref(value: &ValueRef) -> String {
    match value {
        ValueRef::Input(index) => format!("input:{index}"),
        ValueRef::Output => "output".to_owned(),
        ValueRef::Node(id) => format!("node:{}", id.index()),
    }
}

fn predicate(predicate: &Predicate) -> String {
    match predicate {
        Predicate::Nonnegative(value) => format!("nonnegative({})", value_ref(value)),
        Predicate::Equal(a, b) => format!("equal({},{})", value_ref(a), value_ref(b)),
        Predicate::PointwiseLe(a, b) => format!("pointwise_le({},{})", value_ref(a), value_ref(b)),
    }
}

fn claims(claims: &[ContractClaim]) -> String {
    let text = claims
        .iter()
        .map(|claim| match claim {
            ContractClaim::Nonnegative => "nonnegative".to_owned(),
            ContractClaim::EqualsInput(index) => format!("equals_input:{index}"),
            ContractClaim::Predicate(value) => predicate(value),
        })
        .collect::<Vec<_>>()
        .join("+");
    if text.is_empty() {
        "-".to_owned()
    } else {
        text
    }
}

fn ports(ports: impl IntoIterator<Item = String>) -> String {
    let text = ports.into_iter().collect::<Vec<_>>().join(",");
    if text.is_empty() {
        "-".to_owned()
    } else {
        text
    }
}

fn gpu_semantics(gpu: GpuSemantics) -> &'static str {
    match gpu {
        GpuSemantics::NotLowered => "not_lowered",
    }
}

pub(crate) fn graph_ir(graph: &NeuralGraph) -> String {
    let mut text = String::new();
    writeln!(text, "{FORMAT_VERSION}").expect("writing to String");
    writeln!(text, "policy={}", policy(graph)).expect("writing to String");

    for (index, node) in graph.nodes().iter().enumerate() {
        writeln!(
            text,
            "node={index};op={};inputs={};shape={};dtype={};domain={};requires={};ensures={};claims={};ideal={};gpu={}",
            operation(&node.operation),
            nodes(node.inputs.iter().map(|id| id.index())),
            shape(&node.shape),
            node.dtype.name(),
            node.domain.name(),
            ports(node.contract.requires.iter().map(port)),
            port(&node.contract.ensures),
            claims(&node.contract.claims),
            operation(&node.semantics.ideal),
            gpu_semantics(node.semantics.gpu),
        )
        .expect("writing to String");
    }

    for (sequence, module) in graph.module_instances().enumerate() {
        match module {
            ModuleInstanceRef::Single(module) => {
                writeln!(
                    text,
                    "module={sequence};kind=single;name={};version={};inputs={};body={}..{};outputs={};input_ports={};output_ports={}",
                    escaped(module.name),
                    escaped(module.semantic_version),
                    nodes(module.inputs.iter().map(|id| id.index())),
                    module.body_start,
                    module.body_end,
                    module.output.index(),
                    ports(module.contract.requires.iter().enumerate().map(|(index, port_contract)| {
                        format!("input_{index}:{}", port(port_contract))
                    })),
                    format_args!("output:{}!{}", port(&module.contract.ensures), claims(&module.contract.claims)),
                )
                .expect("writing to String");
            }
            ModuleInstanceRef::Multi(module) => {
                writeln!(
                    text,
                    "module={sequence};kind=multi;name={};version={};inputs={};body={}..{};outputs={};input_ports={};output_ports={}",
                    escaped(module.name),
                    escaped(module.semantic_version),
                    nodes(module.inputs.iter().map(|id| id.index())),
                    module.body_start,
                    module.body_end,
                    nodes(module.outputs.iter().map(|id| id.index())),
                    ports(module.inputs.iter().zip(&module.contract.inputs).map(|(id, named)| {
                        format!("{}:{}:{}", escaped(named.name), id.index(), port(&named.port))
                    })),
                    ports(module.outputs.iter().zip(&module.contract.outputs).map(|(id, named)| {
                        format!(
                            "{}:{}:{}!{}",
                            escaped(named.name),
                            id.index(),
                            port(&named.port),
                            claims(&named.claims)
                        )
                    })),
                )
                .expect("writing to String");
            }
        }
    }

    text
}

impl NeuralGraph {
    /// Stable Semantic Neural IR serialization used by every artifact writer.
    /// Graph identity, Rust `Debug` formatting and artifact paths are excluded.
    pub fn canonical_ir(&self) -> String {
        graph_ir(self)
    }
}
