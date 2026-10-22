//! Deterministic, architecture-neutral Semantic IR and complete Graphviz export.
//!
//! Export is a representation and integrity aid, NOT a proof that LLVM/PTX,
//! CUDA JIT or the GPU implements the graph's Lean/FP32 semantics. All nodes,
//! including unreachable ones, are displayed, and nested module boundaries
//! remain visible. Source input roles are caller-declared metadata.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::io::{self, Write as _};
use std::path::Path;
use std::process::{Command, Stdio};

use crate::{GpuProgram, NeuralGraph, NodeId, Operation};

/// Stable provenance and presentation metadata for one architecture.
/// An input role may be "input" or "parameter"; IR has no state/control nodes.
#[derive(Clone, Debug)]
pub struct ArchitectureMetadata {
    pub id: String,
    pub version: String,
    pub input_roles: BTreeMap<usize, String>,
    pub global_property: String,
}

#[derive(Clone, Debug)]
pub struct ArchitectureSnapshot {
    pub semantic_ir: String,
    pub dot: String,
    pub ir_sha256: String,
}

fn bad(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn sha256(data: &[u8]) -> io::Result<String> {
    let mut process = Command::new("sha256sum")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    process
        .stdin
        .take()
        .ok_or_else(|| bad("sha256sum stdin unavailable"))?
        .write_all(data)?;
    let result = process.wait_with_output()?;
    if !result.status.success() {
        return Err(bad("sha256sum rejected Semantic IR input"));
    }
    let printed = String::from_utf8(result.stdout).map_err(|_| bad("invalid sha256sum output"))?;
    let digest = printed.split_whitespace().next().unwrap_or("");
    if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(bad("sha256sum did not return a full SHA-256 digest"));
    }
    Ok(digest.to_owned())
}

fn quoted(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push(' '),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn attrs(pairs: &[(&str, String)]) -> String {
    pairs
        .iter()
        .map(|(key, value)| format!("{key}={}", quoted(value)))
        .collect::<Vec<_>>()
        .join(", ")
}

/// A presentation-only theme. Canonical vn_* attributes remain independent of
/// color, DPI, font, and the Graphviz layout algorithm.
fn node_theme(kind: &str, op: &Operation) -> (&'static str, &'static str) {
    match kind {
        "input" => ("#EAF2FF", "#3972BD"),
        "parameter" => ("#FFF1DB", "#B9772A"),
        "architecture_output" => ("#DDF8E9", "#26805A"),
        _ => match op {
            Operation::Add => ("#F1F5F9", "#64748B"),
            Operation::Mul => ("#EEF2FF", "#6477B5"),
            Operation::BiasAdd => ("#FDEEF3", "#B76A91"),
            Operation::Relu => ("#F0EDFF", "#8160BF"),
            Operation::MatMul => ("#E8F5FE", "#4C90B5"),
            _ => ("#FFFFFF", "#64748B"),
        },
    }
}

fn operation_title(operation: &Operation) -> &'static str {
    match operation {
        Operation::Input => "Input",
        Operation::Add => "Add",
        Operation::Mul => "Multiply",
        Operation::BiasAdd => "Bias add",
        Operation::Relu => "ReLU",
        Operation::MatMul => "MatMul",
        Operation::Sqrt => "Sqrt",
        Operation::Reshape(_) => "Reshape",
    }
}

fn cluster_theme(depth: usize) -> (&'static str, &'static str) {
    match depth % 3 {
        0 => ("#F5F8FF", "#93A9D4"),
        1 => ("#F1F5FD", "#A4B4CB"),
        _ => ("#F8FAFD", "#BCC8D7"),
    }
}

/// Presentation-only edge routing. Distinct compass ports stop Graphviz from
/// stacking every incoming dependency onto the same short line segment.
fn edge_tail_port(role: &str, input_port: usize) -> &'static str {
    if role == "parameter" {
        "e"
    } else {
        match input_port {
            0 => "ne",
            1 => "se",
            _ => "e",
        }
    }
}

fn edge_head_port(operation: &Operation, input_port: usize) -> &'static str {
    match operation {
        Operation::BiasAdd if input_port == 1 => "s",
        Operation::Add | Operation::Mul | Operation::MatMul => match input_port {
            0 => "nw",
            1 => "sw",
            _ => "w",
        },
        _ => "w",
    }
}

fn edge_theme(
    role: &str,
    input_port: usize,
) -> (&'static str, &'static str, &'static str, &'static str) {
    if role == "parameter" {
        ("#C58A3A", "1.35", "1", "2")
    } else {
        let color = match input_port {
            0 => "#7489A8",
            1 => "#95A3B5",
            _ => "#8496AD",
        };
        (color, "1.15", "4", "1")
    }
}

/// Stable visual identity for a module's declared input, distinct from the
/// semantic source node. Module interfaces are explicit relay points.
fn input_anchor_id(module: usize, port: usize) -> String {
    format!("m{module}_in{port}")
}

fn source_is_outside_module(source: usize, module: &crate::ModuleInvocation) -> bool {
    source < module.body_start || source > module.output.index()
}

impl ArchitectureSnapshot {
    /// Re-render the pinned DOT with Graphviz into architecture.png.
    /// Only the configured artifact root receives writes.
    pub fn write(&self, root: &Path) -> io::Result<()> {
        fs::create_dir_all(root)?;
        fs::write(root.join("semantic.ir"), &self.semantic_ir)?;
        fs::write(root.join("architecture.dot"), &self.dot)?;
        let tmp = root.join("architecture.png.tmp");
        let render = Command::new("dot")
            .arg("-Tpng")
            .arg(root.join("architecture.dot"))
            .arg("-o")
            .arg(&tmp)
            .output()?;
        if !render.status.success() {
            let _ = fs::remove_file(&tmp);
            return Err(bad(format!(
                "Graphviz failed: {}",
                String::from_utf8_lossy(&render.stderr)
            )));
        }
        fs::rename(tmp, root.join("architecture.png"))?;
        Ok(())
    }

    /// Checks the exact deterministic Semantic IR and DOT content independently
    /// of the SHA-256 artifact manifest verified by the consumer.
    pub fn verify(&self, root: &Path) -> io::Result<()> {
        if fs::read_to_string(root.join("semantic.ir"))? != self.semantic_ir
            || fs::read_to_string(root.join("architecture.dot"))? != self.dot
        {
            return Err(bad(
                "architecture IR or DOT disagrees with current Rust graph",
            ));
        }
        if !self
            .dot
            .contains(&format!("vn_ir_sha256={}", quoted(&self.ir_sha256)))
        {
            return Err(bad(
                "architecture DOT lacks matching Semantic IR fingerprint",
            ));
        }
        if !root.join("architecture.png").is_file() {
            return Err(bad("rendered architecture PNG is missing"));
        }
        Ok(())
    }
}

/// Export all nodes, all module instances and the selected output.
/// The GPU plan is derived from the SAME IR, not from LLVM/PTX textual output.
pub fn snapshot_architecture(
    graph: &NeuralGraph,
    output: NodeId,
    plan: &GpuProgram,
    metadata: &ArchitectureMetadata,
) -> io::Result<ArchitectureSnapshot> {
    graph.node(output).map_err(|err| bad(err.to_string()))?;
    if plan.output().index() != output.index() {
        return Err(bad(
            "GPU program output differs from requested architecture output",
        ));
    }
    for v in [&metadata.id, &metadata.version] {
        if v.is_empty()
            || !v
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "-._".contains(c))
        {
            return Err(bad(
                "architecture id/version must use ASCII alphanumeric, - . _",
            ));
        }
    }
    for (&idx, role) in &metadata.input_roles {
        if !matches!(
            graph.nodes().get(idx).map(|node| &node.operation),
            Some(Operation::Input)
        ) || !matches!(role.as_str(), "input" | "parameter")
        {
            return Err(bad(
                "unsupported input role or non-input node role assignment",
            ));
        }
    }
    if graph.nodes().iter().enumerate().any(|(idx, node)| {
        matches!(node.operation, Operation::Input) && !metadata.input_roles.contains_key(&idx)
    }) {
        return Err(bad(
            "every architecture source input needs an explicit input/parameter role",
        ));
    }
    if metadata.global_property != "architecture_global_nonnegative" {
        return Err(bad(
            "only explicit, proved ideal-Int global nonnegativity is supported",
        ));
    }
    let gpu = plan.translation_plan().map_err(|e| bad(e.to_string()))?;
    let mut ir = format!(
        "verineural-architecture-ir-v1\nid={}\nversion={}\noutput={}\nfp32={:?}\nproperty={}\n",
        metadata.id,
        metadata.version,
        output.index(),
        graph.policy(),
        metadata.global_property,
    );
    for (index, node) in graph.nodes().iter().enumerate() {
        writeln!(ir, "node={index};op={:?};inputs={:?};shape={:?};dtype=binary32;layout=row-major;domain={:?};semantics={:?};requires={:?};ensures={:?};role={}",
            node.operation, node.inputs.iter().map(|x|x.index()).collect::<Vec<_>>(),
            node.shape.dims(), node.domain, node.semantics,
            node.contract.requires, node.contract.ensures,
            metadata.input_roles.get(&index).map(String::as_str).unwrap_or("primitive"),
        ).expect("writing to string");
    }
    for (i, module) in graph.modules().iter().enumerate() {
        writeln!(
            ir,
            "module={i};name={};inputs={:?};body_start={};output={};requires={:?};ensures={:?}",
            module.name,
            module
                .inputs
                .iter()
                .map(|id| id.index())
                .collect::<Vec<_>>(),
            module.body_start,
            module.output.index(),
            module.contract.requires,
            module.contract.ensures,
        )
        .expect("writing to string");
    }
    ir.push_str("translation_plan_begin\n");
    ir.push_str(&gpu.to_text());
    ir.push_str("translation_plan_end\n");
    let digest = sha256(ir.as_bytes())?;

    // Build a deterministic hierarchy from nested intervals. An inner module
    // must be wholly contained in one outer module, never cross boundaries.
    let mods = graph.modules();
    let mut parents: Vec<Option<usize>> = vec![None; mods.len()];
    for (i, module) in mods.iter().enumerate() {
        let first = module.body_start;
        let last = module.output.index();
        if first > last || last >= graph.nodes().len() {
            return Err(bad("invalid module body interval"));
        }
        let mut candidates: Vec<usize> = mods
            .iter()
            .enumerate()
            .filter_map(|(j, other)| {
                (i != j
                    && other.body_start <= first
                    && other.output.index() >= last
                    && (other.body_start < first || other.output.index() > last))
                    .then_some(j)
            })
            .collect();
        candidates.sort_by_key(|&j| (mods[j].output.index() - mods[j].body_start, j));
        parents[i] = candidates.first().copied();
        for (j, other) in mods.iter().enumerate() {
            if i != j && first <= other.output.index() && other.body_start <= last {
                let contains = (first <= other.body_start && last >= other.output.index())
                    || (other.body_start <= first && other.output.index() >= last);
                if !contains {
                    return Err(bad("crossing module body intervals"));
                }
            }
        }
    }
    let mut paths = vec![String::new(); mods.len()];
    fn module_path(
        i: usize,
        parents: &[Option<usize>],
        names: &[crate::ModuleInvocation],
    ) -> String {
        let local = format!("m{i}_{}", names[i].name);
        parents[i].map_or_else(
            || local.clone(),
            |p| format!("{}/{}", module_path(p, parents, names), local),
        )
    }
    for (i, path) in paths.iter_mut().enumerate() {
        *path = module_path(i, &parents, mods);
    }
    let mut owners = vec![None; graph.nodes().len()];
    for (index, slot) in owners.iter_mut().enumerate() {
        *slot = (0..mods.len())
            .filter(|&i| index >= mods[i].body_start && index <= mods[i].output.index())
            .min_by_key(|&i| (mods[i].output.index() - mods[i].body_start, i));
    }
    let mut reachable = BTreeSet::new();
    let mut pending = vec![output];
    while let Some(id) = pending.pop() {
        if reachable.insert(id.index()) {
            pending.extend(
                graph
                    .node(id)
                    .map_err(|e| bad(e.to_string()))?
                    .inputs
                    .iter()
                    .copied(),
            );
        }
    }
    let kernel_map: BTreeMap<_, _> = gpu.kernels.iter().map(|k| (k.node, k)).collect();
    let fused: BTreeSet<_> = plan
        .fused_relus()
        .values()
        .copied()
        .chain(
            plan.fused_pointwise_chains()
                .values()
                .flat_map(|x| x.iter().copied()),
        )
        .collect();
    // Stable, spacious presentation: source nodes on one left-hand column,
    // actual data dependencies only, and rounded colored module boundaries.
    // DPI belongs in DOT so rerendering via plain "dot -Tpng" is consistent.
    let mut dot = String::from(
        r##"digraph verineural {
  graph [rankdir=LR, compound=true, newrank=true, splines=spline,
         concentrate=false, outputorder=edgesfirst,
         nodesep=0.90, ranksep=1.28, pad=0.60, margin=0.18, dpi=180,
         bgcolor="#FFFFFF", fontname="DejaVu Sans", fontcolor="#24324B",
         labelloc=t, labeljust=l];
  node [shape=box, style="rounded,filled", fontname="DejaVu Sans",
        fontsize=12, fontcolor="#24324B", penwidth=1.5, margin="0.17,0.13"];
  edge [color="#8796AA", penwidth=1.1, arrowsize=0.72];
"##,
    );
    writeln!(
        dot,
        "  label={};",
        quoted(&format!(
            "{}  ·  architecture v{}\nSemantic IR · FP32 · module composition",
            metadata.id.replace('-', " "),
            metadata.version
        ))
    )
    .expect("writing to string");
    for (key, value) in [
        ("vn_export_version", "1".to_owned()),
        ("vn_architecture_id", metadata.id.clone()),
        ("vn_architecture_version", metadata.version.clone()),
        ("vn_ir_sha256", digest.clone()),
        ("vn_fp32_profile", format!("{:?}", graph.policy())),
        ("vn_global_property", metadata.global_property.clone()),
        ("vn_global_proof", "Lean-check-required".into()),
        (
            "vn_gpu_assurance",
            "structural-LLVM-PTX-validation;JIT-SASS-trusted".into(),
        ),
        ("vn_architecture_output", format!("n{}", output.index())),
        (
            "vn_supported_roles",
            "input,parameter,primitive,module-boundary,output".into(),
        ),
        ("vn_unmodeled", "state,control,training".into()),
    ] {
        writeln!(dot, "  {key}={};", quoted(&value)).expect("writing to string");
    }

    // One immutable rendering context for all recursive module scopes.
    // The recursion varies only in parent, depth and the output DOT buffer.
    struct RenderContext<'a> {
        graph: &'a NeuralGraph,
        owners: &'a [Option<usize>],
        parents: &'a [Option<usize>],
        paths: &'a [String],
        reachable: &'a BTreeSet<usize>,
        roles: &'a BTreeMap<usize, String>,
        kernel_map: &'a BTreeMap<usize, &'a crate::TranslationKernel>,
        fused: &'a BTreeSet<usize>,
        output: NodeId,
    }

    fn render_scope(
        parent: Option<usize>,
        context: &RenderContext<'_>,
        depth: usize,
        dot: &mut String,
    ) {
        let RenderContext {
            graph,
            owners,
            parents,
            paths,
            reachable,
            roles,
            kernel_map,
            fused,
            output,
        } = context;
        let indent = "  ".repeat(depth + 1);
        for (index, node) in graph
            .nodes()
            .iter()
            .enumerate()
            .filter(|(i, _)| owners[*i] == parent)
        {
            let dims = format!("{:?}", node.shape.dims());
            let kind = if index == output.index() {
                "architecture_output"
            } else if matches!(node.operation, Operation::Input) {
                roles.get(&index).map(String::as_str).unwrap_or("input")
            } else {
                "primitive"
            };
            let kernel = kernel_map.get(&index);
            let lowering = kernel.map_or_else(
                || {
                    if fused.contains(&index) {
                        "fused-into-consumer"
                    } else if matches!(node.operation, Operation::Reshape(_)) {
                        "zero-copy-alias"
                    } else {
                        "not-lowered"
                    }
                },
                |_| "kernel",
            );
            let policy = format!("{:?}", node.semantics.fp32);
            let (fill, outline) = node_theme(kind, &node.operation);
            // Actual newline, not a literal backslash-n. quoted() escapes it
            // once, allowing Graphviz to render a true two-line label.
            let title = match kind {
                "input" => "INPUT".to_owned(),
                "parameter" => "PARAMETER".to_owned(),
                "architecture_output" => format!("OUTPUT · {}", operation_title(&node.operation)),
                _ => operation_title(&node.operation).to_owned(),
            };
            let label = format!("{title}  ·  n{index}\n{dims}");
            let properties = [
                ("label", label),
                ("fillcolor", fill.into()),
                ("color", outline.into()),
                (
                    "tooltip",
                    format!("node {index}: {}", node.operation.name()),
                ),
                ("vn_id", format!("n{index}")),
                ("vn_kind", kind.into()),
                ("vn_op", node.operation.name().into()),
                (
                    "vn_instance_path",
                    parent.map_or_else(|| "root".to_owned(), |p| paths[p].clone()),
                ),
                (
                    "vn_inputs",
                    format!(
                        "{:?}",
                        node.inputs.iter().map(|n| n.index()).collect::<Vec<_>>()
                    ),
                ),
                ("vn_shape", dims),
                ("vn_dtype", "binary32".into()),
                ("vn_layout", "row-major".into()),
                ("vn_domain", format!("{:?}", node.domain)),
                ("vn_semantics", format!("{:?}", node.semantics)),
                ("vn_requires", format!("{:?}", node.contract.requires)),
                ("vn_ensures", format!("{:?}", node.contract.ensures)),
                ("vn_fp32_policy", policy),
                (
                    "vn_reachable_from_output",
                    reachable.contains(&index).to_string(),
                ),
                ("vn_gpu_lowering", lowering.into()),
                (
                    "vn_gpu_kernel",
                    kernel.map_or_else(|| "none".into(), |k| format!("vn_node_{}", k.node)),
                ),
                (
                    "vn_gpu_memory",
                    kernel.map_or_else(
                        || "none".into(),
                        |k| format!("reads={:?},writes={}", k.reads, k.writes),
                    ),
                ),
                (
                    "vn_assurance",
                    "ideal-Int-Lean-conditional;FP32-test-only;JIT-trusted".into(),
                ),
            ];
            writeln!(dot, "{indent}n{index} [{}];", attrs(&properties)).expect("writing to string");
        }
        for (i, module) in graph
            .modules()
            .iter()
            .enumerate()
            .filter(|(i, _)| parents[*i] == parent)
        {
            writeln!(dot, "{indent}subgraph cluster_m{i} {{").expect("writing to string");
            let inner = format!("{indent}  ");
            let (background, outline) = cluster_theme(depth);
            for (key, value) in [
                ("style", "rounded,filled".to_owned()),
                ("fillcolor", background.to_owned()),
                ("color", outline.to_owned()),
                ("fontname", "DejaVu Sans".to_owned()),
                ("fontcolor", "#344664".to_owned()),
                ("fontsize", "15".to_owned()),
                ("penwidth", "1.7".to_owned()),
                ("margin", "22".to_owned()),
                ("labelloc", "t".to_owned()),
                ("labeljust", "l".to_owned()),
            ] {
                writeln!(dot, "{inner}{key}={};", quoted(&value)).expect("writing to string");
            }
            writeln!(
                dot,
                "{inner}label={};",
                quoted(&format!(
                    "{}  ·  m{i}",
                    module.name.replace('_', " ").to_uppercase()
                ))
            )
            .expect("writing to string");
            for (key, value) in [
                ("vn_instance_path", paths[i].clone()),
                ("vn_definition_id", module.name.into()),
                (
                    "vn_module_inputs",
                    format!(
                        "{:?}",
                        module.inputs.iter().map(|n| n.index()).collect::<Vec<_>>()
                    ),
                ),
                ("vn_module_output", format!("n{}", module.output.index())),
                ("vn_requires", format!("{:?}", module.contract.requires)),
                ("vn_ensures", format!("{:?}", module.contract.ensures)),
                (
                    "vn_proof_status",
                    "conditional-ideal-Int;Lean-check-required".into(),
                ),
            ] {
                writeln!(dot, "{inner}{key}={};", quoted(&value)).expect("writing to string");
            }
            // Interface edges would duplicate real data dependencies and
            // route dozens of dashed lines across the PNG. Preserve every
            // positional port as machine-readable cluster metadata instead.
            for (port, source) in module.inputs.iter().enumerate() {
                writeln!(
                    dot,
                    "{inner}vn_input_port_{port}={};",
                    quoted(&format!("n{}", source.index()))
                )
                .expect("writing to string");
            }
            // Tiny visible inlet per declared port. Unlike duplicate
            // annotation edges to module outputs, this is an actual relay:
            // captured data can enter a nested module only through its
            // interface, never via a long diagonal across several clusters.
            for (port, source) in module.inputs.iter().enumerate() {
                let role = roles
                    .get(&source.index())
                    .map(String::as_str)
                    .filter(|role| *role == "parameter")
                    .unwrap_or("data");
                let (fill, outline) = if role == "parameter" {
                    ("#FFF1DB", "#B9772A")
                } else {
                    ("#EAF2FF", "#3972BD")
                };
                let properties = [
                    ("label", format!("p{port}")),
                    ("shape", "circle".into()),
                    ("style", "filled".into()),
                    ("width", "0.32".into()),
                    ("height", "0.32".into()),
                    ("fixedsize", "true".into()),
                    ("fontsize", "9".into()),
                    ("margin", "0.01".into()),
                    ("fillcolor", fill.into()),
                    ("color", outline.into()),
                    ("fontcolor", outline.into()),
                    (
                        "tooltip",
                        format!("{} input {port} from n{}", module.name, source.index()),
                    ),
                    ("vn_kind", "module_input_port".into()),
                    ("vn_visual_only", "true".into()),
                    ("vn_instance_path", paths[i].clone()),
                    ("vn_port_index", port.to_string()),
                    ("vn_source_node", format!("n{}", source.index())),
                ];
                writeln!(
                    dot,
                    "{inner}{} [{}];",
                    input_anchor_id(i, port),
                    attrs(&properties)
                )
                .expect("writing to string");
            }
            render_scope(Some(i), context, depth + 1, dot);
            writeln!(dot, "{indent}}}").expect("writing to string");
        }
    }
    let context = RenderContext {
        graph,
        owners: &owners,
        parents: &parents,
        paths: &paths,
        reachable: &reachable,
        roles: &metadata.input_roles,
        kernel_map: &kernel_map,
        fused: &fused,
        output,
    };
    render_scope(None, &context, 0, &mut dot);
    // Keep all source inputs and parameters aligned to the left, regardless
    // of the number of downstream consumers or nested module boundaries.
    let sources = graph
        .nodes()
        .iter()
        .enumerate()
        .filter_map(|(i, node)| matches!(node.operation, Operation::Input).then_some(i))
        .collect::<Vec<_>>();
    if !sources.is_empty() {
        let positions = sources
            .iter()
            .map(|i| format!("n{i};"))
            .collect::<Vec<_>>()
            .join(" ");
        writeln!(dot, "  {{ rank=same; {positions} }}").expect("writing to string");
    }
    // Route each *declared* interface from its nearest containing module.
    // This turns a source-to-deep-node diagonal into short parent-to-child
    // segments. No synthetic edge is treated as a computational dependency:
    // vn_kind and vn_semantic_src explicitly distinguish interface relays.
    for (i, module) in mods.iter().enumerate() {
        for (port, source) in module.inputs.iter().enumerate() {
            let source_id = source.index();
            if !source_is_outside_module(source_id, module) {
                return Err(bad(format!(
                    "module {i} input {port} is not external to its body"
                )));
            }
            let visual_source = if let Some(parent) = parents[i] {
                let outer = &mods[parent];
                if source_is_outside_module(source_id, outer) {
                    let outer_port = outer
                        .inputs
                        .iter()
                        .position(|id| id.index() == source_id)
                        .ok_or_else(|| bad(format!(
                            "module m{i} captures n{source_id} without a declared parent m{parent} port"
                        )))?;
                    input_anchor_id(parent, outer_port)
                } else {
                    format!("n{source_id}")
                }
            } else {
                format!("n{source_id}")
            };
            let role = metadata
                .input_roles
                .get(&source_id)
                .map(String::as_str)
                .filter(|role| *role == "parameter")
                .unwrap_or("data");
            let (color, width, _, _) = edge_theme(role, port);
            let data = [
                ("color", color.into()),
                ("penwidth", width.into()),
                ("tailport", "e".into()),
                ("headport", "w".into()),
                ("weight", "2".into()),
                ("minlen", "1".into()),
                ("vn_kind", "module_interface".into()),
                ("vn_visual_only", "true".into()),
                ("vn_semantic_src", format!("n{source_id}")),
                ("vn_visual_src", visual_source.clone()),
                ("vn_src_port", "output".into()),
                ("vn_dst_port", format!("module_input_{port}")),
                ("vn_role", role.into()),
                (
                    "vn_shape",
                    format!(
                        "{:?}",
                        graph
                            .node(*source)
                            .map_err(|err| bad(err.to_string()))?
                            .shape
                            .dims()
                    ),
                ),
                ("vn_dtype", "binary32".into()),
                ("vn_transform", "identity".into()),
                ("vn_order", port.to_string()),
                ("vn_instance_path", paths[i].clone()),
            ];
            writeln!(
                dot,
                "  {visual_source} -> {} [{}];",
                input_anchor_id(i, port),
                attrs(&data)
            )
            .expect("writing to string");
        }
    }
    // Every original IR input receives exactly one edge with its original
    // vn_* contract; only the *visual* source may be a nearby module inlet.
    for (i, node) in graph.nodes().iter().enumerate() {
        for (port, source) in node.inputs.iter().enumerate() {
            let from = graph.node(*source).map_err(|e| bad(e.to_string()))?;
            let transform = match node.operation {
                Operation::BiasAdd if port == 1 => "row-wise-broadcast",
                Operation::Reshape(_) => "zero-copy-reshape",
                Operation::MatMul => "matrix-product-indexing",
                _ => "identity",
            };
            let role = metadata
                .input_roles
                .get(&source.index())
                .map(String::as_str)
                .filter(|r| *r == "parameter")
                .unwrap_or("data");
            let (edge_color, edge_width, edge_weight, edge_minlen) = edge_theme(role, port);
            let visual_source = if let Some(owner) = owners[i] {
                if source_is_outside_module(source.index(), &mods[owner]) {
                    let module_port = mods[owner]
                        .inputs
                        .iter()
                        .position(|id| id.index() == source.index())
                        .ok_or_else(|| bad(format!(
                            "node n{i} captures n{} outside module m{owner} without a declared input",
                            source.index()
                        )))?;
                    input_anchor_id(owner, module_port)
                } else {
                    format!("n{}", source.index())
                }
            } else {
                format!("n{}", source.index())
            };
            let tailport = edge_tail_port(role, port);
            let headport = edge_head_port(&node.operation, port);
            let data = [
                ("color", edge_color.into()),
                ("penwidth", edge_width.into()),
                ("tailport", tailport.into()),
                ("headport", headport.into()),
                ("weight", edge_weight.into()),
                ("minlen", edge_minlen.into()),
                (
                    "constraint",
                    if role == "parameter" { "false" } else { "true" }.into(),
                ),
                ("vn_src_port", "output".into()),
                ("vn_dst_port", format!("input_{port}")),
                ("vn_semantic_src", format!("n{}", source.index())),
                ("vn_semantic_dst", format!("n{i}")),
                ("vn_visual_src", visual_source.clone()),
                ("vn_role", role.into()),
                ("vn_shape", format!("{:?}", from.shape.dims())),
                ("vn_dtype", "binary32".into()),
                ("vn_transform", transform.into()),
                ("vn_order", port.to_string()),
                ("vn_constraint", "source-shape-checked".into()),
            ];
            writeln!(dot, "  {visual_source} -> n{i} [{}];", attrs(&data))
                .expect("writing to string");
        }
    }
    dot.push_str("}\n");
    Ok(ArchitectureSnapshot {
        semantic_ir: ir,
        dot,
        ir_sha256: digest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Domain, GpuEngine, Shape};

    #[test]
    fn graph_id_independent_complete_export_and_detects_tampering() {
        fn build() -> (NeuralGraph, NodeId, GpuProgram) {
            let mut g = NeuralGraph::default();
            let a = g.input(Shape::new(vec![2]).unwrap(), Domain::Finite);
            let b = g.input(Shape::new(vec![2]).unwrap(), Domain::Finite);
            let _unreachable = g.add(a, b).unwrap();
            let product = g.mul(a, b).unwrap();
            let program = GpuEngine::default().compile(&g, product).unwrap();
            (g, product, program)
        }
        let meta = ArchitectureMetadata {
            id: "coupled-cell".into(),
            version: "0.1.0".into(),
            input_roles: BTreeMap::from([(0, "input".into()), (1, "parameter".into())]),
            global_property: "architecture_global_nonnegative".into(),
        };
        let (g1, o1, p1) = build();
        let (g2, o2, p2) = build();
        let a = snapshot_architecture(&g1, o1, &p1, &meta).unwrap();
        let b = snapshot_architecture(&g2, o2, &p2, &meta).unwrap();
        assert_eq!(a.semantic_ir, b.semantic_ir);
        assert_eq!(a.dot, b.dot);
        assert!(a.dot.contains("vn_reachable_from_output=\"false\""));
        assert!(a.dot.contains("vn_ir_sha256="));
        assert_eq!(a.dot.matches("vn_id=").count(), g1.nodes().len());
        // Visual edges must correspond one-to-one with real typed IR inputs,
        // not synthetic module-port annotations that obscure the graph.
        let real_edges: usize = g1.nodes().iter().map(|n| n.inputs.len()).sum();
        assert_eq!(a.dot.matches(" -> ").count(), real_edges);
        assert_eq!(a.dot.matches("vn_semantic_dst=").count(), real_edges);
        assert!(!a.dot.contains("module-input-port"));
        assert!(!a.dot.contains(r#"style="dashed""#));
        assert!(!a.dot.contains("vn_input_port_")); // no modules in this graph
        assert!(a.dot.contains("rank=same; n0; n1;"));
        assert!(a.dot.contains("splines=spline"));
        assert!(a.dot.contains("concentrate=false"));
        assert!(a.dot.contains("nodesep=0.90"));
        assert!(a.dot.contains("ranksep=1.28"));
        assert!(a.dot.contains("tailport=\"ne\""));
        assert!(a.dot.contains("headport=\"nw\""));
        assert!(a.dot.contains("tailport=\"e\""));
        assert!(a.dot.contains("headport=\"sw\""));
        assert!(a.dot.contains("constraint=\"false\""));
        assert!(a.dot.contains("minlen=\"2\""));
        assert!(a.dot.contains("dpi=180"));
        assert!(a.dot.contains("fillcolor=\"#FFF1DB\""));
        // Labels must be multi-line, not a literal printed backslash-n.
        assert!(a.dot.contains(r#"n0 [label="INPUT  ·  n0\n[2]""#));
        assert!(!a.dot.contains(r#"INPUT  ·  n0\\n[2]"#));
        let mut changed = meta.clone();
        changed.version = "0.2.0".into();
        assert_ne!(
            a.ir_sha256,
            snapshot_architecture(&g1, o1, &p1, &changed)
                .unwrap()
                .ir_sha256
        );
        changed.input_roles.insert(20, "parameter".into());
        assert!(snapshot_architecture(&g1, o1, &p1, &changed).is_err());
    }
}
