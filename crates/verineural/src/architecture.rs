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
        Operation::PositiveMask => "Positive mask",
        Operation::SelectPositive => "Select positive",
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

/// Distinguish parameter and tensor edges without forcing compass ports.
/// Graphviz can then place endpoints to minimize crossings.
fn edge_theme(role: &str, input_port: usize) -> (&'static str, &'static str) {
    if role == "parameter" {
        ("#C58A3A", "1.35")
    } else {
        let color = match input_port {
            0 => "#7489A8",
            1 => "#95A3B5",
            _ => "#8496AD",
        };
        (color, "1.15")
    }
}

fn source_is_outside_module(source: usize, module: &crate::ModuleInvocation) -> bool {
    source < module.body_start || source > module.output.index()
}

/// All architecture writers invoke the same Graphviz executable from the
/// pinned Nix environment. PNG is always derived from the saved DOT.
pub(crate) fn render_dot_png(dot: &Path, png: &Path) -> io::Result<()> {
    let tmp = png.with_extension("png.tmp");
    let output = Command::new("dot")
        .arg("-Tpng")
        .arg(dot)
        .arg("-o")
        .arg(&tmp)
        .output()?;
    if !output.status.success() {
        let _ = fs::remove_file(&tmp);
        return Err(bad(format!(
            "Graphviz failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    fs::rename(tmp, png)?;
    Ok(())
}

impl ArchitectureSnapshot {
    /// Re-render the pinned DOT with Graphviz into architecture.png.
    /// Only the configured artifact root receives writes.
    pub fn write(&self, root: &Path) -> io::Result<()> {
        fs::create_dir_all(root)?;
        fs::write(root.join("semantic.ir"), &self.semantic_ir)?;
        fs::write(root.join("architecture.dot"), &self.dot)?;
        render_dot_png(&root.join("architecture.dot"), &root.join("architecture.png"))
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
            "module={i};name={};semantic_version={};inputs={:?};body_start={};output={};requires={:?};ensures={:?};claims={:?}",
            module.name,
            module.semantic_version,
            module
                .inputs
                .iter()
                .map(|id| id.index())
                .collect::<Vec<_>>(),
            module.body_start,
            module.output.index(),
            module.contract.requires,
            module.contract.ensures,
            module.contract.claims,
        )
        .expect("writing to string");
    }
    for (i, module) in graph.multi_modules().iter().enumerate() {
        writeln!(
            ir,
            "multi_module={i};name={};version={};inputs={:?};outputs={:?};ports={:?};body=[{},{})",
            module.name,
            module.semantic_version,
            module
                .inputs
                .iter()
                .map(|id| id.index())
                .collect::<Vec<_>>(),
            module
                .outputs
                .iter()
                .map(|id| id.index())
                .collect::<Vec<_>>(),
            module.contract,
            module.body_start,
            module.body_end
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
    // Visual topology is the real IR DAG, not a second network of relays.
    // Automatic edge ports let dot optimize crossing count and edge lengths.
    // DPI belongs in DOT so rerendering via plain "dot -Tpng" is consistent.
    let mut dot = String::from(
        r##"digraph verineural {
  graph [rankdir=LR, compound=true, newrank=true, splines=spline,
         concentrate=false, outputorder=edgesfirst, mclimit=4,
         nodesep=0.75, ranksep=1.15, pad=0.60, margin=0.18, dpi=180,
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
        (
            "vn_multi_output_modules",
            format!("{:?}", graph.multi_modules()),
        ),
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
                ("vn_claims", format!("{:?}", node.contract.claims)),
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
            let visible_ports = module
                .inputs
                .iter()
                .enumerate()
                .map(|(port, source)| format!("p{port}:n{}", source.index()))
                .collect::<Vec<_>>()
                .join("  ");
            writeln!(
                dot,
                "{inner}label={};",
                quoted(&format!(
                    "{}  ·  m{i}\n{visible_ports}  →  n{}",
                    module.name.replace('_', " ").to_uppercase(),
                    module.output.index()
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
                ("vn_claims", format!("{:?}", module.contract.claims)),
                ("vn_semantic_version", module.semantic_version.into()),
                (
                    "vn_proof_status",
                    "conditional-ideal-Int;Lean-check-required".into(),
                ),
            ] {
                writeln!(dot, "{inner}{key}={};", quoted(&value)).expect("writing to string");
            }
            // The label exposes module ports to readers; the full positional
            // mapping stays available as machine-readable cluster metadata.
            for (port, source) in module.inputs.iter().enumerate() {
                writeln!(
                    dot,
                    "{inner}vn_input_port_{port}={};",
                    quoted(&format!("n{}", source.index()))
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
    // Only external data inputs share the leftmost rank. Shared parameters
    // can sit nearer their consumers instead of drawing long diagonal wires.
    let sources = graph
        .nodes()
        .iter()
        .enumerate()
        .filter_map(|(i, node)| {
            (matches!(node.operation, Operation::Input)
                && metadata
                    .input_roles
                    .get(&i)
                    .is_some_and(|role| role.as_str() == "input"))
            .then_some(i)
        })
        .collect::<Vec<_>>();
    if !sources.is_empty() {
        let positions = sources
            .iter()
            .map(|i| format!("n{i};"))
            .collect::<Vec<_>>()
            .join(" ");
        writeln!(dot, "  {{ rank=same; {positions} }}").expect("writing to string");
    }
    // Validate the nested capture chain without adding visual relay edges.
    for (i, module) in mods.iter().enumerate() {
        for (port, source) in module.inputs.iter().enumerate() {
            let source_id = source.index();
            if !source_is_outside_module(source_id, module) {
                return Err(bad(format!(
                    "module {i} input {port} is not external to its body"
                )));
            }
            if let Some(parent) = parents[i] {
                if source_is_outside_module(source_id, &mods[parent])
                    && !mods[parent].inputs.contains(source)
                {
                    return Err(bad(format!(
                        "module m{i} captures n{source_id} without a declared parent m{parent} port"
                    )));
                }
            }
        }
    }
    // One visible edge per real typed IR dependency, with complete vn_* data.
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
            let (edge_color, edge_width) = edge_theme(role, port);
            if let Some(owner) = owners[i] {
                if source_is_outside_module(source.index(), &mods[owner])
                    && !mods[owner].inputs.contains(source)
                {
                    return Err(bad(format!(
                        "node n{i} captures n{} outside module m{owner} without a declared input",
                        source.index()
                    )));
                }
            }
            // Emphasize local chains, but still rank all cross-module data
            // and parameter dependencies in the correct direction.
            let weight = if owners[i] == owners[source.index()] {
                "5"
            } else {
                "2"
            };
            let data = [
                ("color", edge_color.into()),
                ("penwidth", edge_width.into()),
                ("weight", weight.into()),
                ("minlen", "1".into()),
                ("constraint", "true".into()),
                ("vn_src_port", "output".into()),
                ("vn_dst_port", format!("input_{port}")),
                ("vn_semantic_src", format!("n{}", source.index())),
                ("vn_semantic_dst", format!("n{i}")),
                ("vn_visual_src", format!("n{}", source.index())),
                ("vn_role", role.into()),
                ("vn_shape", format!("{:?}", from.shape.dims())),
                ("vn_dtype", "binary32".into()),
                ("vn_transform", transform.into()),
                ("vn_order", port.to_string()),
                ("vn_constraint", "source-shape-checked".into()),
            ];
            writeln!(dot, "  n{} -> n{i} [{}];", source.index(), attrs(&data))
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


/// One architecture-neutral DOT containing *all* Semantic IR nodes, including
/// unselected nodes, and typed port mappings of both module kinds.
///
/// The digest belongs to the caller's canonical auto-build semantic.ir, not
/// to the picture. The DOT is informational; it is not an independent proof.
pub(crate) fn auto_architecture_dot(
    graph: &NeuralGraph,
    plan: &crate::AutoBuildPlan,
    semantic_ir_sha256: &str,
) -> io::Result<String> {
    if semantic_ir_sha256.len() != 64
        || !semantic_ir_sha256.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err(bad("invalid canonical Semantic IR SHA-256 for DOT"));
    }

    struct Scope {
        name: &'static str,
        version: &'static str,
        id: String,
        start: usize,
        end: usize,
        inputs: Vec<(String, usize)>,
        outputs: Vec<(String, usize)>,
        contract: String,
    }
    let mut scopes = Vec::new();
    for (i, module) in graph.modules().iter().enumerate() {
        scopes.push(Scope {
            name: module.name,
            version: module.semantic_version,
            id: format!("m{i}"),
            start: module.body_start,
            end: module.output.index() + 1,
            inputs: module.inputs.iter().enumerate()
                .map(|(p, id)| (format!("input_{p}"), id.index())).collect(),
            outputs: vec![("output".to_owned(), module.output.index())],
            contract: format!("{:?}", module.contract),
        });
    }
    for (i, module) in graph.multi_modules().iter().enumerate() {
        scopes.push(Scope {
            name: module.name,
            version: module.semantic_version,
            id: format!("multi{i}"),
            start: module.body_start,
            end: module.body_end,
            inputs: module.inputs.iter().zip(&module.contract.inputs)
                .map(|(id, port)| (port.name.to_owned(), id.index())).collect(),
            outputs: module.outputs.iter().zip(&module.contract.outputs)
                .map(|(id, port)| (port.name.to_owned(), id.index())).collect(),
            contract: format!("{:?}", module.contract),
        });
    }
    let n = graph.nodes().len();
    for scope in &scopes {
        if scope.start >= scope.end || scope.end > n {
            return Err(bad("module contains invalid node interval"));
        }
    }
    // Nest only intervals which genuinely contain one another. For identical
    // intervals, the later registered scope is the outer scope; this supports
    // modules that forward a nested module's result without producing a node.
    let mut parents = vec![None; scopes.len()];
    for (i, scope) in scopes.iter().enumerate() {
        let mut candidates = Vec::new();
        for (j, other) in scopes.iter().enumerate() {
            if i == j { continue; }
            let overlaps = scope.start < other.end && other.start < scope.end;
            let contained = other.start <= scope.start && scope.end <= other.end;
            let contains = scope.start <= other.start && other.end <= scope.end;
            if overlaps && !contained && !contains {
                return Err(bad("crossing module intervals in auto DOT"));
            }
            if contained && (!contains || j > i) {
                candidates.push(j);
            }
        }
        candidates.sort_by_key(|&j| (scopes[j].end - scopes[j].start, j));
        parents[i] = candidates.first().copied();
    }
    let mut paths = vec![String::new(); scopes.len()];
    fn path_of(i: usize, scopes: &[Scope], parents: &[Option<usize>]) -> String {
        let local = format!("{}_{}", scopes[i].id, scopes[i].name);
        match parents[i] {
            Some(p) => format!("{}/{}", path_of(p, scopes, parents), local),
            None => local,
        }
    }
    for (i, p) in paths.iter_mut().enumerate() {
        *p = path_of(i, &scopes, &parents);
    }
    let owners = (0..n).map(|i| {
        (0..scopes.len())
            .filter(|&j| i >= scopes[j].start && i < scopes[j].end)
            .min_by_key(|&j| (scopes[j].end - scopes[j].start, j))
    }).collect::<Vec<_>>();
    let selected = plan.outputs.iter().map(|id| id.index()).collect::<BTreeSet<_>>();
    // All nodes are drawn, but Lean only models reachable dependencies.
    let mut reachable = BTreeSet::new();
    let mut pending = plan.outputs.clone();
    while let Some(id) = pending.pop() {
        if !reachable.insert(id.index()) { continue; }
        pending.extend(graph.node(id).map_err(|err| bad(err.to_string()))?
            .inputs.iter().copied());
        for module in graph.modules() {
            if module.output == id { pending.extend(module.inputs.iter().copied()); }
        }
        for module in graph.multi_modules() {
            if module.outputs.contains(&id) {
                pending.extend(module.inputs.iter().copied());
            }
        }
    }
    let gpu_status = plan.gpu.iter().map(|entry| match entry {
        crate::PlannedGpu::Supported { output, .. } =>
            (output.index(), "nvptx-supported"),
        crate::PlannedGpu::Unsupported { output, .. } =>
            (output.index(), "unsupported"),
    }).collect::<BTreeMap<_, _>>();
    let mut dot = String::from(
        r##"digraph verineural_auto {
  graph [rankdir=LR, compound=true, newrank=true, splines=spline,
         concentrate=false, outputorder=edgesfirst, mclimit=4,
         nodesep=0.75, ranksep=1.15, pad=0.60, margin=0.18, dpi=180,
         bgcolor="#FFFFFF", fontname="DejaVu Sans", fontcolor="#24324B",
         labelloc=t, labeljust=l];
  node [shape=box, style="rounded,filled", fontname="DejaVu Sans",
        fontsize=12, fontcolor="#24324B", penwidth=1.5, margin="0.17,0.13"];
  edge [color="#8796AA", penwidth=1.1, arrowsize=0.72];
"##,
    );
    for (key, value) in [
        ("label", format!(
            "VeriNeural · auto architecture\n{} nodes · {} module instances · {} outputs",
            n, scopes.len(), plan.outputs.len()
        )),
        ("vn_export_version", "auto-v1".into()),
        ("vn_ir_sha256", semantic_ir_sha256.to_owned()),
        ("vn_fp32_profile", format!("{:?}", graph.policy())),
        ("vn_selected_outputs", format!("{:?}", selected)),
        ("vn_lean_assurance", "ideal-Int;conditional-source-assumptions;kernel-checked-at-build".into()),
        ("vn_gpu_assurance", "LLVM/PTX-structural;JIT-SASS-not-proven".into()),
        ("vn_connectivity", "Rust-structural-validation".into()),
        ("vn_unmodeled", "arbitrary-dtype;implicit-broadcast;machine-equivalence".into()),
    ] {
        writeln!(dot, "  {key}={};", quoted(&value)).expect("writing DOT");
    }
    struct Context<'a> {
        graph: &'a NeuralGraph,
        scopes: &'a [Scope],
        parents: &'a [Option<usize>],
        owners: &'a [Option<usize>],
        paths: &'a [String],
        selected: &'a BTreeSet<usize>,
        reachable: &'a BTreeSet<usize>,
        gpu_status: &'a BTreeMap<usize, &'static str>,
    }
    fn render_scope(parent: Option<usize>, context: &Context<'_>, depth: usize, dot: &mut String) {
        let pad = "  ".repeat(depth + 1);
        for (i, node) in context.graph.nodes().iter().enumerate()
            .filter(|(i, _)| context.owners[*i] == parent)
        {
            let kind = if context.selected.contains(&i) {
                "architecture_output"
            } else if matches!(node.operation, Operation::Input) {
                "input"
            } else {
                "primitive"
            };
            let (fill, outline) = node_theme(kind, &node.operation);
            let title = if kind == "architecture_output" {
                format!("OUTPUT · {}", operation_title(&node.operation))
            } else {
                operation_title(&node.operation).to_uppercase()
            };
            let properties = [
                ("label", format!("{title} · n{i}\n{:?}", node.shape.dims())),
                ("fillcolor", fill.to_owned()),
                ("color", outline.to_owned()),
                ("vn_id", format!("n{i}")),
                ("vn_kind", kind.to_owned()),
                ("vn_op", node.operation.name().to_owned()),
                ("vn_instance_path", parent.map_or_else(|| "root".to_owned(),
                    |p| context.paths[p].clone())),
                ("vn_inputs", format!("{:?}", node.inputs.iter()
                    .map(|id| id.index()).collect::<Vec<_>>())),
                ("vn_shape", format!("{:?}", node.shape.dims())),
                ("vn_dtype", "binary32".into()),
                ("vn_layout", "row-major".into()),
                ("vn_domain", format!("{:?}", node.domain)),
                ("vn_contract", format!("{:?}", node.contract)),
                ("vn_semantics", format!("{:?}", node.semantics)),
                ("vn_selected_output", context.selected.contains(&i).to_string()),
                ("vn_reachable_from_outputs", context.reachable.contains(&i).to_string()),
                ("vn_gpu_lowering", context.gpu_status.get(&i).copied()
                    .unwrap_or("no-selected-output-profile").to_owned()),
                ("vn_proof_status", if context.reachable.contains(&i) {
                    "ideal-Int;checked-in-auto-bundle"
                } else {
                    "unselected;no-Lean-obligation"
                }.into()),
            ];
            writeln!(dot, "{pad}n{i} [{}];", attrs(&properties)).expect("writing DOT");
        }
        for (i, scope) in context.scopes.iter().enumerate()
            .filter(|(i, _)| context.parents[*i] == parent)
        {
            writeln!(dot, "{pad}subgraph cluster_{} {{", scope.id).expect("writing DOT");
            let inner = format!("{pad}  ");
            let (fill, outline) = cluster_theme(depth);
            let in_ports = scope.inputs.iter()
                .map(|(name, index)| format!("{name}:n{index}"))
                .collect::<Vec<_>>().join("  ");
            let out_ports = scope.outputs.iter()
                .map(|(name, index)| format!("{name}:n{index}"))
                .collect::<Vec<_>>().join("  ");
            for (key, value) in [
                ("style", "rounded,filled".into()),
                ("fillcolor", fill.into()),
                ("color", outline.into()),
                ("fontname", "DejaVu Sans".into()),
                ("fontsize", "14".into()),
                ("labelloc", "t".into()),
                ("labeljust", "l".into()),
                ("margin", "22".into()),
                ("label", format!(
                    "{} · {}\n{} → {}",
                    scope.name.replace('_', " ").to_uppercase(),
                    scope.id, in_ports, out_ports,
                )),
                ("vn_instance_path", context.paths[i].clone()),
                ("vn_definition_id", scope.name.to_owned()),
                ("vn_semantic_version", scope.version.to_owned()),
                ("vn_contract", scope.contract.clone()),
                ("vn_input_ports", format!("{:?}", scope.inputs)),
                ("vn_output_ports", format!("{:?}", scope.outputs)),
                ("vn_proof_status", "module-contracts-checkable;see-Lean-artifact".into()),
            ] {
                writeln!(dot, "{inner}{key}={};", quoted(&value)).expect("writing DOT");
            }
            render_scope(Some(i), context, depth + 1, dot);
            writeln!(dot, "{pad}}}").expect("writing DOT");
        }
    }
    render_scope(None, &Context {
        graph,
        scopes: &scopes,
        parents: &parents,
        owners: &owners,
        paths: &paths,
        selected: &selected,
        reachable: &reachable,
        gpu_status: &gpu_status,
    }, 0, &mut dot);
    let inputs = graph.nodes().iter().enumerate()
        .filter_map(|(i, node)| matches!(node.operation, Operation::Input).then_some(i))
        .collect::<Vec<_>>();
    if !inputs.is_empty() {
        writeln!(dot, "  {{ rank=same; {} }}", inputs.iter()
            .map(|i| format!("n{i};")).collect::<Vec<_>>().join(" "))
            .expect("writing DOT");
    }
    // Exactly one rendered edge for every real Semantic IR dependency.
    // Module port mappings are attributes, never extra visual relay edges.
    for (i, node) in graph.nodes().iter().enumerate() {
        for (port, input) in node.inputs.iter().enumerate() {
            let source = graph.node(*input).map_err(|err| bad(err.to_string()))?;
            let (color, width) = edge_theme("data", port);
            let properties = [
                ("color", color.into()),
                ("penwidth", width.into()),
                ("weight", if owners[i] == owners[input.index()] {
                    "5".into()
                } else { "2".into() }),
                ("minlen", "1".into()),
                ("constraint", "true".into()),
                ("vn_semantic_src", format!("n{}", input.index())),
                ("vn_semantic_dst", format!("n{i}")),
                ("vn_dst_port", format!("input_{port}")),
                ("vn_order", port.to_string()),
                ("vn_shape", format!("{:?}", source.shape.dims())),
                ("vn_dtype", "binary32".into()),
                ("vn_role", "data".into()),
            ];
            writeln!(dot, "  n{} -> n{i} [{}];", input.index(), attrs(&properties))
                .expect("writing DOT");
        }
    }
    dot.push_str("}\n");
    Ok(dot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Domain, GpuEngine, Shape};

    #[test]
    fn automatic_dot_represents_nested_multi_ports_and_all_real_edges() {
        use crate::{
            ArchitectureAutoCompiler, GpuRequirement, MathContract, MultiOutputContract,
            MultiOutputModule, NamedPortContract, NeuralError, NeuralModule, PortContract,
        };
        struct Activation { contract: MathContract }
        impl NeuralModule for Activation {
            fn name(&self) -> &'static str { "consumer_activation" }
            fn contract(&self) -> &MathContract { &self.contract }
            fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId])
                -> Result<NodeId, NeuralError> {
                graph.relu(inputs[0])
            }
        }
        struct Step { contract: MultiOutputContract, child: Activation }
        impl MultiOutputModule for Step {
            fn name(&self) -> &'static str { "consumer_step" }
            fn contract(&self) -> &MultiOutputContract { &self.contract }
            fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId])
                -> Result<Vec<NodeId>, NeuralError> {
                let next = graph.apply_module(&self.child, &[inputs[0]])?;
                let readout = graph.add(next, inputs[1])?;
                Ok(vec![next, readout])
            }
        }
        fn prepare() -> (NeuralGraph, crate::AutoBuildPlan) {
            let mut graph = NeuralGraph::default();
            let shape = Shape::new(vec![2]).unwrap();
            let positive = PortContract::new(shape.clone(), Domain::NonNegativeFinite);
            let any = PortContract::new(shape.clone(), Domain::Any);
            let a = graph.input(shape.clone(), Domain::NonNegativeFinite);
            let b = graph.input(shape, Domain::NonNegativeFinite);
            let _detached = graph.relu(b).unwrap();
            let module = Step {
                contract: MultiOutputContract {
                    inputs: vec![
                        NamedPortContract::new("state", positive.clone()),
                        NamedPortContract::new("input", positive.clone()),
                    ],
                    outputs: vec![
                        NamedPortContract::new("next_state", positive.clone()),
                        NamedPortContract::new("readout", any),
                    ],
                },
                child: Activation {
                    contract: MathContract::new(vec![positive.clone()], positive),
                },
            };
            let outputs = graph.apply_multi_module(&module, &[a, b]).unwrap();
            let plan = ArchitectureAutoCompiler::default()
                .plan(&graph, &outputs, &[], GpuRequirement::AllOutputs).unwrap();
            (graph, plan)
        }

        let (g1, p1) = prepare();
        let (g2, p2) = prepare();
        let hash = "a".repeat(64);
        let dot = auto_architecture_dot(&g1, &p1, &hash).unwrap();
        assert_eq!(dot, auto_architecture_dot(&g2, &p2, &hash).unwrap());
        assert!(dot.contains(&format!(r#"vn_ir_sha256="{hash}""#)));
        assert!(dot.contains("subgraph cluster_m0"));
        assert!(dot.contains("subgraph cluster_multi0"));
        assert!(dot.contains("next_state"));
        assert!(dot.contains("readout"));
        assert!(dot.contains("vn_input_ports"));
        assert!(dot.contains("vn_output_ports"));
        assert!(dot.contains(r#"vn_reachable_from_outputs="false""#));
        assert!(dot.contains(r#"vn_gpu_lowering="nvptx-supported""#));
        assert_eq!(dot.matches("vn_id=").count(), g1.nodes().len());
        let edges: usize = g1.nodes().iter().map(|n| n.inputs.len()).sum();
        assert_eq!(dot.matches(" -> n").count(), edges);
        assert!(!dot.contains("module-input-port"));
        assert!(!dot.contains("headport="));
        assert!(!dot.contains("tailport="));
        assert!(auto_architecture_dot(&g1, &p1, "bad").is_err());
    }

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
        assert!(a.dot.contains("rank=same; n0;"));
        assert!(a.dot.contains("splines=spline"));
        assert!(a.dot.contains("concentrate=false"));
        assert!(a.dot.contains("mclimit=4"));
        assert!(a.dot.contains("nodesep=0.75"));
        assert!(a.dot.contains("ranksep=1.15"));
        assert!(!a.dot.contains("tailport="));
        assert!(!a.dot.contains("headport="));
        assert!(!a.dot.contains("constraint=\"false\""));
        assert!(a.dot.contains("constraint=\"true\""));
        assert!(a.dot.contains("minlen=\"1\""));
        assert!(a.dot.contains("vn_visual_src=\"n0\""));
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
