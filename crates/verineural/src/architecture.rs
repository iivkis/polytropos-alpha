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
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{GpuProgram, ModuleInstanceRef, NeuralGraph, NodeId, Operation};

/// Stable provenance and presentation metadata for one architecture.
/// Input, parameter and temporal state are metadata on the same pure IR source.
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
        "state" => ("#E9F6F5", "#36827C"),
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
        Operation::Pointwise(_) => "User-defined pointwise",
        Operation::ReduceSum { .. } => "Reduce sum",
        Operation::ReduceMax { .. } => "Reduce max",
        Operation::BroadcastTo(_) => "Broadcast",
        Operation::Permute { .. } => "Permute",
        Operation::ReverseAxis { .. } => "Reverse axis",
        Operation::Concat { .. } => "Concat",
        Operation::ArgMax { .. } => "Argmax",
        Operation::Take { .. } => "Take",
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

#[derive(Clone, Debug)]
struct ModuleScope {
    id: String,
    name: &'static str,
    version: &'static str,
    start: usize,
    end: usize,
    inputs: Vec<(String, usize)>,
    outputs: Vec<(String, usize)>,
    contract: String,
    requires: String,
    ensures: String,
    claims: String,
}

#[derive(Clone, Debug)]
struct ModuleLayout {
    scopes: Vec<ModuleScope>,
    parents: Vec<Option<usize>>,
    paths: Vec<String>,
    owners: Vec<Option<usize>>,
}

fn module_layout(graph: &NeuralGraph) -> io::Result<ModuleLayout> {
    let mut scopes = Vec::new();
    for (sequence, module) in graph.module_instances().enumerate() {
        match module {
            ModuleInstanceRef::Single(module) => scopes.push(ModuleScope {
                id: format!("m{sequence}"),
                name: module.name,
                version: module.semantic_version,
                start: module.body_start,
                end: module.body_end,
                inputs: module
                    .inputs
                    .iter()
                    .enumerate()
                    .map(|(port, id)| (format!("input_{port}"), id.index()))
                    .collect(),
                outputs: vec![("output".to_owned(), module.output.index())],
                contract: format!("{:?}", module.contract),
                requires: format!("{:?}", module.contract.requires),
                ensures: format!("{:?}", module.contract.ensures),
                claims: format!("{:?}", module.contract.claims),
            }),
            ModuleInstanceRef::Multi(module) => scopes.push(ModuleScope {
                id: format!("m{sequence}"),
                name: module.name,
                version: module.semantic_version,
                start: module.body_start,
                end: module.body_end,
                inputs: module
                    .inputs
                    .iter()
                    .zip(&module.contract.inputs)
                    .map(|(id, port)| (port.name.to_owned(), id.index()))
                    .collect(),
                outputs: module
                    .outputs
                    .iter()
                    .zip(&module.contract.outputs)
                    .map(|(id, port)| (port.name.to_owned(), id.index()))
                    .collect(),
                contract: format!("{:?}", module.contract),
                requires: format!("{:?}", module.contract.inputs),
                ensures: format!("{:?}", module.contract.outputs),
                claims: format!(
                    "{:?}",
                    module
                        .contract
                        .outputs
                        .iter()
                        .map(|port| (&port.name, &port.claims))
                        .collect::<Vec<_>>()
                ),
            }),
        }
    }

    let node_count = graph.nodes().len();
    for scope in &scopes {
        if scope.start >= scope.end || scope.end > node_count {
            return Err(bad("module contains invalid node interval"));
        }
        if scope
            .outputs
            .iter()
            .any(|(_, output)| *output < scope.start || *output >= scope.end)
        {
            return Err(bad("module output lies outside its owned body"));
        }
    }

    let mut parents = vec![None; scopes.len()];
    for (i, scope) in scopes.iter().enumerate() {
        let mut candidates = Vec::new();
        for (j, other) in scopes.iter().enumerate() {
            if i == j {
                continue;
            }
            let overlaps = scope.start < other.end && other.start < scope.end;
            let contained = other.start <= scope.start && scope.end <= other.end;
            let contains = scope.start <= other.start && other.end <= scope.end;
            if overlaps && !contained && !contains {
                return Err(bad("crossing module body intervals"));
            }
            if contained && (!contains || j > i) {
                candidates.push(j);
            }
        }
        candidates.sort_by_key(|&j| (scopes[j].end - scopes[j].start, j));
        parents[i] = candidates.first().copied();
    }

    fn path_of(i: usize, scopes: &[ModuleScope], parents: &[Option<usize>]) -> String {
        let local = format!("{}_{}", scopes[i].id, scopes[i].name);
        parents[i].map_or_else(
            || local.clone(),
            |parent| format!("{}/{}", path_of(parent, scopes, parents), local),
        )
    }

    let paths = (0..scopes.len())
        .map(|index| path_of(index, &scopes, &parents))
        .collect::<Vec<_>>();
    let owners = (0..node_count)
        .map(|node| {
            (0..scopes.len())
                .filter(|&scope| node >= scopes[scope].start && node < scopes[scope].end)
                .min_by_key(|&scope| (scopes[scope].end - scopes[scope].start, scope))
        })
        .collect::<Vec<_>>();

    Ok(ModuleLayout {
        scopes,
        parents,
        paths,
        owners,
    })
}

fn source_is_outside_scope(source: usize, scope: &ModuleScope) -> bool {
    source < scope.start || source >= scope.end
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

/// Re-render from saved DOT and compare bytes, not just presence or checksums.
/// The unique sibling path avoids trampling another verifier's temporary image.
pub(crate) fn verify_rendered_dot_png(dot: &Path, png: &Path) -> io::Result<()> {
    static NEXT_REPLAY: AtomicU64 = AtomicU64::new(0);
    let id = NEXT_REPLAY.fetch_add(1, Ordering::Relaxed);
    let replay = dot.with_file_name(format!(
        ".verineural-png-replay-{}-{id}.png",
        std::process::id()
    ));
    let result = (|| {
        render_dot_png(dot, &replay)?;
        if fs::read(&replay)? != fs::read(png)? {
            return Err(bad(
                "architecture.png differs from Graphviz rendering of saved DOT",
            ));
        }
        Ok(())
    })();
    let _ = fs::remove_file(&replay);
    result
}

impl ArchitectureSnapshot {
    /// Re-render the pinned DOT with Graphviz into architecture.png.
    /// Only the configured artifact root receives writes.
    pub fn write(&self, root: &Path) -> io::Result<()> {
        fs::create_dir_all(root)?;
        fs::write(root.join("semantic.ir"), &self.semantic_ir)?;
        fs::write(root.join("architecture.dot"), &self.dot)?;
        render_dot_png(
            &root.join("architecture.dot"),
            &root.join("architecture.png"),
        )
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
        verify_rendered_dot_png(
            &root.join("architecture.dot"),
            &root.join("architecture.png"),
        )
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
        ) || !matches!(role.as_str(), "input" | "parameter" | "state")
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
            "every architecture source input needs an explicit input/parameter/state role",
        ));
    }
    if metadata.global_property != "architecture_global_nonnegative" {
        return Err(bad(
            "only explicit, proved ideal-Int global nonnegativity is supported",
        ));
    }
    let gpu = plan.translation_plan().map_err(|e| bad(e.to_string()))?;
    let ir = graph.canonical_ir();
    let digest = sha256(ir.as_bytes())?;

    let layout = module_layout(graph)?;
    let scopes = &layout.scopes;
    let parents = &layout.parents;
    let paths = &layout.paths;
    let owners = &layout.owners;
    let reachable = graph
        .dependency_scope(&[output])
        .map_err(|err| bad(err.to_string()))?;
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
        ("vn_export_version", "3".to_owned()),
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
            "input,parameter,state,primitive,module-boundary,output".into(),
        ),
        ("vn_unmodeled", "control,training".into()),
        ("vn_module_instances", scopes.len().to_string()),
    ] {
        writeln!(dot, "  {key}={};", quoted(&value)).expect("writing to string");
    }

    // One immutable rendering context for all recursive module scopes.
    // Single- and multi-output modules use the same normalized hierarchy.
    struct RenderContext<'a> {
        graph: &'a NeuralGraph,
        scopes: &'a [ModuleScope],
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
            scopes,
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
            let title = match kind {
                "input" => "INPUT".to_owned(),
                "parameter" => "PARAMETER".to_owned(),
                "state" => "STATE".to_owned(),
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
                    "vn_reduction_axis",
                    match node.operation {
                        Operation::ReduceSum { axis } | Operation::ReduceMax { axis } => {
                            axis.to_string()
                        }
                        _ => String::new(),
                    },
                ),
                (
                    "vn_axis",
                    match node.operation {
                        Operation::ReduceSum { axis }
                        | Operation::ReduceMax { axis }
                        | Operation::ReverseAxis { axis }
                        | Operation::Concat { axis }
                        | Operation::SliceAxis { axis, .. }
                        | Operation::ArgMax { axis, .. }
                        | Operation::Take { axis } => axis.to_string(),
                        _ => String::new(),
                    },
                ),
                (
                    "vn_permutation",
                    match &node.operation {
                        Operation::Permute { axes } => format!("{axes:?}"),
                        _ => String::new(),
                    },
                ),
                (
                    "vn_tie_break",
                    match node.operation {
                        Operation::ArgMax { tie_break, .. } => tie_break.name().to_owned(),
                        _ => String::new(),
                    },
                ),
                (
                    "vn_custom_semantics",
                    match &node.operation {
                        Operation::Pointwise(spec) => format!(
                            "{}@{}:{}",
                            spec.id(),
                            spec.version(),
                            spec.expression().canonical()
                        ),
                        _ => String::new(),
                    },
                ),
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
                ("vn_dtype", node.dtype.name().into()),
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
        for (i, scope) in scopes
            .iter()
            .enumerate()
            .filter(|(i, _)| parents[*i] == parent)
        {
            writeln!(dot, "{indent}subgraph cluster_{} {{", scope.id).expect("writing to string");
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
            let visible_inputs = scope
                .inputs
                .iter()
                .map(|(name, source)| format!("{name}:n{source}"))
                .collect::<Vec<_>>()
                .join("  ");
            let visible_outputs = scope
                .outputs
                .iter()
                .map(|(name, output)| format!("{name}:n{output}"))
                .collect::<Vec<_>>()
                .join("  ");
            writeln!(
                dot,
                "{inner}label={};",
                quoted(&format!(
                    "{}  ·  {}\n{visible_inputs}  →  {visible_outputs}",
                    scope.name.replace('_', " ").to_uppercase(),
                    scope.id,
                ))
            )
            .expect("writing to string");
            for (key, value) in [
                ("vn_instance_path", paths[i].clone()),
                ("vn_definition_id", scope.name.into()),
                ("vn_input_ports", format!("{:?}", scope.inputs)),
                ("vn_output_ports", format!("{:?}", scope.outputs)),
                ("vn_requires", scope.requires.clone()),
                ("vn_ensures", scope.ensures.clone()),
                ("vn_claims", scope.claims.clone()),
                ("vn_semantic_version", scope.version.into()),
                (
                    "vn_proof_status",
                    "conditional-ideal-Int;Lean-check-required".into(),
                ),
            ] {
                writeln!(dot, "{inner}{key}={};", quoted(&value)).expect("writing to string");
            }
            render_scope(Some(i), context, depth + 1, dot);
            writeln!(dot, "{indent}}}").expect("writing to string");
        }
    }
    let context = RenderContext {
        graph,
        scopes,
        owners,
        parents,
        paths,
        reachable: &reachable,
        roles: &metadata.input_roles,
        kernel_map: &kernel_map,
        fused: &fused,
        output,
    };
    render_scope(None, &context, 0, &mut dot);
    // Let dot choose input ranks: hard rank=same constraints create crossing
    // bundles when shared inputs and parameters feed distant submodules.
    // Validate the nested capture chain without adding visual relay edges.
    for (i, scope) in scopes.iter().enumerate() {
        for (port, (_, source_id)) in scope.inputs.iter().enumerate() {
            if !source_is_outside_scope(*source_id, scope) {
                return Err(bad(format!(
                    "module {} input {port} is not external to its body",
                    scope.id
                )));
            }
            if let Some(parent) = parents[i] {
                let parent_scope = &scopes[parent];
                if source_is_outside_scope(*source_id, parent_scope)
                    && !parent_scope
                        .inputs
                        .iter()
                        .any(|(_, parent_source)| parent_source == source_id)
                {
                    return Err(bad(format!(
                        "module {} captures n{} without a declared parent {} port",
                        scope.id, source_id, parent_scope.id
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
                Operation::Permute { .. } => "axis-permutation",
                Operation::BroadcastTo(_) => "explicit-broadcast",
                Operation::ReduceMax { .. } => "axis-reduction-max",
                Operation::ReverseAxis { .. } => "axis-reversal",
                Operation::Concat { .. } => "axis-concatenation",
                Operation::SliceAxis { .. } => "axis-slice",
                Operation::ArgMax { .. } => "argmax-indexing",
                Operation::Take { .. } if port == 1 => "index-selection",
                Operation::Take { .. } => "axis-selection",
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
                let scope = &scopes[owner];
                if source_is_outside_scope(source.index(), scope)
                    && !scope
                        .inputs
                        .iter()
                        .any(|(_, declared)| *declared == source.index())
                {
                    return Err(bad(format!(
                        "node n{i} captures n{} outside module {} without a declared input",
                        source.index(),
                        scope.id
                    )));
                }
            }
            // Emphasize local chains, but still rank all cross-module data
            // and parameter dependencies in the correct direction.
            let weight = if role == "parameter" {
                "1"
            } else if owners[i] == owners[source.index()] {
                "5"
            } else {
                "2"
            };
            let src_module_port = owners[source.index()]
                .and_then(|owner| {
                    scopes[owner]
                        .outputs
                        .iter()
                        .find(|(_, index)| *index == source.index())
                })
                .map_or("none", |(name, _)| name.as_str());
            let dst_module_port = owners[i]
                .and_then(|owner| {
                    let scope = &scopes[owner];
                    source_is_outside_scope(source.index(), scope)
                        .then(|| {
                            scope
                                .inputs
                                .iter()
                                .find(|(_, index)| *index == source.index())
                        })
                        .flatten()
                })
                .map_or("none", |(name, _)| name.as_str());
            let data = [
                ("color", edge_color.into()),
                ("penwidth", edge_width.into()),
                ("weight", weight.into()),
                ("minlen", "1".into()),
                (
                    "constraint",
                    if role == "parameter" { "false" } else { "true" }.into(),
                ),
                ("vn_src_port", "output".into()),
                ("vn_dst_port", format!("input_{port}")),
                ("vn_module_src_port", src_module_port.into()),
                ("vn_module_dst_port", dst_module_port.into()),
                ("vn_semantic_src", format!("n{}", source.index())),
                ("vn_semantic_dst", format!("n{i}")),
                ("vn_visual_src", format!("n{}", source.index())),
                ("vn_role", role.into()),
                ("vn_shape", format!("{:?}", from.shape.dims())),
                ("vn_dtype", from.dtype.name().into()),
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
    if semantic_ir_sha256.len() != 64 || !semantic_ir_sha256.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err(bad("invalid canonical Semantic IR SHA-256 for DOT"));
    }

    let layout = module_layout(graph)?;
    let scopes = &layout.scopes;
    let parents = &layout.parents;
    let paths = &layout.paths;
    let owners = &layout.owners;
    let n = graph.nodes().len();
    let selected = plan
        .outputs
        .iter()
        .map(|id| id.index())
        .collect::<BTreeSet<_>>();
    // All nodes are drawn, but Lean only models reachable dependencies.
    let reachable = graph
        .dependency_scope(&plan.outputs)
        .map_err(|err| bad(err.to_string()))?;
    let gpu_status = plan
        .gpu
        .iter()
        .map(|entry| match entry {
            crate::PlannedGpu::Supported { output, .. } => (output.index(), "nvptx-supported"),
            crate::PlannedGpu::Unsupported { output, .. } => (output.index(), "unsupported"),
        })
        .collect::<BTreeMap<_, _>>();
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
        (
            "label",
            format!(
                "VeriNeural · auto architecture\n{} nodes · {} module instances · {} outputs",
                n,
                scopes.len(),
                plan.outputs.len()
            ),
        ),
        ("vn_export_version", "auto-v3".into()),
        ("vn_ir_sha256", semantic_ir_sha256.to_owned()),
        ("vn_fp32_profile", format!("{:?}", graph.policy())),
        ("vn_selected_outputs", format!("{:?}", selected)),
        (
            "vn_lean_assurance",
            "ideal-Int;conditional-source-assumptions;kernel-checked-at-build".into(),
        ),
        (
            "vn_gpu_assurance",
            "LLVM/PTX-structural;JIT-SASS-not-proven".into(),
        ),
        ("vn_connectivity", "Rust-structural-validation".into()),
        (
            "vn_unmodeled",
            "general-dtype;implicit-broadcast;machine-equivalence".into(),
        ),
    ] {
        writeln!(dot, "  {key}={};", quoted(&value)).expect("writing DOT");
    }
    struct Context<'a> {
        graph: &'a NeuralGraph,
        scopes: &'a [ModuleScope],
        parents: &'a [Option<usize>],
        owners: &'a [Option<usize>],
        paths: &'a [String],
        selected: &'a BTreeSet<usize>,
        reachable: &'a BTreeSet<usize>,
        gpu_status: &'a BTreeMap<usize, &'static str>,
    }
    fn render_scope(parent: Option<usize>, context: &Context<'_>, depth: usize, dot: &mut String) {
        let pad = "  ".repeat(depth + 1);
        for (i, node) in context
            .graph
            .nodes()
            .iter()
            .enumerate()
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
                (
                    "vn_reduction_axis",
                    match node.operation {
                        Operation::ReduceSum { axis } | Operation::ReduceMax { axis } => {
                            axis.to_string()
                        }
                        _ => String::new(),
                    },
                ),
                (
                    "vn_axis",
                    match node.operation {
                        Operation::ReduceSum { axis }
                        | Operation::ReduceMax { axis }
                        | Operation::ReverseAxis { axis }
                        | Operation::Concat { axis }
                        | Operation::SliceAxis { axis, .. }
                        | Operation::ArgMax { axis, .. }
                        | Operation::Take { axis } => axis.to_string(),
                        _ => String::new(),
                    },
                ),
                (
                    "vn_permutation",
                    match &node.operation {
                        Operation::Permute { axes } => format!("{axes:?}"),
                        _ => String::new(),
                    },
                ),
                (
                    "vn_tie_break",
                    match node.operation {
                        Operation::ArgMax { tie_break, .. } => tie_break.name().to_owned(),
                        _ => String::new(),
                    },
                ),
                (
                    "vn_custom_semantics",
                    match &node.operation {
                        Operation::Pointwise(spec) => format!(
                            "{}@{}:{}",
                            spec.id(),
                            spec.version(),
                            spec.expression().canonical()
                        ),
                        _ => String::new(),
                    },
                ),
                (
                    "vn_instance_path",
                    parent.map_or_else(|| "root".to_owned(), |p| context.paths[p].clone()),
                ),
                (
                    "vn_inputs",
                    format!(
                        "{:?}",
                        node.inputs.iter().map(|id| id.index()).collect::<Vec<_>>()
                    ),
                ),
                ("vn_shape", format!("{:?}", node.shape.dims())),
                ("vn_dtype", node.dtype.name().into()),
                ("vn_layout", "row-major".into()),
                ("vn_domain", format!("{:?}", node.domain)),
                ("vn_contract", format!("{:?}", node.contract)),
                ("vn_semantics", format!("{:?}", node.semantics)),
                (
                    "vn_selected_output",
                    context.selected.contains(&i).to_string(),
                ),
                (
                    "vn_reachable_from_outputs",
                    context.reachable.contains(&i).to_string(),
                ),
                (
                    "vn_gpu_lowering",
                    context
                        .gpu_status
                        .get(&i)
                        .copied()
                        .unwrap_or("no-selected-output-profile")
                        .to_owned(),
                ),
                (
                    "vn_proof_status",
                    if context.reachable.contains(&i) {
                        "ideal-Int;checked-in-auto-bundle"
                    } else {
                        "unselected;no-Lean-obligation"
                    }
                    .into(),
                ),
            ];
            writeln!(dot, "{pad}n{i} [{}];", attrs(&properties)).expect("writing DOT");
        }
        for (i, scope) in context
            .scopes
            .iter()
            .enumerate()
            .filter(|(i, _)| context.parents[*i] == parent)
        {
            writeln!(dot, "{pad}subgraph cluster_{} {{", scope.id).expect("writing DOT");
            let inner = format!("{pad}  ");
            let (fill, outline) = cluster_theme(depth);
            let in_ports = scope
                .inputs
                .iter()
                .map(|(name, index)| format!("{name}:n{index}"))
                .collect::<Vec<_>>()
                .join("  ");
            let out_ports = scope
                .outputs
                .iter()
                .map(|(name, index)| format!("{name}:n{index}"))
                .collect::<Vec<_>>()
                .join("  ");
            for (key, value) in [
                ("style", "rounded,filled".into()),
                ("fillcolor", fill.into()),
                ("color", outline.into()),
                ("fontname", "DejaVu Sans".into()),
                ("fontsize", "14".into()),
                ("labelloc", "t".into()),
                ("labeljust", "l".into()),
                ("margin", "22".into()),
                (
                    "label",
                    format!(
                        "{} · {}\n{} → {}",
                        scope.name.replace('_', " ").to_uppercase(),
                        scope.id,
                        in_ports,
                        out_ports,
                    ),
                ),
                ("vn_instance_path", context.paths[i].clone()),
                ("vn_definition_id", scope.name.to_owned()),
                ("vn_semantic_version", scope.version.to_owned()),
                ("vn_contract", scope.contract.clone()),
                ("vn_input_ports", format!("{:?}", scope.inputs)),
                ("vn_output_ports", format!("{:?}", scope.outputs)),
                (
                    "vn_proof_status",
                    "module-contracts-checkable;see-Lean-artifact".into(),
                ),
            ] {
                writeln!(dot, "{inner}{key}={};", quoted(&value)).expect("writing DOT");
            }
            render_scope(Some(i), context, depth + 1, dot);
            writeln!(dot, "{pad}}}").expect("writing DOT");
        }
    }
    render_scope(
        None,
        &Context {
            graph,
            scopes,
            parents,
            owners,
            paths,
            selected: &selected,
            reachable: &reachable,
            gpu_status: &gpu_status,
        },
        0,
        &mut dot,
    );
    // Inputs are not forced onto a single rank; semantic input identity is
    // recorded in vn_kind, independently of the visual layout.
    // Exactly one rendered edge for every real Semantic IR dependency.
    // Module port mappings are attributes, never extra visual relay edges.
    for (i, node) in graph.nodes().iter().enumerate() {
        for (port, input) in node.inputs.iter().enumerate() {
            let source = graph.node(*input).map_err(|err| bad(err.to_string()))?;
            let (color, width) = edge_theme("data", port);
            let transform = match node.operation {
                Operation::BiasAdd if port == 1 => "row-wise-broadcast",
                Operation::Reshape(_) => "zero-copy-reshape",
                Operation::Permute { .. } => "axis-permutation",
                Operation::BroadcastTo(_) => "explicit-broadcast",
                Operation::ReduceMax { .. } => "axis-reduction-max",
                Operation::ReverseAxis { .. } => "axis-reversal",
                Operation::Concat { .. } => "axis-concatenation",
                Operation::SliceAxis { .. } => "axis-slice",
                Operation::ArgMax { .. } => "argmax-indexing",
                Operation::Take { .. } if port == 1 => "index-selection",
                Operation::Take { .. } => "axis-selection",
                Operation::MatMul => "matrix-product-indexing",
                _ => "identity",
            };
            let module_dst_port = owners[i]
                .and_then(|owner| {
                    let scope = &scopes[owner];
                    source_is_outside_scope(input.index(), scope)
                        .then(|| scope.inputs.iter().find(|(_, id)| *id == input.index()))
                })
                .flatten()
                .map_or("none", |(name, _)| name.as_str());
            let module_src_port = owners[input.index()]
                .and_then(|owner| {
                    scopes[owner]
                        .outputs
                        .iter()
                        .find(|(_, index)| *index == input.index())
                })
                .map_or("none", |(name, _)| name.as_str());
            let properties = [
                ("color", color.into()),
                ("penwidth", width.into()),
                (
                    "weight",
                    if owners[i] == owners[input.index()] {
                        "5".into()
                    } else {
                        "2".into()
                    },
                ),
                ("minlen", "1".into()),
                ("constraint", "true".into()),
                ("vn_src_port", "output".into()),
                ("vn_dst_port", format!("input_{port}")),
                ("vn_module_src_port", module_src_port.into()),
                ("vn_module_dst_port", module_dst_port.into()),
                ("vn_semantic_src", format!("n{}", input.index())),
                ("vn_semantic_dst", format!("n{i}")),
                ("vn_visual_src", format!("n{}", input.index())),
                ("vn_order", port.to_string()),
                ("vn_shape", format!("{:?}", source.shape.dims())),
                ("vn_dtype", source.dtype.name().into()),
                ("vn_role", "data".into()),
                ("vn_transform", transform.into()),
                ("vn_constraint", "source-shape-checked".into()),
            ];
            writeln!(
                dot,
                "  n{} -> n{i} [{}];",
                input.index(),
                attrs(&properties)
            )
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
        struct Activation {
            contract: MathContract,
        }
        impl NeuralModule for Activation {
            fn name(&self) -> &'static str {
                "consumer_activation"
            }
            fn contract(&self) -> &MathContract {
                &self.contract
            }
            fn build(
                &self,
                graph: &mut NeuralGraph,
                inputs: &[NodeId],
            ) -> Result<NodeId, NeuralError> {
                graph.relu(inputs[0])
            }
        }
        struct Step {
            contract: MultiOutputContract,
            child: Activation,
        }
        impl MultiOutputModule for Step {
            fn name(&self) -> &'static str {
                "consumer_step"
            }
            fn contract(&self) -> &MultiOutputContract {
                &self.contract
            }
            fn build(
                &self,
                graph: &mut NeuralGraph,
                inputs: &[NodeId],
            ) -> Result<Vec<NodeId>, NeuralError> {
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
                .plan(&graph, &outputs, &[], GpuRequirement::AllOutputs)
                .unwrap();
            (graph, plan)
        }

        let (g1, p1) = prepare();
        let (g2, p2) = prepare();
        let hash = "a".repeat(64);
        let dot = auto_architecture_dot(&g1, &p1, &hash).unwrap();
        assert_eq!(dot, auto_architecture_dot(&g2, &p2, &hash).unwrap());
        assert!(dot.contains(&format!(r#"vn_ir_sha256="{hash}""#)));
        assert!(dot.contains("subgraph cluster_m0"));
        assert!(dot.contains("subgraph cluster_m1"));
        assert!(dot.contains("next_state"));
        assert!(dot.contains("readout"));
        assert!(dot.contains("vn_src_port=\"output\""));
        assert!(dot.contains("vn_module_src_port="));
        assert!(dot.contains("vn_module_dst_port="));
        assert!(!dot.contains("rank=same;"));
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
        assert!(!a.dot.contains("rank=same;"));
        assert!(a.dot.contains("splines=spline"));
        assert!(a.dot.contains("concentrate=false"));
        assert!(a.dot.contains("mclimit=4"));
        assert!(a.dot.contains("nodesep=0.75"));
        assert!(a.dot.contains("ranksep=1.15"));
        assert!(!a.dot.contains("tailport="));
        assert!(!a.dot.contains("headport="));
        assert!(a.dot.contains("constraint=\"false\""));
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
        let changed_snapshot = snapshot_architecture(&g1, o1, &p1, &changed).unwrap();
        assert_eq!(a.ir_sha256, changed_snapshot.ir_sha256);
        assert_ne!(a.dot, changed_snapshot.dot);
        changed.input_roles.insert(20, "parameter".into());
        assert!(snapshot_architecture(&g1, o1, &p1, &changed).is_err());
    }
    #[test]
    fn snapshot_png_replay_detects_image_tampering() {
        if Command::new("dot").arg("-V").output().is_err() {
            // Graphviz is supplied by the project's pinned Nix shell.
            return;
        }
        let mut graph = NeuralGraph::default();
        let input = graph.input(Shape::new(vec![2]).unwrap(), Domain::Finite);
        let output = graph.relu(input).unwrap();
        let program = GpuEngine::default().compile(&graph, output).unwrap();
        let metadata = ArchitectureMetadata {
            id: "png-replay".into(),
            version: "0.1.0".into(),
            input_roles: BTreeMap::from([(input.index(), "input".into())]),
            global_property: "architecture_global_nonnegative".into(),
        };
        let snapshot = snapshot_architecture(&graph, output, &program, &metadata).unwrap();
        let root = std::env::temp_dir().join(format!(
            "verineural-png-replay-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::create_dir_all(&root).unwrap();
        snapshot.write(&root).unwrap();
        snapshot.verify(&root).unwrap();
        fs::write(root.join("architecture.png"), b"not the Graphviz image").unwrap();
        assert!(snapshot.verify(&root).is_err());
        fs::remove_dir_all(&root).unwrap();
    }
}
