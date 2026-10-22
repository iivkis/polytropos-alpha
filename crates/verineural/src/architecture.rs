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

use crate::{GpuProgram, NeuralGraph, NodeId, Operation, TranslationPlan};

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
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn()?;
    process.stdin.take().ok_or_else(|| bad("sha256sum stdin unavailable"))?
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
    pairs.iter().map(|(key,value)| format!("{key}={}", quoted(value)))
        .collect::<Vec<_>>().join(", ")
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
            .arg("-Tpng").arg(root.join("architecture.dot"))
            .arg("-o").arg(&tmp).output()?;
        if !render.status.success() {
            let _ = fs::remove_file(&tmp);
            return Err(bad(format!(
                "Graphviz failed: {}", String::from_utf8_lossy(&render.stderr)
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
            return Err(bad("architecture IR or DOT disagrees with current Rust graph"));
        }
        if !self.dot.contains(&format!("vn_ir_sha256={}", quoted(&self.ir_sha256))) {
            return Err(bad("architecture DOT lacks matching Semantic IR fingerprint"));
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
    graph: &NeuralGraph, output: NodeId, plan: &GpuProgram, metadata: &ArchitectureMetadata,
) -> io::Result<ArchitectureSnapshot> {
    graph.node(output).map_err(|err| bad(err.to_string()))?;
    if plan.output().index() != output.index() {
        return Err(bad("GPU program output differs from requested architecture output"));
    }
    for v in [&metadata.id, &metadata.version] {
        if v.is_empty() || !v.chars().all(|c| c.is_ascii_alphanumeric() || "-._".contains(c)) {
            return Err(bad("architecture id/version must use ASCII alphanumeric, - . _"));
        }
    }
    for (&idx, role) in &metadata.input_roles {
        if !matches!(graph.nodes().get(idx).map(|node| &node.operation), Some(Operation::Input))
            || !matches!(role.as_str(), "input" | "parameter")
        {
            return Err(bad("unsupported input role or non-input node role assignment"));
        }
    }
    if graph.nodes().iter().enumerate().any(|(idx,node)|
        matches!(node.operation, Operation::Input) && !metadata.input_roles.contains_key(&idx)
    ) {
        return Err(bad("every architecture source input needs an explicit input/parameter role"));
    }
    if metadata.global_property != "architecture_global_nonnegative" {
        return Err(bad("only explicit, proved ideal-Int global nonnegativity is supported"));
    }
    let gpu = plan.translation_plan().map_err(|e| bad(e.to_string()))?;
    let mut ir = format!(
        "verineural-architecture-ir-v1\nid={}\nversion={}\noutput={}\nfp32={:?}\nproperty={}\n",
        metadata.id, metadata.version, output.index(), graph.policy(),
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
        writeln!(ir,"module={i};name={};inputs={:?};body_start={};output={};requires={:?};ensures={:?}",
            module.name,module.inputs.iter().map(|id|id.index()).collect::<Vec<_>>(),
            module.body_start,module.output.index(),module.contract.requires,module.contract.ensures,
        ).expect("writing to string");
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
        let mut candidates: Vec<usize> = mods.iter().enumerate()
            .filter_map(|(j, other)| {
                (i != j && other.body_start <= first && other.output.index() >= last
                    && (other.body_start < first || other.output.index() > last))
                    .then_some(j)
            }).collect();
        candidates.sort_by_key(|&j| (mods[j].output.index()-mods[j].body_start,j));
        parents[i] = candidates.first().copied();
        for (j,other) in mods.iter().enumerate() {
            if i != j && first <= other.output.index() && other.body_start <= last {
                let contains = (first <= other.body_start && last >= other.output.index())
                    || (other.body_start <= first && other.output.index() >= last);
                if !contains { return Err(bad("crossing module body intervals")); }
            }
        }
    }
    let mut paths = vec![String::new();mods.len()];
    fn module_path(i:usize, parents:&[Option<usize>], names:&[crate::ModuleInvocation])->String {
        let local = format!("m{i}_{}",names[i].name);
        parents[i].map_or_else(||local.clone(),|p|format!("{}/{}",module_path(p,parents,names),local))
    }
    for (i,path) in paths.iter_mut().enumerate() { *path=module_path(i,&parents,mods); }
    let mut owners = vec![None;graph.nodes().len()];
    for (index,slot) in owners.iter_mut().enumerate() {
        *slot=(0..mods.len()).filter(|&i|
            index >= mods[i].body_start && index <= mods[i].output.index())
            .min_by_key(|&i|(mods[i].output.index()-mods[i].body_start,i));
    }
    let mut reachable = BTreeSet::new();
    let mut pending = vec![output];
    while let Some(id)=pending.pop() {
        if reachable.insert(id.index()) {
            pending.extend(graph.node(id).map_err(|e|bad(e.to_string()))?.inputs.iter().copied());
        }
    }
    let kernel_map: BTreeMap<_,_> = gpu.kernels.iter().map(|k|(k.node,k)).collect();
    let fused: BTreeSet<_> = plan.fused_relus().values().copied()
        .chain(plan.fused_pointwise_chains().values().flat_map(|x|x.iter().copied())).collect();
    let mut dot = String::from("digraph verineural {\n  graph [rankdir=LR, compound=true];\n  node [shape=box, fontsize=10];\n");
    for (key,value) in [
        ("vn_export_version","1".to_owned()),("vn_architecture_id",metadata.id.clone()),
        ("vn_architecture_version",metadata.version.clone()),
        ("vn_ir_sha256",digest.clone()),("vn_fp32_profile",format!("{:?}",graph.policy())),
        ("vn_global_property",metadata.global_property.clone()),
        ("vn_global_proof","Lean-check-required".into()),
        ("vn_gpu_assurance","structural-LLVM-PTX-validation;JIT-SASS-trusted".into()),
        ("vn_architecture_output",format!("n{}",output.index())),
        ("vn_supported_roles","input,parameter,primitive,module-boundary,output".into()),
        ("vn_unmodeled","state,control,training".into()),
    ] { writeln!(dot,"  {key}={};",quoted(&value)).expect("writing to string"); }

    fn render_scope(
        parent:Option<usize>,graph:&NeuralGraph,owners:&[Option<usize>],
        parents:&[Option<usize>],paths:&[String],reachable:&BTreeSet<usize>,
        roles:&BTreeMap<usize,String>,kernel_map:&BTreeMap<usize,&crate::TranslationKernel>,
        fused:&BTreeSet<usize>,output:NodeId,depth:usize,dot:&mut String,
    ) {
        let indent="  ".repeat(depth+1);
        for (index,node) in graph.nodes().iter().enumerate().filter(|(i,_)|owners[*i]==parent) {
            let dims=format!("{:?}",node.shape.dims());
            let kind=if index==output.index(){"architecture_output"}else if matches!(node.operation,Operation::Input){
                roles.get(&index).map(String::as_str).unwrap_or("input")
            }else{"primitive"};
            let kernel=kernel_map.get(&index);
            let lowering=kernel.map_or_else(|| if fused.contains(&index){"fused-into-consumer"}else if matches!(node.operation,Operation::Reshape(_)){"zero-copy-alias"}else{"not-lowered"}, |_|"kernel");
            let policy=format!("{:?}",node.semantics.fp32);
            let properties=[
                ("label",format!("n{index} {}\\n{dims}",node.operation.name())),
                ("vn_id",format!("n{index}")),
                ("vn_kind",kind.into()),
                ("vn_op",node.operation.name().into()),
                ("vn_instance_path",parent.map_or_else(||"root".to_owned(),|p|paths[p].clone())),
                ("vn_inputs",format!("{:?}",node.inputs.iter().map(|n|n.index()).collect::<Vec<_>>())),
                ("vn_shape",dims),
                ("vn_dtype","binary32".into()),
                ("vn_layout","row-major".into()),
                ("vn_domain",format!("{:?}",node.domain)),
                ("vn_semantics",format!("{:?}",node.semantics)),
                ("vn_requires",format!("{:?}",node.contract.requires)),
                ("vn_ensures",format!("{:?}",node.contract.ensures)),
                ("vn_fp32_policy",policy),
                ("vn_reachable_from_output",reachable.contains(&index).to_string()),
                ("vn_gpu_lowering",lowering.into()),
                ("vn_gpu_kernel",kernel.map_or_else(||"none".into(),|k|format!("vn_node_{}",k.node))),
                ("vn_gpu_memory",kernel.map_or_else(||"none".into(),|k|format!("reads={:?},writes={}",k.reads,k.writes))),
                ("vn_assurance","ideal-Int-Lean-conditional;FP32-test-only;JIT-trusted".into()),
            ];
            writeln!(dot,"{indent}n{index} [{}];",attrs(&properties)).expect("writing to string");
        }
        for (i,module) in graph.modules().iter().enumerate().filter(|(i,_)|parents[*i]==parent) {
            writeln!(dot,"{indent}subgraph cluster_m{i} {{").expect("writing to string");
            let inner=format!("{indent}  ");
            writeln!(dot,"{inner}label={};",quoted(&format!("{} (m{i})",module.name))).expect("writing to string");
            for (key,value) in [
                ("vn_instance_path",paths[i].clone()),
                ("vn_definition_id",module.name.into()),
                ("vn_module_inputs",format!("{:?}",module.inputs.iter().map(|n|n.index()).collect::<Vec<_>>())),
                ("vn_module_output",format!("n{}",module.output.index())),
                ("vn_requires",format!("{:?}",module.contract.requires)),
                ("vn_ensures",format!("{:?}",module.contract.ensures)),
                ("vn_proof_status","conditional-ideal-Int;Lean-check-required".into()),
            ] { writeln!(dot,"{inner}{key}={};",quoted(&value)).expect("writing to string"); }
            render_scope(Some(i),graph,owners,parents,paths,reachable,roles,kernel_map,fused,output,depth+1,dot);
            writeln!(dot,"{indent}}}").expect("writing to string");
        }
    }
    render_scope(None,graph,&owners,&parents,&paths,&reachable,&metadata.input_roles,&kernel_map,&fused,output,0,&mut dot);
    // Edges are drawn globally to expose captures, module ports and sharing.
    for (i,node) in graph.nodes().iter().enumerate() {
        for (port,source) in node.inputs.iter().enumerate() {
            let from=graph.node(*source).map_err(|e|bad(e.to_string()))?;
            let transform=match node.operation {
                Operation::BiasAdd if port==1=>"row-wise-broadcast",
                Operation::Reshape(_)=>"zero-copy-reshape",
                Operation::MatMul=>"matrix-product-indexing",
                _=>"identity",
            };
            let role=metadata.input_roles.get(&source.index())
                .map(String::as_str).filter(|r|*r=="parameter").unwrap_or("data");
            let data=[
                ("vn_src_port","output".into()),
                ("vn_dst_port",format!("input_{port}")),
                ("vn_role",role.into()),
                ("vn_shape",format!("{:?}",from.shape.dims())),
                ("vn_dtype","binary32".into()),
                ("vn_transform",transform.into()),
                ("vn_order",port.to_string()),
                ("vn_constraint","source-shape-checked".into()),
            ];
            writeln!(dot,"  n{} -> n{i} [{}];",source.index(),attrs(&data)).expect("writing to string");
        }
    }
    for (i,module) in mods.iter().enumerate() {
        for (port,source) in module.inputs.iter().enumerate() {
            let properties=[
                ("style","dashed".into()),("constraint","false".into()),
                ("vn_role","module-input-port".into()),
                ("vn_dst_port",format!("m{i}:input_{port}")),
                ("vn_instance_path",paths[i].clone()),
            ];
            // Edge to module output is annotation only, not an IR dependency.
            writeln!(dot,"  n{} -> n{} [{}];",source.index(),module.output.index(),attrs(&properties)).expect("writing to string");
        }
    }
    dot.push_str("}\n");
    Ok(ArchitectureSnapshot { semantic_ir:ir,dot,ir_sha256:digest })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Domain, GpuEngine, Shape};

    #[test]
    fn graph_id_independent_complete_export_and_detects_tampering() {
        fn build()->(NeuralGraph,NodeId,GpuProgram) {
            let mut g=NeuralGraph::default();
            let a=g.input(Shape::new(vec![2]).unwrap(),Domain::Finite);
            let b=g.input(Shape::new(vec![2]).unwrap(),Domain::Finite);
            let _unreachable=g.add(a,b).unwrap();
            let product=g.mul(a,b).unwrap();
            let program=GpuEngine::default().compile(&g,product).unwrap();
            (g,product,program)
        }
        let meta=ArchitectureMetadata {
            id:"coupled-cell".into(),version:"0.1.0".into(),
            input_roles:BTreeMap::from([(0,"input".into()),(1,"parameter".into())]),
            global_property:"architecture_global_nonnegative".into(),
        };
        let (g1,o1,p1)=build();let (g2,o2,p2)=build();
        let a=snapshot_architecture(&g1,o1,&p1,&meta).unwrap();
        let b=snapshot_architecture(&g2,o2,&p2,&meta).unwrap();
        assert_eq!(a.semantic_ir,b.semantic_ir);
        assert_eq!(a.dot,b.dot);
        assert!(a.dot.contains("vn_reachable_from_output=\"false\""));
        assert!(a.dot.contains("vn_ir_sha256="));
        assert_eq!(a.dot.matches("vn_id=").count(),g1.nodes().len());
        let mut changed=meta.clone();changed.version="0.2.0".into();
        assert_ne!(a.ir_sha256,snapshot_architecture(&g1,o1,&p1,&changed).unwrap().ir_sha256);
        changed.input_roles.insert(20,"parameter".into());
        assert!(snapshot_architecture(&g1,o1,&p1,&changed).is_err());
    }
}
