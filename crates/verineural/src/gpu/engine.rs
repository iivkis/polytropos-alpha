//! GPU compilation, validation and CUDA Graph session preparation.

use super::*;

impl GpuEngine {
    /// CUDA architecture must be an explicit LLVM -mcpu value (e.g. sm_86).
    pub fn new(llc: impl Into<PathBuf>, arch: impl Into<String>) -> Self {
        Self {
            llc: llc.into(),
            arch: arch.into(),
            launch: GpuLaunchConfig::default(),
            codegen_opt_level: GpuCodegenOptLevel::O0,
            allocation_slab: false,
        }
    }

    /// Select a checked launch policy. Fusion remains experimental.
    pub fn with_launch_config(mut self, launch: GpuLaunchConfig) -> Self {
        self.launch = launch;
        self
    }

    /// Experimental NVPTX codegen profile; O0 remains the safe default.
    /// The emitted PTX must still pass translation validation before execution.
    pub fn with_codegen_opt_level(mut self, level: GpuCodegenOptLevel) -> Self {
        self.codegen_opt_level = level;
        self
    }

    /// Opt into one CUDA allocation containing disjoint tensor buffers.
    /// Physical addresses change but Semantic IR, arithmetic and kernel
    /// ordering do not. Disabled by default pending GPU parity/benchmarks.
    /// The entire live graph must fit in one device allocation; allocation
    /// failures are returned, never silently retried using a different plan.
    pub fn with_allocation_slab(mut self, enabled: bool) -> Self {
        self.allocation_slab = enabled;
        self
    }

    /// Backward-compatible single-output entrypoint, using the same shared
    /// compilation path as multi-output graphs.
    pub fn compile(&self, graph: &NeuralGraph, output: NodeId) -> Result<GpuProgram, GpuError> {
        self.compile_outputs(graph, &[output])
    }

    /// Compile the union of all requested outputs into one LLVM/PTX module.
    /// No duplicated producer kernels, no disconnected output slices, and
    /// no fusion that discards an explicitly selected observable output.
    pub fn compile_outputs(
        &self,
        graph: &NeuralGraph,
        outputs: &[NodeId],
    ) -> Result<GpuProgram, GpuError> {
        if outputs.is_empty() {
            return Err(GpuError::InvalidInput {
                node: 0,
                reason: "multi-output GPU program requires at least one output",
            });
        }
        let mut seen_outputs = BTreeSet::new();
        for &id in outputs {
            graph.node(id)?;
            if !seen_outputs.insert(id.index()) {
                return Err(GpuError::InvalidInput {
                    node: id.index(),
                    reason: "multi-output GPU program has duplicate output",
                });
            }
        }
        let output = outputs[0];
        let mut reachable = BTreeSet::new();
        let mut pending = outputs.to_vec();
        while let Some(id) = pending.pop() {
            let node = graph.node(id)?;
            if !reachable.insert(id.index()) {
                continue;
            }
            if node.shape.element_count() > i32::MAX as usize {
                return Err(GpuError::UnsupportedSize {
                    elements: node.shape.element_count(),
                });
            }
            if !matches!(
                node.operation,
                Operation::Input
                    | Operation::Add
                    | Operation::Mul
                    | Operation::MatMul
                    | Operation::BiasAdd
                    | Operation::Relu
                    | Operation::Permute { .. }
                    | Operation::ReverseAxis { .. }
                    | Operation::Concat { .. }
                    | Operation::SliceAxis { .. }
                    | Operation::BroadcastTo(_)
                    | Operation::ArgMax { .. }
                    | Operation::Take { .. }
                    | Operation::Reshape(_)
                    | Operation::Pointwise(_)
            ) {
                return Err(GpuError::UnsupportedOperation {
                    node: id.index(),
                    operation: node.operation.name(),
                });
            }
            if let Operation::Pointwise(ref spec) = node.operation {
                let kind = supported_pointwise(spec);
                if kind.is_none()
                    || (matches!(kind, Some(TranslationKind::ExpApprox))
                        && !self.launch.approximate_exp())
                {
                    return Err(GpuError::UnsupportedOperation {
                        node: id.index(),
                        operation: node.operation.name(),
                    });
                }
            }
            pending.extend(node.inputs.iter().copied());
        }
        let live_nodes = reachable.into_iter().collect::<Vec<_>>();
        let source_inputs = live_nodes
            .iter()
            .filter_map(|&index| {
                let node = &graph.nodes()[index];
                matches!(node.operation, Operation::Input).then_some(node.id)
            })
            .collect::<Vec<_>>();
        // A producer can be skipped only when its sole reachable use is the
        // consuming Relu. Shared DAG nodes and Reshape aliases are left intact.
        let mut uses = vec![0usize; graph.nodes().len()];
        for &index in &live_nodes {
            for id in &graph.nodes()[index].inputs {
                uses[id.index()] += 1;
            }
        }
        // Selected roots are observable even when an in-graph consumer also
        // reads them. Never fold one into a different output's kernel.
        for &selected in outputs {
            uses[selected.index()] += 1;
        }
        // A materialized additional output may have later DAG consumers.
        // It can move to a later kernel only if no live consumer executes
        // before that kernel. This is independent of output observability.
        let mut earliest_consumer = vec![usize::MAX; graph.nodes().len()];
        for &index in &live_nodes {
            for id in &graph.nodes()[index].inputs {
                earliest_consumer[id.index()] = earliest_consumer[id.index()].min(index);
            }
        }
        let mut fused_relus = BTreeMap::new();
        let mut fused_pointwise_chains = BTreeMap::new();
        let mut fused_mul_adds = BTreeMap::new();
        let mut fused_mul_add_muls = BTreeMap::new();
        let mut fused_binary_pairs = BTreeMap::new();
        let mut fused_sqrt_add_divs = BTreeMap::new();
        let mut fused_mul_mul_add_divs = BTreeMap::new();
        let mut fused_mul_diamond_adds = BTreeMap::new();
        let mut fused_dual_outputs = BTreeMap::new();
        if self.launch.relu_fusion == ReluFusion::Experimental {
            for &index in &live_nodes {
                let node = &graph.nodes()[index];
                if !matches!(node.operation, Operation::Relu) {
                    continue;
                }
                let producer_id = node.inputs[0];
                let producer = graph.node(producer_id)?;
                if uses[producer_id.index()] != 1 {
                    continue;
                }
                match producer.operation {
                    Operation::Add | Operation::Mul => {
                        fused_relus.insert(index, producer_id.index());
                    }
                    Operation::BiasAdd => {
                        fused_relus.insert(index, producer_id.index());
                        let inner_id = producer.inputs[0];
                        let inner = graph.node(inner_id)?;
                        if uses[inner_id.index()] == 1
                            && matches!(inner.operation, Operation::Add | Operation::Mul)
                        {
                            fused_pointwise_chains
                                .insert(index, vec![inner_id.index(), producer_id.index()]);
                        }
                    }
                    _ => {}
                }
            }
        }
        let mut skipped = fused_relus.values().copied().collect::<BTreeSet<_>>();
        for chain in fused_pointwise_chains.values() {
            skipped.extend(chain.iter().copied());
        }
        if self.launch.pointwise_fusion() {
            // A four-stage rounded pointwise chain, limited by four
            // full-shape external inputs and explicit reuse of a multiplicand:
            // Div(Add(Mul(Mul(a,b), a|b), c), d).
            // Walk roots in reverse topological order and reserve all
            // exclusive intermediates before selecting shorter rewrites.
            for &index in live_nodes.iter().rev() {
                let root = &graph.nodes()[index];
                let div = matches!(&root.operation, Operation::Pointwise(spec)
                    if supported_pointwise(spec) == Some(TranslationKind::DivRne));
                if !div || skipped.contains(&index) {
                    continue;
                }
                let sum = graph.node(root.inputs[0])?;
                if !matches!(sum.operation, Operation::Add)
                    || uses[sum.id.index()] != 1
                    || skipped.contains(&sum.id.index())
                    || sum.shape != root.shape
                {
                    continue;
                }
                'candidate: for add_slot in 0..2 {
                    let second = graph.node(sum.inputs[add_slot])?;
                    if !matches!(second.operation, Operation::Mul)
                        || uses[second.id.index()] != 1
                        || skipped.contains(&second.id.index())
                        || second.shape != root.shape
                    {
                        continue;
                    }
                    for middle_slot in 0..2 {
                        let first = graph.node(second.inputs[middle_slot])?;
                        if !matches!(first.operation, Operation::Mul)
                            || uses[first.id.index()] != 1
                            || skipped.contains(&first.id.index())
                            || first.shape != root.shape
                        {
                            continue;
                        }
                        let externals = [
                            first.inputs[0],
                            first.inputs[1],
                            sum.inputs[1 - add_slot],
                            root.inputs[1],
                        ];
                        if externals
                            .iter()
                            .any(|id| graph.nodes()[id.index()].shape != root.shape)
                        {
                            continue;
                        }
                        if let Some(reused) = first
                            .inputs
                            .iter()
                            .position(|&id| id == second.inputs[1 - middle_slot])
                        {
                            fused_mul_mul_add_divs.insert(
                                index,
                                (
                                    first.id.index(),
                                    second.id.index(),
                                    sum.id.index(),
                                    reused,
                                    middle_slot,
                                    add_slot,
                                ),
                            );
                            skipped.extend([first.id.index(), second.id.index(), sum.id.index()]);
                            break 'candidate;
                        }
                    }
                }
            }
            // Opt-in architecture-neutral strict-order fusion:
            // Sqrt(x) -> Add(sqrt(x), epsilon) -> Div(numerator, sum).
            // Sources must have exactly the same shape and both internal
            // nodes exactly one reachable use (including selected outputs).
            for &index in live_nodes.iter().rev() {
                let root = &graph.nodes()[index];
                let is_div = matches!(&root.operation, Operation::Pointwise(spec)
                    if supported_pointwise(spec) == Some(TranslationKind::DivRne));
                if !is_div || skipped.contains(&index) {
                    continue;
                }
                let middle = graph.node(root.inputs[1])?;
                if !matches!(middle.operation, Operation::Add)
                    || uses[middle.id.index()] != 1
                    || skipped.contains(&middle.id.index())
                    || root.shape != middle.shape
                {
                    continue;
                }
                for slot in 0..2 {
                    let inner = graph.node(middle.inputs[slot])?;
                    let is_sqrt = matches!(&inner.operation, Operation::Pointwise(spec)
                        if supported_pointwise(spec) == Some(TranslationKind::SqrtRne));
                    if !is_sqrt
                        || uses[inner.id.index()] != 1
                        || skipped.contains(&inner.id.index())
                        || inner.shape != root.shape
                        || [inner.inputs[0], middle.inputs[1 - slot], root.inputs[0]]
                            .iter()
                            .any(|&id| graph.nodes()[id.index()].shape != root.shape)
                    {
                        continue;
                    }
                    fused_sqrt_add_divs.insert(index, (inner.id.index(), middle.id.index(), slot));
                    skipped.insert(inner.id.index());
                    skipped.insert(middle.id.index());
                    break;
                }
            }
            // Four-stage pointwise diamond: Add(Mul(a,b), Mul(Mul(c,d), c|d)).
            // The root Add may be selected, unlike its three Mul producers.
            for &index in live_nodes.iter().rev() {
                let root = &graph.nodes()[index];
                if !matches!(root.operation, Operation::Add) || skipped.contains(&index) {
                    continue;
                }
                'diamond: for left_slot in 0..2 {
                    let left = graph.node(root.inputs[left_slot])?;
                    let second = graph.node(root.inputs[1 - left_slot])?;
                    if !matches!(left.operation, Operation::Mul)
                        || !matches!(second.operation, Operation::Mul)
                        || uses[left.id.index()] != 1
                        || uses[second.id.index()] != 1
                        || skipped.contains(&left.id.index())
                        || skipped.contains(&second.id.index())
                        || left.id == second.id
                        || left.shape != root.shape
                        || second.shape != root.shape
                    {
                        continue;
                    }
                    for right_slot in 0..2 {
                        let first = graph.node(second.inputs[right_slot])?;
                        if !matches!(first.operation, Operation::Mul)
                            || uses[first.id.index()] != 1
                            || skipped.contains(&first.id.index())
                            || first.id == left.id
                            || first.shape != root.shape
                        {
                            continue;
                        }
                        let sources = [
                            left.inputs[0],
                            left.inputs[1],
                            first.inputs[0],
                            first.inputs[1],
                        ];
                        if sources
                            .iter()
                            .any(|&id| graph.nodes()[id.index()].shape != root.shape)
                        {
                            continue;
                        }
                        if let Some(reused) = first
                            .inputs
                            .iter()
                            .position(|&id| id == second.inputs[1 - right_slot])
                        {
                            fused_mul_diamond_adds.insert(
                                index,
                                (
                                    left.id.index(),
                                    first.id.index(),
                                    second.id.index(),
                                    reused,
                                    right_slot,
                                    left_slot,
                                ),
                            );
                            skipped.extend([left.id.index(), first.id.index(), second.id.index()]);
                            break 'diamond;
                        }
                    }
                }
            }
            // Longer chains take priority, but never cross observed values,
            // shared nodes or existing ReLU fusion boundaries.
            for &index in live_nodes.iter().rev() {
                let root = &graph.nodes()[index];
                if !matches!(root.operation, Operation::Mul) || skipped.contains(&index) {
                    continue;
                }
                'outer: for outer_slot in 0..2 {
                    let middle = graph.node(root.inputs[outer_slot])?;
                    if !matches!(middle.operation, Operation::Add)
                        || uses[middle.id.index()] != 1
                        || skipped.contains(&middle.id.index())
                        || middle.shape != root.shape
                    {
                        continue;
                    }
                    for add_slot in 0..2 {
                        let inner = graph.node(middle.inputs[add_slot])?;
                        if !matches!(inner.operation, Operation::Mul)
                            || uses[inner.id.index()] != 1
                            || skipped.contains(&inner.id.index())
                            || inner.shape != root.shape
                        {
                            continue;
                        }
                        let externals = [
                            inner.inputs[0],
                            inner.inputs[1],
                            middle.inputs[1 - add_slot],
                        ];
                        if externals
                            .iter()
                            .any(|&id| graph.nodes()[id.index()].shape != root.shape)
                        {
                            continue;
                        }
                        if let Some(reuse) = externals
                            .iter()
                            .position(|&id| id == root.inputs[1 - outer_slot])
                        {
                            fused_mul_add_muls.insert(
                                index,
                                (
                                    inner.id.index(),
                                    middle.id.index(),
                                    add_slot,
                                    outer_slot,
                                    reuse,
                                ),
                            );
                            skipped.insert(inner.id.index());
                            skipped.insert(middle.id.index());
                            break 'outer;
                        }
                    }
                }
            }
            for &index in &live_nodes {
                let node = &graph.nodes()[index];
                if !matches!(node.operation, Operation::Add) || skipped.contains(&index) {
                    continue;
                }
                for slot in 0..2 {
                    let producer = graph.node(node.inputs[slot])?;
                    if !matches!(producer.operation, Operation::Mul)
                        || uses[producer.id.index()] != 1
                        || skipped.contains(&producer.id.index())
                        || producer.shape != node.shape
                        || graph.node(node.inputs[1 - slot])?.shape != node.shape
                        || producer
                            .inputs
                            .iter()
                            .any(|&id| graph.nodes()[id.index()].shape != node.shape)
                    {
                        continue;
                    }
                    fused_mul_adds.insert(index, (producer.id.index(), slot));
                    skipped.insert(producer.id.index());
                    break;
                }
            }
            // Extend strict-order pair fusion to Add→Add, Add→Mul and
            // Mul→Mul. Existing Mul→Add has priority via fused_mul_adds.
            // No shared/selected intermediate, no alias-driven shape
            // broadcast, and no overlapping fusion roots.
            for &index in live_nodes.iter().rev() {
                let root = &graph.nodes()[index];
                if !matches!(root.operation, Operation::Add | Operation::Mul)
                    || skipped.contains(&index)
                    || fused_mul_adds.contains_key(&index)
                    || fused_mul_add_muls.contains_key(&index)
                {
                    continue;
                }
                for slot in 0..2 {
                    let inner = graph.node(root.inputs[slot])?;
                    if !matches!(inner.operation, Operation::Add | Operation::Mul)
                        || (matches!(root.operation, Operation::Add)
                            && matches!(inner.operation, Operation::Mul))
                        || uses[inner.id.index()] != 1
                        || skipped.contains(&inner.id.index())
                        || fused_mul_adds.contains_key(&inner.id.index())
                        || fused_mul_add_muls.contains_key(&inner.id.index())
                        || root.shape != inner.shape
                        || graph.nodes()[root.inputs[1 - slot].index()].shape != root.shape
                        || inner
                            .inputs
                            .iter()
                            .any(|&id| graph.nodes()[id.index()].shape != root.shape)
                    {
                        continue;
                    }
                    fused_binary_pairs.insert(index, (inner.id.index(), slot));
                    skipped.insert(inner.id.index());
                    break;
                }
            }
        }
        // Horizontal fusion of two independent live Add/Mul nodes of the same
        // shape. Both results are materialized, even for non-public nodes.
        // No consumer of the earlier result may run before the second node.
        // A bounded lookback avoids quadratic compilation on large BPTT DAGs.
        // FP32 operation order and all observable values are preserved.
        if self.launch.pointwise_fusion() {
            let candidates = if self.launch.internal_pointwise_fusion() {
                live_nodes
                    .iter()
                    .copied()
                    .filter(|&index| {
                        matches!(
                            graph.nodes()[index].operation,
                            Operation::Add | Operation::Mul
                        )
                    })
                    .collect::<Vec<_>>()
            } else {
                let mut selected = outputs.iter().map(|id| id.index()).collect::<Vec<_>>();
                selected.sort_unstable();
                selected
            };
            let mut reserved = BTreeSet::new();
            for (later_pos, &later) in candidates.iter().enumerate() {
                let second = &graph.nodes()[later];
                if !matches!(second.operation, Operation::Add | Operation::Mul)
                    || uses[later] == 0
                    || skipped.contains(&later)
                    || reserved.contains(&later)
                    || second.dtype != DType::F32
                    || second.inputs.iter().any(|id| skipped.contains(&id.index()))
                    || fused_mul_adds.contains_key(&later)
                    || fused_binary_pairs.contains_key(&later)
                    || fused_mul_add_muls.contains_key(&later)
                    || fused_mul_mul_add_divs.contains_key(&later)
                    || fused_mul_diamond_adds.contains_key(&later)
                {
                    continue;
                }
                for &earlier in candidates[..later_pos].iter().rev().take(64) {
                    let first = &graph.nodes()[earlier];
                    if reserved.contains(&earlier)
                        || skipped.contains(&earlier)
                        || uses[earlier] == 0
                        || earliest_consumer[earlier] <= later
                        || !matches!(first.operation, Operation::Add | Operation::Mul)
                        || first.dtype != DType::F32
                        || first.shape != second.shape
                        || first.inputs.iter().chain(&second.inputs).any(|id| {
                            skipped.contains(&id.index())
                                || graph.nodes()[id.index()].dtype != DType::F32
                                || graph.nodes()[id.index()].shape != second.shape
                        })
                        || fused_mul_adds.contains_key(&earlier)
                        || fused_binary_pairs.contains_key(&earlier)
                        || fused_mul_add_muls.contains_key(&earlier)
                        || fused_mul_mul_add_divs.contains_key(&earlier)
                        || fused_mul_diamond_adds.contains_key(&earlier)
                    {
                        continue;
                    }
                    fused_dual_outputs.insert(later, earlier);
                    reserved.extend([earlier, later]);
                    skipped.insert(earlier);
                    break;
                }
            }
        }
        let mut llvm_ir = String::from(
            "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64\"\n\
             target triple = \"nvptx64-nvidia-cuda\"\n\n\
             declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
             declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
             declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()\n\n",
        );
        let mut specializations = BTreeMap::<String, usize>::new();
        let mut kernel_entries = BTreeMap::new();
        for &index in &live_nodes {
            let node = &graph.nodes()[index];
            if matches!(node.operation, Operation::Input)
                || graph.storage(node.id)?.is_alias()
                || skipped.contains(&index)
            {
                continue;
            }
            let mut body = String::new();
            if let Some(&earlier) = fused_dual_outputs.get(&index) {
                body.push_str(&dual_pointwise_kernel_ir(
                    index,
                    matches!(graph.nodes()[earlier].operation, Operation::Mul),
                    matches!(node.operation, Operation::Mul),
                ));
            } else if let Some(&(_, _, _, reused, right_slot, left_slot)) =
                fused_mul_diamond_adds.get(&index)
            {
                body.push_str(&mul_diamond_add_kernel_ir(
                    index,
                    reused,
                    right_slot == 1,
                    left_slot == 1,
                ));
            } else if let Some(&(_, _, _, reused, middle_slot, add_slot)) =
                fused_mul_mul_add_divs.get(&index)
            {
                body.push_str(&mul_mul_add_div_kernel_ir(
                    index,
                    reused,
                    middle_slot == 1,
                    add_slot == 1,
                ));
            } else if let Some(&(_, _, slot)) = fused_sqrt_add_divs.get(&index) {
                body.push_str(&sqrt_add_div_kernel_ir(index, slot == 1));
            } else if let Some(&(_, _, add_slot, outer_slot, reuse)) =
                fused_mul_add_muls.get(&index)
            {
                body.push_str(&mul_add_mul_kernel_ir(
                    index,
                    add_slot == 1,
                    outer_slot == 1,
                    reuse,
                ));
            } else if let Some(&(mul, slot)) = fused_mul_adds.get(&index) {
                debug_assert!(matches!(graph.nodes()[mul].operation, Operation::Mul));
                body.push_str(&mul_add_kernel_ir(index, slot == 1));
            } else if let Some(&(producer, slot)) = fused_binary_pairs.get(&index) {
                body.push_str(&binary_chain_kernel_ir(
                    index,
                    matches!(graph.nodes()[producer].operation, Operation::Add),
                    matches!(node.operation, Operation::Add),
                    slot == 1,
                ));
            } else if let Some(chain) = fused_pointwise_chains.get(&index) {
                let inner = &graph.nodes()[chain[0]];
                let bias_add = &graph.nodes()[chain[1]];
                body.push_str(&pointwise_chain_kernel_ir(
                    index,
                    &inner.operation,
                    bias_add.shape.dims()[1],
                ));
            } else if let Some(&producer) = fused_relus.get(&index) {
                let first = &graph.nodes()[producer];
                let columns = if matches!(first.operation, Operation::BiasAdd) {
                    first.shape.dims()[1]
                } else {
                    0
                };
                body.push_str(&kernel_ir(index, &first.operation, columns, true));
            } else if matches!(node.operation, Operation::MatMul) {
                let left = &graph.nodes()[node.inputs[0].index()];
                body.push_str(&matmul_kernel_ir(
                    index,
                    left.shape.dims()[1],
                    node.shape.dims()[1],
                ));
            } else {
                match &node.operation {
                    Operation::Permute { axes } if axes.as_slice() == [1, 0] => {
                        let input = &graph.nodes()[node.inputs[0].index()];
                        if input.shape.rank() != 2 {
                            return Err(GpuError::UnsupportedOperation {
                                node: index,
                                operation: node.operation.name(),
                            });
                        }
                        body.push_str(&transpose2d_kernel_ir(
                            index,
                            input.shape.dims()[0],
                            input.shape.dims()[1],
                        ));
                    }
                    Operation::ReverseAxis { axis } => {
                        let input = &graph.nodes()[node.inputs[0].index()];
                        if input.shape.rank() != 2 || *axis > 1 {
                            return Err(GpuError::UnsupportedOperation {
                                node: index,
                                operation: node.operation.name(),
                            });
                        }
                        body.push_str(&reverse2d_kernel_ir(
                            index,
                            input.shape.dims()[0],
                            input.shape.dims()[1],
                            *axis,
                        ));
                    }
                    Operation::Concat { axis }
                        if *axis == 0
                            || (*axis == 1
                                && node.shape.rank() == 2
                                && node.shape.dims()[0] == 1) =>
                    {
                        let left = &graph.nodes()[node.inputs[0].index()];
                        body.push_str(&concat_axis0_kernel_ir(index, left.shape.element_count()));
                    }
                    Operation::SliceAxis { axis: 1, start, .. } => {
                        body.push_str(&slice_last_row_kernel_ir(index, *start));
                    }
                    Operation::BroadcastTo(_) => {
                        body.push_str(&broadcast_scalar_kernel_ir(index));
                    }
                    Operation::ArgMax {
                        axis: 1,
                        tie_break: crate::ArgMaxTieBreak::First,
                    } => {
                        let input = &graph.nodes()[node.inputs[0].index()];
                        if input.shape.rank() != 2 {
                            return Err(GpuError::UnsupportedOperation {
                                node: index,
                                operation: node.operation.name(),
                            });
                        }
                        body.push_str(&argmax_last_kernel_ir(index, input.shape.dims()[1]));
                    }
                    Operation::Take { axis: 0 } => {
                        body.push_str(&take_axis0_kernel_ir(index, node.shape.element_count()));
                    }
                    Operation::Permute { .. }
                    | Operation::Concat { .. }
                    | Operation::SliceAxis { .. }
                    | Operation::ArgMax { .. }
                    | Operation::Take { .. } => {
                        return Err(GpuError::UnsupportedOperation {
                            node: index,
                            operation: node.operation.name(),
                        });
                    }
                    _ => {
                        let columns = if matches!(node.operation, Operation::BiasAdd) {
                            node.shape.dims()[1]
                        } else {
                            0
                        };
                        body.push_str(&kernel_ir(index, &node.operation, columns, false));
                    }
                }
            }
            // The generated LLVM body, including its ABI, exact FP32
            // instructions and shape-dependent constants, defines the key.
            // Only the synthetic function name is excluded. No heuristic
            // merging of merely similar graph operations is permitted.
            let name = format!("define ptx_kernel void @vn_node_{index}(");
            let signature = body.strip_prefix(&name).ok_or(GpuError::Fp32PtxRejected {
                reason: format!("generated kernel {index} lacks expected LLVM entrypoint"),
            })?;
            if let Some(&owner) = specializations.get(signature) {
                kernel_entries.insert(index, owner);
            } else {
                specializations.insert(signature.to_owned(), index);
                kernel_entries.insert(index, index);
                llvm_ir.push_str(&body);
            }
        }
        let program = GpuProgram {
            graph: graph.clone(),
            output,
            outputs: outputs.to_vec(),
            live_nodes,
            source_inputs,
            llvm_ir,
            kernel_entries,
            launch: self.launch,
            fused_relus,
            fused_pointwise_chains,
            fused_mul_adds,
            fused_mul_add_muls,
            fused_binary_pairs,
            fused_sqrt_add_divs,
            fused_mul_mul_add_divs,
            fused_mul_diamond_adds,
            fused_dual_outputs,
        };
        // Fail closed unless every launched kernel has matching memory and
        // translation evidence and every applied optimization has valid
        // structural preconditions.
        program.capability_plan()?;
        Ok(program)
    }

    /// Write the inspectable LLVM IR and lower it with LLVM's NVPTX backend.
    /// LLVM must be available in the devShell; a nonzero exit is an error.
    pub fn lower(&self, program: &GpuProgram, dir: &Path) -> Result<PathBuf, GpuError> {
        let profile = std::env::var_os("VERINEURAL_GPU_PREPARE_PROFILE").is_some();
        let lower_started = Instant::now();
        fs::create_dir_all(dir)?;
        let ir = dir.join("graph.ll");
        let ptx = dir.join("graph.ptx");
        fs::write(&ir, program.llvm_ir())?;
        let output = Command::new(&self.llc)
            .arg("-march=nvptx64")
            .arg(format!("-mcpu={}", self.arch))
            .arg(self.codegen_opt_level.llc_flag())
            .arg("-o")
            .arg(&ptx)
            .arg(&ir)
            .output()
            .map_err(|err| GpuError::LlvmFailed {
                diagnostics: format!(
                    "cannot launch LLVM NVPTX compiler '{}': {err}; \
                     verify that it is in PATH (project devShell: `nix develop`)",
                    self.llc.display()
                ),
            })?;
        if !output.status.success() {
            return Err(GpuError::LlvmFailed {
                diagnostics: format!(
                    "status={:?}\nstdout={}\nstderr={}",
                    output.status.code(),
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                ),
            });
        }
        if profile {
            eprintln!(
                "gpu_nvptx_stage=llc elapsed_ms={:.3} artifact_dir={}",
                lower_started.elapsed().as_secs_f64() * 1000.,
                dir.display()
            );
        }
        let audit_started = Instant::now();
        if profile {
            eprintln!(
                "gpu_nvptx_stage=intern graph_kernels={} unique_functions={}",
                program.kernel_count()?,
                program.unique_kernel_count()
            );
        }
        // Translation validation must succeed before the PTX reaches CUDA JIT.
        // The generated plan is derived from Semantic IR and fusion/memory
        // contracts, while the report is checked from both emitted artifacts.
        let ptx_source = fs::read_to_string(&ptx)?;
        let plan = program.translation_plan()?;
        let validated = plan
            .validate(program.llvm_ir(), &ptx_source)
            .map_err(|err| GpuError::Fp32PtxRejected {
                reason: format!("per-kernel translation validation: {err}"),
            });
        let report = match validated {
            Ok(report) => report,
            Err(err) => {
                let _ = fs::remove_file(&ptx);
                return Err(err);
            }
        };
        fs::write(dir.join("translation.plan"), plan.to_text())?;
        fs::write(dir.join("translation.report"), report.to_text())?;
        fs::write(
            dir.join("capability.plan"),
            program.capability_plan()?.to_text(),
        )?;
        if profile {
            eprintln!(
                "gpu_nvptx_stage=translation_audit elapsed_ms={:.3} artifact_dir={}",
                audit_started.elapsed().as_secs_f64() * 1000.,
                dir.display()
            );
        }
        Ok(ptx)
    }

    pub fn prepare(
        &self,
        program: &GpuProgram,
        inputs: &[(NodeId, Tensor<f32>)],
        output_dir: &Path,
    ) -> Result<GpuExecution, GpuError> {
        // Opt-in setup attribution. Never alter kernel generation, stream
        // ordering, input bindings, FP32 arithmetic or proof claims.
        let profile = std::env::var_os("VERINEURAL_GPU_PREPARE_PROFILE").is_some();
        let total_started = Instant::now();
        let report = |phase: &str, started: Instant| {
            if profile {
                eprintln!(
                    "gpu_prepare_phase={phase} elapsed_ms={:.3} artifact_dir={}",
                    started.elapsed().as_secs_f64() * 1000.,
                    output_dir.display(),
                );
            }
        };
        let contracts = program.memory_contracts()?;
        let given = validate_inputs(program, inputs)?;
        report("preflight", total_started);
        let phase_started = Instant::now();
        let ptx_path = self.lower(program, output_dir)?;
        let ptx = fs::read(ptx_path)?;
        report("lower_and_audit", phase_started);
        let phase_started = Instant::now();
        let mut session = CudaSession::new(&ptx)?;
        report("cuda_module_load", phase_started);
        let phase_started = Instant::now();
        let mut buffers: BTreeMap<usize, u64> = BTreeMap::new();
        let mut input_addresses = BTreeMap::new();
        let skipped = program.skipped_nodes();

        // Preassign all materialized source/output allocations before any
        // CUDA Graph capture. A fused secondary output still needs storage;
        // zero-copy aliases share the exact address of their source.
        // Every checked offset belongs to the single owned CUDA allocation.
        let mut slab_addresses = BTreeMap::new();
        if self.allocation_slab {
            let mut entries = Vec::new();
            for &index in &program.live_nodes {
                if skipped.contains(&index)
                    && !program.fused_dual_outputs.values().any(|&n| n == index)
                {
                    continue;
                }
                let node = &program.graph.nodes()[index];
                if matches!(node.operation, Operation::Input)
                    || matches!(
                        program.graph.storage(node.id)?,
                        NodeStorage::OwnedContiguous { .. }
                    )
                {
                    entries.push((index, node.shape.element_count()));
                }
            }
            let sizes = entries.iter().map(|&(_, size)| size).collect::<Vec<_>>();
            let addresses = session.allocate_batch(&sizes)?;
            slab_addresses.extend(
                entries
                    .into_iter()
                    .zip(addresses)
                    .map(|((index, _), address)| (index, address)),
            );
        }
        let allocate_node = |session: &mut CudaSession, index: usize, elements: usize| {
            if self.allocation_slab {
                slab_addresses
                    .get(&index)
                    .copied()
                    .ok_or(GpuError::InvalidInput {
                        node: index,
                        reason: "missing preallocated slab tensor",
                    })
            } else {
                session.allocate(elements)
            }
        };
        for &index in &program.live_nodes {
            if skipped.contains(&index) && !program.fused_dual_outputs.values().any(|&n| n == index)
            {
                continue;
            }
            let node = &program.graph.nodes()[index];
            let storage = program.graph.storage(node.id)?;
            match (&node.operation, storage) {
                (Operation::Input, _) => {
                    let value = given[&index];
                    let ptr = allocate_node(&mut session, index, value.values().len())?;
                    session.copy_to_device(ptr, value.values())?;
                    input_addresses.insert(index, ptr);
                    buffers.insert(index, ptr);
                }
                (_, NodeStorage::Alias { source, .. }) => {
                    let ptr = buffers[&source.index()];
                    buffers.insert(index, ptr);
                }
                (
                    Operation::Add
                    | Operation::Mul
                    | Operation::MatMul
                    | Operation::BiasAdd
                    | Operation::Relu
                    | Operation::Permute { .. }
                    | Operation::ReverseAxis { .. }
                    | Operation::Concat { .. }
                    | Operation::SliceAxis { .. }
                    | Operation::BroadcastTo(_)
                    | Operation::ArgMax { .. }
                    | Operation::Take { .. }
                    | Operation::Pointwise(_),
                    NodeStorage::OwnedContiguous { .. },
                ) => {
                    let ptr = allocate_node(&mut session, index, node.shape.element_count())?;
                    buffers.insert(index, ptr);
                }
                _ => unreachable!("the compilation pass rejected unsupported nodes"),
            }
        }

        if profile {
            let materialized_tensors = buffers.values().copied().collect::<BTreeSet<_>>().len();
            eprintln!(
                "gpu_prepare_allocations mode={} materialized_tensors={} cuda_allocations={} artifact_dir={}",
                if self.allocation_slab {
                    "slab"
                } else {
                    "separate"
                },
                materialized_tensors,
                if self.allocation_slab {
                    usize::from(materialized_tensors != 0)
                } else {
                    materialized_tensors
                },
                output_dir.display(),
            );
        }
        report("allocate_and_initial_h2d", phase_started);
        let phase_started = Instant::now();
        // Validate the *allocated* buffers as well as the abstract shapes
        // before capturing any kernel into the CUDA Graph. Each output has
        // its own allocation; aliases are allowed only for zero-copy Reshape
        // inputs, never for an output being written by this kernel.
        let mut written_buffers = BTreeSet::new();
        for contract in &contracts {
            let output_ptr = buffers[&contract.kernel_node];
            session.require_allocation(output_ptr, contract.writes)?;
            for &(other, elements) in &contract.additional_writes {
                let ptr = buffers[&other];
                session.require_allocation(ptr, elements)?;
                if ptr == output_ptr || !written_buffers.insert(ptr) {
                    return Err(GpuError::InvalidInput {
                        node: contract.kernel_node,
                        reason: "multi-output CUDA kernel has overlapping output allocations",
                    });
                }
            }
            if !written_buffers.insert(output_ptr) {
                return Err(GpuError::InvalidInput {
                    node: contract.kernel_node,
                    reason: "two CUDA kernel nodes write the same output allocation",
                });
            }
            for &(source, elements) in &contract.reads {
                let input_ptr = buffers[&source];
                session.require_allocation(input_ptr, elements)?;
                if input_ptr == output_ptr
                    || contract
                        .additional_writes
                        .iter()
                        .any(|&(other, _)| input_ptr == buffers[&other])
                {
                    return Err(GpuError::InvalidInput {
                        node: contract.kernel_node,
                        reason: "in-place CUDA kernel write is not supported",
                    });
                }
            }
        }

        report("memory_contracts", phase_started);
        let kernel_count = contracts.len();
        let has_graph = kernel_count > 0;
        if has_graph {
            let phase_started = Instant::now();
            for entry in program
                .kernel_entries
                .values()
                .copied()
                .collect::<BTreeSet<_>>()
            {
                session.prepare_kernel(&format!("vn_node_{entry}"))?;
            }
            report("kernel_resolution", phase_started);
            let phase_started = Instant::now();
            session.begin_graph()?;
            for contract in &contracts {
                let index = contract.kernel_node;
                let entry = program.kernel_entries[&index];
                let inputs = contract
                    .reads
                    .iter()
                    .map(|&(node, _)| buffers[&node])
                    .collect::<Vec<_>>();
                if let Some(&(other, _)) = contract.additional_writes.first() {
                    if contract.additional_writes.len() != 1 || inputs.len() != 4 {
                        return Err(GpuError::InvalidInput {
                            node: index,
                            reason: "invalid dual-output kernel ABI",
                        });
                    }
                    session.launch4_dual(
                        &format!("vn_node_{entry}"),
                        [inputs[0], inputs[1], inputs[2], inputs[3]],
                        [buffers[&other], buffers[&index]],
                        contract.writes,
                        program.launch.threads_per_block,
                    )?;
                } else if inputs.len() == 4 {
                    session.launch4(
                        &format!("vn_node_{entry}"),
                        [inputs[0], inputs[1], inputs[2], inputs[3]],
                        buffers[&index],
                        contract.writes,
                        program.launch.threads_per_block,
                    )?;
                } else if inputs.len() == 3 {
                    session.launch3(
                        &format!("vn_node_{entry}"),
                        [inputs[0], inputs[1], inputs[2]],
                        buffers[&index],
                        contract.writes,
                        program.launch.threads_per_block,
                    )?;
                } else {
                    session.launch(
                        &format!("vn_node_{entry}"),
                        inputs[0],
                        inputs.get(1).copied().unwrap_or(0),
                        buffers[&index],
                        contract.writes,
                        program.launch.threads_per_block,
                    )?;
                }
            }
            report("graph_capture", phase_started);
            let phase_started = Instant::now();
            session.finish_graph()?;
            report("graph_instantiate", phase_started);
        }
        report("total_before_execution", total_started);

        let last = program.graph.node(program.output)?;
        let output_addresses = program
            .outputs
            .iter()
            .map(|&id| {
                let node = program.graph.node(id)?;
                let address = *buffers.get(&id.index()).ok_or(GpuError::InvalidInput {
                    node: id.index(),
                    reason: "selected output does not own a live CUDA buffer",
                })?;
                Ok((id, node.shape.clone(), address))
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        let output_address_lookup = output_addresses
            .iter()
            .enumerate()
            .map(|(slot, (id, _, _))| (id.index(), slot))
            .collect();
        Ok(GpuExecution {
            session,
            program: program.clone(),
            input_addresses,
            output_shape: last.shape.clone(),
            output_address: buffers[&program.output.index()],
            output_addresses,
            output_address_lookup,
            preserved_output_addresses: BTreeMap::new(),
            feedback_observable_outputs: None,
            feedback_snapshot_destinations: BTreeSet::new(),
            prepared_feedback: None,
            preserved_output_valid: Cell::new(false),
            execution_health: ExecutionHealth::default(),
            has_graph,
        })
    }

    /// Prepare independent executions of one compiled graph, each with its
    /// own CUDA allocations, non-blocking stream and source inventory.
    /// Validate every member's source set before preparing *any* session,
    /// so a malformed late member cannot trigger partial CUDA allocation.
    /// Each prepared session may be fed different tokens/state independently;
    /// this does not aggregate gradients or share trainable weights.
    pub fn prepare_independent(
        &self,
        program: &GpuProgram,
        batches: &[Vec<(NodeId, Tensor<f32>)>],
        output_dir: &Path,
    ) -> Result<Vec<GpuExecution>, GpuError> {
        if batches.is_empty() {
            return Err(GpuError::InvalidInput {
                node: program.output.index(),
                reason: "independent GPU batch must not be empty",
            });
        }
        for inputs in batches {
            validate_inputs(program, inputs)?;
        }
        batches
            .iter()
            .enumerate()
            .map(|(i, inputs)| {
                self.prepare(
                    program,
                    inputs,
                    &output_dir.join(format!("instance_{i:04}")),
                )
            })
            .collect()
    }

    /// Evaluate all outputs with one shared CUDA Graph capture and replay,
    /// preserving each output's Semantic IR dtype.
    pub fn execute_values(
        &self,
        program: &GpuProgram,
        inputs: &[(NodeId, Tensor<f32>)],
        output_dir: &Path,
    ) -> Result<Vec<(NodeId, TensorValue)>, GpuError> {
        let mut execution = self.prepare(program, inputs, output_dir)?;
        execution.replay()?;
        execution.read_values()
    }

    /// Backward-compatible all-F32 shared execution.
    pub fn execute_outputs(
        &self,
        program: &GpuProgram,
        inputs: &[(NodeId, Tensor<f32>)],
        output_dir: &Path,
    ) -> Result<Vec<(NodeId, Tensor<f32>)>, GpuError> {
        let mut execution = self.prepare(program, inputs, output_dir)?;
        execution.replay()?;
        execution.read_outputs()
    }

    /// Evaluate the primary output while preserving its Semantic IR dtype.
    pub fn execute_value(
        &self,
        program: &GpuProgram,
        inputs: &[(NodeId, Tensor<f32>)],
        output_dir: &Path,
    ) -> Result<TensorValue, GpuError> {
        let mut execution = self.prepare(program, inputs, output_dir)?;
        execution.replay()?;
        execution.read_value()
    }

    /// Backward-compatible F32 primary execution.
    pub fn execute(
        &self,
        program: &GpuProgram,
        inputs: &[(NodeId, Tensor<f32>)],
        output_dir: &Path,
    ) -> Result<Tensor<f32>, GpuError> {
        let mut execution = self.prepare(program, inputs, output_dir)?;
        execution.replay()?;
        execution.read_output()
    }
}

