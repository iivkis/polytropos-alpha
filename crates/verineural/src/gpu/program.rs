//! Semantic IR GPU program lowering, capability and proof/memory contracts.

use super::*;

impl GpuProgram {
    pub fn llvm_ir(&self) -> &str {
        &self.llvm_ir
    }

    pub fn source_inputs(&self) -> &[NodeId] {
        &self.source_inputs
    }

    pub fn output(&self) -> NodeId {
        self.output
    }

    /// Ordered, distinct architecture outputs served by the same kernel DAG.
    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    /// Exactly one kernel per live Semantic IR producer, even when multiple
    /// selected outputs depend on it. Zero-copy Reshape retains its alias.
    pub fn kernel_count(&self) -> Result<usize, GpuError> {
        Ok(self.memory_contracts()?.len())
    }

    /// Number of distinct compiled NVPTX functions, not CUDA Graph launches.
    /// Identical code templates share an entrypoint but never share arguments
    /// or output storage. The physical graph still has `kernel_count()` nodes.
    pub fn unique_kernel_count(&self) -> usize {
        self.kernel_entries
            .values()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
    }

    /// Function owner for a launched Semantic IR producer (not an alias).
    pub fn kernel_entry(&self, node: usize) -> Option<usize> {
        self.kernel_entries.get(&node).copied()
    }

    /// Conservative pre-JIT size of graph tensor allocations, in bytes.
    /// Exactly matches the materialized Input/OwnedContiguous allocations
    /// used by GpuEngine::prepare(), including secondary fused outputs.
    /// Excludes PTX/JIT, CUDA Graph metadata, transfers and auxiliary buffers;
    /// it is a budget guard, NOT a VRAM peak measurement.
    pub fn tensor_allocation_bytes(&self) -> Result<usize, GpuError> {
        let skipped = self.skipped_nodes();
        let mut total = 0usize;
        for &index in &self.live_nodes {
            if skipped.contains(&index) && !self.fused_dual_outputs.values().any(|&n| n == index) {
                continue;
            }
            let node = &self.graph.nodes()[index];
            if matches!(node.operation, Operation::Input)
                || matches!(
                    self.graph.storage(node.id)?,
                    NodeStorage::OwnedContiguous { .. }
                )
            {
                let elements = node.shape.element_count();
                let bytes = elements
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or(GpuError::UnsupportedSize { elements })?;
                total = total
                    .checked_add(bytes)
                    .ok_or(GpuError::UnsupportedSize { elements })?;
            }
        }
        Ok(total)
    }

    pub const fn launch_config(&self) -> GpuLaunchConfig {
        self.launch
    }

    /// (consumer Relu node, producer binary node) pairs for the selected IR rewrite.
    pub fn fused_relus(&self) -> &BTreeMap<usize, usize> {
        &self.fused_relus
    }

    pub fn fused_pointwise_chains(&self) -> &BTreeMap<usize, Vec<usize>> {
        &self.fused_pointwise_chains
    }

    pub fn fused_mul_adds(&self) -> &BTreeMap<usize, (usize, usize)> {
        &self.fused_mul_adds
    }

    pub fn fused_mul_add_muls(&self) -> &BTreeMap<usize, (usize, usize, usize, usize, usize)> {
        &self.fused_mul_add_muls
    }

    /// Other strict-order binary pointwise pairs fused into one CUDA kernel.
    pub fn fused_binary_pairs(&self) -> &BTreeMap<usize, (usize, usize)> {
        &self.fused_binary_pairs
    }

    /// Opt-in rounded Sqrt→Add→Div chains with exclusive intermediate uses.
    pub fn fused_sqrt_add_divs(&self) -> &BTreeMap<usize, (usize, usize, usize)> {
        &self.fused_sqrt_add_divs
    }

    pub fn fused_mul_mul_add_divs(
        &self,
    ) -> &BTreeMap<usize, (usize, usize, usize, usize, usize, usize)> {
        &self.fused_mul_mul_add_divs
    }

    /// Four-op pointwise DAG fusion which preserves the selected Add root.
    pub fn fused_mul_diamond_adds(
        &self,
    ) -> &BTreeMap<usize, (usize, usize, usize, usize, usize, usize)> {
        &self.fused_mul_diamond_adds
    }

    pub fn removed_kernel_count(&self) -> usize {
        self.skipped_nodes().len()
    }

    /// Bounded, multi-output, strict-RNE fusion of independent live pointwise
    /// roots, including non-public intermediates whose consumers occur later.
    /// Both buffers remain materialized; IR and observable values are unchanged.
    pub fn fused_dual_outputs(&self) -> &BTreeMap<usize, usize> {
        &self.fused_dual_outputs
    }

    fn skipped_nodes(&self) -> BTreeSet<usize> {
        let mut skipped = self.fused_relus.values().copied().collect::<BTreeSet<_>>();
        for chain in self.fused_pointwise_chains.values() {
            skipped.extend(chain.iter().copied());
        }
        skipped.extend(self.fused_mul_adds.values().map(|&(mul, _)| mul));
        skipped.extend(
            self.fused_binary_pairs
                .values()
                .map(|&(producer, _)| producer),
        );
        for &(mul, add, _, _, _) in self.fused_mul_add_muls.values() {
            skipped.insert(mul);
            skipped.insert(add);
        }
        for &(sqrt, add, _) in self.fused_sqrt_add_divs.values() {
            skipped.insert(sqrt);
            skipped.insert(add);
        }
        for &(inner, middle, add, _, _, _) in self.fused_mul_mul_add_divs.values() {
            skipped.extend([inner, middle, add]);
        }
        for &(left, first, second, _, _, _) in self.fused_mul_diamond_adds.values() {
            skipped.extend([left, first, second]);
        }
        skipped.extend(self.fused_dual_outputs.values().copied());
        skipped
    }

    /// Independently describe and check every launched kernel's element
    /// addresses. The same input selection drives CUDA graph capture.
    pub fn memory_contracts(&self) -> Result<Vec<GpuMemoryContract>, GpuError> {
        let mut result = Vec::new();
        let skipped = self.skipped_nodes();
        for &index in &self.live_nodes {
            if skipped.contains(&index) {
                continue;
            }
            let node = &self.graph.nodes()[index];
            if matches!(node.operation, Operation::Input) || self.graph.storage(node.id)?.is_alias()
            {
                continue;
            }
            let writes = node.shape.element_count();
            let (read_nodes, pattern): (Vec<usize>, GpuMemoryPattern) = if let Some(&earlier) =
                self.fused_dual_outputs.get(&index)
            {
                let first = &self.graph.nodes()[earlier];
                (
                    first
                        .inputs
                        .iter()
                        .chain(&node.inputs)
                        .map(|id| id.index())
                        .collect(),
                    GpuMemoryPattern::Pointwise,
                )
            } else if let Some(&(left, first, _, _, _, _)) = self.fused_mul_diamond_adds.get(&index)
            {
                let lhs = &self.graph.nodes()[left];
                let rhs = &self.graph.nodes()[first];
                (
                    vec![
                        lhs.inputs[0].index(),
                        lhs.inputs[1].index(),
                        rhs.inputs[0].index(),
                        rhs.inputs[1].index(),
                    ],
                    GpuMemoryPattern::Pointwise,
                )
            } else if let Some(&(inner, _, add, _, _, add_slot)) =
                self.fused_mul_mul_add_divs.get(&index)
            {
                let first = &self.graph.nodes()[inner];
                let sum = &self.graph.nodes()[add];
                (
                    vec![
                        first.inputs[0].index(),
                        first.inputs[1].index(),
                        sum.inputs[1 - add_slot].index(),
                        node.inputs[1].index(),
                    ],
                    GpuMemoryPattern::Pointwise,
                )
            } else if let Some(&(sqrt, add, slot)) = self.fused_sqrt_add_divs.get(&index) {
                let inner = &self.graph.nodes()[sqrt];
                let middle = &self.graph.nodes()[add];
                (
                    vec![
                        inner.inputs[0].index(),
                        middle.inputs[1 - slot].index(),
                        node.inputs[0].index(),
                    ],
                    GpuMemoryPattern::Pointwise,
                )
            } else if let Some(&(mul, add, slot, _, _)) = self.fused_mul_add_muls.get(&index) {
                let inner = &self.graph.nodes()[mul];
                let middle = &self.graph.nodes()[add];
                (
                    vec![
                        inner.inputs[0].index(),
                        inner.inputs[1].index(),
                        middle.inputs[1 - slot].index(),
                    ],
                    GpuMemoryPattern::Pointwise,
                )
            } else if let Some(&(mul, slot)) = self.fused_mul_adds.get(&index) {
                let producer = &self.graph.nodes()[mul];
                (
                    vec![
                        producer.inputs[0].index(),
                        producer.inputs[1].index(),
                        node.inputs[1 - slot].index(),
                    ],
                    GpuMemoryPattern::Pointwise,
                )
            } else if let Some(&(producer, slot)) = self.fused_binary_pairs.get(&index) {
                let inner = &self.graph.nodes()[producer];
                (
                    vec![
                        inner.inputs[0].index(),
                        inner.inputs[1].index(),
                        node.inputs[1 - slot].index(),
                    ],
                    GpuMemoryPattern::Pointwise,
                )
            } else if let Some(chain) = self.fused_pointwise_chains.get(&index) {
                let inner = &self.graph.nodes()[chain[0]];
                let bias = &self.graph.nodes()[chain[1]];
                (
                    vec![
                        inner.inputs[0].index(),
                        inner.inputs[1].index(),
                        bias.inputs[1].index(),
                    ],
                    GpuMemoryPattern::BiasAdd {
                        columns: node.shape.dims()[1],
                    },
                )
            } else {
                let producer = self.fused_relus.get(&index).copied().unwrap_or(index);
                let op_node = &self.graph.nodes()[producer];
                let pattern = match &op_node.operation {
                    Operation::BiasAdd => GpuMemoryPattern::BiasAdd {
                        columns: node.shape.dims()[1],
                    },
                    Operation::MatMul => {
                        let left = &self.graph.nodes()[op_node.inputs[0].index()];
                        GpuMemoryPattern::MatMul {
                            rows: left.shape.dims()[0],
                            inner: left.shape.dims()[1],
                            columns: node.shape.dims()[1],
                        }
                    }
                    Operation::Permute { axes } => {
                        let input = &self.graph.nodes()[op_node.inputs[0].index()];
                        if input.shape.rank() != 2 || axes.as_slice() != [1, 0] {
                            return Err(GpuError::UnsupportedOperation {
                                node: index,
                                operation: op_node.operation.name(),
                            });
                        }
                        GpuMemoryPattern::Transpose2D {
                            rows: input.shape.dims()[0],
                            columns: input.shape.dims()[1],
                        }
                    }
                    Operation::ReverseAxis { axis } => {
                        let input = &self.graph.nodes()[op_node.inputs[0].index()];
                        if input.shape.rank() != 2 {
                            return Err(GpuError::UnsupportedOperation {
                                node: index,
                                operation: op_node.operation.name(),
                            });
                        }
                        match axis {
                            0 => GpuMemoryPattern::ReverseRows2D {
                                rows: input.shape.dims()[0],
                                columns: input.shape.dims()[1],
                            },
                            1 => GpuMemoryPattern::ReverseColumns2D {
                                rows: input.shape.dims()[0],
                                columns: input.shape.dims()[1],
                            },
                            _ => {
                                return Err(GpuError::UnsupportedOperation {
                                    node: index,
                                    operation: op_node.operation.name(),
                                });
                            }
                        }
                    }
                    Operation::Concat { axis } => {
                        let left = &self.graph.nodes()[op_node.inputs[0].index()];
                        let right = &self.graph.nodes()[op_node.inputs[1].index()];
                        if *axis != 0
                            && !(*axis == 1
                                && left.shape.rank() == 2
                                && left.shape.dims()[0] == 1
                                && right.shape.dims()[0] == 1)
                        {
                            return Err(GpuError::UnsupportedOperation {
                                node: index,
                                operation: op_node.operation.name(),
                            });
                        }
                        GpuMemoryPattern::ConcatAxis0 {
                            left: left.shape.element_count(),
                        }
                    }
                    Operation::SliceAxis {
                        axis: 1,
                        start,
                        len,
                    } => {
                        let input = &self.graph.nodes()[op_node.inputs[0].index()];
                        if input.shape.rank() != 2
                            || input.shape.dims()[0] != 1
                            || *len != writes
                            || start
                                .checked_add(*len)
                                .is_none_or(|end| end > input.shape.dims()[1])
                        {
                            return Err(GpuError::UnsupportedOperation {
                                node: index,
                                operation: op_node.operation.name(),
                            });
                        }
                        GpuMemoryPattern::SliceLastRow {
                            start: *start,
                            source_width: input.shape.dims()[1],
                        }
                    }
                    Operation::BroadcastTo(_) => {
                        let input = &self.graph.nodes()[op_node.inputs[0].index()];
                        // Broadcasting a one-element tensor uses source
                        // address zero for every output lane, independently
                        // of output rank. General axis-wise broadcasting
                        // still has no GPU lowering and must fail closed.
                        if input.shape.element_count() != 1 {
                            return Err(GpuError::UnsupportedOperation {
                                node: index,
                                operation: op_node.operation.name(),
                            });
                        }
                        GpuMemoryPattern::BroadcastScalar
                    }
                    Operation::ArgMax { axis, tie_break } => {
                        let input = &self.graph.nodes()[op_node.inputs[0].index()];
                        if input.shape.rank() != 2
                            || *axis != 1
                            || !matches!(tie_break, crate::ArgMaxTieBreak::First)
                        {
                            return Err(GpuError::UnsupportedOperation {
                                node: index,
                                operation: op_node.operation.name(),
                            });
                        }
                        GpuMemoryPattern::ArgMaxLast {
                            width: input.shape.dims()[1],
                        }
                    }
                    Operation::Take { axis: 0 } => {
                        let data = &self.graph.nodes()[op_node.inputs[0].index()];
                        GpuMemoryPattern::TakeAxis0 {
                            lane: writes,
                            choices: data.shape.dims()[0],
                        }
                    }
                    _ => GpuMemoryPattern::Pointwise,
                };
                (
                    op_node.inputs.iter().map(|id| id.index()).collect(),
                    pattern,
                )
            };
            let required = match pattern {
                GpuMemoryPattern::Pointwise => vec![writes; read_nodes.len()],
                GpuMemoryPattern::BiasAdd { columns } => {
                    if read_nodes.len() == 3 {
                        vec![writes, writes, columns]
                    } else {
                        vec![writes, columns]
                    }
                }
                GpuMemoryPattern::MatMul {
                    rows,
                    inner,
                    columns,
                } => {
                    if rows.checked_mul(columns) != Some(writes) {
                        return Err(GpuError::InvalidInput {
                            node: index,
                            reason: "MatMul output extent does not match row-major indexing",
                        });
                    }
                    vec![
                        rows.checked_mul(inner)
                            .ok_or(GpuError::UnsupportedSize { elements: writes })?,
                        inner
                            .checked_mul(columns)
                            .ok_or(GpuError::UnsupportedSize { elements: writes })?,
                    ]
                }
                GpuMemoryPattern::Transpose2D { rows, columns } => {
                    if rows.checked_mul(columns) != Some(writes) {
                        return Err(GpuError::InvalidInput {
                            node: index,
                            reason: "transpose2d output extent mismatch",
                        });
                    }
                    vec![writes]
                }
                GpuMemoryPattern::ReverseRows2D { rows, columns }
                | GpuMemoryPattern::ReverseColumns2D { rows, columns } => {
                    if rows == 0 || columns == 0 || rows.checked_mul(columns) != Some(writes) {
                        return Err(GpuError::InvalidInput {
                            node: index,
                            reason: "reverse2d output extent mismatch",
                        });
                    }
                    vec![writes]
                }
                GpuMemoryPattern::SliceLastRow {
                    start,
                    source_width,
                } => {
                    if read_nodes.len() != 1
                        || start
                            .checked_add(writes)
                            .is_none_or(|end| end > source_width)
                    {
                        return Err(GpuError::InvalidInput {
                            node: index,
                            reason: "slice source extent mismatch",
                        });
                    }
                    vec![source_width]
                }
                GpuMemoryPattern::BroadcastScalar => vec![1],
                GpuMemoryPattern::ConcatAxis0 { left } => {
                    if read_nodes.len() != 2 {
                        return Err(GpuError::InvalidInput {
                            node: index,
                            reason: "concat_axis0 requires two source tensors",
                        });
                    }
                    let right = writes.checked_sub(left).ok_or(GpuError::InvalidInput {
                        node: index,
                        reason: "concat_axis0 left extent exceeds output",
                    })?;
                    vec![left, right]
                }
                GpuMemoryPattern::ArgMaxLast { width } => vec![
                    writes
                        .checked_mul(width)
                        .ok_or(GpuError::UnsupportedSize { elements: writes })?,
                ],
                GpuMemoryPattern::TakeAxis0 { lane, choices } => {
                    let data = &self.graph.nodes()[read_nodes[0]];
                    if lane != writes
                        || choices == 0
                        || choices.checked_mul(lane) != Some(data.shape.element_count())
                    {
                        return Err(GpuError::InvalidInput {
                            node: index,
                            reason: "take_axis0 source extent mismatch",
                        });
                    }
                    vec![data.shape.element_count(), 1]
                }
            };
            // CUDA templates form idx = block * width + lane with u32
            // wrap semantics, then test idx < length as unsigned. Bound the
            // *whole* grid so even unused trailing lanes cannot wrap.
            let width = self.launch.threads_per_block as usize;
            let launch_extent = writes.div_ceil(width).checked_mul(width);
            if !matches!(read_nodes.len(), 1..=4)
                || read_nodes.len() != required.len()
                || writes == 0
                || writes > i32::MAX as usize
                || launch_extent.is_none_or(|n| n > u32::MAX as usize)
                || read_nodes.iter().any(|&source| source >= index)
            {
                return Err(GpuError::InvalidInput {
                    node: index,
                    reason: "unsupported CUDA kernel memory access or launch bounds",
                });
            }
            if let GpuMemoryPattern::BiasAdd { columns } = pattern {
                if columns == 0 || writes % columns != 0 {
                    return Err(GpuError::InvalidInput {
                        node: index,
                        reason: "BiasAdd requires nonempty aligned rows",
                    });
                }
            }
            let mut reads = Vec::with_capacity(read_nodes.len());
            for (source, needed) in read_nodes.into_iter().zip(required) {
                if self.graph.nodes()[source].shape.element_count() != needed
                    || needed > i32::MAX as usize
                {
                    return Err(GpuError::InvalidInput {
                        node: index,
                        reason: "kernel input extent differs from its indexed access",
                    });
                }
                reads.push((source, needed));
            }
            result.push(GpuMemoryContract {
                kernel_node: index,
                reads,
                writes,
                additional_writes: self
                    .fused_dual_outputs
                    .get(&index)
                    .map_or_else(Vec::new, |&earlier| vec![(earlier, writes)]),
                pattern,
            });
        }
        Ok(result)
    }

    /// Derive the expected kernel composition from the original reachable
    /// graph, checked fusion decisions and exact memory contracts, not PTX.
    /// Each plan row is later cross-checked against LLVM and PTX per kernel.
    pub fn translation_plan(&self) -> Result<TranslationPlan, GpuError> {
        let mut kernels = Vec::new();
        for contract in self.memory_contracts()? {
            let node = &self.graph.nodes()[contract.kernel_node];
            let kind = if let Some(&earlier) = self.fused_dual_outputs.get(&contract.kernel_node) {
                TranslationKind::DualPointwise {
                    earlier,
                    first_mul: matches!(self.graph.nodes()[earlier].operation, Operation::Mul),
                    second_mul: matches!(node.operation, Operation::Mul),
                }
            } else if let Some(&(_, _, _, reused, right_slot, left_slot)) =
                self.fused_mul_diamond_adds.get(&contract.kernel_node)
            {
                TranslationKind::MulDiamondAdd {
                    reused_source: reused as u8,
                    right_mul_on_right: right_slot == 1,
                    left_mul_on_right: left_slot == 1,
                }
            } else if let Some(&(_, _, _, reused, middle_slot, add_slot)) =
                self.fused_mul_mul_add_divs.get(&contract.kernel_node)
            {
                TranslationKind::MulMulAddDiv {
                    reused_source: reused as u8,
                    multiply_on_right: middle_slot == 1,
                    add_on_right: add_slot == 1,
                }
            } else if let Some(&(_, _, slot)) = self.fused_sqrt_add_divs.get(&contract.kernel_node)
            {
                TranslationKind::SqrtAddDiv {
                    sqrt_on_right: slot == 1,
                }
            } else if let Some(&(_, _, add_slot, outer_slot, reuse)) =
                self.fused_mul_add_muls.get(&contract.kernel_node)
            {
                TranslationKind::MulAddMul {
                    add_mul_on_right: add_slot == 1,
                    mul_add_on_right: outer_slot == 1,
                    reused_source: reuse as u8,
                }
            } else if let Some(&(_, slot)) = self.fused_mul_adds.get(&contract.kernel_node) {
                TranslationKind::MulAdd {
                    multiply_on_right: slot == 1,
                }
            } else if let Some(&(producer, slot)) =
                self.fused_binary_pairs.get(&contract.kernel_node)
            {
                TranslationKind::BinaryChain {
                    first_add: matches!(self.graph.nodes()[producer].operation, Operation::Add),
                    second_add: matches!(node.operation, Operation::Add),
                    producer_on_right: slot == 1,
                }
            } else if let Some(chain) = self.fused_pointwise_chains.get(&contract.kernel_node) {
                let inner = &self.graph.nodes()[chain[0]].operation;
                let columns = match contract.pattern {
                    GpuMemoryPattern::BiasAdd { columns } => columns,
                    _ => {
                        return Err(GpuError::Fp32PtxRejected {
                            reason: "fused chain lacks BiasAdd memory contract".into(),
                        });
                    }
                };
                match inner {
                    Operation::Add => TranslationKind::AddBiasRelu { columns },
                    Operation::Mul => TranslationKind::MulBiasRelu { columns },
                    _ => {
                        return Err(GpuError::Fp32PtxRejected {
                            reason: "unsupported fused inner operation".into(),
                        });
                    }
                }
            } else if let Some(&producer) = self.fused_relus.get(&contract.kernel_node) {
                match self.graph.nodes()[producer].operation {
                    Operation::Add => TranslationKind::AddRelu,
                    Operation::Mul => TranslationKind::MulRelu,
                    Operation::BiasAdd => {
                        let columns = match contract.pattern {
                            GpuMemoryPattern::BiasAdd { columns } => columns,
                            _ => {
                                return Err(GpuError::Fp32PtxRejected {
                                    reason: "fused bias lacks broadcast memory contract".into(),
                                });
                            }
                        };
                        TranslationKind::BiasAddRelu { columns }
                    }
                    _ => {
                        return Err(GpuError::Fp32PtxRejected {
                            reason: "unsupported fused producer operation".into(),
                        });
                    }
                }
            } else {
                match node.operation {
                    Operation::Add => TranslationKind::Add,
                    Operation::Mul => TranslationKind::Mul,
                    Operation::Pointwise(ref spec) => {
                        supported_pointwise(spec).ok_or(GpuError::UnsupportedOperation {
                            node: contract.kernel_node,
                            operation: node.operation.name(),
                        })?
                    }
                    Operation::BiasAdd => {
                        let columns = match contract.pattern {
                            GpuMemoryPattern::BiasAdd { columns } => columns,
                            _ => {
                                return Err(GpuError::Fp32PtxRejected {
                                    reason: "bias lacks broadcast memory contract".into(),
                                });
                            }
                        };
                        TranslationKind::BiasAdd { columns }
                    }
                    Operation::MatMul => {
                        let (inner, columns) = match contract.pattern {
                            GpuMemoryPattern::MatMul { inner, columns, .. } => (inner, columns),
                            _ => {
                                return Err(GpuError::Fp32PtxRejected {
                                    reason: "MatMul lacks reduction memory contract".into(),
                                });
                            }
                        };
                        TranslationKind::MatMul { inner, columns }
                    }
                    Operation::Permute { .. } => match contract.pattern {
                        GpuMemoryPattern::Transpose2D { rows, columns } => {
                            TranslationKind::Transpose2D { rows, columns }
                        }
                        _ => {
                            return Err(GpuError::Fp32PtxRejected {
                                reason: "Permute lacks transpose2d memory contract".into(),
                            });
                        }
                    },
                    Operation::ReverseAxis { axis: 0 } => match contract.pattern {
                        GpuMemoryPattern::ReverseRows2D { rows, columns } => {
                            TranslationKind::ReverseRows2D { rows, columns }
                        }
                        _ => {
                            return Err(GpuError::Fp32PtxRejected {
                                reason: "ReverseAxis(0) lacks memory contract".into(),
                            });
                        }
                    },
                    Operation::ReverseAxis { axis: 1 } => match contract.pattern {
                        GpuMemoryPattern::ReverseColumns2D { rows, columns } => {
                            TranslationKind::ReverseColumns2D { rows, columns }
                        }
                        _ => {
                            return Err(GpuError::Fp32PtxRejected {
                                reason: "ReverseAxis(1) lacks memory contract".into(),
                            });
                        }
                    },
                    Operation::BroadcastTo(_) => match contract.pattern {
                        GpuMemoryPattern::BroadcastScalar => TranslationKind::BroadcastScalar,
                        _ => {
                            return Err(GpuError::Fp32PtxRejected {
                                reason: "BroadcastTo lacks scalar memory contract".into(),
                            });
                        }
                    },
                    Operation::SliceAxis { axis: 1, .. } => match contract.pattern {
                        GpuMemoryPattern::SliceLastRow {
                            start,
                            source_width,
                        } => TranslationKind::SliceLastRow {
                            start,
                            source_width,
                        },
                        _ => {
                            return Err(GpuError::Fp32PtxRejected {
                                reason: "SliceAxis(1) lacks memory contract".into(),
                            });
                        }
                    },
                    Operation::Concat { .. } => match contract.pattern {
                        GpuMemoryPattern::ConcatAxis0 { left } => {
                            TranslationKind::ConcatAxis0 { left }
                        }
                        _ => {
                            return Err(GpuError::Fp32PtxRejected {
                                reason: "Concat(0) lacks memory contract".into(),
                            });
                        }
                    },
                    Operation::ArgMax { axis: 1, .. } => match contract.pattern {
                        GpuMemoryPattern::ArgMaxLast { width } => {
                            TranslationKind::ArgMaxLast { width }
                        }
                        _ => {
                            return Err(GpuError::Fp32PtxRejected {
                                reason: "ArgMax(last) lacks memory contract".into(),
                            });
                        }
                    },
                    Operation::Take { axis: 0 } => match contract.pattern {
                        GpuMemoryPattern::TakeAxis0 { lane, .. } => {
                            TranslationKind::TakeAxis0 { lane }
                        }
                        _ => {
                            return Err(GpuError::Fp32PtxRejected {
                                reason: "Take(0) lacks memory contract".into(),
                            });
                        }
                    },
                    Operation::Relu => TranslationKind::Relu,
                    _ => {
                        return Err(GpuError::Fp32PtxRejected {
                            reason: "unexpected operation in kernel translation plan".into(),
                        });
                    }
                }
            };
            kernels.push(TranslationKernel {
                node: contract.kernel_node,
                entry_node: self.kernel_entries[&contract.kernel_node],
                kind,
                reads: contract.reads,
                writes: contract.writes,
            });
        }
        Ok(TranslationPlan {
            threads_per_block: self.launch.threads_per_block(),
            kernels,
        })
    }

    /// Cross-check the concrete lowering profile before it can be called
    /// supported by higher layers. This validates evidence coverage and fusion
    /// preconditions only; it does not execute Lean, LLVM, CUDA or the GPU.
    pub fn capability_plan(&self) -> Result<GpuCapabilityPlan, GpuError> {
        let contracts = self.memory_contracts()?;
        let translation = self.translation_plan()?;
        if contracts.len() != translation.kernels.len() {
            return Err(GpuError::Fp32PtxRejected {
                reason: "capability plan kernel/memory coverage mismatch".into(),
            });
        }

        let mut kernels = Vec::with_capacity(contracts.len());
        for (contract, rule) in contracts.iter().zip(&translation.kernels) {
            if contract.kernel_node != rule.node
                || contract.reads != rule.reads
                || contract.writes != rule.writes
                || contract.additional_writes != rule.kind.extra_writes(contract.writes)
            {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!(
                        "capability evidence mismatch for kernel node {}",
                        contract.kernel_node
                    ),
                });
            }
            kernels.push(GpuKernelCapability {
                node: contract.kernel_node,
                operation: self.graph.nodes()[contract.kernel_node].operation.name(),
                memory_pattern: contract.pattern,
                translation_kind: rule.kind,
                reads: contract.reads.clone(),
                writes: contract.writes,
                additional_writes: contract.additional_writes.clone(),
            });
        }

        let skipped = self.skipped_nodes();
        let launched = kernels
            .iter()
            .map(|kernel| kernel.node)
            .collect::<BTreeSet<_>>();
        let mut expected = BTreeSet::new();
        for &index in &self.live_nodes {
            let node = &self.graph.nodes()[index];
            if !skipped.contains(&index)
                && !matches!(node.operation, Operation::Input)
                && !self.graph.storage(node.id)?.is_alias()
            {
                expected.insert(index);
            }
        }
        if launched != expected {
            return Err(GpuError::Fp32PtxRejected {
                reason: "capability plan does not cover every launched semantic node".into(),
            });
        }

        let mut uses = vec![0usize; self.graph.nodes().len()];
        for &index in &self.live_nodes {
            for input in &self.graph.nodes()[index].inputs {
                uses[input.index()] += 1;
            }
        }
        for output in &self.outputs {
            uses[output.index()] += 1;
        }
        let selected = self
            .outputs
            .iter()
            .map(|id| id.index())
            .collect::<BTreeSet<_>>();
        let mut optimizations = Vec::new();
        let mut covered_removed = BTreeSet::new();

        for (&root, chain) in &self.fused_pointwise_chains {
            if chain.len() != 2 {
                return Err(GpuError::Fp32PtxRejected {
                    reason: "fused pointwise chain has invalid length".into(),
                });
            }
            let inner = chain[0];
            let bias = chain[1];
            let root_node = &self.graph.nodes()[root];
            let inner_node = &self.graph.nodes()[inner];
            let bias_node = &self.graph.nodes()[bias];
            let exclusive = uses[inner] == 1 && uses[bias] == 1;
            let preserves = !selected.contains(&inner) && !selected.contains(&bias);
            let compatible = matches!(root_node.operation, Operation::Relu)
                && root_node.inputs.len() == 1
                && root_node.inputs[0].index() == bias
                && matches!(bias_node.operation, Operation::BiasAdd)
                && bias_node.inputs[0].index() == inner
                && matches!(inner_node.operation, Operation::Add | Operation::Mul)
                && self.fused_relus.get(&root).copied() == Some(bias);
            if !exclusive || !preserves || !compatible {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!("invalid pointwise fusion capability at node {root}"),
                });
            }
            covered_removed.insert(inner);
            covered_removed.insert(bias);
            optimizations.push(GpuOptimizationCapability {
                root,
                kind: GpuOptimizationKind::PointwiseBiasReluFusion,
                removed_nodes: chain.clone(),
                exclusive_reachable_use: true,
                selected_outputs_preserved: true,
            });
        }

        for (&root, &producer) in &self.fused_relus {
            if self.fused_pointwise_chains.contains_key(&root) {
                continue;
            }
            let root_node = &self.graph.nodes()[root];
            let producer_node = &self.graph.nodes()[producer];
            let exclusive = uses[producer] == 1;
            let preserves = !selected.contains(&producer);
            let compatible = matches!(root_node.operation, Operation::Relu)
                && root_node.inputs.len() == 1
                && root_node.inputs[0].index() == producer
                && matches!(
                    producer_node.operation,
                    Operation::Add | Operation::Mul | Operation::BiasAdd
                );
            if !exclusive || !preserves || !compatible {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!("invalid relu fusion capability at node {root}"),
                });
            }
            covered_removed.insert(producer);
            optimizations.push(GpuOptimizationCapability {
                root,
                kind: GpuOptimizationKind::ReluProducerFusion,
                removed_nodes: vec![producer],
                exclusive_reachable_use: true,
                selected_outputs_preserved: true,
            });
        }

        for (&root, &(mul, add, add_slot, outer_slot, reuse)) in &self.fused_mul_add_muls {
            let root_node = &self.graph.nodes()[root];
            let middle = &self.graph.nodes()[add];
            let inner = &self.graph.nodes()[mul];
            let valid_slots = add_slot < 2 && outer_slot < 2 && reuse < 3;
            let external = if valid_slots {
                [
                    inner.inputs[0],
                    inner.inputs[1],
                    middle.inputs[1 - add_slot],
                ]
            } else {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!("invalid mul-add-mul slots at node {root}"),
                });
            };
            let compatible = matches!(root_node.operation, Operation::Mul)
                && matches!(middle.operation, Operation::Add)
                && matches!(inner.operation, Operation::Mul)
                && root_node.inputs[outer_slot].index() == add
                && middle.inputs[add_slot].index() == mul
                && root_node.inputs[1 - outer_slot] == external[reuse]
                && root_node.shape == middle.shape
                && middle.shape == inner.shape
                && external
                    .iter()
                    .all(|id| self.graph.nodes()[id.index()].shape == root_node.shape)
                && uses[mul] == 1
                && uses[add] == 1
                && !selected.contains(&mul)
                && !selected.contains(&add)
                && !covered_removed.contains(&mul)
                && !covered_removed.contains(&add)
                && !self.fused_relus.contains_key(&root)
                && !self.fused_pointwise_chains.contains_key(&root)
                && !self.fused_mul_adds.contains_key(&root);
            if !compatible {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!("invalid mul-add-mul fusion capability at node {root}"),
                });
            }
            covered_removed.insert(mul);
            covered_removed.insert(add);
            optimizations.push(GpuOptimizationCapability {
                root,
                kind: GpuOptimizationKind::MulAddMulFusion,
                removed_nodes: vec![mul, add],
                exclusive_reachable_use: true,
                selected_outputs_preserved: true,
            });
        }

        for (&root, &(mul, slot)) in &self.fused_mul_adds {
            let root_node = &self.graph.nodes()[root];
            let producer = &self.graph.nodes()[mul];
            let exclusive = uses[mul] == 1;
            let preserves = !selected.contains(&mul);
            let compatible = slot < 2
                && matches!(root_node.operation, Operation::Add)
                && matches!(producer.operation, Operation::Mul)
                && root_node.inputs[slot].index() == mul
                && root_node.shape == producer.shape
                && producer
                    .inputs
                    .iter()
                    .all(|&id| self.graph.nodes()[id.index()].shape == root_node.shape)
                && self.graph.nodes()[root_node.inputs[1 - slot].index()].shape == root_node.shape
                && !covered_removed.contains(&mul)
                && !self.fused_relus.contains_key(&root)
                && !self.fused_pointwise_chains.contains_key(&root);
            if !exclusive || !preserves || !compatible {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!("invalid mul-add fusion capability at node {root}"),
                });
            }
            covered_removed.insert(mul);
            optimizations.push(GpuOptimizationCapability {
                root,
                kind: GpuOptimizationKind::MulAddFusion,
                removed_nodes: vec![mul],
                exclusive_reachable_use: true,
                selected_outputs_preserved: true,
            });
        }

        for (&root, &(left, right_first, right_second, reused, right_slot, left_slot)) in
            &self.fused_mul_diamond_adds
        {
            if reused > 1 || right_slot > 1 || left_slot > 1 {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!("invalid MulDiamondAdd selectors at node {root}"),
                });
            }
            let output = &self.graph.nodes()[root];
            let lhs = &self.graph.nodes()[left];
            let rhs = &self.graph.nodes()[right_first];
            let repeated = &self.graph.nodes()[right_second];
            let compatible = matches!(output.operation, Operation::Add)
                && matches!(lhs.operation, Operation::Mul)
                && matches!(rhs.operation, Operation::Mul)
                && matches!(repeated.operation, Operation::Mul)
                && output.inputs[left_slot].index() == left
                && output.inputs[1 - left_slot].index() == right_second
                && repeated.inputs[right_slot].index() == right_first
                && repeated.inputs[1 - right_slot] == rhs.inputs[reused]
                && [left, right_first, right_second]
                    .iter()
                    .all(|&id| uses[id] == 1)
                && [left, right_first, right_second]
                    .iter()
                    .all(|id| !selected.contains(id))
                && [left, right_first, right_second]
                    .iter()
                    .all(|id| !covered_removed.contains(id))
                && lhs.shape == output.shape
                && rhs.shape == output.shape
                && repeated.shape == output.shape
                && lhs
                    .inputs
                    .iter()
                    .chain(&rhs.inputs)
                    .all(|&id| self.graph.nodes()[id.index()].shape == output.shape)
                && !self.fused_mul_adds.contains_key(&root)
                && !self.fused_binary_pairs.contains_key(&root)
                && !self.fused_mul_add_muls.contains_key(&root);
            if !compatible {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!("invalid MulDiamondAdd fusion at node {root}"),
                });
            }
            covered_removed.extend([left, right_first, right_second]);
            optimizations.push(GpuOptimizationCapability {
                root,
                kind: GpuOptimizationKind::MulDiamondAddFusion,
                removed_nodes: vec![left, right_first, right_second],
                exclusive_reachable_use: true,
                selected_outputs_preserved: true,
            });
        }

        for (&root, &(inner, middle, add, reused, middle_slot, add_slot)) in
            &self.fused_mul_mul_add_divs
        {
            if reused > 1 || middle_slot > 1 || add_slot > 1 {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!("invalid four-operation fusion selectors at {root}"),
                });
            }
            let output = &self.graph.nodes()[root];
            let first = &self.graph.nodes()[inner];
            let second = &self.graph.nodes()[middle];
            let sum = &self.graph.nodes()[add];
            let div = matches!(&output.operation, Operation::Pointwise(spec)
                if supported_pointwise(spec) == Some(TranslationKind::DivRne));
            let sources = [
                first.inputs[0],
                first.inputs[1],
                sum.inputs[1 - add_slot],
                output.inputs[1],
            ];
            let compatible = div
                && matches!(first.operation, Operation::Mul)
                && matches!(second.operation, Operation::Mul)
                && matches!(sum.operation, Operation::Add)
                && output.inputs[0].index() == add
                && sum.inputs[add_slot].index() == middle
                && second.inputs[middle_slot].index() == inner
                && second.inputs[1 - middle_slot] == first.inputs[reused]
                && output.shape == first.shape
                && output.shape == second.shape
                && output.shape == sum.shape
                && sources
                    .iter()
                    .all(|id| self.graph.nodes()[id.index()].shape == output.shape)
                && [inner, middle, add].iter().all(|&id| uses[id] == 1)
                && [inner, middle, add].iter().all(|id| !selected.contains(id))
                && [inner, middle, add]
                    .iter()
                    .all(|id| !covered_removed.contains(id))
                && !self.fused_mul_adds.contains_key(&add)
                && !self.fused_binary_pairs.contains_key(&add);
            if !compatible {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!("invalid four-operation fusion at node {root}"),
                });
            }
            covered_removed.extend([inner, middle, add]);
            optimizations.push(GpuOptimizationCapability {
                root,
                kind: GpuOptimizationKind::MulMulAddDivFusion,
                removed_nodes: vec![inner, middle, add],
                exclusive_reachable_use: true,
                selected_outputs_preserved: true,
            });
        }

        for (&root, &(sqrt, add, slot)) in &self.fused_sqrt_add_divs {
            let output = &self.graph.nodes()[root];
            let middle = &self.graph.nodes()[add];
            let inner = &self.graph.nodes()[sqrt];
            let div = matches!(&output.operation, Operation::Pointwise(spec)
                if supported_pointwise(spec) == Some(TranslationKind::DivRne));
            let square_root = matches!(&inner.operation, Operation::Pointwise(spec)
                if supported_pointwise(spec) == Some(TranslationKind::SqrtRne));
            let compatible = slot < 2
                && div
                && square_root
                && matches!(middle.operation, Operation::Add)
                && output.inputs[1].index() == add
                && middle.inputs[slot].index() == sqrt
                && output.shape == middle.shape
                && middle.shape == inner.shape
                && [inner.inputs[0], middle.inputs[1 - slot], output.inputs[0]]
                    .iter()
                    .all(|&id| self.graph.nodes()[id.index()].shape == output.shape)
                && uses[sqrt] == 1
                && uses[add] == 1
                && !selected.contains(&sqrt)
                && !selected.contains(&add)
                && !covered_removed.contains(&sqrt)
                && !covered_removed.contains(&add)
                && !self.fused_mul_adds.contains_key(&add)
                && !self.fused_mul_add_muls.contains_key(&add)
                && !self.fused_binary_pairs.contains_key(&add);
            if !compatible {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!("invalid sqrt-add-div fusion at node {root}"),
                });
            }
            covered_removed.insert(sqrt);
            covered_removed.insert(add);
            optimizations.push(GpuOptimizationCapability {
                root,
                kind: GpuOptimizationKind::SqrtAddDivFusion,
                removed_nodes: vec![sqrt, add],
                exclusive_reachable_use: true,
                selected_outputs_preserved: true,
            });
        }

        for (&root, &(producer, slot)) in &self.fused_binary_pairs {
            let output = &self.graph.nodes()[root];
            let inner = &self.graph.nodes()[producer];
            let compatible = slot < 2
                && matches!(output.operation, Operation::Add | Operation::Mul)
                && matches!(inner.operation, Operation::Add | Operation::Mul)
                && !(matches!(output.operation, Operation::Add)
                    && matches!(inner.operation, Operation::Mul))
                && output.inputs[slot].index() == producer
                && output.shape == inner.shape
                && inner
                    .inputs
                    .iter()
                    .all(|id| self.graph.nodes()[id.index()].shape == output.shape)
                && self.graph.nodes()[output.inputs[1 - slot].index()].shape == output.shape
                && uses[producer] == 1
                && !selected.contains(&producer)
                && !covered_removed.contains(&producer)
                && !self.fused_mul_adds.contains_key(&root)
                && !self.fused_mul_add_muls.contains_key(&root)
                && !self.fused_mul_adds.contains_key(&producer)
                && !self.fused_mul_add_muls.contains_key(&producer)
                && !self.fused_relus.contains_key(&root)
                && !self.fused_pointwise_chains.contains_key(&root);
            if !compatible {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!("invalid binary pointwise fusion at node {root}"),
                });
            }
            covered_removed.insert(producer);
            optimizations.push(GpuOptimizationCapability {
                root,
                kind: GpuOptimizationKind::BinaryPointwiseFusion,
                removed_nodes: vec![producer],
                exclusive_reachable_use: true,
                selected_outputs_preserved: true,
            });
        }

        for (&root, &earlier) in &self.fused_dual_outputs {
            let left = &self.graph.nodes()[earlier];
            let right = &self.graph.nodes()[root];
            let valid = earlier < root
                && self.live_nodes.binary_search(&earlier).is_ok()
                && self.live_nodes.binary_search(&root).is_ok()
                && uses[earlier] >= 1
                && uses[root] >= 1
                && self.live_nodes.iter().all(|&consumer| {
                    consumer > root
                        || self.graph.nodes()[consumer]
                            .inputs
                            .iter()
                            .all(|input| input.index() != earlier)
                })
                && matches!(left.operation, Operation::Add | Operation::Mul)
                && matches!(right.operation, Operation::Add | Operation::Mul)
                && left.shape == right.shape
                && left.dtype == DType::F32
                && right.dtype == DType::F32
                && left.inputs.iter().chain(&right.inputs).all(|id| {
                    self.graph.nodes()[id.index()].shape == right.shape
                        && self.graph.nodes()[id.index()].dtype == DType::F32
                        && !skipped.contains(&id.index())
                })
                && !covered_removed.contains(&earlier);
            if !valid {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!("invalid dual-output pointwise fusion at node {root}"),
                });
            }
            covered_removed.insert(earlier);
            optimizations.push(GpuOptimizationCapability {
                root,
                kind: GpuOptimizationKind::DualPointwiseOutputFusion,
                removed_nodes: vec![earlier],
                // The standalone kernel is removed, not the materialized
                // value. Downstream fanout is permitted after this root.
                exclusive_reachable_use: uses[earlier] == 1,
                selected_outputs_preserved: true,
            });
        }
        if covered_removed != skipped {
            return Err(GpuError::Fp32PtxRejected {
                reason: "capability plan does not explain every removed fusion kernel".into(),
            });
        }

        Ok(GpuCapabilityPlan {
            kernels,
            optimizations,
            boundary: GpuAssuranceBoundary {
                memory_address_arithmetic: GpuAssuranceStatus::GeneratedLeanObligation,
                llvm_ptx_structure: GpuAssuranceStatus::RequiredAtLowering,
                ideal_to_fp32: GpuAssuranceStatus::SeparateNumericProofRequired,
                cuda_runtime: GpuAssuranceStatus::RuntimeAcceptanceRequired,
                jit_sass_correspondence: GpuAssuranceStatus::NotEstablished,
            },
        })
    }

    /// Named Lean theorems for natural-number address bounds of the actual
    /// launched kernel contracts. Generation is not proof; certified builders
    /// must kernel-check and axiom-audit this program.
    pub fn memory_lean_program(&self) -> Result<LeanProgram, GpuError> {
        let mut source = format!(
            "{}\n\n{}\n",
            include_str!("../../../../lean/VeriNeural/Basic.lean"),
            include_str!("../../../../lean/VeriNeural/Memory.lean")
        );
        source.push_str("\nnamespace VeriNeural.Generated\n");
        let mut theorems = Vec::new();
        for contract in self.memory_contracts()? {
            source.push_str(&format!(
                "-- IR kernel {}: {} source tensors, {} output elements.\n",
                contract.kernel_node,
                contract.reads.len(),
                contract.writes
            ));
            let output_name = format!("memory_node_{}_output_bound", contract.kernel_node);
            source.push_str(&format!(
                "theorem {output_name} (index : Nat) (active : index < {}) : index < {} := by\n  exact VeriNeural.pointwise_address_bound active\n\n",
                contract.writes, contract.writes
            ));
            theorems.push(output_name);
            for &(other, extent) in &contract.additional_writes {
                let name = format!(
                    "memory_node_{}_additional_{}_output_bound",
                    contract.kernel_node, other
                );
                source.push_str(&format!(
                    "theorem {name} (index : Nat) (active : index < {extent}) : index < {extent} := by\n  exact VeriNeural.pointwise_address_bound active\n\n"
                ));
                theorems.push(name);
            }
            match contract.pattern {
                GpuMemoryPattern::Pointwise => {}
                GpuMemoryPattern::BiasAdd { columns } => {
                    let name = format!("memory_node_{}_bias_bound", contract.kernel_node);
                    source.push_str(&format!(
                        "theorem {name} (index : Nat) : index % {columns} < {columns} := by\n  exact VeriNeural.bias_address_bound (by decide)\n\n"
                    ));
                    theorems.push(name);
                }
                GpuMemoryPattern::MatMul {
                    rows,
                    inner,
                    columns,
                } => {
                    let name = format!("memory_node_{}_matmul_bounds", contract.kernel_node);
                    source.push_str(&format!(
                        "theorem {name} (index j : Nat) (active : index < {rows} * {columns}) (reduction : j < {inner}) :\n    (index / {columns}) * {inner} + j < {rows} * {inner} ∧\n    j * {columns} + index % {columns} < {inner} * {columns} := by\n  exact VeriNeural.matmul_address_bounds (by decide) active reduction\n\n"
                    ));
                    theorems.push(name);
                }
                GpuMemoryPattern::Transpose2D { rows, columns } => {
                    let name = format!("memory_node_{}_transpose_bound", contract.kernel_node);
                    source.push_str(&format!(
                        "theorem {name} (index : Nat) (active : index < {rows} * {columns}) :\n    (index % {rows}) * {columns} + index / {rows} < {rows} * {columns} := by\n  exact VeriNeural.transpose2d_address_bound (by decide) active\n\n"
                    ));
                    theorems.push(name);
                }
                GpuMemoryPattern::ReverseRows2D { rows, columns } => {
                    let name = format!("memory_node_{}_reverse_rows_bound", contract.kernel_node);
                    source.push_str(&format!(
                        "theorem {name} (index : Nat) (active : index < {rows} * {columns}) :\n    ({rows} - 1 - index / {columns}) * {columns} + index % {columns} < {rows} * {columns} := by\n  exact VeriNeural.reverse_rows2d_address_bound (by decide) (by decide) active\n\n"
                    ));
                    theorems.push(name);
                }
                GpuMemoryPattern::ReverseColumns2D { rows, columns } => {
                    let name =
                        format!("memory_node_{}_reverse_columns_bound", contract.kernel_node);
                    source.push_str(&format!(
                        "theorem {name} (index : Nat) (active : index < {rows} * {columns}) :\n    (index / {columns}) * {columns} + ({columns} - 1 - index % {columns}) < {rows} * {columns} := by\n  exact VeriNeural.reverse_columns2d_address_bound (by decide) active\n\n"
                    ));
                    theorems.push(name);
                }
                GpuMemoryPattern::BroadcastScalar => {
                    let name = format!(
                        "memory_node_{}_broadcast_scalar_bound",
                        contract.kernel_node
                    );
                    source.push_str(&format!("theorem {name} : 0 < 1 := by decide\n\n"));
                    theorems.push(name);
                }
                GpuMemoryPattern::SliceLastRow {
                    start,
                    source_width,
                } => {
                    let name = format!("memory_node_{}_slice_bound", contract.kernel_node);
                    source.push_str(&format!(
                        "theorem {name} (index : Nat) (active : index < {}) : index + {start} < {source_width} := by\n  exact VeriNeural.slice_last_row_address_bound active (by decide)\n\n",
                        contract.writes
                    ));
                    theorems.push(name);
                }
                GpuMemoryPattern::ConcatAxis0 { left } => {
                    let right = contract.reads[1].1;
                    let left_name =
                        format!("memory_node_{}_concat_left_bound", contract.kernel_node);
                    source.push_str(&format!(
                        "theorem {left_name} (index : Nat) (active : index < {left}) : index < {left} := by\n  exact VeriNeural.pointwise_address_bound active\n\n"
                    ));
                    theorems.push(left_name);
                    let right_name =
                        format!("memory_node_{}_concat_right_bound", contract.kernel_node);
                    source.push_str(&format!(
                        "theorem {right_name} (index : Nat) (branch : {left} ≤ index) (active : index < {left} + {right}) : index - {left} < {right} := by\n  exact VeriNeural.concat_axis0_right_address_bound branch active\n\n"
                    ));
                    theorems.push(right_name);
                }
                GpuMemoryPattern::ArgMaxLast { width } => {
                    let rows = contract.writes;
                    let first_name =
                        format!("memory_node_{}_argmax_first_bound", contract.kernel_node);
                    source.push_str(&format!(
                        "theorem {first_name} (index : Nat) (active : index < {rows}) : index * {width} < {rows} * {width} := by\n  exact VeriNeural.argmax_first_address_bound (by decide) active\n\n"
                    ));
                    theorems.push(first_name);
                    let candidate_name = format!(
                        "memory_node_{}_argmax_candidate_bound",
                        contract.kernel_node
                    );
                    source.push_str(&format!(
                        "theorem {candidate_name} (index lane : Nat) (active : index < {rows}) (candidate : lane < {width}) : index * {width} + lane < {rows} * {width} := by\n  exact VeriNeural.argmax_candidate_address_bound active candidate\n\n"
                    ));
                    theorems.push(candidate_name);
                }
                GpuMemoryPattern::TakeAxis0 { lane, choices } => {
                    let index_name =
                        format!("memory_node_{}_take_index_nonempty", contract.kernel_node);
                    source.push_str(&format!("theorem {index_name} : 0 < 1 := by decide\n\n"));
                    theorems.push(index_name);
                    let bound_name = format!("memory_node_{}_take_bound", contract.kernel_node);
                    source.push_str(&format!(
                        "theorem {bound_name} (selected index : Nat) (selectedBound : selected < {choices}) (active : index < {lane}) : selected * {lane} + index < {choices} * {lane} := by\n  exact VeriNeural.take_axis0_address_bound selectedBound active\n\n"
                    ));
                    theorems.push(bound_name);
                }
            }
        }
        source.push_str("end VeriNeural.Generated\n");
        Ok(LeanProgram::generated(source, theorems, Vec::new()))
    }

    /// Backwards-compatible source view over the named memory obligations.
    pub fn memory_lean_source(&self) -> Result<String, GpuError> {
        Ok(self.memory_lean_program()?.source().to_owned())
    }

    /// The Lean proof of the *ideal integer* fusion rule, not PTX/FP32 equivalence.
    /// None when no fusion was actually applied.
    pub fn fusion_lean_source(&self) -> Option<String> {
        (!self.fused_relus.is_empty()).then(|| {
            format!(
                "{}\n\n{}",
                include_str!("../../../../lean/VeriNeural/Basic.lean"),
                include_str!("../../../../lean/VeriNeural/Fusion.lean"),
            )
        })
    }
}

