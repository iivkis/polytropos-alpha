//! Bounded structural translation validation, not a semantic proof of LLVM or PTX.
//! The expected kernel plan comes from the checked Semantic Neural IR, fusion
//! decisions and memory contracts; it is NOT inferred from LLVM/PTX text.
//! Unknown kernels, missing entries, changed FP32 operation sequences, shapes,
//! address templates or rounding modes fail closed. CUDA JIT/SASS stays trusted.

use std::collections::BTreeMap;
use std::fmt;

use crate::gpu::audit_fp32_ptx;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranslationKind {
    Add,
    Mul,
    BiasAdd { columns: usize },
    MatMul { inner: usize, columns: usize },
    Transpose2D { rows: usize, columns: usize },
    ReverseRows2D { rows: usize, columns: usize },
    ReverseColumns2D { rows: usize, columns: usize },
    ConcatAxis0 { left: usize },
    SliceLastRow { start: usize, source_width: usize },
    BroadcastScalar,
    ArgMaxLast { width: usize },
    TakeAxis0 { lane: usize },
    Relu,
    AddRelu,
    MulRelu,
    BiasAddRelu { columns: usize },
    AddBiasRelu { columns: usize },
    MulBiasRelu { columns: usize },
}

impl TranslationKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Mul => "mul",
            Self::BiasAdd { .. } => "bias_add",
            Self::MatMul { .. } => "matmul",
            Self::Transpose2D { .. } => "transpose2d",
            Self::ReverseRows2D { .. } => "reverse_rows2d",
            Self::ReverseColumns2D { .. } => "reverse_columns2d",
            Self::ConcatAxis0 { .. } => "concat_axis0",
            Self::SliceLastRow { .. } => "slice_last_row",
            Self::BroadcastScalar => "broadcast_scalar",
            Self::ArgMaxLast { .. } => "argmax_last",
            Self::TakeAxis0 { .. } => "take_axis0",
            Self::Relu => "relu",
            Self::AddRelu => "add_relu",
            Self::MulRelu => "mul_relu",
            Self::BiasAddRelu { .. } => "bias_relu",
            Self::AddBiasRelu { .. } => "add_bias_relu",
            Self::MulBiasRelu { .. } => "mul_bias_relu",
        }
    }

    fn counts(self) -> (usize, usize, usize) {
        match self {
            Self::Add | Self::BiasAdd { .. } => (1, 0, 0),
            Self::Mul => (0, 1, 0),
            Self::MatMul { .. } => (1, 1, 0),
            Self::Transpose2D { .. }
            | Self::ReverseRows2D { .. }
            | Self::ReverseColumns2D { .. }
            | Self::ConcatAxis0 { .. }
            | Self::SliceLastRow { .. }
            | Self::BroadcastScalar
            | Self::ArgMaxLast { .. }
            | Self::TakeAxis0 { .. } => (0, 0, 0),
            Self::Relu => (0, 0, 1),
            Self::AddRelu | Self::BiasAddRelu { .. } => (1, 0, 1),
            Self::MulRelu => (0, 1, 1),
            Self::AddBiasRelu { .. } => (2, 0, 1),
            Self::MulBiasRelu { .. } => (1, 1, 1),
        }
    }

    fn expected_inputs(self) -> usize {
        match self {
            Self::Relu
            | Self::Transpose2D { .. }
            | Self::ReverseRows2D { .. }
            | Self::ReverseColumns2D { .. }
            | Self::SliceLastRow { .. }
            | Self::BroadcastScalar
            | Self::ArgMaxLast { .. } => 1,
            Self::AddBiasRelu { .. } | Self::MulBiasRelu { .. } => 3,
            _ => 2,
        }
    }

    fn columns(self) -> usize {
        match self {
            Self::BiasAdd { columns }
            | Self::BiasAddRelu { columns }
            | Self::AddBiasRelu { columns }
            | Self::MulBiasRelu { columns }
            | Self::MatMul { columns, .. }
            | Self::Transpose2D { columns, .. }
            | Self::ReverseRows2D { columns, .. }
            | Self::ReverseColumns2D { columns, .. } => columns,
            Self::SliceLastRow { source_width, .. } => source_width,
            _ => 0,
        }
    }

    fn inner(self) -> usize {
        match self {
            Self::MatMul { inner, .. } => inner,
            Self::Transpose2D { rows, .. }
            | Self::ReverseRows2D { rows, .. }
            | Self::ReverseColumns2D { rows, .. } => rows,
            Self::ConcatAxis0 { left } => left,
            Self::SliceLastRow { start, .. } => start,
            Self::ArgMaxLast { width } => width,
            Self::TakeAxis0 { lane } => lane,
            _ => 0,
        }
    }

    fn parse(name: &str, columns: usize, inner: usize) -> Result<Self, TranslationError> {
        let kind = match name {
            "add" => Self::Add,
            "mul" => Self::Mul,
            "bias_add" => Self::BiasAdd { columns },
            "matmul" => Self::MatMul { inner, columns },
            "transpose2d" => Self::Transpose2D {
                rows: inner,
                columns,
            },
            "reverse_rows2d" => Self::ReverseRows2D {
                rows: inner,
                columns,
            },
            "reverse_columns2d" => Self::ReverseColumns2D {
                rows: inner,
                columns,
            },
            "concat_axis0" => Self::ConcatAxis0 { left: inner },
            "slice_last_row" => Self::SliceLastRow { start: inner, source_width: columns },
            "broadcast_scalar" => Self::BroadcastScalar,
            "argmax_last" => Self::ArgMaxLast { width: inner },
            "take_axis0" => Self::TakeAxis0 { lane: inner },
            "relu" => Self::Relu,
            "add_relu" => Self::AddRelu,
            "mul_relu" => Self::MulRelu,
            "bias_relu" => Self::BiasAddRelu { columns },
            "add_bias_relu" => Self::AddBiasRelu { columns },
            "mul_bias_relu" => Self::MulBiasRelu { columns },
            _ => return Err(TranslationError(format!("unknown lowering kind: {name}"))),
        };
        if kind.columns() != columns || kind.inner() != inner {
            return Err(TranslationError("unexpected lowering dimensions".into()));
        }
        Ok(kind)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranslationKernel {
    pub node: usize,
    pub kind: TranslationKind,
    pub reads: Vec<(usize, usize)>,
    pub writes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranslationPlan {
    pub threads_per_block: u32,
    pub kernels: Vec<TranslationKernel>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranslationReport {
    pub kernel_count: usize,
    pub rounded_adds: usize,
    pub rounded_muls: usize,
    pub relu_guards: usize,
    pub details: String,
}

impl TranslationReport {
    pub fn to_text(&self) -> String {
        format!(
            "assurance=structural-translation-validation\n\
             semantic_fp32_equivalence=not-proven\n\
             llvm_ptx_memory_execution=not-proven\n\
             cuda_jit_sass_hardware=trusted\n\
             kernels={}\nrounded_adds={}\nrounded_muls={}\nrelu_guards={}\n{}",
            self.kernel_count, self.rounded_adds, self.rounded_muls, self.relu_guards, self.details
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranslationError(pub String);

impl fmt::Display for TranslationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TranslationError {}

fn invalid(reason: impl Into<String>) -> TranslationError {
    TranslationError(reason.into())
}

impl TranslationPlan {
    fn check(&self) -> Result<(), TranslationError> {
        if !matches!(self.threads_per_block, 64 | 128 | 256 | 512) {
            return Err(invalid("unsupported CUDA launch width in translation plan"));
        }
        let mut previous = None;
        for kernel in &self.kernels {
            if previous.is_some_and(|n| kernel.node <= n) {
                return Err(invalid("kernel roots not strictly ordered"));
            }
            previous = Some(kernel.node);
            if kernel.writes == 0
                || kernel.writes > i32::MAX as usize
                || kernel.reads.len() != kernel.kind.expected_inputs()
                || kernel.reads.iter().any(|&(source, size)| {
                    source >= kernel.node || size == 0 || size > i32::MAX as usize
                })
            {
                return Err(invalid(format!(
                    "invalid source/write extents for node {}",
                    kernel.node
                )));
            }
            let columns = kernel.kind.columns();
            if (matches!(
                kernel.kind,
                TranslationKind::BiasAdd { .. }
                    | TranslationKind::BiasAddRelu { .. }
                    | TranslationKind::AddBiasRelu { .. }
                    | TranslationKind::MulBiasRelu { .. }
                    | TranslationKind::MatMul { .. }
            ) && columns == 0)
                || (columns != 0 && (columns > i32::MAX as usize
                    || (!matches!(kernel.kind, TranslationKind::SliceLastRow { .. })
                        && kernel.writes % columns != 0)))
            {
                return Err(invalid("unrepresentable column extent"));
            }
            if let TranslationKind::MatMul { inner, columns } = kernel.kind {
                if inner == 0
                    || inner > i32::MAX as usize
                    || kernel.reads.len() != 2
                    || kernel.reads[0].1 != (kernel.writes / columns).saturating_mul(inner)
                    || kernel.reads[1].1 != inner.saturating_mul(columns)
                {
                    return Err(invalid("MatMul source shape differs from reduction shape"));
                }
            }
            if let TranslationKind::BiasAdd { columns } | TranslationKind::BiasAddRelu { columns } =
                kernel.kind
            {
                if kernel.reads[0].1 != kernel.writes || kernel.reads[1].1 != columns {
                    return Err(invalid("BiasAdd source shape mismatch"));
                }
            }
            if let TranslationKind::AddBiasRelu { columns }
            | TranslationKind::MulBiasRelu { columns } = kernel.kind
            {
                if kernel.reads[0].1 != kernel.writes
                    || kernel.reads[1].1 != kernel.writes
                    || kernel.reads[2].1 != columns
                {
                    return Err(invalid("fused chain shape mismatch"));
                }
            }
            match kernel.kind {
                TranslationKind::Transpose2D { rows, columns }
                    if rows == 0
                        || columns == 0
                        || rows.saturating_mul(columns) != kernel.writes
                        || kernel.reads[0].1 != kernel.writes =>
                {
                    return Err(invalid("transpose2d source shape mismatch"));
                }
                TranslationKind::ReverseRows2D { rows, columns }
                | TranslationKind::ReverseColumns2D { rows, columns }
                    if rows == 0
                        || columns == 0
                        || rows.saturating_mul(columns) != kernel.writes
                        || kernel.reads[0].1 != kernel.writes =>
                {
                    return Err(invalid("reverse2d source shape mismatch"));
                }
                TranslationKind::SliceLastRow { start, source_width }
                    if source_width == 0
                        || source_width > i32::MAX as usize
                        || kernel.reads[0].1 != source_width
                        || start.checked_add(kernel.writes)
                            .is_none_or(|end| end > source_width) =>
                {
                    return Err(invalid("last-row slice source mismatch"));
                }
                TranslationKind::BroadcastScalar if kernel.reads[0].1 != 1 => {
                    return Err(invalid("scalar broadcast source mismatch"));
                }
                TranslationKind::ConcatAxis0 { left }
                    if left == 0
                        || kernel.reads[0].1 != left
                        || kernel.reads[0].1.saturating_add(kernel.reads[1].1) != kernel.writes =>
                {
                    return Err(invalid("concat_axis0 source shape mismatch"));
                }
                TranslationKind::ArgMaxLast { width }
                    if width == 0 || kernel.reads[0].1 != kernel.writes.saturating_mul(width) =>
                {
                    return Err(invalid("argmax_last source shape mismatch"));
                }
                TranslationKind::TakeAxis0 { lane }
                    if lane != kernel.writes
                        || kernel.reads[1].1 != 1
                        || kernel.reads[0].1 < lane
                        || kernel.reads[0].1 % lane != 0 =>
                {
                    return Err(invalid("take_axis0 source shape mismatch"));
                }
                _ => {}
            }
            if matches!(
                kernel.kind,
                TranslationKind::Add
                    | TranslationKind::Mul
                    | TranslationKind::Relu
                    | TranslationKind::AddRelu
                    | TranslationKind::MulRelu
            ) && kernel.reads.iter().any(|&(_, size)| size != kernel.writes)
            {
                return Err(invalid("pointwise source shape mismatch"));
            }
        }
        Ok(())
    }

    /// Deterministic, versioned plan. Generated from a graph, never from PTX.
    pub fn to_text(&self) -> String {
        let mut out = format!(
            "vn-translation-plan-v1\nthreads={}\n",
            self.threads_per_block
        );
        for kernel in &self.kernels {
            let reads = kernel
                .reads
                .iter()
                .map(|(source, size)| format!("{source}:{size}"))
                .collect::<Vec<_>>()
                .join(",");
            out.push_str(&format!(
                "kernel\t{}\t{}\t{}\t{}\t{}\t{}\n",
                kernel.node,
                kernel.kind.name(),
                kernel.writes,
                reads,
                kernel.kind.columns(),
                kernel.kind.inner()
            ));
        }
        out
    }

    /// Offline reader rejects unknown schema/kinds, malformed nodes and sizes.
    /// The saved plan is SHA-pinned evidence, NOT by itself a semantic proof.
    pub fn from_text(source: &str) -> Result<Self, TranslationError> {
        let mut lines = source.lines();
        if lines.next() != Some("vn-translation-plan-v1") {
            return Err(invalid("unsupported translation plan version"));
        }
        let threads_per_block = lines
            .next()
            .and_then(|line| line.strip_prefix("threads="))
            .and_then(|value| value.parse::<u32>().ok())
            .ok_or_else(|| invalid("invalid launch width in translation plan"))?;
        let mut kernels = Vec::new();
        for line in lines {
            let fields = line.split('\t').collect::<Vec<_>>();
            if fields.len() != 7 || fields[0] != "kernel" {
                return Err(invalid("malformed translation kernel row"));
            }
            let number = |field: &str| -> Result<usize, TranslationError> {
                field
                    .parse()
                    .map_err(|_| invalid("invalid translation integer"))
            };
            let node = number(fields[1])?;
            let writes = number(fields[3])?;
            let reads = fields[4]
                .split(',')
                .map(|pair| {
                    let (source, count) = pair
                        .split_once(':')
                        .ok_or_else(|| invalid("malformed translation source"))?;
                    Ok((number(source)?, number(count)?))
                })
                .collect::<Result<Vec<_>, TranslationError>>()?;
            let columns = number(fields[5])?;
            let inner = number(fields[6])?;
            let kind = TranslationKind::parse(fields[2], columns, inner)?;
            kernels.push(TranslationKernel {
                node,
                kind,
                reads,
                writes,
            });
        }
        let plan = Self {
            threads_per_block,
            kernels,
        };
        plan.check()?;
        Ok(plan)
    }

    /// Compares each IR kernel against its independent lowering contract,
    /// and each PTX entry against that kernel, rather than global opcode sums.
    /// This is a bounded structural validation of known templates only.
    pub fn validate(&self, llvm: &str, ptx: &str) -> Result<TranslationReport, TranslationError> {
        self.check()?;
        let mut llvm_parts = BTreeMap::new();
        let sections = llvm
            .split("define ptx_kernel void @vn_node_")
            .collect::<Vec<_>>();
        if sections[0].contains("define ") {
            return Err(invalid("unrecognized LLVM function"));
        }
        if llvm.matches("define ").count() != sections.len() - 1 {
            return Err(invalid("unknown LLVM function or entrypoint"));
        }
        for section in sections.into_iter().skip(1) {
            let digits = section
                .chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>();
            let node = digits
                .parse::<usize>()
                .map_err(|_| invalid("invalid LLVM kernel name"))?;
            if !section[digits.len()..].starts_with('(')
                || !section.contains("i32 %length) {")
                || !section.lines().any(|line| line.trim() == "}")
            {
                return Err(invalid(format!("malformed LLVM kernel {node}")));
            }
            if llvm_parts.insert(node, section).is_some() {
                return Err(invalid("duplicate LLVM kernel"));
            }
        }

        let mut ptx_parts: BTreeMap<usize, String> = BTreeMap::new();
        let mut current = None;
        for raw_line in ptx.lines() {
            let line = raw_line.split("//").next().unwrap_or("");
            let fields = line.split_whitespace().collect::<Vec<_>>();
            if let Some(i) = fields.iter().position(|&token| token == ".entry") {
                let name = fields
                    .get(i + 1)
                    .ok_or_else(|| invalid("PTX entry missing name"))?;
                let stripped = name
                    .strip_prefix("vn_node_")
                    .ok_or_else(|| invalid("unexpected PTX entry"))?;
                let digits = stripped
                    .chars()
                    .take_while(char::is_ascii_digit)
                    .collect::<String>();
                let node = digits
                    .parse::<usize>()
                    .map_err(|_| invalid("invalid PTX entry name"))?;
                if !stripped[digits.len()..].starts_with('(')
                    || ptx_parts.insert(node, String::new()).is_some()
                {
                    return Err(invalid("duplicate or malformed PTX entrypoint"));
                }
                current = Some(node);
            }
            if let Some(node) = current {
                ptx_parts
                    .get_mut(&node)
                    .expect("current PTX kernel inserted")
                    .push_str(&format!("{raw_line}\n"));
            }
        }
        if llvm_parts.len() != self.kernels.len() || ptx_parts.len() != self.kernels.len() {
            return Err(invalid(format!(
                "kernel set mismatch: semantic={}, LLVM={}, PTX={}",
                self.kernels.len(),
                llvm_parts.len(),
                ptx_parts.len()
            )));
        }

        let mut report = TranslationReport {
            kernel_count: self.kernels.len(),
            rounded_adds: 0,
            rounded_muls: 0,
            relu_guards: 0,
            details: String::new(),
        };
        for kernel in &self.kernels {
            let node = kernel.node;
            let ir = llvm_parts
                .get(&node)
                .ok_or_else(|| invalid(format!("LLVM kernel {node} missing")))?;
            let asm = ptx_parts
                .get(&node)
                .ok_or_else(|| invalid(format!("PTX kernel {node} missing")))?;
            let (add, mul, relu) = kernel.kind.counts();
            let structural_requirements: Option<Vec<String>> = match kernel.kind {
                TranslationKind::Transpose2D { rows, columns } => Some(vec![
                    format!("urem i32 %idx, {rows}"),
                    format!("udiv i32 %idx, {rows}"),
                    format!("mul i32 %in_row, {columns}"),
                    "%ap = getelementptr float, ptr addrspace(1) %a, i32 %source_idx".into(),
                    "store float %x, ptr addrspace(1) %op, align 4".into(),
                ]),
                TranslationKind::ReverseRows2D { rows, columns } => Some(vec![
                    format!("udiv i32 %idx, {columns}"),
                    format!("urem i32 %idx, {columns}"),
                    format!("%rows_minus_one = add i32 0, {}", rows - 1),
                    "%reversed_row = sub i32 %rows_minus_one, %row".into(),
                    "%ap = getelementptr float, ptr addrspace(1) %a, i32 %source_idx".into(),
                    "store float %x, ptr addrspace(1) %op, align 4".into(),
                ]),
                TranslationKind::ReverseColumns2D { columns, .. } => Some(vec![
                    format!("udiv i32 %idx, {columns}"),
                    format!("urem i32 %idx, {columns}"),
                    format!("%columns_minus_one = add i32 0, {}", columns - 1),
                    "%reversed_col = sub i32 %columns_minus_one, %col".into(),
                    "%ap = getelementptr float, ptr addrspace(1) %a, i32 %source_idx".into(),
                    "store float %x, ptr addrspace(1) %op, align 4".into(),
                ]),
                TranslationKind::SliceLastRow { start, .. } => Some(vec![
                    format!("%source_idx = add i32 %idx, {start}"),
                    "%ap = getelementptr float, ptr addrspace(1) %a, i32 %source_idx".into(),
                    "store float %x, ptr addrspace(1) %op, align 4".into(),
                ]),
                TranslationKind::BroadcastScalar => Some(vec![
                    "%ap = getelementptr float, ptr addrspace(1) %a, i32 0".into(),
                    "%x = load float, ptr addrspace(1) %ap, align 4".into(),
                    "store float %x, ptr addrspace(1) %op, align 4".into(),
                ]),
                TranslationKind::ConcatAxis0 { left } => Some(vec![
                    format!("icmp ult i32 %idx, {left}"),
                    format!("sub i32 %idx, {left}"),
                    "phi float [ %left_value, %left ], [ %right_value, %right ]".into(),
                    "store float %value, ptr addrspace(1) %op, align 4".into(),
                ]),
                TranslationKind::ArgMaxLast { width } => Some(vec![
                    format!("mul i32 %idx, {width}"),
                    format!("icmp ult i32 %lane, {width}"),
                    "%candidate_numeric = fcmp ord float %candidate, %candidate".into(),
                    "%best_nan = fcmp uno float %best_value, %best_value".into(),
                    "%greater = fcmp ogt float %candidate, %best_value".into(),
                    "store i32 %best_lane, ptr addrspace(1) %op, align 4".into(),
                ]),
                TranslationKind::TakeAxis0 { lane } => Some(vec![
                    "load i32, ptr addrspace(1) %bp, align 4".into(),
                    format!("mul i32 %selected, {lane}"),
                    "%ap = getelementptr float, ptr addrspace(1) %a, i32 %source_idx".into(),
                    "store float %x, ptr addrspace(1) %op, align 4".into(),
                ]),
                _ => None,
            };
            if let Some(requirements) = structural_requirements {
                if !ir.contains("icmp ult i32 %idx, %length")
                    || !ir.contains("br i1 %valid, label %compute, label %done")
                    || requirements.iter().any(|required| !ir.contains(required))
                    || ir.contains("call float asm sideeffect \"add.rn.f32")
                    || ir.contains("call float asm sideeffect \"mul.rn.f32")
                {
                    return Err(invalid(format!(
                        "structural LLVM lowering mismatch at node {node}"
                    )));
                }
                audit_fp32_ptx("", asm)
                    .map_err(|err| invalid(format!("PTX node {node}: {err}")))?;
                let reads = kernel
                    .reads
                    .iter()
                    .map(|(source, size)| format!("{source}:{size}"))
                    .collect::<Vec<_>>()
                    .join(",");
                report.details.push_str(&format!(
                    "node={node} kind={} reads={reads} writes={} add.rn.f32=0 mul.rn.f32=0 relu=0 status=structural-pass\n",
                    kernel.kind.name(), kernel.writes
                ));
                continue;
            }
            let actual_add = ir.matches("call float asm sideeffect \"add.rn.f32").count();
            let actual_mul = ir.matches("call float asm sideeffect \"mul.rn.f32").count();
            let actual_relu = ir.matches("fcmp uno float").count();
            if (actual_add, actual_mul, actual_relu) != (add, mul, relu)
                || ir.matches("call float asm").count() != add + mul
                || ir.matches("fcmp ogt float").count() != relu
                || ir.matches("select i1 %").count() != relu
                || !ir.contains("icmp ult i32 %idx, %length")
                || !ir.contains("br i1 %valid, label %compute, label %done")
                || !ir.contains("store float ")
                || !ir.contains("ptr addrspace(1) %op, align 4")
                || !ir.contains("%ap = getelementptr float, ptr addrspace(1) %a")
                || !ir.contains("%x = load float, ptr addrspace(1) %ap")
            {
                return Err(invalid(format!(
                    "LLVM skeleton or FP32 semantics mismatch at node {node}"
                )));
            }
            if kernel.reads.len() >= 2
                && (!ir.contains("%bp = getelementptr float, ptr addrspace(1) %b")
                    || !ir.contains("%y = load float, ptr addrspace(1) %bp"))
            {
                return Err(invalid(format!("LLVM second input missing at node {node}")));
            }
            // Index-address obligations are checked against the known lowering
            // templates. This is a text-level check, NOT pointer/race safety
            // or translation to the physical SASS address calculations.
            let is_matmul = matches!(kernel.kind, TranslationKind::MatMul { .. });
            let lhs_index = if is_matmul { "%lhsIdx" } else { "%idx" };
            let lhs_address =
                format!("%ap = getelementptr float, ptr addrspace(1) %a, i32 {lhs_index}");
            if !ir.contains(&lhs_address) {
                return Err(invalid(format!("LLVM lhs address changed at node {node}")));
            }
            if kernel.reads.len() >= 2 {
                let rhs_index = if is_matmul {
                    "%rhsIdx"
                } else if matches!(
                    kernel.kind,
                    TranslationKind::BiasAdd { .. } | TranslationKind::BiasAddRelu { .. }
                ) {
                    "%col"
                } else {
                    "%idx"
                };
                let rhs_address =
                    format!("%bp = getelementptr float, ptr addrspace(1) %b, i32 {rhs_index}");
                if !ir.contains(&rhs_address) {
                    return Err(invalid(format!("LLVM rhs address changed at node {node}")));
                }
            }
            if kernel.reads.len() == 3
                && !ir.contains("%biasp = getelementptr float, ptr addrspace(1) %bias, i32 %col")
            {
                return Err(invalid(format!(
                    "LLVM broadcast source changed at node {node}"
                )));
            }
            let third = kernel.reads.len() == 3;
            if ir.contains("ptr addrspace(1) %bias,") != third
                || (third && !ir.contains("%bias_value = load float, ptr addrspace(1) %biasp"))
            {
                return Err(invalid(format!(
                    "LLVM kernel signature mismatch at node {node}"
                )));
            }
            let columns = kernel.kind.columns();
            if columns > 0 {
                let expected = format!("urem i32 %idx, {columns}");
                if !ir.contains(&expected) {
                    return Err(invalid(format!("broadcast column mismatch at node {node}")));
                }
            }
            if let TranslationKind::MatMul { inner, columns } = kernel.kind {
                let requirements = [
                    format!("icmp ult i32 %j, {inner}"),
                    format!("udiv i32 %idx, {columns}"),
                    format!("urem i32 %idx, {columns}"),
                    format!("mul i32 %row, {inner}"),
                    format!("mul i32 %j, {columns}"),
                    "phi float [ 0.000000e+00, %compute ], [ %updated, %body ]".into(),
                    "%product = call float asm sideeffect \"mul.rn.f32".into(),
                    "%updated = call float asm sideeffect \"add.rn.f32".into(),
                ];
                if requirements.iter().any(|needed| !ir.contains(needed)) {
                    return Err(invalid(format!(
                        "MatMul accumulation or addressing changed at node {node}"
                    )));
                }
            } else if matches!(
                kernel.kind,
                TranslationKind::AddBiasRelu { .. } | TranslationKind::MulBiasRelu { .. }
            ) {
                if !ir.contains("%inner = call float asm sideeffect")
                    || !ir.contains("%biased = call float asm sideeffect \"add.rn.f32")
                    || !ir.contains("%result = select i1 %keep")
                {
                    return Err(invalid(format!(
                        "fused three-operation chain changed at node {node}"
                    )));
                }
            } else if relu != 0
                && !(ir.contains("%fused_result = select i1 %keep_fused")
                    || ir.contains("%value = select i1 %keep"))
            {
                return Err(invalid(format!(
                    "ReLU branch semantics changed at node {node}"
                )));
            }
            // Each supported template has a fixed operand/result binding.
            // Opcode counts alone would allow an optimizer to use a different
            // source register while preserving the same totals.
            let required_binding = match kernel.kind {
                TranslationKind::Add
                | TranslationKind::BiasAdd { .. }
                | TranslationKind::AddRelu
                | TranslationKind::BiasAddRelu { .. } => Some(
                    "%value = call float asm sideeffect \"add.rn.f32 $0, $1, $2;\", \"=f,f,f\"(float %x, float %y)",
                ),
                TranslationKind::Mul | TranslationKind::MulRelu => Some(
                    "%value = call float asm sideeffect \"mul.rn.f32 $0, $1, $2;\", \"=f,f,f\"(float %x, float %y)",
                ),
                TranslationKind::Relu => {
                    Some("%value = select i1 %keep, float %x, float 0.000000e+00")
                }
                TranslationKind::MatMul { .. }
                | TranslationKind::Transpose2D { .. }
                | TranslationKind::ReverseRows2D { .. }
                | TranslationKind::ReverseColumns2D { .. }
                | TranslationKind::ConcatAxis0 { .. }
                | TranslationKind::SliceLastRow { .. }
                | TranslationKind::BroadcastScalar
                | TranslationKind::ArgMaxLast { .. }
                | TranslationKind::TakeAxis0 { .. }
                | TranslationKind::AddBiasRelu { .. }
                | TranslationKind::MulBiasRelu { .. } => None,
            };
            if required_binding.is_some_and(|binding| !ir.contains(binding)) {
                return Err(invalid(format!(
                    "LLVM result operands changed at node {node}"
                )));
            }
            if add != 0 && mul != 0 {
                let mul_at = ir
                    .find("call float asm sideeffect \"mul.rn.f32")
                    .ok_or_else(|| invalid("missing rounded multiply"))?;
                let add_at = ir
                    .find("call float asm sideeffect \"add.rn.f32")
                    .ok_or_else(|| invalid("missing rounded add"))?;
                if mul_at > add_at {
                    return Err(invalid(format!(
                        "incorrect separate multiply/add order at node {node}"
                    )));
                }
            }
            // The existing strict PTX parser checks *all* FP32 opcodes and
            // exact round counts inside each entry. An instruction in a
            // different kernel can never satisfy this kernel's obligation.
            let llvm_expectation = format!(
                "{}{}",
                "call float asm sideeffect \"add.rn.f32\n".repeat(add),
                "call float asm sideeffect \"mul.rn.f32\n".repeat(mul)
            );
            audit_fp32_ptx(&llvm_expectation, asm)
                .map_err(|err| invalid(format!("PTX node {node}: {err}")))?;
            if add != 0 && mul != 0 {
                let operations = asm
                    .lines()
                    .map(|line| line.split("//").next().unwrap_or(""))
                    .filter_map(|line| {
                        if line.contains("mul.rn.f32") {
                            Some("mul")
                        } else if line.contains("add.rn.f32") {
                            Some("add")
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                if operations != ["mul", "add"] {
                    return Err(invalid(format!(
                        "PTX separate mul/add order changed at node {node}"
                    )));
                }
            }
            report.rounded_adds += add;
            report.rounded_muls += mul;
            report.relu_guards += relu;
            let reads = kernel
                .reads
                .iter()
                .map(|(source, size)| format!("{source}:{size}"))
                .collect::<Vec<_>>()
                .join(",");
            report.details.push_str(&format!(
                "node={node} kind={} reads={reads} writes={} add.rn.f32={add} mul.rn.f32={mul} relu={relu} status=structural-pass\n",
                kernel.kind.name(), kernel.writes
            ));
        }
        // Reject unguarded instructions outside named kernels, too.
        audit_fp32_ptx(llvm, ptx).map_err(|err| invalid(format!("global PTX check: {err}")))?;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_fail_closed_on_unknown_schemas() {
        let plan = TranslationPlan {
            threads_per_block: 256,
            kernels: vec![TranslationKernel {
                node: 2,
                kind: TranslationKind::MulBiasRelu { columns: 1 },
                reads: vec![(0, 1), (1, 1), (1, 1)],
                writes: 1,
            }],
        };
        assert_eq!(TranslationPlan::from_text(&plan.to_text()).unwrap(), plan);

        let reverse = TranslationPlan {
            threads_per_block: 128,
            kernels: vec![TranslationKernel {
                node: 1,
                kind: TranslationKind::ReverseRows2D {
                    rows: 2,
                    columns: 3,
                },
                reads: vec![(0, 6)],
                writes: 6,
            }],
        };
        let reverse_text = reverse.to_text();
        assert!(reverse_text.contains("reverse_rows2d\t6\t0:6\t3\t2"));
        assert_eq!(TranslationPlan::from_text(&reverse_text).unwrap(), reverse);

        for invalid_text in [
            plan.to_text()
                .replace("vn-translation-plan-v1", "vn-translation-plan-v0"),
            plan.to_text().replace("mul_bias_relu", "unknown"),
            plan.to_text().replace("threads=256", "threads=0"),
            plan.to_text().replace("0:1", "3:1"),
            plan.to_text().replace("\t1\t0\n", "\t0\t0\n"),
        ] {
            assert!(TranslationPlan::from_text(&invalid_text).is_err());
        }
    }
}
