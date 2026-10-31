//! Architecture-neutral LLVM/NVPTX kernel templates.

use super::*;

pub(super) fn kernel_ir(index: usize, op: &Operation, columns: usize, fuse_relu: bool) -> String {
    // This is a single output per CUDA thread, without shared memory or
    // synchronization. The optional rewrite consumes two source buffers.
    let computation = match op {
        Operation::Add => String::from(
            "  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx\n\
               %y = load float, ptr addrspace(1) %bp, align 4\n\
               %value = call float asm sideeffect \"add.rn.f32 $0, $1, $2;\", \"=f,f,f\"(float %x, float %y)\n",
        ),
        Operation::Mul => String::from(
            "  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx\n\
               %y = load float, ptr addrspace(1) %bp, align 4\n\
               %value = call float asm sideeffect \"mul.rn.f32 $0, $1, $2;\", \"=f,f,f\"(float %x, float %y)\n",
        ),
        Operation::Pointwise(spec) => match supported_pointwise(spec) {
            Some(TranslationKind::Sub) => String::from(
                "  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx\n\
                   %y = load float, ptr addrspace(1) %bp, align 4\n\
                   %value = call float asm sideeffect \"sub.rn.f32 $0, $1, $2;\", \"=f,f,f\"(float %x, float %y)\n",
            ),
            Some(TranslationKind::DivRne) => String::from(
                "  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx\n\
                   %y = load float, ptr addrspace(1) %bp, align 4\n\
                   %value = call float asm sideeffect \"div.rn.f32 $0, $1, $2;\", \"=f,f,f\"(float %x, float %y)\n",
            ),
            Some(TranslationKind::SqrtRne) => String::from(
                "  %value = call float asm sideeffect \"sqrt.rn.f32 $0, $1;\", \"=f,f\"(float %x)\n",
            ),
            Some(TranslationKind::ExpApprox) => format!(
                "  %scaled = call float asm sideeffect \"mul.rn.f32 $0, $1, $2;\", \"=f,f,f\"(float %x, float {LOG2_E_F32_LLVM})\n\
                   %value = call float asm sideeffect \"ex2.approx.f32 $0, $1;\", \"=f,f\"(float %scaled)\n",
            ),
            _ => unreachable!("pointwise signatures were validated before codegen"),
        },
        Operation::BiasAdd => format!(
            "  %col = urem i32 %idx, {columns}\n\
               %bp = getelementptr float, ptr addrspace(1) %b, i32 %col\n\
               %y = load float, ptr addrspace(1) %bp, align 4\n\
               %value = call float asm sideeffect \"add.rn.f32 $0, $1, $2;\", \"=f,f,f\"(float %x, float %y)\n",
        ),
        Operation::Relu => String::from(
            "  %nan = fcmp uno float %x, %x\n\
               %positive = fcmp ogt float %x, 0.000000e+00\n\
               %keep = or i1 %nan, %positive\n\
               %value = select i1 %keep, float %x, float 0.000000e+00\n",
        ),
        _ => unreachable!("only supported pointwise operations require this kernel"),
    };
    let tail = if fuse_relu {
        "  %nan_fused = fcmp uno float %value, %value\n\
           %positive_fused = fcmp ogt float %value, 0.000000e+00\n\
           %keep_fused = or i1 %nan_fused, %positive_fused\n\
           %fused_result = select i1 %keep_fused, float %value, float 0.000000e+00\n"
    } else {
        ""
    };
    let result = if fuse_relu { "%fused_result" } else { "%value" };
    pointwise_kernel_ir(index, &format!("{computation}{tail}"), result)
}

/// Opt-in strict-order fusion of a pointwise Mul followed by Add.
/// Two explicit PTX RN instructions prevent LLVM from contracting the pair
/// into FMA. %a/%b are the Mul operands; %bias is the other Add operand.
/// The original Add operand order is preserved, including signed zero/NaN
/// cases where commutation must not be assumed.
/// 3-op, 3-source fusion: Mul(a,b) -> Add(product,c) -> Mul(sum,one-of-a/b/c).
/// Strictly preserve BOTH binary operand orders and two explicit mul.rn
/// instructions; no reassociation or implicit FMA.
/// Strictly rounded, architecture-neutral 3-op pointwise chain:
/// sqrt.rn(x) → add.rn(sqrt(x), epsilon) → div.rn(numerator, sum).
/// Never contract, reorder the Add operands, or skip intermediate rounding.
/// Four sequential, individually rounded FP32 operations. Operand reuse
/// is explicit and cannot be replaced with reassociation or FMA.
/// Four separate RN operations on a full-shape, four-source pointwise
/// diamond with explicit repeated input. No reassociation or FMA.
pub(super) fn mul_diamond_add_kernel_ir(
    index: usize,
    reused: usize,
    right_mul_on_right: bool,
    left_mul_on_right: bool,
) -> String {
    let factor = if reused == 0 { "%z" } else { "%w" };
    let repeat_args = if right_mul_on_right {
        format!("(float {factor}, float %right)")
    } else {
        format!("(float %right, float {factor})")
    };
    let add_args = if left_mul_on_right {
        "(float %repeated, float %left)"
    } else {
        "(float %left, float %repeated)"
    };
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %idx
  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx
  %cp = getelementptr float, ptr addrspace(1) %c, i32 %idx
  %dp = getelementptr float, ptr addrspace(1) %d, i32 %idx
  %x = load float, ptr addrspace(1) %ap, align 4
  %y = load float, ptr addrspace(1) %bp, align 4
  %z = load float, ptr addrspace(1) %cp, align 4
  %w = load float, ptr addrspace(1) %dp, align 4
  %left = call float asm sideeffect "mul.rn.f32 $0, $1, $2;", "=f,f,f"(float %x, float %y)
  %right = call float asm sideeffect "mul.rn.f32 $0, $1, $2;", "=f,f,f"(float %z, float %w)
  %repeated = call float asm sideeffect "mul.rn.f32 $0, $1, $2;", "=f,f,f"{repeat_args}
  %value = call float asm sideeffect "add.rn.f32 $0, $1, $2;", "=f,f,f"{add_args}
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %value, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

pub(super) fn mul_mul_add_div_kernel_ir(
    index: usize,
    reused: usize,
    multiply_on_right: bool,
    add_on_right: bool,
) -> String {
    let factor = if reused == 0 { "%x" } else { "%y" };
    let product_args = if multiply_on_right {
        format!("(float {factor}, float %first)")
    } else {
        format!("(float %first, float {factor})")
    };
    let add_args = if add_on_right {
        "(float %z, float %second)"
    } else {
        "(float %second, float %z)"
    };
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %idx
  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx
  %cp = getelementptr float, ptr addrspace(1) %c, i32 %idx
  %dp = getelementptr float, ptr addrspace(1) %d, i32 %idx
  %x = load float, ptr addrspace(1) %ap, align 4
  %y = load float, ptr addrspace(1) %bp, align 4
  %z = load float, ptr addrspace(1) %cp, align 4
  %w = load float, ptr addrspace(1) %dp, align 4
  %first = call float asm sideeffect "mul.rn.f32 $0, $1, $2;", "=f,f,f"(float %x, float %y)
  %second = call float asm sideeffect "mul.rn.f32 $0, $1, $2;", "=f,f,f"{product_args}
  %third = call float asm sideeffect "add.rn.f32 $0, $1, $2;", "=f,f,f"{add_args}
  %value = call float asm sideeffect "div.rn.f32 $0, $1, $2;", "=f,f,f"(float %third, float %w)
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %value, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

pub(super) fn sqrt_add_div_kernel_ir(index: usize, sqrt_on_right: bool) -> String {
    let add_operands = if sqrt_on_right {
        "(float %y, float %root)"
    } else {
        "(float %root, float %y)"
    };
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %bias, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %idx
  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx
  %biasp = getelementptr float, ptr addrspace(1) %bias, i32 %idx
  %x = load float, ptr addrspace(1) %ap, align 4
  %y = load float, ptr addrspace(1) %bp, align 4
  %z = load float, ptr addrspace(1) %biasp, align 4
  %root = call float asm sideeffect "sqrt.rn.f32 $0, $1;", "=f,f"(float %x)
  %shifted = call float asm sideeffect "add.rn.f32 $0, $1, $2;", "=f,f,f"{add_operands}
  %value = call float asm sideeffect "div.rn.f32 $0, $1, $2;", "=f,f,f"(float %z, float %shifted)
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %value, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

pub(super) fn mul_add_mul_kernel_ir(index: usize, add_right: bool, outer_right: bool, reuse: usize) -> String {
    let add_operands = if add_right {
        "(float %z, float %product)"
    } else {
        "(float %product, float %z)"
    };
    let factor = ["%x", "%y", "%z"][reuse];
    let outer_operands = if outer_right {
        format!("(float {factor}, float %middle)")
    } else {
        format!("(float %middle, float {factor})")
    };
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %bias, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %idx
  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx
  %biasp = getelementptr float, ptr addrspace(1) %bias, i32 %idx
  %x = load float, ptr addrspace(1) %ap, align 4
  %y = load float, ptr addrspace(1) %bp, align 4
  %z = load float, ptr addrspace(1) %biasp, align 4
  %product = call float asm sideeffect "mul.rn.f32 $0, $1, $2;", "=f,f,f"(float %x, float %y)
  %middle = call float asm sideeffect "add.rn.f32 $0, $1, $2;", "=f,f,f"{add_operands}
  %value = call float asm sideeffect "mul.rn.f32 $0, $1, $2;", "=f,f,f"{outer_operands}
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %value, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

/// Three-input pointwise chain with two explicit independent RN
/// instructions, preserving both operand orders and intermediate rounding.
/// One physical CUDA Graph kernel for two independent public F32 results.
/// The four source pointers preserve the exact source operand order for both
/// operations. No FMA, fast-math, reassociation or output aliasing is allowed.
pub(super) fn dual_pointwise_kernel_ir(index: usize, first_mul: bool, second_mul: bool) -> String {
    let first = if first_mul {
        "mul.rn.f32"
    } else {
        "add.rn.f32"
    };
    let second = if second_mul {
        "mul.rn.f32"
    } else {
        "add.rn.f32"
    };
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %extra, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %idx
  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx
  %cp = getelementptr float, ptr addrspace(1) %c, i32 %idx
  %dp = getelementptr float, ptr addrspace(1) %d, i32 %idx
  %x = load float, ptr addrspace(1) %ap, align 4
  %y = load float, ptr addrspace(1) %bp, align 4
  %z = load float, ptr addrspace(1) %cp, align 4
  %w = load float, ptr addrspace(1) %dp, align 4
  %first = call float asm sideeffect "{first} $0, $1, $2;", "=f,f,f"(float %x, float %y)
  %second = call float asm sideeffect "{second} $0, $1, $2;", "=f,f,f"(float %z, float %w)
  %extra_ptr = getelementptr float, ptr addrspace(1) %extra, i32 %idx
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %first, ptr addrspace(1) %extra_ptr, align 4
  store float %second, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

pub(super) fn binary_chain_kernel_ir(
    index: usize,
    first_add: bool,
    second_add: bool,
    producer_on_right: bool,
) -> String {
    let first = if first_add {
        "add.rn.f32"
    } else {
        "mul.rn.f32"
    };
    let second = if second_add {
        "add.rn.f32"
    } else {
        "mul.rn.f32"
    };
    let operands = if producer_on_right {
        "(float %z, float %middle)"
    } else {
        "(float %middle, float %z)"
    };
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %bias, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %idx
  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx
  %biasp = getelementptr float, ptr addrspace(1) %bias, i32 %idx
  %x = load float, ptr addrspace(1) %ap, align 4
  %y = load float, ptr addrspace(1) %bp, align 4
  %z = load float, ptr addrspace(1) %biasp, align 4
  %middle = call float asm sideeffect "{first} $0, $1, $2;", "=f,f,f"(float %x, float %y)
  %value = call float asm sideeffect "{second} $0, $1, $2;", "=f,f,f"{operands}
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %value, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

pub(super) fn mul_add_kernel_ir(index: usize, multiply_on_right: bool) -> String {
    let operands = if multiply_on_right {
        "(float %z, float %product)"
    } else {
        "(float %product, float %z)"
    };
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %bias, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %idx
  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx
  %biasp = getelementptr float, ptr addrspace(1) %bias, i32 %idx
  %x = load float, ptr addrspace(1) %ap, align 4
  %y = load float, ptr addrspace(1) %bp, align 4
  %z = load float, ptr addrspace(1) %biasp, align 4
  %product = call float asm sideeffect "mul.rn.f32 $0, $1, $2;", "=f,f,f"(float %x, float %y)
  %value = call float asm sideeffect "add.rn.f32 $0, $1, $2;", "=f,f,f"{operands}
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %value, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

pub(super) fn pointwise_chain_kernel_ir(index: usize, inner_op: &Operation, columns: usize) -> String {
    let inner = match inner_op {
        Operation::Add => "add.rn.f32",
        Operation::Mul => "mul.rn.f32",
        _ => unreachable!("only Add/Mul start the supported pointwise chain"),
    };
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %bias, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %idx
  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx
  %x = load float, ptr addrspace(1) %ap, align 4
  %y = load float, ptr addrspace(1) %bp, align 4
  %inner = call float asm sideeffect "{inner} $0, $1, $2;", "=f,f,f"(float %x, float %y)
  %col = urem i32 %idx, {columns}
  %biasp = getelementptr float, ptr addrspace(1) %bias, i32 %col
  %bias_value = load float, ptr addrspace(1) %biasp, align 4
  %biased = call float asm sideeffect "add.rn.f32 $0, $1, $2;", "=f,f,f"(float %inner, float %bias_value)
  %nan = fcmp uno float %biased, %biased
  %positive = fcmp ogt float %biased, 0.000000e+00
  %keep = or i1 %nan, %positive
  %result = select i1 %keep, float %biased, float 0.000000e+00
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %result, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

pub(super) fn pointwise_kernel_ir(index: usize, computation: &str, result: &str) -> String {
    format!(
        "define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %length) {{\n\
         entry:\n\
           %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
           %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
           %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()\n\
           %base = mul i32 %block, %width\n\
           %idx = add i32 %base, %tid\n\
           %valid = icmp ult i32 %idx, %length\n\
           br i1 %valid, label %compute, label %done\n\
         compute:\n\
           %ap = getelementptr float, ptr addrspace(1) %a, i32 %idx\n\
           %x = load float, ptr addrspace(1) %ap, align 4\n\
         {computation}\
           %op = getelementptr float, ptr addrspace(1) %out, i32 %idx\n\
           store float {result}, ptr addrspace(1) %op, align 4\n\
           br label %done\n\
         done:\n\
           ret void\n\
         }}\n\n"
    )
}

// Stage 6: simple one-output-per-thread row-major MatMul.
// Deliberately no fusion, tiling, TF32 or fast-math. LLVM/CUDA remain TCB.
pub(super) fn matmul_kernel_ir(index: usize, inner: usize, columns: usize) -> String {
    format!(
        "define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %length) {{\n\
         entry:\n\
           %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
           %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
           %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()\n\
           %base = mul i32 %block, %width\n\
           %idx = add i32 %base, %tid\n\
           %valid = icmp ult i32 %idx, %length\n\
           br i1 %valid, label %compute, label %done\n\
         compute:\n\
           %row = udiv i32 %idx, {columns}\n\
           %col = urem i32 %idx, {columns}\n\
           br label %loop\n\
         loop:\n\
           %j = phi i32 [ 0, %compute ], [ %next, %body ]\n\
           %sum = phi float [ 0.000000e+00, %compute ], [ %updated, %body ]\n\
           %more = icmp ult i32 %j, {inner}\n\
           br i1 %more, label %body, label %store\n\
         body:\n\
           %lhsBase = mul i32 %row, {inner}\n\
           %lhsIdx = add i32 %lhsBase, %j\n\
           %rhsBase = mul i32 %j, {columns}\n\
           %rhsIdx = add i32 %rhsBase, %col\n\
           %ap = getelementptr float, ptr addrspace(1) %a, i32 %lhsIdx\n\
           %bp = getelementptr float, ptr addrspace(1) %b, i32 %rhsIdx\n\
           %x = load float, ptr addrspace(1) %ap, align 4\n\
           %y = load float, ptr addrspace(1) %bp, align 4\n\
           %product = call float asm sideeffect \"mul.rn.f32 $0, $1, $2;\", \"=f,f,f\"(float %x, float %y)\n\
           %updated = call float asm sideeffect \"add.rn.f32 $0, $1, $2;\", \"=f,f,f\"(float %sum, float %product)\n\
           %next = add i32 %j, 1\n\
           br label %loop\n\
         store:\n\
           %op = getelementptr float, ptr addrspace(1) %out, i32 %idx\n\
           store float %sum, ptr addrspace(1) %op, align 4\n\
           br label %done\n\
         done:\n\
           ret void\n\
         }}\n\n"
    )
}

pub(super) fn transpose2d_kernel_ir(index: usize, rows: usize, columns: usize) -> String {
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %in_row = urem i32 %idx, {rows}
  %in_col = udiv i32 %idx, {rows}
  %row_base = mul i32 %in_row, {columns}
  %source_idx = add i32 %row_base, %in_col
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %source_idx
  %x = load float, ptr addrspace(1) %ap, align 4
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %x, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

pub(super) fn reverse2d_kernel_ir(index: usize, rows: usize, columns: usize, axis: usize) -> String {
    let source = if axis == 0 {
        format!(
            "  %row = udiv i32 %idx, {columns}\n\
             %col = urem i32 %idx, {columns}\n\
             %rows_minus_one = add i32 0, {}\n\
             %reversed_row = sub i32 %rows_minus_one, %row\n\
             %row_base = mul i32 %reversed_row, {columns}\n\
             %source_idx = add i32 %row_base, %col\n",
            rows - 1
        )
    } else {
        format!(
            "  %row = udiv i32 %idx, {columns}\n\
             %col = urem i32 %idx, {columns}\n\
             %columns_minus_one = add i32 0, {}\n\
             %reversed_col = sub i32 %columns_minus_one, %col\n\
             %row_base = mul i32 %row, {columns}\n\
             %source_idx = add i32 %row_base, %reversed_col\n",
            columns - 1
        )
    };
    format!(
        "define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %length) {{\n\
         entry:\n\
           %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
           %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
           %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()\n\
           %base = mul i32 %block, %width\n\
           %idx = add i32 %base, %tid\n\
           %valid = icmp ult i32 %idx, %length\n\
           br i1 %valid, label %compute, label %done\n\
         compute:\n\
         {source}\
           %ap = getelementptr float, ptr addrspace(1) %a, i32 %source_idx\n\
           %x = load float, ptr addrspace(1) %ap, align 4\n\
           %op = getelementptr float, ptr addrspace(1) %out, i32 %idx\n\
           store float %x, ptr addrspace(1) %op, align 4\n\
           br label %done\n\
         done:\n\
           ret void\n\
         }}\n\n"
    )
}

// Contiguous structural kernels. No neural-architecture operation is hidden here.
pub(super) fn slice_last_row_kernel_ir(index: usize, start: usize) -> String {
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %source_idx = add i32 %idx, {start}
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %source_idx
  %x = load float, ptr addrspace(1) %ap, align 4
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %x, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

pub(super) fn broadcast_scalar_kernel_ir(index: usize) -> String {
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %ap = getelementptr float, ptr addrspace(1) %a, i32 0
  %x = load float, ptr addrspace(1) %ap, align 4
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %x, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

pub(super) fn concat_axis0_kernel_ir(index: usize, left: usize) -> String {
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %from_left = icmp ult i32 %idx, {left}
  br i1 %from_left, label %left, label %right
left:
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %idx
  %left_value = load float, ptr addrspace(1) %ap, align 4
  br label %store
right:
  %right_idx = sub i32 %idx, {left}
  %bp = getelementptr float, ptr addrspace(1) %b, i32 %right_idx
  %right_value = load float, ptr addrspace(1) %bp, align 4
  br label %store
store:
  %value = phi float [ %left_value, %left ], [ %right_value, %right ]
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %value, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

pub(super) fn argmax_last_kernel_ir(index: usize, width: usize) -> String {
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %threads = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %grid_base = mul i32 %block, %threads
  %idx = add i32 %grid_base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %row_base = mul i32 %idx, {width}
  %first_ptr = getelementptr float, ptr addrspace(1) %a, i32 %row_base
  %first = load float, ptr addrspace(1) %first_ptr, align 4
  br label %loop
loop:
  %lane = phi i32 [ 1, %compute ], [ %next, %body ]
  %best_lane = phi i32 [ 0, %compute ], [ %chosen_lane, %body ]
  %best_value = phi float [ %first, %compute ], [ %chosen_value, %body ]
  %more = icmp ult i32 %lane, {width}
  br i1 %more, label %body, label %store
body:
  %candidate_idx = add i32 %row_base, %lane
  %candidate_ptr = getelementptr float, ptr addrspace(1) %a, i32 %candidate_idx
  %candidate = load float, ptr addrspace(1) %candidate_ptr, align 4
  %candidate_numeric = fcmp ord float %candidate, %candidate
  %best_nan = fcmp uno float %best_value, %best_value
  %greater = fcmp ogt float %candidate, %best_value
  %better_than_best = or i1 %best_nan, %greater
  %replace = and i1 %candidate_numeric, %better_than_best
  %chosen_lane = select i1 %replace, i32 %lane, i32 %best_lane
  %chosen_value = select i1 %replace, float %candidate, float %best_value
  %next = add i32 %lane, 1
  br label %loop
store:
  %op = getelementptr i32, ptr addrspace(1) %out, i32 %idx
  store i32 %best_lane, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

pub(super) fn take_axis0_kernel_ir(index: usize, lane: usize) -> String {
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %bp = getelementptr i32, ptr addrspace(1) %b, i32 0
  %selected = load i32, ptr addrspace(1) %bp, align 4
  %selected_base = mul i32 %selected, {lane}
  %source_idx = add i32 %selected_base, %idx
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %source_idx
  %x = load float, ptr addrspace(1) %ap, align 4
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %x, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

