target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()

define ptx_kernel void @vn_node_4(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %length) {
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
%x = load float, ptr addrspace(1) %ap, align 4
  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx
%y = load float, ptr addrspace(1) %bp, align 4
%value = call float asm sideeffect "add.rn.f32 $0, $1, $2;", "=f,f,f"(float %x, float %y)
%op = getelementptr float, ptr addrspace(1) %out, i32 %idx
store float %value, ptr addrspace(1) %op, align 4
br label %done
done:
ret void
}

define ptx_kernel void @vn_node_5(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %length) {
entry:
%tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
%block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
%width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
%base = mul i32 %block, %width
%idx = add i32 %base, %tid
%valid = icmp ult i32 %idx, %length
br i1 %valid, label %compute, label %done
compute:
%row = udiv i32 %idx, 64
%col = urem i32 %idx, 64
br label %loop
loop:
%j = phi i32 [ 0, %compute ], [ %next, %body ]
%sum = phi float [ 0.000000e+00, %compute ], [ %updated, %body ]
%more = icmp ult i32 %j, 64
br i1 %more, label %body, label %store
body:
%lhsBase = mul i32 %row, 64
%lhsIdx = add i32 %lhsBase, %j
%rhsBase = mul i32 %j, 64
%rhsIdx = add i32 %rhsBase, %col
%ap = getelementptr float, ptr addrspace(1) %a, i32 %lhsIdx
%bp = getelementptr float, ptr addrspace(1) %b, i32 %rhsIdx
%x = load float, ptr addrspace(1) %ap, align 4
%y = load float, ptr addrspace(1) %bp, align 4
%product = call float asm sideeffect "mul.rn.f32 $0, $1, $2;", "=f,f,f"(float %x, float %y)
%updated = call float asm sideeffect "add.rn.f32 $0, $1, $2;", "=f,f,f"(float %sum, float %product)
%next = add i32 %j, 1
br label %loop
store:
%op = getelementptr float, ptr addrspace(1) %out, i32 %idx
store float %sum, ptr addrspace(1) %op, align 4
br label %done
done:
ret void
}

define ptx_kernel void @vn_node_6(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %length) {
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
%x = load float, ptr addrspace(1) %ap, align 4
  %col = urem i32 %idx, 64
%bp = getelementptr float, ptr addrspace(1) %b, i32 %col
%y = load float, ptr addrspace(1) %bp, align 4
%value = call float asm sideeffect "add.rn.f32 $0, $1, $2;", "=f,f,f"(float %x, float %y)
%op = getelementptr float, ptr addrspace(1) %out, i32 %idx
store float %value, ptr addrspace(1) %op, align 4
br label %done
done:
ret void
}

define ptx_kernel void @vn_node_7(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %length) {
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
%x = load float, ptr addrspace(1) %ap, align 4
  %nan = fcmp uno float %x, %x
%positive = fcmp ogt float %x, 0.000000e+00
%keep = or i1 %nan, %positive
%value = select i1 %keep, float %x, float 0.000000e+00
%op = getelementptr float, ptr addrspace(1) %out, i32 %idx
store float %value, ptr addrspace(1) %op, align 4
br label %done
done:
ret void
}

