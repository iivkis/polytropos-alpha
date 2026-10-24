target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()

define ptx_kernel void @vn_node_1(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %length) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %threads = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %grid_base = mul i32 %block, %threads
  %idx = add i32 %grid_base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %row_base = mul i32 %idx, 4
  %first_ptr = getelementptr float, ptr addrspace(1) %a, i32 %row_base
  %first = load float, ptr addrspace(1) %first_ptr, align 4
  br label %loop
loop:
  %lane = phi i32 [ 1, %compute ], [ %next, %body ]
  %best_lane = phi i32 [ 0, %compute ], [ %chosen_lane, %body ]
  %best_value = phi float [ %first, %compute ], [ %chosen_value, %body ]
  %more = icmp ult i32 %lane, 4
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
}

