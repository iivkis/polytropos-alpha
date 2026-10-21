/-! Stage 7 ideal Int rules for experimental Relu-after-binary fusion.
    These theorems do not describe FP32 rounding, LLVM, PTX, or a GPU device.
    The Rust code generator and mapping from IR to these definitions remain TCB.
-/

namespace VeriNeural

theorem relu_add_fusion {n : Nat} (a b : Tensor n) :
    relu (add a b) = (fun i => max (a i + b i) 0) := by
  funext i
  rfl

theorem relu_mul_fusion {n : Nat} (a b : Tensor n) :
    relu (mul a b) = (fun i => max (a i * b i) 0) := by
  funext i
  rfl

theorem relu_bias_fusion {m n : Nat} (a : Tensor (m * n)) (b : Tensor n) :
    relu (biasAdd a b) = (fun i => max (a i + flatGet b (i.val % n)) 0) := by
  funext i
  rfl

end VeriNeural
