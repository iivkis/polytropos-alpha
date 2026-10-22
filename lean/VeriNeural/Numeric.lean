/-!
Stage 8: compositional integer-range lemmas for the exact-integer subset of
IEEE 754 binary32 (integers with magnitude <= 2^24). This file proves only
properties of ideal Int tensors. Connecting those bounds to *actual* FP32
arithmetic requires a separately trusted IEEE rounding/LLVM/PTX assumption:
neither the GPU, the driver nor the FP32 arithmetic is modeled here.
-/

namespace VeriNeural

def Within (bound value : Int) : Prop :=
  -bound ≤ value ∧ value ≤ bound

def TensorWithin {n : Nat} (bound : Int) (a : Tensor n) : Prop :=
  ∀ i : Fin n, Within bound (a i)

theorem within_add {a b boundA boundB : Int}
    (ha : Within boundA a) (hb : Within boundB b) :
    Within (boundA + boundB) (a + b) := by
  have ha1 := ha.1
  have ha2 := ha.2
  have hb1 := hb.1
  have hb2 := hb.2
  constructor <;> omega

theorem within_relu {a bound : Int}
    (nonnegative : 0 ≤ bound) (ha : Within bound a) :
    Within bound (max a 0) := by
  unfold Within at ha ⊢
  by_cases h : a ≤ 0
  · rw [Int.max_eq_right h]
    constructor <;> omega
  · have hpos : 0 ≤ a := by omega
    rw [Int.max_eq_left hpos]
    exact ha

theorem tensor_within_add {n : Nat} {a b : Tensor n}
    {boundA boundB : Int}
    (ha : TensorWithin boundA a) (hb : TensorWithin boundB b) :
    TensorWithin (boundA + boundB) (add a b) := by
  intro i
  exact within_add (ha i) (hb i)

theorem tensor_within_biasAdd {m n : Nat}
    {a : Tensor (m * n)} {b : Tensor n}
    {boundA boundB : Int} (positive : 0 < n)
    (ha : TensorWithin boundA a) (hb : TensorWithin boundB b) :
    TensorWithin (boundA + boundB) (biasAdd a b) := by
  intro i
  have hindex : i.val % n < n := Nat.mod_lt _ positive
  simpa [biasAdd, flatGet, hindex] using
    (within_add (ha i) (hb ⟨i.val % n, hindex⟩))

theorem tensor_within_relu {n : Nat} {a : Tensor n} {bound : Int}
    (nonnegative : 0 ≤ bound) (ha : TensorWithin bound a) :
    TensorWithin bound (relu a) := by
  intro i
  exact within_relu nonnegative (ha i)

theorem tensor_within_reshape {n : Nat} {a : Tensor n} {bound : Int}
    (ha : TensorWithin bound a) :
    TensorWithin bound (reshape a) := ha

end VeriNeural
