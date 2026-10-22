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

/-- A symbolic multiplication bound, including all sign combinations. -/
theorem within_mul {a b boundA boundB : Int}
    (ha : Within boundA a) (hb : Within boundB b) :
    Within (boundA * boundB) (a * b) := by
  rcases ha with ⟨haLower, haUpper⟩
  rcases hb with ⟨hbLower, hbUpper⟩
  have hA : (a.natAbs : Int) ≤ boundA := by
    rcases Int.natAbs_eq a with h | h <;> omega
  have hB : (b.natAbs : Int) ≤ boundB := by
    rcases Int.natAbs_eq b with h | h <;> omega
  have nonnegA : 0 ≤ boundA := by omega
  have hprod : ((a * b).natAbs : Int) ≤ boundA * boundB := by
    rw [Int.natAbs_mul]
    simpa only [Int.natCast_mul] using
      (Int.mul_le_mul hA hB (by omega) nonnegA)
  unfold Within
  rcases Int.natAbs_eq (a * b) with h | h <;> omega

theorem tensor_within_mul {n : Nat} {a b : Tensor n}
    {boundA boundB : Int}
    (ha : TensorWithin boundA a) (hb : TensorWithin boundB b) :
    TensorWithin (boundA * boundB) (mul a b) := by
  intro i
  exact within_mul (ha i) (hb i)

/-- Accumulation bound for an arbitrary left-to-right sequence of terms. -/
theorem foldl_within_sum (f : Nat → Int) (bound : Int) :
    ∀ (xs : List Nat) (acc budget : Int),
      (∀ j ∈ xs, Within bound (f j)) →
      Within budget acc →
      Within (budget + (xs.length : Int) * bound)
        (xs.foldl (fun sum j => sum + f j) acc) := by
  intro xs
  induction xs with
  | nil =>
      intro acc budget _ hacc
      simpa using hacc
  | cons j rest ih =>
      intro acc budget hxs hacc
      have hj : Within bound (f j) := hxs j (by simp)
      have hrest : ∀ x ∈ rest, Within bound (f x) := by
        intro x hx
        exact hxs x (by simp [hx])
      have hstep := within_add hacc hj
      simpa [List.foldl_cons, Nat.cast_add, add_mul, add_assoc, add_comm,
        add_left_comm] using
        (ih (acc + f j) (budget + bound) hrest hstep)

/-- Row-major dot product with no FMA and a left-to-right reduction.
    Every intermediate product and partial sum has a symbolic bound. -/
theorem tensor_within_matmul {m k n : Nat}
    {a : Tensor (m * k)} {b : Tensor (k * n)}
    {boundA boundB : Int} (positive : 0 < n)
    (ha : TensorWithin boundA a) (hb : TensorWithin boundB b) :
    TensorWithin ((k : Int) * (boundA * boundB)) (matmul a b) := by
  intro i
  let f : Nat → Int := fun j =>
    flatGet a ((i.val / n) * k + j) * flatGet b (j * n + i.val % n)
  have hf : ∀ j ∈ List.range k, Within (boundA * boundB) (f j) := by
    intro j hj
    have hjlt : j < k := List.mem_range.mp hj
    have haddr := matmul_address_bounds positive i.isLt hjlt
    simpa [f, flatGet, haddr.1, haddr.2] using
      (within_mul (ha ⟨(i.val / n) * k + j, haddr.1⟩)
        (hb ⟨j * n + i.val % n, haddr.2⟩))
  have hzero : Within 0 (0 : Int) := by decide
  have h := foldl_within_sum f (boundA * boundB) (List.range k)
    0 0 hf hzero
  simpa [matmul, f, List.length_range] using h

end VeriNeural
