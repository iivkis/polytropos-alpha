/-!
Stage 7: bounds for the mathematical indexing schemes emitted by VeriNeural.

These lemmas describe natural-number indices in the *selected kernel
templates*. A caller must separately establish that grid coordinates have
not overflowed their i32 representation, that the corresponding device
allocation has the expected length, that threads use the checked predicate,
and that LLVM/PTX preserves these expressions. They do not prove those
assumptions, race freedom, FP32 semantics or GPU machine-code behavior.
-/

namespace VeriNeural

/-- An active one-output-per-thread lane accesses its output in bounds. -/
theorem pointwise_address_bound {count index : Nat}
    (active : index < count) : index < count := active

/-- Every row-wise bias address is in bounds when the bias is nonempty. -/
theorem bias_address_bound {columns index : Nat}
    (positive : 0 < columns) : index % columns < columns :=
  Nat.mod_lt _ positive

/-- Both row-major MatMul source addresses are in bounds for an active
    output lane and a valid reduction index. -/
theorem matmul_address_bounds {rows inner columns index j : Nat}
    (positive : 0 < columns)
    (active : index < rows * columns)
    (reduction : j < inner) :
    (index / columns) * inner + j < rows * inner ∧
    j * columns + index % columns < inner * columns := by
  have row : index / columns < rows :=
    (Nat.div_lt_iff_lt_mul positive).2 active
  have column : index % columns < columns :=
    Nat.mod_lt _ positive
  constructor
  · calc
      (index / columns) * inner + j <
          (index / columns) * inner + inner :=
        Nat.add_lt_add_left reduction _
      _ = (index / columns + 1) * inner := by
        simp [Nat.add_mul]
      _ ≤ rows * inner :=
        Nat.mul_le_mul_right inner (Nat.succ_le_of_lt row)
  · calc
      j * columns + index % columns < j * columns + columns :=
        Nat.add_lt_add_left column _
      _ = (j + 1) * columns := by
        simp [Nat.add_mul]
      _ ≤ inner * columns :=
        Nat.mul_le_mul_right columns (Nat.succ_le_of_lt reduction)

/-- Any two active threads with the same flat output index and the same
    block have the same lane: no duplicate output write within a block. -/
theorem same_block_lane_unique {block width laneA laneB : Nat}
    (equal_index : block * width + laneA = block * width + laneB) :
    laneA = laneB :=
  Nat.add_left_cancel equal_index

end VeriNeural
