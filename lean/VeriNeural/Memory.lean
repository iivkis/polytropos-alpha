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

/-- A row-major coordinate is in bounds when both coordinates are. -/
theorem row_major_address_bound {rows columns row column : Nat}
    (rowBound : row < rows)
    (columnBound : column < columns) :
    row * columns + column < rows * columns := by
  calc
    row * columns + column < row * columns + columns :=
      Nat.add_lt_add_left columnBound _
    _ = (row + 1) * columns := by
      simp [Nat.add_mul]
    _ ≤ rows * columns :=
      Nat.mul_le_mul_right columns (Nat.succ_le_of_lt rowBound)

/-- Rank-2 transpose source address: output [column,row] reads input [row,column]. -/
theorem transpose2d_address_bound {rows columns index : Nat}
    (positiveRows : 0 < rows)
    (active : index < rows * columns) :
    (index % rows) * columns + index / rows < rows * columns := by
  have sourceRow : index % rows < rows :=
    Nat.mod_lt _ positiveRows
  have sourceColumn : index / rows < columns := by
    apply (Nat.div_lt_iff_lt_mul positiveRows).2
    simpa [Nat.mul_comm] using active
  exact row_major_address_bound sourceRow sourceColumn

/-- Reversing a coordinate by width - 1 - lane stays inside a positive axis. -/
theorem reverse_coordinate_bound {width lane : Nat}
    (positive : 0 < width) : width - 1 - lane < width := by
  have predecessor : width - 1 < width := by
    cases width with
    | zero => simp at positive
    | succ n => simp
  calc
    width - 1 - lane ≤ width - 1 := Nat.sub_le _ _
    _ < width := predecessor

/-- Rank-2 axis-0 reversal reads an in-bounds row-major source address. -/
theorem reverse_rows2d_address_bound {rows columns index : Nat}
    (positiveRows : 0 < rows)
    (positiveColumns : 0 < columns)
    (_active : index < rows * columns) :
    (rows - 1 - index / columns) * columns + index % columns <
      rows * columns := by
  have column : index % columns < columns :=
    Nat.mod_lt _ positiveColumns
  exact row_major_address_bound (reverse_coordinate_bound positiveRows) column

/-- Rank-2 axis-1 reversal reads an in-bounds row-major source address. -/
theorem reverse_columns2d_address_bound {rows columns index : Nat}
    (positiveColumns : 0 < columns)
    (active : index < rows * columns) :
    (index / columns) * columns + (columns - 1 - index % columns) <
      rows * columns := by
  have row : index / columns < rows :=
    (Nat.div_lt_iff_lt_mul positiveColumns).2 active
  exact row_major_address_bound row (reverse_coordinate_bound positiveColumns)

/-- The right branch of a flat axis-0 concat subtracts the left extent safely. -/
theorem concat_axis0_right_address_bound {left right index : Nat}
    (rightBranch : left ≤ index)
    (active : index < left + right) :
    index - left < right :=
  Nat.sub_lt_left_of_lt_add rightBranch active

/-- The first ArgMax load is inside the selected row. -/
theorem argmax_first_address_bound {rows width index : Nat}
    (positiveWidth : 0 < width)
    (active : index < rows) :
    index * width < rows * width := by
  simpa using
    (row_major_address_bound (columns := width) active positiveWidth)

/-- Every ArgMax candidate lane is inside the selected row. -/
theorem argmax_candidate_address_bound {rows width index lane : Nat}
    (active : index < rows)
    (laneBound : lane < width) :
    index * width + lane < rows * width :=
  row_major_address_bound active laneBound

/-- Scalar axis-0 take stays inside a contiguous [choices,lane] source tensor
    when the selected lane is known to be in range. -/
theorem take_axis0_address_bound {choices lane selected index : Nat}
    (selectedBound : selected < choices)
    (active : index < lane) :
    selected * lane + index < choices * lane :=
  row_major_address_bound selectedBound active

/-- Any two active threads with the same flat output index and the same
    block have the same lane: no duplicate output write within a block. -/
theorem same_block_lane_unique {block width laneA laneB : Nat}
    (equal_index : block * width + laneA = block * width + laneB) :
    laneA = laneB :=
  Nat.add_left_cancel equal_index

/-- A contiguous single-row slice reads only within its source allocation. -/
theorem slice_last_row_address_bound {index len start sourceWidth : Nat}
    (active : index < len)
    (fits : start + len ≤ sourceWidth) :
    index + start < sourceWidth := by
  have bound : index + start < len + start :=
    Nat.add_lt_add_right active start
  exact lt_of_lt_of_le bound (by simpa [Nat.add_comm] using fits)

end VeriNeural
