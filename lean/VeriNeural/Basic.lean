import Init

/-!
Stage 3 trusted Lean definitions for the supported *ideal integer* fragment.
This is NOT a model of IEEE-754, GPU instructions, or real-valued square roots.
No unproven placeholders or extra axioms are used.
-/

set_option autoImplicit false

namespace VeriNeural

def shapeSize (dims : List Nat) : Nat :=
  dims.foldl (fun acc dim => acc * dim) 1

/-- A flat tensor with exactly n addressable elements. -/
abbrev Tensor (n : Nat) := Fin n → Int

/-- A flat tensor of structural indices. This is deliberately distinct from
    floating-point/data tensors instead of encoding indices as numbers. -/
abbrev IndexTensor (n : Nat) := Fin n → Nat

def add {n : Nat} (a b : Tensor n) : Tensor n :=
  fun i => a i + b i

def mul {n : Nat} (a b : Tensor n) : Tensor n :=
  fun i => a i * b i

def relu {n : Nat} (a : Tensor n) : Tensor n :=
  fun i => max (a i) 0

/-- Ideal integer mask: strict positivity becomes 1, everything else 0.
    Finite integer values have no NaN, unlike the separate f32 host profile. -/
def positiveMask {n : Nat} (a : Tensor n) : Tensor n :=
  fun i => if 0 < a i then 1 else 0

/-- Elementwise conditional selection, no broadcasting or mutation. -/
def selectPositive {n : Nat} (mask yes no : Tensor n) : Tensor n :=
  fun i => if 0 < mask i then yes i else no i

/-- Reshape keeps the flat element order, after proving equal element counts. -/
def reshape {n : Nat} (a : Tensor n) : Tensor n := a

/-- Total access to a flat tensor. For valid Rust shapes, MatMul addresses
    lie in range; proving that correspondence is a separate obligation. -/
def flatGet {n : Nat} (a : Tensor n) (index : Nat) : Int :=
  if h : index < n then a ⟨index, h⟩ else 0

/-- Total access to an index tensor. -/
def indexGet {n : Nat} (a : IndexTensor n) (index : Nat) : Nat :=
  if h : index < n then a ⟨index, h⟩ else 0

/-- Rank-2 row-major transpose, the first supported nontrivial permutation. -/
def transpose2D {rows cols : Nat} (a : Tensor (rows * cols)) : Tensor (cols * rows) :=
  fun i => flatGet a ((i.val % rows) * cols + i.val / rows)

/-- Reverse one row-major axis encoded as prefix/width/suffix extents. -/
def reverseAxis {outer width inner : Nat}
    (a : Tensor (outer * width * inner)) : Tensor (outer * width * inner) :=
  fun i =>
    let outerIndex := i.val / (width * inner)
    let lane := (i.val / inner) % width
    let suffix := i.val % inner
    flatGet a ((outerIndex * width + (width - 1 - lane)) * inner + suffix)

/-- Concatenate two row-major tensors along one axis. -/
def concatAxis {outer leftWidth rightWidth inner : Nat}
    (a : Tensor (outer * leftWidth * inner))
    (b : Tensor (outer * rightWidth * inner)) :
    Tensor (outer * (leftWidth + rightWidth) * inner) :=
  fun i =>
    let totalWidth := leftWidth + rightWidth
    let outerIndex := i.val / (totalWidth * inner)
    let lane := (i.val / inner) % totalWidth
    let suffix := i.val % inner
    if lane < leftWidth then
      flatGet a ((outerIndex * leftWidth + lane) * inner + suffix)
    else
      flatGet b ((outerIndex * rightWidth + (lane - leftWidth)) * inner + suffix)

/-- Static contiguous row-major interval of one axis. Rust construction proves
    start + len <= width; the total Lean definition still uses flatGet. -/
def sliceAxis {outer width inner start len : Nat}
    (a : Tensor (outer * width * inner)) : Tensor (outer * len * inner) :=
  fun i =>
    let outerIndex := i.val / (len * inner)
    let lane := (i.val / inner) % len
    let suffix := i.val % inner
    flatGet a ((outerIndex * width + (start + lane)) * inner + suffix)

/-- Deterministic argmax along one axis. Strict `>` keeps the first lane on ties. -/
def argMaxFirstAxis {outer width inner : Nat}
    (a : Tensor (outer * width * inner)) : IndexTensor (outer * inner) :=
  fun i =>
    let outerIndex := i.val / inner
    let suffix := i.val % inner
    (List.range width).foldl
      (fun best lane =>
        if flatGet a ((outerIndex * width + best) * inner + suffix) <
            flatGet a ((outerIndex * width + lane) * inner + suffix) then
          lane
        else
          best)
      0

/-- Select one lane using a one-element index tensor. Out-of-range indices map
    to zero in the total Lean model; Rust construction currently exposes no
    arbitrary Index source and `argMaxFirstAxis` always returns an in-range lane. -/
def takeAxis {outer width inner : Nat}
    (a : Tensor (outer * width * inner)) (selected : IndexTensor 1) :
    Tensor (outer * inner) :=
  fun i =>
    let lane := indexGet selected 0
    flatGet a ((i.val / inner) * width * inner + lane * inner + i.val % inner)

/-- Row-major sum across an arbitrary axis, encoded by prefix, width and
    suffix element counts. The sum is a left fold starting at zero. -/
def reduceSumAxis {outer width inner : Nat}
    (a : Tensor (outer * width * inner)) : Tensor (outer * inner) :=
  fun i =>
    (List.range width).foldl
      (fun acc j =>
        acc + flatGet a ((i.val / inner) * width * inner + j * inner + i.val % inner))
      0

/-- Ideal integer row-major matrix multiplication. -/
def matmul {m k n : Nat} (a : Tensor (m * k))
    (b : Tensor (k * n)) : Tensor (m * n) :=
  fun i => (List.range k).foldl
    (fun sum j => sum + flatGet a ((i.val / n) * k + j) * flatGet b (j * n + i.val % n)) 0

/-- Ideal integer addition of a length-n bias to each of m rows. -/
def biasAdd {m n : Nat} (a : Tensor (m * n))
    (b : Tensor n) : Tensor (m * n) :=
  fun i => a i + flatGet b (i.val % n)

def Nonnegative {n : Nat} (a : Tensor n) : Prop :=
  ∀ i : Fin n, 0 ≤ a i

/-- Independent relational specification; not the definition of a kernel. -/
def PointwiseLe {n : Nat} (lower upper : Tensor n) : Prop :=
  ∀ i : Fin n, lower i ≤ upper i

/-- ReLU never decreases an ideal integer value. -/
theorem pointwise_le_relu {n : Nat} (a : Tensor n) :
    PointwiseLe a (relu a) := by
  intro i
  change a i ≤ max (a i) 0
  exact Int.le_max_left (a i) 0

/-- Nontrivial composition property: adding a proven nonnegative branch
    cannot push the other branch below its value, even through final ReLU. -/
theorem pointwise_le_relu_add_of_nonnegative {n : Nat} (a b : Tensor n)
    (hb : Nonnegative b) : PointwiseLe a (relu (add a b)) := by
  intro i
  change a i ≤ max (a i + b i) 0
  exact Int.le_trans
    (Int.le_add_of_nonneg_right (hb i))
    (Int.le_max_left (a i + b i) 0)

theorem add_pointwise {n : Nat} (a b : Tensor n) (i : Fin n) :
    add a b i = a i + b i := by
  rfl

theorem mul_pointwise {n : Nat} (a b : Tensor n) (i : Fin n) :
    mul a b i = a i * b i := by
  rfl

theorem relu_pointwise {n : Nat} (a : Tensor n) (i : Fin n) :
    relu a i = max (a i) 0 := by
  rfl

theorem positiveMask_pointwise {n : Nat} (a : Tensor n) (i : Fin n) :
    positiveMask a i = (if 0 < a i then 1 else 0) := by
  rfl

theorem selectPositive_pointwise {n : Nat} (mask yes no : Tensor n) (i : Fin n) :
    selectPositive mask yes no i = (if 0 < mask i then yes i else no i) := by
  rfl

theorem reshape_pointwise {n : Nat} (a : Tensor n) (i : Fin n) :
    reshape a i = a i := by
  rfl

theorem add_nonnegative {n : Nat} (a b : Tensor n)
    (ha : Nonnegative a) (hb : Nonnegative b) :
    Nonnegative (add a b) := by
  intro i
  exact Int.add_nonneg (ha i) (hb i)

theorem mul_nonnegative {n : Nat} (a b : Tensor n)
    (ha : Nonnegative a) (hb : Nonnegative b) :
    Nonnegative (mul a b) := by
  intro i
  exact Int.mul_nonneg (ha i) (hb i)

/-- Out-of-range flat reads yield zero, which preserves nonnegativity. -/
theorem flatGet_nonnegative {n : Nat} (a : Tensor n)
    (ha : Nonnegative a) (index : Nat) : 0 ≤ flatGet a index := by
  unfold flatGet
  split
  · rename_i h
    exact ha ⟨index, h⟩
  · decide

theorem transpose2D_nonnegative {rows cols : Nat} (a : Tensor (rows * cols))
    (ha : Nonnegative a) : Nonnegative (transpose2D a) := by
  intro i
  exact flatGet_nonnegative a ha _

theorem reverseAxis_nonnegative {outer width inner : Nat}
    (a : Tensor (outer * width * inner)) (ha : Nonnegative a) :
    Nonnegative (reverseAxis a) := by
  intro i
  exact flatGet_nonnegative a ha _

theorem concatAxis_nonnegative {outer leftWidth rightWidth inner : Nat}
    (a : Tensor (outer * leftWidth * inner))
    (b : Tensor (outer * rightWidth * inner))
    (ha : Nonnegative a) (hb : Nonnegative b) :
    Nonnegative (concatAxis a b) := by
  intro i
  unfold concatAxis
  dsimp
  split
  · exact flatGet_nonnegative a ha _
  · exact flatGet_nonnegative b hb _

theorem takeAxis_nonnegative {outer width inner : Nat}
    (a : Tensor (outer * width * inner)) (selected : IndexTensor 1)
    (ha : Nonnegative a) : Nonnegative (takeAxis a selected) := by
  intro i
  exact flatGet_nonnegative a ha _

theorem sliceAxis_nonnegative {outer width inner start len : Nat}
    (a : Tensor (outer * width * inner)) (ha : Nonnegative a) :
    Nonnegative (sliceAxis (start := start) (len := len) a) := by
  intro i
  exact flatGet_nonnegative a ha _

/-- Every intermediate accumulator stays nonnegative when the input is
    nonnegative; unlike a post-hoc range assertion, this is kernel checked. -/
theorem reduceSumAxis_nonnegative {outer width inner : Nat}
    (a : Tensor (outer * width * inner)) (ha : Nonnegative a) :
    Nonnegative (reduceSumAxis a) := by
  intro i
  have hfold :
      ∀ xs : List Nat, ∀ acc : Int, 0 ≤ acc →
        0 ≤ xs.foldl
          (fun sum j =>
            sum + flatGet a ((i.val / inner) * width * inner + j * inner + i.val % inner))
          acc := by
    intro xs
    induction xs with
    | nil =>
        intro acc hacc
        exact hacc
    | cons j rest ih =>
        intro acc hacc
        change 0 ≤ rest.foldl
          (fun sum lane =>
            sum + flatGet a ((i.val / inner) * width * inner + lane * inner + i.val % inner))
          (acc + flatGet a ((i.val / inner) * width * inner + j * inner + i.val % inner))
        exact ih _ (Int.add_nonneg hacc (flatGet_nonnegative a ha _))
  exact hfold (List.range width) 0 (by decide)

/-- Every left-to-right partial sum of row-major MatMul is nonnegative. -/
theorem matmul_nonnegative {m k n : Nat} (a : Tensor (m * k))
    (b : Tensor (k * n)) (ha : Nonnegative a) (hb : Nonnegative b) :
    Nonnegative (matmul a b) := by
  intro i
  have fold_nonnegative :
      ∀ (xs : List Nat) (acc : Int), 0 ≤ acc →
        0 ≤ xs.foldl (fun sum j =>
          sum + flatGet a ((i.val / n) * k + j) *
                flatGet b (j * n + i.val % n)) acc := by
    intro xs
    induction xs with
    | nil =>
        intro acc hacc
        exact hacc
    | cons j js ih =>
        intro acc hacc
        apply ih
        exact Int.add_nonneg hacc
          (Int.mul_nonneg
            (flatGet_nonnegative a ha _)
            (flatGet_nonnegative b hb _))
  exact fold_nonnegative (List.range k) 0 (by decide)

/-- BiasAdd uses the same row-major broadcast as the semantic IR. -/
theorem biasAdd_nonnegative {m n : Nat} (a : Tensor (m * n))
    (b : Tensor n) (ha : Nonnegative a) (hb : Nonnegative b) :
    Nonnegative (biasAdd a b) := by
  intro i
  exact Int.add_nonneg (ha i) (flatGet_nonnegative b hb (i.val % n))

theorem relu_nonnegative {n : Nat} (a : Tensor n) :
    Nonnegative (relu a) := by
  intro i
  exact Int.le_max_right (a i) 0

theorem positiveMask_nonnegative {n : Nat} (a : Tensor n) :
    Nonnegative (positiveMask a) := by
  intro i
  unfold positiveMask
  split <;> decide

theorem selectPositive_nonnegative {n : Nat} (mask yes no : Tensor n)
    (hy : Nonnegative yes) (hn : Nonnegative no) :
    Nonnegative (selectPositive mask yes no) := by
  intro i
  unfold selectPositive
  split
  · exact hy i
  · exact hn i

theorem reshape_nonnegative {n : Nat} (a : Tensor n)
    (ha : Nonnegative a) : Nonnegative (reshape a) := by
  exact ha

end VeriNeural
