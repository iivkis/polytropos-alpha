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
  have hsum : a i ≤ a i + b i := by
    have := hb i
    omega
  have hmax : a i + b i ≤ max (a i + b i) 0 :=
    Int.le_max_left (a i + b i) 0
  omega

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


namespace VeriNeural.Generated

def node_0 (x_0 : VeriNeural.Tensor 4) (_x_1 : VeriNeural.Tensor 4) (_x_2 : VeriNeural.Tensor 4) (_x_3 : VeriNeural.Tensor 4) (_x_4 : VeriNeural.Tensor 4) : VeriNeural.Tensor 4 :=
  x_0

theorem node_0_shape :
  VeriNeural.shapeSize ([1, 4] : List Nat) = 4 := by
  decide

theorem node_0_semantics (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (i : Fin 4) :
  node_0 x_0 x_1 x_2 x_3 x_4 i = (x_0) i := by
  rfl

theorem node_0_nonnegative (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (h_0 : VeriNeural.Nonnegative x_0) (_h_1 : VeriNeural.Nonnegative x_1) (_h_2 : VeriNeural.Nonnegative x_2) (_h_3 : VeriNeural.Nonnegative x_3) (_h_4 : VeriNeural.Nonnegative x_4) :
  VeriNeural.Nonnegative (node_0 x_0 x_1 x_2 x_3 x_4) := by
  exact h_0

def node_1 (_x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (_x_2 : VeriNeural.Tensor 4) (_x_3 : VeriNeural.Tensor 4) (_x_4 : VeriNeural.Tensor 4) : VeriNeural.Tensor 4 :=
  x_1

theorem node_1_shape :
  VeriNeural.shapeSize ([1, 4] : List Nat) = 4 := by
  decide

theorem node_1_semantics (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (i : Fin 4) :
  node_1 x_0 x_1 x_2 x_3 x_4 i = (x_1) i := by
  rfl

theorem node_1_nonnegative (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (_h_0 : VeriNeural.Nonnegative x_0) (h_1 : VeriNeural.Nonnegative x_1) (_h_2 : VeriNeural.Nonnegative x_2) (_h_3 : VeriNeural.Nonnegative x_3) (_h_4 : VeriNeural.Nonnegative x_4) :
  VeriNeural.Nonnegative (node_1 x_0 x_1 x_2 x_3 x_4) := by
  exact h_1

def node_2 (_x_0 : VeriNeural.Tensor 4) (_x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (_x_3 : VeriNeural.Tensor 4) (_x_4 : VeriNeural.Tensor 4) : VeriNeural.Tensor 4 :=
  x_2

theorem node_2_shape :
  VeriNeural.shapeSize ([1, 4] : List Nat) = 4 := by
  decide

theorem node_2_semantics (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (i : Fin 4) :
  node_2 x_0 x_1 x_2 x_3 x_4 i = (x_2) i := by
  rfl

theorem node_2_nonnegative (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (_h_0 : VeriNeural.Nonnegative x_0) (_h_1 : VeriNeural.Nonnegative x_1) (h_2 : VeriNeural.Nonnegative x_2) (_h_3 : VeriNeural.Nonnegative x_3) (_h_4 : VeriNeural.Nonnegative x_4) :
  VeriNeural.Nonnegative (node_2 x_0 x_1 x_2 x_3 x_4) := by
  exact h_2

def node_3 (_x_0 : VeriNeural.Tensor 4) (_x_1 : VeriNeural.Tensor 4) (_x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (_x_4 : VeriNeural.Tensor 4) : VeriNeural.Tensor 4 :=
  x_3

theorem node_3_shape :
  VeriNeural.shapeSize ([4] : List Nat) = 4 := by
  decide

theorem node_3_semantics (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (i : Fin 4) :
  node_3 x_0 x_1 x_2 x_3 x_4 i = (x_3) i := by
  rfl

theorem node_3_nonnegative (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (_h_0 : VeriNeural.Nonnegative x_0) (_h_1 : VeriNeural.Nonnegative x_1) (_h_2 : VeriNeural.Nonnegative x_2) (h_3 : VeriNeural.Nonnegative x_3) (_h_4 : VeriNeural.Nonnegative x_4) :
  VeriNeural.Nonnegative (node_3 x_0 x_1 x_2 x_3 x_4) := by
  exact h_3

def node_4 (_x_0 : VeriNeural.Tensor 4) (_x_1 : VeriNeural.Tensor 4) (_x_2 : VeriNeural.Tensor 4) (_x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) : VeriNeural.Tensor 4 :=
  x_4

theorem node_4_shape :
  VeriNeural.shapeSize ([4] : List Nat) = 4 := by
  decide

theorem node_4_semantics (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (i : Fin 4) :
  node_4 x_0 x_1 x_2 x_3 x_4 i = (x_4) i := by
  rfl

theorem node_4_nonnegative (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (_h_0 : VeriNeural.Nonnegative x_0) (_h_1 : VeriNeural.Nonnegative x_1) (_h_2 : VeriNeural.Nonnegative x_2) (_h_3 : VeriNeural.Nonnegative x_3) (h_4 : VeriNeural.Nonnegative x_4) :
  VeriNeural.Nonnegative (node_4 x_0 x_1 x_2 x_3 x_4) := by
  exact h_4

def node_5 (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) : VeriNeural.Tensor 4 :=
  VeriNeural.mul (node_0 x_0 x_1 x_2 x_3 x_4) (node_1 x_0 x_1 x_2 x_3 x_4)

theorem node_5_shape :
  VeriNeural.shapeSize ([1, 4] : List Nat) = 4 := by
  decide

theorem node_5_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_5_input_1_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_5_semantics (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (i : Fin 4) :
  node_5 x_0 x_1 x_2 x_3 x_4 i = (VeriNeural.mul (node_0 x_0 x_1 x_2 x_3 x_4) (node_1 x_0 x_1 x_2 x_3 x_4)) i := by
  rfl

def node_6 (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) : VeriNeural.Tensor 4 :=
  VeriNeural.biasAdd (m := 1) (n := 4) (node_5 x_0 x_1 x_2 x_3 x_4) (node_3 x_0 x_1 x_2 x_3 x_4)

theorem node_6_shape :
  VeriNeural.shapeSize ([1, 4] : List Nat) = 4 := by
  decide

theorem node_6_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_6_input_1_shape :
  4 = 4 := by
  decide

theorem node_6_semantics (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (i : Fin 4) :
  node_6 x_0 x_1 x_2 x_3 x_4 i = (VeriNeural.biasAdd (m := 1) (n := 4) (node_5 x_0 x_1 x_2 x_3 x_4) (node_3 x_0 x_1 x_2 x_3 x_4)) i := by
  rfl

def node_7 (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) : VeriNeural.Tensor 4 :=
  VeriNeural.relu (node_6 x_0 x_1 x_2 x_3 x_4)

theorem node_7_shape :
  VeriNeural.shapeSize ([1, 4] : List Nat) = 4 := by
  decide

theorem node_7_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_7_semantics (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (i : Fin 4) :
  node_7 x_0 x_1 x_2 x_3 x_4 i = (VeriNeural.relu (node_6 x_0 x_1 x_2 x_3 x_4)) i := by
  rfl

def node_8 (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) : VeriNeural.Tensor 4 :=
  VeriNeural.mul (node_1 x_0 x_1 x_2 x_3 x_4) (node_2 x_0 x_1 x_2 x_3 x_4)

theorem node_8_shape :
  VeriNeural.shapeSize ([1, 4] : List Nat) = 4 := by
  decide

theorem node_8_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_8_input_1_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_8_semantics (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (i : Fin 4) :
  node_8 x_0 x_1 x_2 x_3 x_4 i = (VeriNeural.mul (node_1 x_0 x_1 x_2 x_3 x_4) (node_2 x_0 x_1 x_2 x_3 x_4)) i := by
  rfl

def node_9 (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) : VeriNeural.Tensor 4 :=
  VeriNeural.biasAdd (m := 1) (n := 4) (node_8 x_0 x_1 x_2 x_3 x_4) (node_4 x_0 x_1 x_2 x_3 x_4)

theorem node_9_shape :
  VeriNeural.shapeSize ([1, 4] : List Nat) = 4 := by
  decide

theorem node_9_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_9_input_1_shape :
  4 = 4 := by
  decide

theorem node_9_semantics (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (i : Fin 4) :
  node_9 x_0 x_1 x_2 x_3 x_4 i = (VeriNeural.biasAdd (m := 1) (n := 4) (node_8 x_0 x_1 x_2 x_3 x_4) (node_4 x_0 x_1 x_2 x_3 x_4)) i := by
  rfl

def node_10 (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) : VeriNeural.Tensor 4 :=
  VeriNeural.relu (node_9 x_0 x_1 x_2 x_3 x_4)

theorem node_10_shape :
  VeriNeural.shapeSize ([1, 4] : List Nat) = 4 := by
  decide

theorem node_10_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_10_semantics (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (i : Fin 4) :
  node_10 x_0 x_1 x_2 x_3 x_4 i = (VeriNeural.relu (node_9 x_0 x_1 x_2 x_3 x_4)) i := by
  rfl

def node_11 (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) : VeriNeural.Tensor 4 :=
  VeriNeural.add (node_7 x_0 x_1 x_2 x_3 x_4) (node_10 x_0 x_1 x_2 x_3 x_4)

theorem node_11_shape :
  VeriNeural.shapeSize ([1, 4] : List Nat) = 4 := by
  decide

theorem node_11_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_11_input_1_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_11_semantics (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (i : Fin 4) :
  node_11 x_0 x_1 x_2 x_3 x_4 i = (VeriNeural.add (node_7 x_0 x_1 x_2 x_3 x_4) (node_10 x_0 x_1 x_2 x_3 x_4)) i := by
  rfl

def node_12 (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) : VeriNeural.Tensor 4 :=
  VeriNeural.relu (node_11 x_0 x_1 x_2 x_3 x_4)

theorem node_12_shape :
  VeriNeural.shapeSize ([1, 4] : List Nat) = 4 := by
  decide

theorem node_12_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_12_semantics (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (i : Fin 4) :
  node_12 x_0 x_1 x_2 x_3 x_4 i = (VeriNeural.relu (node_11 x_0 x_1 x_2 x_3 x_4)) i := by
  rfl

def node_13 (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) : VeriNeural.Tensor 4 :=
  VeriNeural.mul (node_0 x_0 x_1 x_2 x_3 x_4) (node_2 x_0 x_1 x_2 x_3 x_4)

theorem node_13_shape :
  VeriNeural.shapeSize ([1, 4] : List Nat) = 4 := by
  decide

theorem node_13_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_13_input_1_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_13_semantics (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (i : Fin 4) :
  node_13 x_0 x_1 x_2 x_3 x_4 i = (VeriNeural.mul (node_0 x_0 x_1 x_2 x_3 x_4) (node_2 x_0 x_1 x_2 x_3 x_4)) i := by
  rfl

def node_14 (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) : VeriNeural.Tensor 4 :=
  VeriNeural.biasAdd (m := 1) (n := 4) (node_13 x_0 x_1 x_2 x_3 x_4) (node_4 x_0 x_1 x_2 x_3 x_4)

theorem node_14_shape :
  VeriNeural.shapeSize ([1, 4] : List Nat) = 4 := by
  decide

theorem node_14_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_14_input_1_shape :
  4 = 4 := by
  decide

theorem node_14_semantics (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (i : Fin 4) :
  node_14 x_0 x_1 x_2 x_3 x_4 i = (VeriNeural.biasAdd (m := 1) (n := 4) (node_13 x_0 x_1 x_2 x_3 x_4) (node_4 x_0 x_1 x_2 x_3 x_4)) i := by
  rfl

def node_15 (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) : VeriNeural.Tensor 4 :=
  VeriNeural.relu (node_14 x_0 x_1 x_2 x_3 x_4)

theorem node_15_shape :
  VeriNeural.shapeSize ([1, 4] : List Nat) = 4 := by
  decide

theorem node_15_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_15_semantics (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (i : Fin 4) :
  node_15 x_0 x_1 x_2 x_3 x_4 i = (VeriNeural.relu (node_14 x_0 x_1 x_2 x_3 x_4)) i := by
  rfl

def node_16 (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) : VeriNeural.Tensor 4 :=
  VeriNeural.mul (node_2 x_0 x_1 x_2 x_3 x_4) (node_1 x_0 x_1 x_2 x_3 x_4)

theorem node_16_shape :
  VeriNeural.shapeSize ([1, 4] : List Nat) = 4 := by
  decide

theorem node_16_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_16_input_1_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_16_semantics (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (i : Fin 4) :
  node_16 x_0 x_1 x_2 x_3 x_4 i = (VeriNeural.mul (node_2 x_0 x_1 x_2 x_3 x_4) (node_1 x_0 x_1 x_2 x_3 x_4)) i := by
  rfl

def node_17 (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) : VeriNeural.Tensor 4 :=
  VeriNeural.biasAdd (m := 1) (n := 4) (node_16 x_0 x_1 x_2 x_3 x_4) (node_3 x_0 x_1 x_2 x_3 x_4)

theorem node_17_shape :
  VeriNeural.shapeSize ([1, 4] : List Nat) = 4 := by
  decide

theorem node_17_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_17_input_1_shape :
  4 = 4 := by
  decide

theorem node_17_semantics (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (i : Fin 4) :
  node_17 x_0 x_1 x_2 x_3 x_4 i = (VeriNeural.biasAdd (m := 1) (n := 4) (node_16 x_0 x_1 x_2 x_3 x_4) (node_3 x_0 x_1 x_2 x_3 x_4)) i := by
  rfl

def node_18 (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) : VeriNeural.Tensor 4 :=
  VeriNeural.relu (node_17 x_0 x_1 x_2 x_3 x_4)

theorem node_18_shape :
  VeriNeural.shapeSize ([1, 4] : List Nat) = 4 := by
  decide

theorem node_18_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_18_semantics (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (i : Fin 4) :
  node_18 x_0 x_1 x_2 x_3 x_4 i = (VeriNeural.relu (node_17 x_0 x_1 x_2 x_3 x_4)) i := by
  rfl

def node_19 (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) : VeriNeural.Tensor 4 :=
  VeriNeural.add (node_15 x_0 x_1 x_2 x_3 x_4) (node_18 x_0 x_1 x_2 x_3 x_4)

theorem node_19_shape :
  VeriNeural.shapeSize ([1, 4] : List Nat) = 4 := by
  decide

theorem node_19_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_19_input_1_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_19_semantics (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (i : Fin 4) :
  node_19 x_0 x_1 x_2 x_3 x_4 i = (VeriNeural.add (node_15 x_0 x_1 x_2 x_3 x_4) (node_18 x_0 x_1 x_2 x_3 x_4)) i := by
  rfl

def node_20 (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) : VeriNeural.Tensor 4 :=
  VeriNeural.relu (node_19 x_0 x_1 x_2 x_3 x_4)

theorem node_20_shape :
  VeriNeural.shapeSize ([1, 4] : List Nat) = 4 := by
  decide

theorem node_20_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_20_semantics (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (i : Fin 4) :
  node_20 x_0 x_1 x_2 x_3 x_4 i = (VeriNeural.relu (node_19 x_0 x_1 x_2 x_3 x_4)) i := by
  rfl

def node_21 (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) : VeriNeural.Tensor 4 :=
  VeriNeural.add (node_12 x_0 x_1 x_2 x_3 x_4) (node_20 x_0 x_1 x_2 x_3 x_4)

theorem node_21_shape :
  VeriNeural.shapeSize ([1, 4] : List Nat) = 4 := by
  decide

theorem node_21_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_21_input_1_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_21_semantics (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (i : Fin 4) :
  node_21 x_0 x_1 x_2 x_3 x_4 i = (VeriNeural.add (node_12 x_0 x_1 x_2 x_3 x_4) (node_20 x_0 x_1 x_2 x_3 x_4)) i := by
  rfl

def node_22 (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) : VeriNeural.Tensor 4 :=
  VeriNeural.relu (node_21 x_0 x_1 x_2 x_3 x_4)

theorem node_22_shape :
  VeriNeural.shapeSize ([1, 4] : List Nat) = 4 := by
  decide

theorem node_22_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem node_22_semantics (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) (i : Fin 4) :
  node_22 x_0 x_1 x_2 x_3 x_4 i = (VeriNeural.relu (node_21 x_0 x_1 x_2 x_3 x_4)) i := by
  rfl

theorem module_0_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem module_0_input_1_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem module_0_input_2_shape :
  ([4] : List Nat) = ([4] : List Nat) := by
  decide

theorem module_0_output_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem module_1_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem module_1_input_1_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem module_1_input_2_shape :
  ([4] : List Nat) = ([4] : List Nat) := by
  decide

theorem module_1_output_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem module_2_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem module_2_input_1_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem module_2_input_2_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem module_2_input_3_shape :
  ([4] : List Nat) = ([4] : List Nat) := by
  decide

theorem module_2_input_4_shape :
  ([4] : List Nat) = ([4] : List Nat) := by
  decide

theorem module_2_output_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem module_3_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem module_3_input_1_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem module_3_input_2_shape :
  ([4] : List Nat) = ([4] : List Nat) := by
  decide

theorem module_3_output_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem module_4_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem module_4_input_1_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem module_4_input_2_shape :
  ([4] : List Nat) = ([4] : List Nat) := by
  decide

theorem module_4_output_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem module_5_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem module_5_input_1_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem module_5_input_2_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem module_5_input_3_shape :
  ([4] : List Nat) = ([4] : List Nat) := by
  decide

theorem module_5_input_4_shape :
  ([4] : List Nat) = ([4] : List Nat) := by
  decide

theorem module_5_output_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem module_6_input_0_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem module_6_input_1_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem module_6_input_2_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem module_6_input_3_shape :
  ([4] : List Nat) = ([4] : List Nat) := by
  decide

theorem module_6_input_4_shape :
  ([4] : List Nat) = ([4] : List Nat) := by
  decide

theorem module_6_output_shape :
  ([1, 4] : List Nat) = ([1, 4] : List Nat) := by
  decide

theorem architecture_property_coupled_branch_lower_bound (x_0 : VeriNeural.Tensor 4) (x_1 : VeriNeural.Tensor 4) (x_2 : VeriNeural.Tensor 4) (x_3 : VeriNeural.Tensor 4) (x_4 : VeriNeural.Tensor 4) : VeriNeural.PointwiseLe (node_12 x_0 x_1 x_2 x_3 x_4) (node_22 x_0 x_1 x_2 x_3 x_4) := by
  change VeriNeural.PointwiseLe (node_12 x_0 x_1 x_2 x_3 x_4) (VeriNeural.relu (VeriNeural.add (node_12 x_0 x_1 x_2 x_3 x_4) (node_20 x_0 x_1 x_2 x_3 x_4)))
  exact VeriNeural.pointwise_le_relu_add_of_nonnegative (node_12 x_0 x_1 x_2 x_3 x_4) (node_20 x_0 x_1 x_2 x_3 x_4) (VeriNeural.relu_nonnegative (node_19 x_0 x_1 x_2 x_3 x_4))

end VeriNeural.Generated
