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

/-- Reshape keeps the flat element order, after proving equal element counts. -/
def reshape {n : Nat} (a : Tensor n) : Tensor n := a

/-- Total access to a flat tensor. For valid Rust shapes, MatMul addresses
    lie in range; proving that correspondence is a separate obligation. -/
def flatGet {n : Nat} (a : Tensor n) (index : Nat) : Int :=
  if h : index < n then a ⟨index, h⟩ else 0

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

theorem add_pointwise {n : Nat} (a b : Tensor n) (i : Fin n) :
    add a b i = a i + b i := by
  rfl

theorem mul_pointwise {n : Nat} (a b : Tensor n) (i : Fin n) :
    mul a b i = a i * b i := by
  rfl

theorem relu_pointwise {n : Nat} (a : Tensor n) (i : Fin n) :
    relu a i = max (a i) 0 := by
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
  · exact le_refl 0

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
  exact fold_nonnegative (List.range k) 0 (le_refl 0)

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

theorem reshape_nonnegative {n : Nat} (a : Tensor n)
    (ha : Nonnegative a) : Nonnegative (reshape a) := by
  exact ha

end VeriNeural
