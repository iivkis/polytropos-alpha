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

theorem relu_nonnegative {n : Nat} (a : Tensor n) :
    Nonnegative (relu a) := by
  intro i
  exact Int.le_max_right (a i) 0

theorem reshape_nonnegative {n : Nat} (a : Tensor n)
    (ha : Nonnegative a) : Nonnegative (reshape a) := by
  exact ha

end VeriNeural
