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

instance withinDecidable (bound value : Int) : Decidable (Within bound value) := by
  unfold Within
  infer_instance

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
      have hbudget :
          budget + ((j :: rest).length : Int) * bound =
            (budget + bound) + (rest.length : Int) * bound := by
        simp only [List.length_cons, Int.natCast_add, Int.natCast_one,
          Int.add_mul, Int.one_mul]
        omega
      rw [List.foldl_cons, hbudget]
      exact ih (acc + f j) (budget + bound) hrest hstep

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
  have hzero : Within 0 (0 : Int) := by
    unfold Within
    constructor <;> omega
  have h := foldl_within_sum f (boundA * boundB) (List.range k)
    0 0 hf hzero
  simpa [matmul, f, List.length_range] using h


/-!
A second Stage 8 slice models round-to-nearest, ties-to-even for concrete
dyadic traces. Rust chooses a power-of-two tick lattice per node: additive
operators align to the finer input lattice, while products use the sum of the
operand fractional widths. Source inputs use 1..30 fractional bits; internal
product lattices are capped separately and every concrete value/budget is
checked for i64 overflow and normal finite binary32 encodability.

Binary32RneTick does not claim that LLVM/PTX/GPU hardware followed this
relation. It is the formal target relation used by the independent integer
rounding oracle and concrete Lean trace.
-/

def ApproxWithin (epsilon ideal actual : Int) : Prop :=
  Within epsilon (actual - ideal)

theorem approx_refl (a : Int) : ApproxWithin 0 a a := by
  unfold ApproxWithin Within
  constructor <;> omega

theorem approx_weaken {small big ideal actual : Int}
    (hbound : small ≤ big) (h : ApproxWithin small ideal actual) :
    ApproxWithin big ideal actual := by
  unfold ApproxWithin Within at h ⊢
  constructor <;> omega

theorem approx_add {idealA actualA idealB actualB epsilonA epsilonB : Int}
    (ha : ApproxWithin epsilonA idealA actualA)
    (hb : ApproxWithin epsilonB idealB actualB) :
    ApproxWithin (epsilonA + epsilonB)
      (idealA + idealB) (actualA + actualB) := by
  unfold ApproxWithin Within at ha hb ⊢
  constructor <;> omega

theorem approx_trans {ideal middle actual epsilonA epsilonB : Int}
    (ha : ApproxWithin epsilonA ideal middle)
    (hb : ApproxWithin epsilonB middle actual) :
    ApproxWithin (epsilonA + epsilonB) ideal actual := by
  unfold ApproxWithin Within at ha hb ⊢
  constructor <;> omega

theorem approx_add_round
    {idealA actualA idealB actualB rounded epsilonA epsilonB epsilonRound : Int}
    (ha : ApproxWithin epsilonA idealA actualA)
    (hb : ApproxWithin epsilonB idealB actualB)
    (hr : ApproxWithin epsilonRound (actualA + actualB) rounded) :
    ApproxWithin (epsilonA + epsilonB + epsilonRound)
      (idealA + idealB) rounded := by
  have hsum := approx_add ha hb
  have h := approx_trans hsum hr
  simpa [Int.add_assoc] using h

theorem approx_scale {epsilon ideal actual scale : Int}
    (nonnegative : 0 ≤ scale)
    (h : ApproxWithin epsilon ideal actual) :
    ApproxWithin (epsilon * scale) (ideal * scale) (actual * scale) := by
  unfold ApproxWithin at h ⊢
  have hscale : Within scale scale := by
    unfold Within
    constructor <;> omega
  have hmul := within_mul h hscale
  simpa only [Int.sub_mul] using hmul

/-- Product error propagation on the product tick lattice.
    The asymmetric form uses a bound on idealA and actualB; callers may choose
    either operand order if a tighter concrete bound is useful. -/
theorem approx_mul
    {idealA actualA idealB actualB epsilonA epsilonB boundIdealA boundActualB : Int}
    (ha : ApproxWithin epsilonA idealA actualA)
    (hb : ApproxWithin epsilonB idealB actualB)
    (hidealA : Within boundIdealA idealA)
    (hactualB : Within boundActualB actualB) :
    ApproxWithin (epsilonA * boundActualB + boundIdealA * epsilonB)
      (idealA * idealB) (actualA * actualB) := by
  unfold ApproxWithin at ha hb ⊢
  have hleft := within_mul ha hactualB
  have hright := within_mul hidealA hb
  have hsum := within_add hleft hright
  unfold Within at hsum ⊢
  rw [Int.sub_mul, Int.mul_sub] at hsum
  rcases hsum with ⟨hlower, hupper⟩
  constructor <;> omega

theorem approx_mul_round
    {idealA actualA idealB actualB rounded epsilonA epsilonB
      boundIdealA boundActualB epsilonRound : Int}
    (ha : ApproxWithin epsilonA idealA actualA)
    (hb : ApproxWithin epsilonB idealB actualB)
    (hidealA : Within boundIdealA idealA)
    (hactualB : Within boundActualB actualB)
    (hr : ApproxWithin epsilonRound (actualA * actualB) rounded) :
    ApproxWithin
      (epsilonA * boundActualB + boundIdealA * epsilonB + epsilonRound)
      (idealA * idealB) rounded := by
  exact approx_trans (approx_mul ha hb hidealA hactualB) hr

theorem approx_relu {ideal actual epsilon : Int}
    (h : ApproxWithin epsilon ideal actual) :
    ApproxWithin epsilon (max ideal 0) (max actual 0) := by
  unfold ApproxWithin Within at h ⊢
  rcases h with ⟨hlower, hupper⟩
  by_cases hi : ideal ≤ 0
  · rw [Int.max_eq_right hi]
    by_cases ha : actual ≤ 0
    · rw [Int.max_eq_right ha]
      constructor <;> omega
    · have ha0 : 0 ≤ actual := by omega
      rw [Int.max_eq_left ha0]
      constructor <;> omega
  · have hi0 : 0 ≤ ideal := by omega
    rw [Int.max_eq_left hi0]
    by_cases ha : actual ≤ 0
    · rw [Int.max_eq_right ha]
      constructor <;> omega
    · have ha0 : 0 ≤ actual := by omega
      rw [Int.max_eq_left ha0]
      exact ⟨hlower, hupper⟩

def binary32Step (shift : Nat) : Int :=
  Int.ofNat (2 ^ shift)

/-- RNE on a dyadic integer lattice.
    shift = 0 means the exact tick magnitude is below 2^24 and therefore
    exactly representable. For shift > 0, one binary32 ulp is 2^shift ticks.
    The final clause is nearest with an explicit ties-to-even rule. -/
def Binary32RneTick (exact rounded : Int) (shift : Nat) : Prop :=
  let step := binary32Step shift
  let magnitude : Int := Int.ofNat exact.natAbs
  0 < step ∧
    ((shift = 0 ∧ magnitude < 16777216) ∨
      (0 < shift ∧ 8388608 * step ≤ magnitude ∧
        magnitude < 16777216 * step)) ∧
    rounded % step = 0 ∧
    let distance : Int := Int.ofNat (rounded - exact).natAbs
    (2 * distance < step ∨
      (2 * distance = step ∧ (rounded / step) % 2 = 0))

instance binary32RneTickDecidable (exact rounded : Int) (shift : Nat) :
    Decidable (Binary32RneTick exact rounded shift) := by
  unfold Binary32RneTick
  infer_instance

/-- RNE implies a conservative one-ulp absolute error bound in tick units.
    This deliberately keeps the proof simple; concrete traces may observe a
    smaller half-ulp error, but composition uses this proved budget. -/
theorem binary32_rne_within_step {exact rounded : Int} {shift : Nat}
    (h : Binary32RneTick exact rounded shift) :
    ApproxWithin (binary32Step shift) exact rounded := by
  simp only [Binary32RneTick] at h
  rcases h with ⟨_hstep, _hbin, _halign, hnearest⟩
  have hdistance :
      Int.ofNat (rounded - exact).natAbs ≤ binary32Step shift := by
    rcases hnearest with hlt | heq
    · omega
    · omega
  have hupper :
      rounded - exact ≤ Int.ofNat (rounded - exact).natAbs :=
    Int.le_natAbs
  have hlower :
      -(rounded - exact) ≤ Int.ofNat (rounded - exact).natAbs := by
    simpa only [Int.natAbs_neg] using
      (Int.le_natAbs (a := -(rounded - exact)))
  unfold ApproxWithin Within
  constructor <;> omega

/-!
Reusable range/error contracts for the dyadic RNE slice.

These theorems are independent of one concrete value trace. They state how an
input ideal-value bound and an absolute error budget propagate through one
scalar operation, assuming the actual arithmetic result satisfies the same
Binary32RneTick target relation used by the concrete oracle. Rust instantiates
the numeric bounds and maximum admissible RNE shift for graph/module contracts.
The LLVM/PTX/GPU implementation remains outside this proof boundary.
-/

theorem within_natAbs_le {bound value : Int} (h : Within bound value) :
    Int.ofNat value.natAbs ≤ bound := by
  rw [Int.ofNat_eq_natCast]
  rcases h with ⟨hlower, hupper⟩
  by_cases hnonneg : 0 ≤ value
  · have habs : (value.natAbs : Int) = value :=
      Int.natAbs_of_nonneg hnonneg
    rw [habs]
    exact hupper
  · have hnonpos : value ≤ 0 := by omega
    have habs : (value.natAbs : Int) = -value :=
      Int.ofNat_natAbs_of_nonpos hnonpos
    rw [habs]
    omega

theorem binary32Step_pos (shift : Nat) : 0 < binary32Step shift := by
  unfold binary32Step
  exact Int.ofNat_lt.mpr (Nat.pow_pos Nat.zero_lt_two)

theorem binary32Step_mono {a b : Nat} (h : a ≤ b) :
    binary32Step a ≤ binary32Step b := by
  unfold binary32Step
  exact Int.ofNat_le.mpr (Nat.pow_le_pow_right Nat.zero_lt_two h)

theorem binary32_rne_shift_le_of_bound
    {exact rounded bound : Int} {shift maxShift : Nat}
    (hbound : Within bound exact)
    (hlimit : bound < 8388608 * binary32Step (maxShift + 1))
    (hrne : Binary32RneTick exact rounded shift) :
    shift ≤ maxShift := by
  have hmag : Int.ofNat exact.natAbs ≤ bound :=
    within_natAbs_le hbound
  simp only [Binary32RneTick] at hrne
  rcases hrne with ⟨_hstep, hbin, _halign, _hnearest⟩
  rcases hbin with hzero | hnormal
  · omega
  · rcases hnormal with ⟨_hpositive, hlower, _hupper⟩
    by_cases hle : shift ≤ maxShift
    · exact hle
    · have hsucc : maxShift + 1 ≤ shift := by omega
      have hstep :
          binary32Step (maxShift + 1) ≤ binary32Step shift :=
        binary32Step_mono hsucc
      have hscaled :
          8388608 * binary32Step (maxShift + 1) ≤
            8388608 * binary32Step shift :=
        Int.mul_le_mul_of_nonneg_left hstep (by decide)
      omega

theorem binary32_rne_within_maxStep
    {exact rounded bound : Int} {shift maxShift : Nat}
    (hbound : Within bound exact)
    (hlimit : bound < 8388608 * binary32Step (maxShift + 1))
    (hrne : Binary32RneTick exact rounded shift) :
    ApproxWithin (binary32Step maxShift) exact rounded := by
  have hshift :=
    binary32_rne_shift_le_of_bound hbound hlimit hrne
  exact approx_weaken (binary32Step_mono hshift)
    (binary32_rne_within_step hrne)

/-- A reusable scalar numerical contract on one dyadic tick lattice. -/
def ApproxContract
    (idealBound epsilon ideal actual : Int) : Prop :=
  0 ≤ idealBound ∧
    0 ≤ epsilon ∧
    Within idealBound ideal ∧
    ApproxWithin epsilon ideal actual

theorem approx_contract_exact {bound value : Int}
    (hbound : 0 ≤ bound)
    (hvalue : Within bound value) :
    ApproxContract bound 0 value value := by
  exact ⟨hbound, by omega, hvalue, approx_refl value⟩

theorem approx_contract_actual
    {idealBound epsilon ideal actual : Int}
    (h : ApproxContract idealBound epsilon ideal actual) :
    Within (idealBound + epsilon) actual := by
  rcases h with ⟨_hbound, _hepsilon, hideal, herr⟩
  unfold ApproxWithin at herr
  unfold Within at herr hideal ⊢
  constructor <;> omega

theorem approx_contract_scale
    {idealBound epsilon ideal actual scale : Int}
    (hscale : 0 ≤ scale)
    (h : ApproxContract idealBound epsilon ideal actual) :
    ApproxContract (idealBound * scale) (epsilon * scale)
      (ideal * scale) (actual * scale) := by
  rcases h with ⟨hbound, hepsilon, hideal, herr⟩
  have hscaleWithin : Within scale scale := by
    unfold Within
    constructor <;> omega
  refine ⟨Int.mul_nonneg hbound hscale, Int.mul_nonneg hepsilon hscale, ?_, ?_⟩
  · exact within_mul hideal hscaleWithin
  · exact approx_scale hscale herr

theorem approx_contract_add_rne
    {idealBoundA epsilonA idealA actualA
      idealBoundB epsilonB idealB actualB rounded : Int}
    {shift maxShift : Nat}
    (ha : ApproxContract idealBoundA epsilonA idealA actualA)
    (hb : ApproxContract idealBoundB epsilonB idealB actualB)
    (hlimit :
      (idealBoundA + epsilonA) + (idealBoundB + epsilonB) <
        8388608 * binary32Step (maxShift + 1))
    (hrne : Binary32RneTick (actualA + actualB) rounded shift) :
    ApproxContract
      (idealBoundA + idealBoundB)
      (epsilonA + epsilonB + binary32Step maxShift)
      (idealA + idealB) rounded := by
  rcases ha with ⟨hboundA, hepsilonA, hidealA, herrA⟩
  rcases hb with ⟨hboundB, hepsilonB, hidealB, herrB⟩
  have hactualA :
      Within (idealBoundA + epsilonA) actualA :=
    approx_contract_actual ⟨hboundA, hepsilonA, hidealA, herrA⟩
  have hactualB :
      Within (idealBoundB + epsilonB) actualB :=
    approx_contract_actual ⟨hboundB, hepsilonB, hidealB, herrB⟩
  have hpre :
      Within ((idealBoundA + epsilonA) + (idealBoundB + epsilonB))
        (actualA + actualB) :=
    within_add hactualA hactualB
  have hround :
      ApproxWithin (binary32Step maxShift) (actualA + actualB) rounded :=
    binary32_rne_within_maxStep hpre hlimit hrne
  have hstep : 0 ≤ binary32Step maxShift :=
    Int.le_of_lt (binary32Step_pos maxShift)
  refine ⟨?_, ?_, within_add hidealA hidealB, ?_⟩
  · omega
  · omega
  · exact approx_add_round herrA herrB hround

theorem approx_contract_mul_rne
    {idealBoundA epsilonA idealA actualA
      idealBoundB epsilonB idealB actualB rounded : Int}
    {shift maxShift : Nat}
    (ha : ApproxContract idealBoundA epsilonA idealA actualA)
    (hb : ApproxContract idealBoundB epsilonB idealB actualB)
    (hlimit :
      (idealBoundA + epsilonA) * (idealBoundB + epsilonB) <
        8388608 * binary32Step (maxShift + 1))
    (hrne : Binary32RneTick (actualA * actualB) rounded shift) :
    ApproxContract
      (idealBoundA * idealBoundB)
      (epsilonA * (idealBoundB + epsilonB) +
        idealBoundA * epsilonB + binary32Step maxShift)
      (idealA * idealB) rounded := by
  rcases ha with ⟨hboundA, hepsilonA, hidealA, herrA⟩
  rcases hb with ⟨hboundB, hepsilonB, hidealB, herrB⟩
  have hactualA :
      Within (idealBoundA + epsilonA) actualA :=
    approx_contract_actual ⟨hboundA, hepsilonA, hidealA, herrA⟩
  have hactualB :
      Within (idealBoundB + epsilonB) actualB :=
    approx_contract_actual ⟨hboundB, hepsilonB, hidealB, herrB⟩
  have hpre :
      Within ((idealBoundA + epsilonA) * (idealBoundB + epsilonB))
        (actualA * actualB) :=
    within_mul hactualA hactualB
  have hround :
      ApproxWithin (binary32Step maxShift) (actualA * actualB) rounded :=
    binary32_rne_within_maxStep hpre hlimit hrne
  have hboundOut :
      0 ≤ idealBoundA * idealBoundB :=
    Int.mul_nonneg hboundA hboundB
  have hactualBoundB : 0 ≤ idealBoundB + epsilonB := by omega
  have htermA :
      0 ≤ epsilonA * (idealBoundB + epsilonB) :=
    Int.mul_nonneg hepsilonA hactualBoundB
  have htermB :
      0 ≤ idealBoundA * epsilonB :=
    Int.mul_nonneg hboundA hepsilonB
  have hstep : 0 ≤ binary32Step maxShift :=
    Int.le_of_lt (binary32Step_pos maxShift)
  refine ⟨hboundOut, ?_, within_mul hidealA hidealB, ?_⟩
  · omega
  · exact approx_mul_round herrA herrB hidealA hactualB hround

theorem approx_contract_relu
    {idealBound epsilon ideal actual : Int}
    (h : ApproxContract idealBound epsilon ideal actual) :
    ApproxContract idealBound epsilon (max ideal 0) (max actual 0) := by
  rcases h with ⟨hbound, hepsilon, hideal, herr⟩
  exact ⟨hbound, hepsilon, within_relu hbound hideal, approx_relu herr⟩



/-!
Stage 8 threshold slice. These lemmas select the ReLU branch only when
the observed pre-activation has enough margin relative to the epsilon bound.
The results depend on ApproxContract, not CUDA/LLVM/PTX conformance.
-/

theorem approx_contract_relu_positive_branch
    {bound epsilon ideal actual : Int}
    (h : ApproxContract bound epsilon ideal actual)
    (hmargin : epsilon < actual) :
    max ideal 0 = ideal ∧ max actual 0 = actual := by
  rcases h with ⟨_hbound, _hepsilon, _hideal, herr⟩
  unfold ApproxWithin Within at herr
  rcases herr with ⟨_hlower, hupper⟩
  constructor
  · exact Int.max_eq_left (by omega)
  · exact Int.max_eq_left (by omega)

theorem approx_contract_relu_nonpositive_branch
    {bound epsilon ideal actual : Int}
    (h : ApproxContract bound epsilon ideal actual)
    (hmargin : actual + epsilon ≤ 0) :
    max ideal 0 = 0 ∧ max actual 0 = 0 := by
  rcases h with ⟨_hbound, hepsilon, _hideal, herr⟩
  unfold ApproxWithin Within at herr
  rcases herr with ⟨hlower, _hupper⟩
  constructor
  · exact Int.max_eq_right (by omega)
  · exact Int.max_eq_right (by omega)

/-!
P1 stable branch slice for PositiveMask → SelectPositive.

The condition contract is independent from either branch contract. A select inherits
the chosen branch's numerical contract only when the observed condition is far
enough from zero to force the same strict-positive decision in the ideal and
actual values. No theorem is provided for an interval crossing zero.
-/

theorem approx_contract_positive_mask_select_positive
    {conditionBound conditionEpsilon conditionIdeal conditionActual
      yesBound yesEpsilon yesIdeal yesActual noIdeal noActual : Int}
    (hcondition :
      ApproxContract conditionBound conditionEpsilon conditionIdeal conditionActual)
    (hmargin : conditionEpsilon < conditionActual)
    (hyes : ApproxContract yesBound yesEpsilon yesIdeal yesActual) :
    ApproxContract yesBound yesEpsilon
      (if 0 < (if 0 < conditionIdeal then 1 else 0) then yesIdeal else noIdeal)
      (if 0 < (if 0 < conditionActual then 1 else 0) then yesActual else noActual) := by
  rcases hcondition with ⟨_hbound, hepsilon, _hideal, herr⟩
  unfold ApproxWithin Within at herr
  rcases herr with ⟨_hlower, hupper⟩
  have hactual : 0 < conditionActual := by omega
  have hideal : 0 < conditionIdeal := by omega
  have hzeroOne : (0 : Nat) < 1 := by decide
  rw [if_pos hideal, if_pos hzeroOne, if_pos hactual, if_pos hzeroOne]
  exact hyes

theorem approx_contract_positive_mask_select_nonpositive
    {conditionBound conditionEpsilon conditionIdeal conditionActual
      noBound noEpsilon noIdeal noActual yesIdeal yesActual : Int}
    (hcondition :
      ApproxContract conditionBound conditionEpsilon conditionIdeal conditionActual)
    (hmargin : conditionActual + conditionEpsilon ≤ 0)
    (hno : ApproxContract noBound noEpsilon noIdeal noActual) :
    ApproxContract noBound noEpsilon
      (if 0 < (if 0 < conditionIdeal then 1 else 0) then yesIdeal else noIdeal)
      (if 0 < (if 0 < conditionActual then 1 else 0) then yesActual else noActual) := by
  rcases hcondition with ⟨_hbound, hepsilon, _hideal, herr⟩
  unfold ApproxWithin Within at herr
  rcases herr with ⟨hlower, _hupper⟩
  have hactual : conditionActual ≤ 0 := by omega
  have hideal : conditionIdeal ≤ 0 := by omega
  have hactualNot : ¬ 0 < conditionActual := Int.not_lt.mpr hactual
  have hidealNot : ¬ 0 < conditionIdeal := Int.not_lt.mpr hideal
  have hzeroZero : ¬ (0 : Nat) < 0 := by decide
  rw [if_neg hidealNot, if_neg hzeroZero, if_neg hactualNot, if_neg hzeroZero]
  exact hno

end VeriNeural
