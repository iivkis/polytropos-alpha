/-!
Stage 8 binary32 bit-class boundary. This proves selected classifications
of concrete 32-bit words under a finite exponent/fraction encoding.
It deliberately does NOT prove floating multiplication/addition, NaN
payload propagation, GPU execution, LLVM/PTX lowering or SASS behavior.
-/

namespace VeriNeural.Special

def exponent (bits : Nat) : Nat := bits / 8388608 % 256

def fraction (bits : Nat) : Nat := bits % 8388608

def isInfinity (bits : Nat) : Prop :=
  exponent bits = 255 ∧ fraction bits = 0

def isQuietNaN (bits : Nat) : Prop :=
  exponent bits = 255 ∧ 4194304 ≤ fraction bits

def isFinite (bits : Nat) : Prop :=
  exponent bits < 255

-- The two zero encodings differ even though their real values coincide.
theorem signed_zeros_have_distinct_bits :
    (0 : Nat) ≠ 2147483648 := by
  decide

theorem signed_zeros_are_finite :
    isFinite 0 ∧ isFinite 2147483648 := by
  simp [isFinite, exponent]

theorem both_infinities_are_encoded :
    isInfinity 2139095040 ∧ isInfinity 4286578688 := by
  simp [isInfinity, exponent, fraction]

theorem quiet_nan_encodings_differ_from_infinity :
    isQuietNaN 2143289344 ∧ isQuietNaN 4290772992 ∧
    ¬ isInfinity 2143289344 ∧ ¬ isInfinity 4290772992 := by
  simp [isQuietNaN, isInfinity, exponent, fraction]

theorem signaling_nan_is_not_quiet :
    ¬ isQuietNaN 2139095041 := by
  simp [isQuietNaN, exponent, fraction]

theorem maximum_finite_is_finite :
    isFinite 2139095039 := by
  simp [isFinite, exponent]

theorem smallest_subnormal_is_finite :
    isFinite 1 ∧ isFinite 2147483649 := by
  simp [isFinite, exponent]

end VeriNeural.Special
