/-!
Stage 8 gradual-underflow boundary arithmetic. This is a restricted
*integer lattice* model for tiny binary32 results, not a model of NVIDIA
hardware, LLVM, PTX, NaN, signed zero, or all IEEE-754 arithmetic.

One subnormal unit is 2^-149; half-units represent exact halfway values.
At a midpoint, round-to-nearest ties-to-even chooses the even integer.
-/

namespace VeriNeural.Subnormal

def roundHalfTicksToEven (twiceTicks : Nat) : Nat :=
  let whole := twiceTicks / 2
  if twiceTicks % 2 == 1 && whole % 2 == 1
  then whole + 1
  else whole

-- One half of the minimum subnormal rounds to +0.
theorem half_min_subnormal_to_zero :
    roundHalfTicksToEven 1 = 0 := by
  decide

-- Three halves of the minimum subnormal rounds to two ticks, not one.
theorem three_halves_round_to_even :
    roundHalfTicksToEven 3 = 2 := by
  decide

-- Half of the smallest positive normal is exactly 2^22 subnormal ticks.
theorem min_normal_half_is_subnormal :
    roundHalfTicksToEven 8388608 = 4194304 := by
  decide

-- 2^23 + 1 half-ticks are exactly between two subnormals;
-- the even candidate wins.
theorem just_above_normal_halfway :
    roundHalfTicksToEven 8388609 = 4194304 := by
  decide

-- The largest subnormal plus one tick becomes the smallest normal.
theorem subnormal_add_to_min_normal :
    (8388607 : Nat) + 1 = 8388608 := by
  decide

-- Do not silently equate flushing to zero with gradual underflow.
theorem gradual_underflow_differs_from_ftz :
    roundHalfTicksToEven 8388608 ≠ 0 := by
  decide

end VeriNeural.Subnormal
