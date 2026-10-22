//! Stage 8 boundary-only binary32 oracle, independent from host FP32 arithmetic.
//!
//! Supports finite binary32 multiplication iff the correctly rounded output
//! is a subnormal, signed zero or the smallest normal. The exact product is
//! evaluated in integer significand/exponent form, then rounded RNE on the
//! 2^-149 subnormal lattice. Addition is only supported when both operands
//! are subnormal/zero/smallest-normal and their exact sum is representable
//! within that same interval. This does NOT establish a general IEEE-754
//! model, machine execution or CUDA correctness.

fn decode_finite(bits: u32) -> Option<(bool, u128, i32)> {
    let biased = (bits >> 23) & 0xff;
    if biased == 0xff {
        return None;
    }
    let negative = (bits >> 31) != 0;
    let fraction = bits & 0x7f_ffff;
    if biased == 0 {
        Some((negative, u128::from(fraction), -149))
    } else {
        Some((
            negative,
            u128::from((1 << 23) | fraction),
            biased as i32 - 127 - 23,
        ))
    }
}

/// Independently compute RNE for a finite binary32 product at gradual
/// underflow, with integer arithmetic only. None means the result leaves
/// the supported subnormal/min-normal boundary profile.
pub fn subnormal_mul_rne_bits(lhs: u32, rhs: u32) -> Option<u32> {
    let (lhs_neg, lhs_sig, lhs_exp) = decode_finite(lhs)?;
    let (rhs_neg, rhs_sig, rhs_exp) = decode_finite(rhs)?;
    let magnitude = lhs_sig.checked_mul(rhs_sig)?;
    let shift = lhs_exp.checked_add(rhs_exp)?.checked_add(149)?;
    let rounded = if shift >= 0 {
        // checked_shl checks the shift count, not discarded high bits.
        // Bound before shifting so a huge normal product cannot wrap into
        // a plausible-looking subnormal result.
        let left = shift as u32;
        if left >= 128 || magnitude > (0x80_0000u128 >> left) {
            return None;
        }
        magnitude << left
    } else {
        let right = (-shift) as u32;
        if right >= 128 {
            0
        } else {
            let whole = magnitude >> right;
            let remainder = magnitude & ((1u128 << right) - 1);
            let half = 1u128 << (right - 1);
            let increment = remainder > half || (remainder == half && whole & 1 != 0);
            whole + u128::from(increment)
        }
    };
    if rounded > 0x80_0000 {
        return None;
    }
    let sign = (u32::from(lhs_neg ^ rhs_neg)) << 31;
    Some(sign | (rounded as u32))
}

/// Exact addition on the 2^-149 lattice, limited to subnormals, signed +0,
/// and the smallest positive or negative normal. Negative zero is not
/// supported as an operand because sign-of-zero needs a separate contract.
pub fn subnormal_add_exact_bits(lhs: u32, rhs: u32) -> Option<u32> {
    fn ticks(bits: u32) -> Option<i64> {
        if bits == 0 {
            return Some(0);
        }
        if bits == 0x8000_0000 {
            return None; // signed-zero addition requires a separate policy
        }
        let magnitude = bits & 0x7fff_ffff;
        if magnitude > 0x0080_0000 {
            return None;
        }
        let value = i64::from(magnitude);
        Some(if bits >> 31 == 0 { value } else { -value })
    }
    let sum = ticks(lhs)?.checked_add(ticks(rhs)?)?;
    if sum.unsigned_abs() > 0x80_0000 {
        return None;
    }
    let sign = if sum < 0 { 1u32 << 31 } else { 0 };
    Some(sign | (sum.unsigned_abs() as u32))
}

/// Bit-level ReLU convention used by the Stage 8 subnormal probe only.
/// Negative values, including -0, become +0. NaNs/infinities are rejected.
pub fn subnormal_relu_bits(bits: u32) -> Option<u32> {
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return None;
    }
    Some(if bits >> 31 != 0 { 0 } else { bits })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tie_to_even_and_gradual_underflow_have_independent_bit_oracles() {
        let half = 0x3f00_0000;
        let quarter_ulp = 0x3380_0000; // 2^-24
        assert_eq!(subnormal_mul_rne_bits(0x0080_0000, half), Some(0x0040_0000));
        assert_eq!(subnormal_mul_rne_bits(0x0080_0000, quarter_ulp), Some(0));
        assert_eq!(subnormal_mul_rne_bits(1, half), Some(0));
        assert_eq!(subnormal_mul_rne_bits(3, half), Some(2));
        assert_eq!(subnormal_mul_rne_bits(0x0080_0001, half), Some(0x0040_0000));
        assert_eq!(subnormal_mul_rne_bits(0x8000_0001, 0x3f80_0000), Some(0x8000_0001));
        assert_eq!(subnormal_add_exact_bits(0x007f_ffff, 1), Some(0x0080_0000));
        assert_eq!(subnormal_add_exact_bits(1, 1), Some(2));
        assert_eq!(subnormal_add_exact_bits(0x8000_0001, 1), Some(0));
        assert_eq!(subnormal_relu_bits(0x8000_0001), Some(0));
        assert_eq!(subnormal_relu_bits(0x0000_0001), Some(1));
    }

    #[test]
    fn boundary_oracle_rejects_values_and_additions_outside_domain() {
        assert_eq!(subnormal_mul_rne_bits(f32::INFINITY.to_bits(), 0x3f00_0000), None);
        assert_eq!(subnormal_mul_rne_bits(f32::NAN.to_bits(), 0x3f00_0000), None);
        assert_eq!(subnormal_mul_rne_bits(0x3f80_0000, 0x3f80_0000), None);
        assert_eq!(subnormal_add_exact_bits(0x0080_0000, 1), None);
        assert_eq!(subnormal_add_exact_bits(0x8000_0000, 0), None);
        assert_eq!(subnormal_add_exact_bits(0x3f80_0000, 0), None);
        assert_eq!(subnormal_relu_bits(f32::NAN.to_bits()), None);
    }
}
