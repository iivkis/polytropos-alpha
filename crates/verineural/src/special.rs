//! Stage 8: narrow binary32 exceptional-value semantics, independent of host FP32.
//!
//! Only the enumerated operations have a defined expected result. None means
//! outside this boundary slice, not "probably correct". Quiet-NaN payloads
//! are deliberately *not* asserted. This is neither an IEEE-754 completeness
//! claim nor a proof of GPU/SASS behavior.

/// Bit-class expected by the restricted special-value oracle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp32SpecialExpected {
    Exact(u32),
    QuietNaN,
}

impl Fp32SpecialExpected {
    /// Exact bits are required except where NaN payload/sign may vary.
    pub fn accepts(self, observed_bits: u32) -> bool {
        match self {
            Self::Exact(bits) => observed_bits == bits,
            Self::QuietNaN => is_quiet_nan(observed_bits),
        }
    }

    pub fn description(self) -> String {
        match self {
            Self::Exact(bits) => format!("exact:{bits:08x}"),
            Self::QuietNaN => "quiet_nan:any_payload".to_owned(),
        }
    }
}

const SIGN: u32 = 0x8000_0000;
const ABS_MASK: u32 = 0x7fff_ffff;
const INFINITY: u32 = 0x7f80_0000;
const MAX_FINITE: u32 = 0x7f7f_ffff;
const ONE: u32 = 0x3f80_0000;
const TWO: u32 = 0x4000_0000;

fn is_infinite(bits: u32) -> bool {
    bits & ABS_MASK == INFINITY
}

fn is_quiet_nan(bits: u32) -> bool {
    bits & 0x7f80_0000 == INFINITY && bits & 0x0040_0000 != 0
}

fn is_signaling_nan(bits: u32) -> bool {
    bits & 0x7f80_0000 == INFINITY
        && bits & 0x007f_ffff != 0
        && bits & 0x0040_0000 == 0
}

fn is_zero(bits: u32) -> bool {
    bits & ABS_MASK == 0
}

fn sign_of(bits: u32) -> u32 {
    bits & SIGN
}

/// Selected exact/special multiplication cases. No floating host operations.
pub fn special_mul_boundary(lhs: u32, rhs: u32) -> Option<Fp32SpecialExpected> {
    if is_signaling_nan(lhs) || is_signaling_nan(rhs) {
        return None;
    }
    if is_quiet_nan(lhs) || is_quiet_nan(rhs) {
        return Some(Fp32SpecialExpected::QuietNaN);
    }
    let sign = sign_of(lhs) ^ sign_of(rhs);
    if is_infinite(lhs) || is_infinite(rhs) {
        return Some(if is_zero(lhs) || is_zero(rhs) {
            Fp32SpecialExpected::QuietNaN
        } else {
            Fp32SpecialExpected::Exact(sign | INFINITY)
        });
    }
    if is_zero(lhs) || is_zero(rhs) {
        return Some(Fp32SpecialExpected::Exact(sign));
    }
    // The two exactly representable overflow witnesses; not general overflow.
    if (lhs & ABS_MASK == MAX_FINITE && rhs & ABS_MASK == TWO)
        || (rhs & ABS_MASK == MAX_FINITE && lhs & ABS_MASK == TWO)
    {
        return Some(Fp32SpecialExpected::Exact(sign | INFINITY));
    }
    // Multiplication by +/-1 preserves all *finite* binary32 magnitudes.
    if lhs & ABS_MASK == ONE {
        return Some(Fp32SpecialExpected::Exact(sign | (rhs & ABS_MASK)));
    }
    if rhs & ABS_MASK == ONE {
        return Some(Fp32SpecialExpected::Exact(sign | (lhs & ABS_MASK)));
    }
    None
}

/// Narrow exact add: signed zero, identity by zero, equal infinities,
/// opposite infinities, +/-1 cancellation and a max-finite overflow witness.
pub fn special_add_boundary(
    lhs: Fp32SpecialExpected,
    rhs_bits: u32,
) -> Option<Fp32SpecialExpected> {
    if is_signaling_nan(rhs_bits) {
        return None;
    }
    if matches!(lhs, Fp32SpecialExpected::QuietNaN) || is_quiet_nan(rhs_bits) {
        return Some(Fp32SpecialExpected::QuietNaN);
    }
    let Fp32SpecialExpected::Exact(lhs_bits) = lhs else {
        return None;
    };
    if is_signaling_nan(lhs_bits) || is_quiet_nan(lhs_bits) {
        return None;
    }
    if is_infinite(lhs_bits) || is_infinite(rhs_bits) {
        return Some(if is_infinite(lhs_bits)
            && is_infinite(rhs_bits)
            && sign_of(lhs_bits) != sign_of(rhs_bits)
        {
            Fp32SpecialExpected::QuietNaN
        } else {
            Fp32SpecialExpected::Exact(if is_infinite(lhs_bits) {
                lhs_bits
            } else {
                rhs_bits
            })
        });
    }
    if is_zero(lhs_bits) && is_zero(rhs_bits) {
        // Round-to-nearest: -0 + -0 is -0; opposite signed zeros give +0.
        return Some(Fp32SpecialExpected::Exact(sign_of(lhs_bits) & sign_of(rhs_bits)));
    }
    if is_zero(lhs_bits) {
        return Some(Fp32SpecialExpected::Exact(rhs_bits));
    }
    if is_zero(rhs_bits) {
        return Some(Fp32SpecialExpected::Exact(lhs_bits));
    }
    if lhs_bits & ABS_MASK == ONE
        && rhs_bits & ABS_MASK == ONE
        && sign_of(lhs_bits) != sign_of(rhs_bits)
    {
        return Some(Fp32SpecialExpected::Exact(0));
    }
    if lhs_bits & ABS_MASK == MAX_FINITE
        && rhs_bits & ABS_MASK == MAX_FINITE
        && sign_of(lhs_bits) == sign_of(rhs_bits)
    {
        return Some(Fp32SpecialExpected::Exact(sign_of(lhs_bits) | INFINITY));
    }
    None
}

/// ReLU as implemented by the IR: qNaN remains qNaN (payload unspecified),
/// positive FP32 values stay unchanged and negative values including -0
/// become +0.
pub fn special_relu_boundary(input: Fp32SpecialExpected) -> Fp32SpecialExpected {
    match input {
        Fp32SpecialExpected::QuietNaN => Fp32SpecialExpected::QuietNaN,
        Fp32SpecialExpected::Exact(bits) if sign_of(bits) != 0 => {
            Fp32SpecialExpected::Exact(0)
        }
        exact => exact,
    }
}

/// Composes exactly the supported Mul -> BiasAdd -> ReLU fragment.
pub fn special_mul_bias_relu_boundary(
    lhs: u32,
    rhs: u32,
    bias: u32,
) -> Option<Fp32SpecialExpected> {
    let product = special_mul_boundary(lhs, rhs)?;
    let biased = special_add_boundary(product, bias)?;
    Some(special_relu_boundary(biased))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exceptional_values_use_class_or_exact_bits_without_floating_host_ops() {
        let lanes = [
            (INFINITY, ONE, 0, Fp32SpecialExpected::Exact(INFINITY)),
            (SIGN | INFINITY, ONE, 0, Fp32SpecialExpected::Exact(0)),
            (0, INFINITY, 0, Fp32SpecialExpected::QuietNaN),
            (0x7fc0_0123, ONE, 0, Fp32SpecialExpected::QuietNaN),
            (MAX_FINITE, TWO, 0, Fp32SpecialExpected::Exact(INFINITY)),
            (SIGN | MAX_FINITE, TWO, 0, Fp32SpecialExpected::Exact(0)),
            (SIGN, ONE, 0, Fp32SpecialExpected::Exact(0)),
            (0, SIGN | ONE, SIGN, Fp32SpecialExpected::Exact(0)),
            (ONE, ONE, SIGN | ONE, Fp32SpecialExpected::Exact(0)),
            (ONE, ONE, 0, Fp32SpecialExpected::Exact(ONE)),
            (SIGN | 1, ONE, 0, Fp32SpecialExpected::Exact(0)),
            (1, ONE, 0, Fp32SpecialExpected::Exact(1)),
            (INFINITY, SIGN | ONE, 0, Fp32SpecialExpected::Exact(0)),
        ];
        for (lhs, rhs, bias, expected) in lanes {
            assert_eq!(special_mul_bias_relu_boundary(lhs, rhs, bias), Some(expected));
        }
        assert!(Fp32SpecialExpected::QuietNaN.accepts(0x7fc0_0000));
        assert!(Fp32SpecialExpected::QuietNaN.accepts(0xffd2_0000));
        assert!(!Fp32SpecialExpected::QuietNaN.accepts(0x7f80_0001));
        assert!(!Fp32SpecialExpected::QuietNaN.accepts(INFINITY));
    }

    #[test]
    fn exceptional_oracle_fails_closed_for_unmodeled_values() {
        assert_eq!(special_mul_boundary(0x7f80_0001, ONE), None);
        assert_eq!(special_add_boundary(Fp32SpecialExpected::Exact(ONE), 0x7f80_0001), None);
        assert_eq!(special_mul_boundary(0x3fc0_0000, 0x3fc0_0000), None);
        assert_eq!(
            special_add_boundary(Fp32SpecialExpected::Exact(ONE), ONE),
            None
        );
        assert_eq!(special_mul_bias_relu_boundary(ONE, ONE, 0x7f80_0001), None);
        assert_eq!(
            special_add_boundary(Fp32SpecialExpected::Exact(SIGN), SIGN),
            Some(Fp32SpecialExpected::Exact(SIGN))
        );
    }
}
