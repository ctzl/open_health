//! Python-exact rounding.
//!
//! `round(x)` and `round(x, n)` in CPython are *correctly rounded decimal*
//! operations on the exact binary value of the float, with ties to even — not
//! "add 0.5 and truncate". Rust's `{:.n}` formatting is correctly rounded the same
//! way, so `format -> parse` reproduces `round(x, n)` bit for bit; `round_ties_even`
//! reproduces `round(x)` (no ndigits, integer result).

/// `round(x, 1)`
pub fn round1(x: f64) -> f64 {
    round_ndigits(x, 1)
}

/// `round(x, 3)`
pub fn round3(x: f64) -> f64 {
    round_ndigits(x, 3)
}

pub fn round_ndigits(x: f64, n: usize) -> f64 {
    // Python keeps -0.0 and NaN/inf out of these paths; formatting handles them.
    let text = format!("{x:.n$}");
    text.parse::<f64>().unwrap_or(x)
}

/// `round(x)` -> int (ties to even, on the exact binary value).
pub fn round0(x: f64) -> i64 {
    x.round_ties_even() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_python_round() {
        // Values checked against CPython: round(2.25, 1) == 2.2, round(0.15, 1) == 0.1,
        // round(2.675, 2) == 2.67, round(0.5) == 0, round(1.5) == 2, round(-2.5) == -2.
        assert_eq!(round1(2.25), 2.2);
        assert_eq!(round1(0.15), 0.1);
        assert_eq!(round1(2.675), 2.7);
        assert_eq!(round1(14.049999999999999), 14.0);
        assert_eq!(round3(0.5005), 0.5);
        assert_eq!(round0(0.5), 0);
        assert_eq!(round0(1.5), 2);
        assert_eq!(round0(-2.5), -2);
        assert_eq!(round0(95.99999999999999), 96);
    }
}
