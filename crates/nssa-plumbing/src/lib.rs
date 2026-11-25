//! Rust port of the pure-Python plumbing behind `tools/run_sleep_model.py`.
//!
//! Mirrors, line for line:
//!
//! * `tools/epoch_time.py` — ring deciseconds -> wall clock ([`clock`]);
//! * `run_sleep_model.ring_bedtimes` — the ring's own bedtime periods ([`bedtimes`]);
//! * `tools/nssa_inputs.py` — ACM / IBI / temperature expansion and windowing
//!   ([`inputs`]);
//! * `tools/respiratory_rate.respiratory_series` + `run_sleep_model.breath_series`
//!   — respiratory rate from the corrected IBI stream ([`breath`]);
//! * `run_sleep_model.grid_stages` + the non-model half of `score_window` —
//!   hypnogram onto the 30 s grid and the summary arithmetic ([`grid`]).
//!
//! IBI correction is NOT re-ported: [`inputs::collect_nssa_inputs`] calls
//! `ibi_correct::correct_ibis` (crates/ibi-correct), the same code the Python shim
//! in `tools/ibi_correction.py` shells out to.
//!
//! The two TorchScript models (NSSA stager, SleepNet-BDI) stay in Python; this
//! crate produces the tensors' inputs and consumes the staged epochs back.
//!
//! Bit-exactness rules this crate follows, because the project has been burned by
//! f64-vs-f32 divergence before:
//!
//! * every value numpy computes in float32 (ACM statistics, temperature) is an
//!   `f32` here and is compared by bit pattern in the golden tests;
//! * integer arithmetic (IBI stamps, clamping, window filters, grid indices) is
//!   exact by construction;
//! * the float64 work in `breath` (resample, FIR, spectrum) is reproduced with the
//!   same formulas; the *reported* series is bit-identical because each window's
//!   value is `round(k / 60 * 60, 1)` — a function of the in-band spectral peak
//!   index, not of the spectrum's low bits. See [`breath`] for what differs.

pub mod bedtimes;
pub mod breath;
pub mod clock;
pub mod grid;
pub mod inputs;
pub mod num;
pub mod rows;

pub use bedtimes::ring_bedtimes;
pub use breath::respiratory_series;
pub use clock::{build_epochs, Clock};
pub use grid::{demographics, grid_stages, score_window, Staged, Stager};
pub use inputs::{collect_nssa_inputs, NssaInputs};
pub use rows::EventRow;
