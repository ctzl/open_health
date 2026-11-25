//! Rust ports of the four uncached model scripts (tools/run_{cva,activity,
//! illness,stress}_model.py). See docs/python-to-rust-plan.md.
//! Each module exposes `run(args: &[String]) -> anyhow::Result<()>` printing
//! the same JSON its Python twin prints.

pub mod activity;
pub mod cva;
pub mod illness;
pub mod stress;

/// Incremental activity recompute: the port of tools/event_cache.py
/// (EventSource/UnitCache) plus run_incremental. Owns its own sidecars next to
/// the DB; see the module docs.
pub mod incremental;

/// Shared helpers: profile.json reading (same file/keys as tools/_common.py
/// read_profile), model resolution under notes/models + ../open_oura/notes/models.
pub mod common;

