//! Shared plumbing: DB open, profile.json (same keys as tools/_common.py),
//! model file resolution (notes/models then ../open_oura/notes/models, newest
//! glob match), and epoch dating of ring timestamps (mirror of
//! tools/epoch_time.py build_epochs/make_unix_s — check for an existing Rust
//! equivalent in oura-summary before writing a new one).
