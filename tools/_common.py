"""Shared helpers for the tools/ model runners (run_models, run_activity_model,
run_sleep_model). Imported as a sibling module — each runner's directory is on
sys.path[0] when invoked as `python tools/run_*.py`.
"""
import json
import os
import sys
from pathlib import Path


def read_profile(db_path, serial=None):
    """Demographics from `profile.json` next to the DB, for one ring.

    `profile.json` is a map of ring serial -> demographics (the dashboard is
    two-person); the `""` key is the default, used for unassigned requests and
    as the fallback for rings with no entry of their own. A legacy flat
    single-object file (pre-profiles) is treated as that default. Returns a
    plain dict of {sex, age, height_m, weight_kg, ring_size}; empty if absent.
    """
    try:
        data = json.loads((Path(db_path).parent / "profile.json").read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    if "age" in data or "sex" in data:  # legacy flat object = the only profile
        return data
    entry = data.get(serial or "") or data.get("") or {}
    return entry if isinstance(entry, dict) else {}


def resolve_models_dir(repo, required=None):
    """Locate the private desktop TorchScript models without coupling repos.

    ``open_health`` owns the runners, while existing decrypted model files may
    still live in the sibling ``open_oura`` checkout after the repository split.
    An explicit environment variable wins so packaged/custom layouts remain
    supported. The returned directory need not exist, allowing each runner to
    report its precise missing model filename.
    """
    configured = os.environ.get("OURA_MODELS_DIR")
    if configured:
        return Path(configured).expanduser()

    candidates = (
        repo / "notes" / "models",
        repo.parent / "open_oura" / "notes" / "models",
    )
    if required:
        return next(
            (path for path in candidates if (path / required).is_file()),
            candidates[0],
        )
    return next((path for path in candidates if path.is_dir()), candidates[0])


def resolve_db(arg, repo):
    """Resolve the SQLite events DB path.

    Explicit ``arg`` wins; otherwise pick the first existing default among
    ./oura.db, repo/oura.db, repo/captures/ring5.db, falling back to
    repo/oura.db. Exit with a clear error if the resolved DB is missing.
    """
    if arg:
        db = Path(arg)
    else:
        db = next(
            (c for c in (Path.cwd() / "oura.db", repo / "oura.db",
                         repo / "captures" / "ring5.db") if c.exists()),
            repo / "oura.db",
        )
    if not db.exists():
        sys.exit(f"error: database not found: {db} (run `oura sync` first)")
    return db
