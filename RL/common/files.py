"""jsonl helpers and the run_config.json that records what code produced an output."""

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

RFT = Path(__file__).resolve().parent.parent


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    """Write atomically: a crash leaves either the old file or the new one, never half of it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_run_config(out_dir, args, external=(), name="run_config.json", **extra):
    """out_dir/run_config.json: arguments, command line, and sha256 of our code and of the external files used.

    The external code is someone else's working copy (no git on the machine), so the hashes are the only record
    of which version produced a result.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ours = sorted(p for p in RFT.rglob("*") if p.suffix in (".py", ".sh") and "runs" not in p.parts)
    cfg = {
        "time": time.strftime("%F %T"),
        "host": platform.node(),
        "argv": sys.argv,
        "args": vars(args) if hasattr(args, "__dict__") else args,
        "code": {str(p.relative_to(RFT)): sha256(p) for p in ours},
        "external": {str(p): sha256(p) for p in map(Path, external)},
    } | extra
    with open(out_dir / name, "w") as f:
        json.dump(cfg, f, indent=1, default=str)
    return cfg
