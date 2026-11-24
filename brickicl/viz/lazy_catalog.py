"""Build a small Viewer index, group lists, and per-model metadata.

The full manifest remains the reproducible source of truth. Only entries whose
MPD files exist are advertised, so a partly completed packing run is safe to view.
"""

import argparse
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


def write_catalog(models_dir, manifest):
    """Write a tiny index; fetch path text only for the selected model."""
    models_dir = Path(models_dir)
    root = models_dir.resolve()
    grouped = OrderedDict()
    skipped = 0
    for entry in manifest:
        rel = entry.get("file", "")
        path = (models_dir / rel).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size == 0:
            skipped += 1
            continue
        task = entry.get("task") or "single_model"
        group = entry.get("group") or "其他"
        grouped.setdefault((task, group), []).append(entry)

    groups = []
    for (task, group), entries in grouped.items():
        key = hashlib.sha256(f"{task}\0{group}".encode()).hexdigest()[:16]
        rel = f"groups/{key}.json"
        items = []
        for entry in entries:
            meta_key = hashlib.sha256(entry["file"].encode()).hexdigest()[:20]
            meta_rel = f"meta/{meta_key}.json"
            _write_json(models_dir / meta_rel, entry)
            items.append({
                "file": entry["file"], "meta": meta_rel, "task": task, "group": group,
                "title": entry.get("title", entry["file"]),
                "caption": entry.get("caption", ""), "n_parts": entry.get("n_parts", 0),
                "parse_error": entry.get("parse_error"), "collisions": entry.get("collisions"),
            })
        _write_json(models_dir / rel, {"task": task, "group": group, "items": items})
        groups.append({
            "id": key, "task": task, "group": group, "file": rel,
            "count": len(entries), "files": [entry["file"] for entry in entries],
        })
    _write_json(models_dir / "catalog.json", {"schema": 2, "groups": groups})
    print(f"catalog: {len(groups)} groups, {sum(g['count'] for g in groups)} ready models, {skipped} missing")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((args.models / "manifest.json").read_text(encoding="utf-8"))
    write_catalog(args.models, manifest)


if __name__ == "__main__":
    main()
