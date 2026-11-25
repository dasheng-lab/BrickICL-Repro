"""Read local JSONL inputs with actionable errors for empty placeholders."""
import json
from pathlib import Path


def read_jsonl(path):
    source = Path(path)
    records = []
    with source.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}:{number}: invalid JSON: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{source}:{number}: expected a JSON object")
            records.append(record)
    if not records:
        raise ValueError(f"{source} is empty. Populate the JSONL with authorized local data before running.")
    return records
