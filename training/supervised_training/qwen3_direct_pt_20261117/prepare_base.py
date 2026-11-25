"""Download the public raw Qwen3 base used by the supervised controls."""
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from prepare import ROOT, atomic, sha


def main():
    model_dir = Path(snapshot_download(
        "Qwen/Qwen3-0.6B",
        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "vocab*", "merges*"],
    )).resolve()
    weights = model_dir / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Expected the public Qwen3 weights at {weights}")
    record = {
        "repository": "Qwen/Qwen3-0.6B",
        "model_directory": str(model_dir),
        "expected_model_sha256": sha(weights),
        "initial_adapters": [],
        "verified": True,
    }
    atomic(ROOT / "BASE_PROVENANCE.json", record)
    print(json.dumps({"model": record["repository"], "directory": str(model_dir)}))


if __name__ == "__main__":
    main()
