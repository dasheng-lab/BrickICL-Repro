#!/usr/bin/env bash
set -euxo pipefail
if ! command -v conda >/dev/null; then
    echo "Conda must be available on PATH" >&2
    exit 2
fi
eval "$(conda shell.bash hook)"
conda env list | grep -q "^vllm " || conda create -y -n vllm --override-channels -c conda-forge python=3.12
conda activate vllm
# pinned to the versions vLLM 0.8.5 was released against; newer transformers / huggingface-hub break it
pip install "vllm==0.8.5.post1" "transformers==4.51.3" "huggingface-hub>=0.30,<1.0" "tokenizers>=0.21,<0.22" \
    "peft==0.15.2" "fastapi<0.116" pybullet scipy platformdirs
python -c "import vllm, torch, transformers, peft, pybullet; print(\"OK\", vllm.__version__, torch.__version__, torch.version.cuda, transformers.__version__, peft.__version__)"
