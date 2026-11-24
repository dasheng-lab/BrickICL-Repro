import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
from common import REPO,sha

def load_model(adapters):
    assert len(adapters)==3 and adapters[0]=='kulits/BrickNet-0.6B-PT'
    assert adapters[1]=='/home/nzd0420/brickICL/runs/nzd-supertry/train_v2/masked/checkpoint-2000'
    assert adapters[2]=='/home/jywu2289/RL/runs/v2.1/train/grpo/checkpoint-400'
    provenance=json.loads((REPO/'runs/qwen3_direct_pt_20261117/BASE_PROVENANCE.json').read_text())
    base=Path(provenance['model_directory'])
    assert provenance['verified'] and sha(base/'model.safetensors')==provenance['expected_model_sha256']
    model=AutoModelForCausalLM.from_pretrained(str(base),dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True)
    # Exactly the documented GRPO training origin, sequential BF16 merges, no SFT adapter.
    for adapter in adapters:model=PeftModel.from_pretrained(model,adapter,local_files_only=True).merge_and_unload()
    model.eval();model.config.use_cache=True
    tok=AutoTokenizer.from_pretrained(str(base),local_files_only=True,padding_side='left');tok.pad_token=tok.eos_token
    model.generation_config.pad_token_id=tok.eos_token_id
    return model,tok
