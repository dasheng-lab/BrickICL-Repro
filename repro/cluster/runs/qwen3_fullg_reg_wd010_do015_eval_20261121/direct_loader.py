import json
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
from common import ROOT,TRAIN,sha

def load_model(adapters):
    assert len(adapters)==1 and str(TRAIN) in adapters[0]
    provenance=json.loads((TRAIN.parent/'BASE_PROVENANCE.json').read_text())
    base=Path(provenance['model_directory'])
    assert provenance['verified'] and sha(base/'model.safetensors')==provenance['expected_model_sha256']
    model=AutoModelForCausalLM.from_pretrained(str(base),dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True)
    model=PeftModel.from_pretrained(model,adapters[0],local_files_only=True).merge_and_unload()
    model.eval();model.config.use_cache=True
    tok=AutoTokenizer.from_pretrained(str(base),local_files_only=True,padding_side='left')
    tok.pad_token=tok.eos_token;model.generation_config.pad_token_id=tok.eos_token_id
    return model,tok
