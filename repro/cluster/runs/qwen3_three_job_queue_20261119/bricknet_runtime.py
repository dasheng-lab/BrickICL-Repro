"""Prepared components for BrickNet continuation training. Importing does not start training."""
import json
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,Sampler
from peft import LoraConfig,PeftModel,get_peft_model
from transformers import AutoModelForCausalLM,AutoTokenizer


def load_bricknet_with_fresh_adapter(size,checkpointing=False,initialization='RAW_QWEN3'):
    if initialization != 'RAW_QWEN3':
        raise ValueError('This experiment must initialize directly from Qwen3, with no BrickNet weights.')
    root=Path(__file__).resolve().parent
    base=json.loads((root/'BASE_PROVENANCE.json').read_text())
    assert base['verified'] and base['initial_adapters']==[]
    model=AutoModelForCausalLM.from_pretrained(base['model_directory'],dtype=torch.bfloat16,
        device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True)
    torch.manual_seed(42)  # identical fresh-adapter initialization for PT/SFT controls
    model=get_peft_model(model,LoraConfig(r=64,lora_alpha=128,lora_dropout=.05,task_type='CAUSAL_LM',
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    model.config.use_cache=False
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
    return model,AutoTokenizer.from_pretrained(base['model_directory'],local_files_only=True)


def completion_loss(model,batch):
    """Exact next-token CE on answer+EOS, equal weight per example, logits only where required."""
    labels=batch['labels'][:,1:]
    keep=(labels!=-100).any(0).nonzero().flatten()
    if not keep.numel():raise ValueError('batch has no supervised targets')
    output=model(input_ids=batch['input_ids'][:,:-1],attention_mask=batch['attention_mask'][:,:-1],
                 logits_to_keep=keep,use_cache=False)
    y=labels[:,keep]
    valid=(y!=-100).sum(1)
    if (valid==0).any():raise ValueError('an example has no supervised targets')
    ce=F.cross_entropy(output.logits.float().flatten(0,1),y.flatten(),ignore_index=-100,reduction='none').view(y.shape)
    return (ce.sum(1)/valid).mean()


class PlannedPathDataset(Dataset):
    def __init__(self,paths,tokenizer,max_length=6400):
        from brickicl.posttrain.data import encode_example
        self.items=[];self.lookup={}
        for dataset,path in paths.items():
            for i,line in enumerate(Path(path).open()):
                row=json.loads(line)
                self.lookup[dataset,i]=len(self.items)
                self.items.append(encode_example(row,tokenizer,max_length=max_length))
    def __len__(self):return len(self.items)
    def __getitem__(self,index):return self.items[index]


class PlannedLengthSampler(Sampler):
    """Same sampled rows across model sizes; local length buckets reduce padding."""
    def __init__(self,dataset,schedule,micro_batch,buffer_size=256,seed=42):
        self.dataset=dataset;self.micro_batch=micro_batch;self.buffer_size=buffer_size;self.seed=seed
        self.indices=[dataset.lookup[r['dataset'],r['row_index']] for r in map(json.loads,Path(schedule).open())]
    def __len__(self):return len(self.indices)
    def __iter__(self):
        rng=random.Random(self.seed)
        for start in range(0,len(self.indices),self.buffer_size):
            buffer=sorted(self.indices[start:start+self.buffer_size],key=lambda i:len(self.dataset[i]['input_ids']))
            batches=[buffer[i:i+self.micro_batch] for i in range(0,len(buffer),self.micro_batch)]
            rng.shuffle(batches)
            for batch in batches:yield from batch
