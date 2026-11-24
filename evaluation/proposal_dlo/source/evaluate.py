"""Score complete long targets with EOS, using the shared strict-DFS evaluation inputs."""
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

project = Path(__file__).resolve().parents[1]
root = project / 'inputs'
arms = 'direct_g direct_m nzd_v2_masked nzd_v2_plain regularized_g regularized_g_wd010_do015 frozen_pt post_pt fullg_pt masked plain grpo_pt'.split()
plan = json.loads((root / 'plan.json').read_text())
specs = {s['arm']: s for s in plan['models']}
inputs = root / 'p1_inputs.jsonl'
assert hashlib.sha256(inputs.read_bytes()).hexdigest() == plan['input_sha256']
rows = [json.loads(line) for line in inputs.read_text().splitlines()]
assert len(rows) == 200 and len({(r['source_key'], r['condition']) for r in rows}) == 200
torch.set_num_threads(2)
tok = AutoTokenizer.from_pretrained(specs['direct_g']['base'], local_files_only=True)
def encode(text):
    return tok.encode(text, add_special_tokens=False)
examples = []
for r in rows:
    prefix, target = encode(r['prefix']), encode(r['target_long'])
    assert target and tok.eos_token_id not in target
    assert tok.decode(prefix + target, clean_up_tokenization_spaces=False) == r['prefix'] + r['target_long']
    examples.append((prefix, target + [tok.eos_token_id]))

@torch.inference_mode()
def run():
    for arm in arms:
        dest = project / 'results' / 'scores' / (arm + '.csv')
        if dest.exists():
            continue
        begin = time.monotonic()
        spec = specs[arm]
        model = AutoModelForCausalLM.from_pretrained(spec['base'], dtype=torch.bfloat16, device_map={'':'cuda'}, attn_implementation='sdpa', local_files_only=True)
        for adapter in spec['adapters']:
            official = adapter == 'kulits/BrickNet-0.6B-PT'
            path = (project / 'weights' / 'official') if official else Path(adapter)
            expected = '9073be9ee47c77085387561113e912e29ac53a71d49fcb20205e598f47d2f553' if official else spec['local_weight_sha256'][adapter]
            assert hashlib.sha256((path/'adapter_model.safetensors').read_bytes()).hexdigest() == expected
            model = PeftModel.from_pretrained(model, str(path.resolve()), local_files_only=True).merge_and_unload()
        model.eval()
        with dest.with_suffix('.tmp').open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['source','condition','target_tokens_including_eos','logp','eos_logp','log_odds'])
            for i, (prefix, target) in enumerate(examples):
                ids = torch.tensor([prefix+target],device='cuda')
                # At position len(prefix)-1 the causal LM predicts target token zero.
                logits = model(ids[:,:-1], use_cache=False, logits_to_keep=len(target)).logits[0].float()
                labels = ids[0,len(prefix):]
                lp = -torch.nn.functional.cross_entropy(logits,labels,reduction='none')
                if i == 0 and arm == arms[0]:
                    masked = ids.clone()
                    masked[:,:len(prefix)] = -100
                    reference = model(ids, labels=masked, use_cache=False).loss.item()
                    assert abs(reference + lp.mean().item()) < .01
                    print('SMOKE',reference,-lp.mean().item(), 'tokens',len(target),'eos',target[-1],flush=True)
                logp = lp.double().sum().item()
                assert math.isfinite(logp) and logp < 0
                log_odds = logp - math.log(-math.expm1(logp))
                writer.writerow([rows[i]['source_key'],rows[i]['condition'],len(target),logp,lp[-1].item(),log_odds])
        dest.with_suffix('.tmp').replace(dest)
        print('DONE',arm,round(time.monotonic()-begin,1),flush=True)
        del model
        torch.cuda.empty_cache()
run()
