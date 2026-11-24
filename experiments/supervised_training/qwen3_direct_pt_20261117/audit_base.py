"""Record exact base-model snapshot and check its content-addressed cache blob."""
import json
from pathlib import Path
from transformers.utils.hub import cached_file
from prepare import ROOT,atomic,sha

result={'repository':'Qwen/Qwen3-0.6B','initial_adapters':[],'files':{}}
for name in ('config.json','model.safetensors','tokenizer.json','tokenizer_config.json'):
    logical=Path(cached_file('Qwen/Qwen3-0.6B',name,local_files_only=True));resolved=logical.resolve()
    digest=sha(resolved);blob=resolved.name
    meta=json.loads((ROOT/'official_model_metadata.json').read_text())
    entry=next(r for r in meta['siblings'] if r['rfilename']==name)
    if 'lfs' in entry:assert digest==entry['lfs']['sha256']
    result['files'][name]={'snapshot_path':str(logical),'resolved_path':str(resolved),'sha256':digest,
                           'matches_official_lfs_sha256':digest==entry['lfs']['sha256'] if 'lfs' in entry else None}
atomic(ROOT/'SHARED_BASE_VERIFIED.json',result);print(json.dumps(result,indent=2))
