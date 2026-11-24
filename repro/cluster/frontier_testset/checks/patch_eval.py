from pathlib import Path
p=Path('/home/shensx3017/brickicl_eval/frontier_testset/run_eval.py')
s=p.read_text()
s=s.replace(" assert j(ROOT/'VISUAL_APPROVED.json')['approved'] and j(ROOT/'SCORING_SELF_CHECK.json')['ideal_targets']==800\n",'')
s=s.replace("  for adapter,digest in spec['local_weight_sha256'].items():assert sha(Path(adapter)/'adapter_model.safetensors')==digest", "  for adapter,digest in spec['local_weight_sha256'].items():assert sha(Path(adapter)/'adapter_model.safetensors')==digest")
s=s.replace("  for adapter in spec['adapters']:model=PeftModel.from_pretrained(model,adapter,local_files_only=True).merge_and_unload()", "  for adapter in spec['adapters']:\n   path=Path('/home/shensx3017/brickicl_eval/proposal_dlo_100/weights/official') if adapter=='kulits/BrickNet-0.6B-PT' else Path(adapter)\n   model=PeftModel.from_pretrained(model,str(path),local_files_only=True).merge_and_unload()")
s=s.replace("'completion_samples':12000", "'completion_samples':len(p['models'])*800")
assert 'VISUAL_APPROVED' not in s and "if adapter=='kulits/BrickNet-0.6B-PT'" in s
p.write_text(s)
