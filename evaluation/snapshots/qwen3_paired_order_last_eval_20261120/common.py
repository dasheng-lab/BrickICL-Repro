"""Frozen protocol and durable, resumable checkpoint evaluation."""
import hashlib,json,os,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
PREVIOUS=REPO/'runs/structural_eval_20261115'
TRAIN=REPO/'runs/qwen3_nzd_v2_20261120/training'
ARMS={'direct_m':'M_only','direct_g':'Full_G','direct_bricknet':'BrickNet'}
LABELS={'direct_m':'Qwen3 直接 PT · M-only','direct_g':'Qwen3 直接 PT · Full-G','direct_bricknet':'Qwen3 直接 PT · BrickNet 数据'}

def read(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def atomic(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)
def jsonl(path,rows):
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));tmp.replace(path)
def plan():return json.loads((ROOT/'plan.json').read_text())
def folder(spec):return ROOT/'results'/spec['key']
def validate_protocol(p):
    for name,digest in p['inputs_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in p['implementation_sha256'].items():assert sha(ROOT/name)==digest,name

