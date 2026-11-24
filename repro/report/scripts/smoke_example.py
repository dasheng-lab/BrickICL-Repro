"""Check a newly authored toy assembly and optional real-tokenizer completion masking; no model inference."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--tokenizer',help='Optional local tokenizer directory or public model ID');args=p.parse_args()
 from bricknet import parse_sample,score_text
 row=json.loads((ROOT/'data/smoke_example.jsonl').read_text(encoding='utf-8'));text=row['prefix']+row['completion'];parsed=parse_sample(text);assert parsed.error is None and len(parsed.tree.parts)==5
 assert len(parse_sample(row['prefix']).tree.parts)==4
 n,invalid,collisions=score_text(text);assert (n,invalid,collisions)==(5,None,[])
 if args.tokenizer:
  from transformers import AutoTokenizer
  tok=AutoTokenizer.from_pretrained(args.tokenizer);prefix=tok.encode(row['prefix'],add_special_tokens=False);target=tok.encode(row['completion'],add_special_tokens=False)+[tok.eos_token_id];ids=prefix+target;labels=[-100]*len(prefix)+target
  assert len(ids)==len(labels) and all(x==-100 for x in labels[:len(prefix)]) and labels[-1]==tok.eos_token_id and any(x!=-100 for x in labels)
 print('Smoke passed: 5 connected, parseable, collision-free bricks; 4-brick prefix; optional tokenizer mask checked when requested.')
if __name__=='__main__':main()
