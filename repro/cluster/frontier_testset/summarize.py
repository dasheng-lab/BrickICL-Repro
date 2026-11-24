"""Compare DFS and frontier on matched works for two BrickNet Full-G models."""
import csv
import json
import math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
DFS=Path('/home/sunshq114514/brickICL/runs/eval100_official_dfs_20261122/results')
TF_DFS=Path('/home/shensx3017/brickicl_eval/proposal_dlo_100/results/scores')
FREE=Path('/home/sunshq114514/brickICL/runs/no_eos_eval_20261120/results')
ARMS=('plain','masked')
CONDITIONS=('original','stretched')
METRICS=('proposal_dlo','sampled_long_feature_dlo','pose_f1_original','pose_f1_stretched',
         'exact_at_4_original','exact_at_4_stretched','parse_rate_original','parse_rate_stretched')

def load(layout,arm):
    source=DFS/arm if layout=='dfs' else HERE/'results'/arm
    path=TF_DFS/(arm+'.csv') if layout=='dfs' else source/'proposal_dlo.csv'
    tf={}
    for row in csv.DictReader(path.open()):
        assert row['condition'] not in tf.setdefault(row['source'],{})
        tf[row['source']][row['condition']]=float(row['log_odds'])
    scored={}
    for line in (source/'p1.scored.jsonl').open():
        row=json.loads(line)
        assert row['condition'] not in scored.setdefault(row['source_key'],{})
        scored[row['source_key']][row['condition']]=row['samples']
    assert len(tf)==len(scored)==100 and tf.keys()==scored.keys()
    values=[]
    for key in sorted(scored):
        assert set(scored[key])==set(tf[key])==set(CONDITIONS)
        original,stretched=[scored[key][c] for c in CONDITIONS]
        assert len(original)==len(stretched)==4
        k0=sum(s['long_feature'] for s in original)
        k1=sum(s['long_feature'] for s in stretched)
        odds=lambda k:math.log((k+.5)/(4-k+.5))
        values.append([tf[key]['stretched']-tf[key]['original'],odds(k1)-odds(k0),
                       np.mean([s['pose_f1'] for s in original]),
                       np.mean([s['pose_f1'] for s in stretched]),
                       float(any(s['exact'] for s in original)),
                       float(any(s['exact'] for s in stretched)),
                       np.mean([s['parse_ok'] for s in original]),
                       np.mean([s['parse_ok'] for s in stretched])])
    values=np.array(values)
    if layout=='dfs':
        original=json.loads((source/'metrics.json').read_text())['completion']['1']
        np.testing.assert_allclose(values[:,1].mean(),original['delta'],atol=1e-12)
        np.testing.assert_allclose(values[:,2:4].mean(0),[original['f1'][c] for c in CONDITIONS],atol=1e-12)
    return values

all_scores={(layout,arm):load(layout,arm) for layout in ('dfs','frontier') for arm in ARMS}
rng=np.random.default_rng(20261123)
weights=rng.multinomial(100,np.full(100,.01),size=100000)
def estimate(x):
    boot=weights@x/100
    return np.vstack((x.mean(0),*np.quantile(boot,[.025,.975],axis=0)))

out=HERE/'results'
with (out/'summary.csv').open('w',newline='') as stream:
    writer=csv.writer(stream)
    writer.writerow(['layout','model',*METRICS,'allow_eos_valid','allow_eos_collision_free',
                     'no_eos_valid','no_eos_collision_free','free_generation_source'])
    for (layout,arm),scores in all_scores.items():
        free=json.loads((FREE/arm/'metrics.json').read_text())
        assert free['collision_coverage']==64 and free['model']['allow_eos_physical']['collision_coverage']==64
        allow=free['model']['allow_eos_physical']
        writer.writerow([layout,arm,*scores.mean(0),allow['valid_length'],allow['collision_free_length'],
                         free['no_eos_valid_length'],free['no_eos_collision_free_length'],str(FREE/arm/'metrics.json')])

comparisons=[('frontier_minus_dfs_plain',('frontier','plain'),('dfs','plain')),
             ('frontier_minus_dfs_aug',('frontier','masked'),('dfs','masked')),
             ('aug_minus_plain_frontier',('frontier','masked'),('frontier','plain')),
             ('aug_minus_plain_dfs',('dfs','masked'),('dfs','plain'))]
with (out/'paired_differences.csv').open('w',newline='') as stream:
    writer=csv.writer(stream)
    writer.writerow(['contrast','metric','difference','lower95','upper95'])
    for name,left,right in comparisons:
        result=estimate(all_scores[left]-all_scores[right])
        for i,metric in enumerate(METRICS):writer.writerow([name,metric,*result[:,i]])
    interaction=(all_scores['frontier','masked']-all_scores['frontier','plain'])-(all_scores['dfs','masked']-all_scores['dfs','plain'])
    result=estimate(interaction)
    for i,metric in enumerate(METRICS):writer.writerow(['aug_advantage_frontier_minus_dfs',metric,*result[:,i]])
print('WROTE',out/'summary.csv',out/'paired_differences.csv',flush=True)
