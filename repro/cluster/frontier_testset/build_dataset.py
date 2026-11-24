"""Reserialize the same 100 strict-DFS test assemblies with masked frontier sampling."""
import hashlib
import json
from pathlib import Path

from bricknet import parse_sample, score_text, tree_to_graph
from brickicl.eval.length_probe import shared_head
from brickicl.sampling.masked_walk import masked_walk

ORIGINAL = Path('/home/sunshq114514/brickICL/runs/eval100_official_dfs_20261122')
OUTPUT = Path('/home/shensx3017/brickicl_eval/frontier_testset')
OUTPUT.mkdir(parents=True, exist_ok=True)
rows = [json.loads(line) for line in (ORIGINAL/'p1_inputs.jsonl').open()]
assert len(rows) == 200
new_rows = []
for index, old in enumerate(rows):
    sequences = {}
    for kind in ('short', 'long'):
        source = parse_sample(old['prefix'] + old['target_' + kind])
        assert source.error is None
        graph = tree_to_graph(source.tree)
        # The primed first target brick is at n_prefix - 1 in the full source.
        missing = range(old['n_prefix'] - 1, len(source.tree.parts))
        sampled = masked_walk(graph, missing, root=0, seed=old['seed'])
        sequences[kind] = sampled
    assert sequences['short'].prefix == sequences['long'].prefix, old['id']
    row = dict(old)
    row.update(prefix=sequences['short'].prefix,
               target_short=sequences['short'].target,
               target_long=sequences['long'].target,
               n_prefix=sequences['short'].walk.n_prefix,
               k_short=len(sequences['short'].walk.order)-sequences['short'].walk.n_prefix,
               k_long=len(sequences['long'].walk.order)-sequences['long'].walk.n_prefix,
               prime=0, layout='masked_frontier_walk',
               dfs_pair_id=old['id'], dfs_prefix_sha256=old['prefix_sha256'])
    head = shared_head(row, 1)
    row.update(prefix=row['prefix'] + head,
               target_short=row['target_short'][len(head):],
               target_long=row['target_long'][len(head):],
               n_prefix=row['n_prefix']+1,
               k_short=row['k_short']-1,
               k_long=row['k_long']-1,
               prime=1)
    for kind in ('short', 'long'):
        text = row['prefix'] + row['target_' + kind]
        parsed = parse_sample(text)
        assert parsed.error is None, (old['id'],kind,parsed.error)
        assert len(parsed.tree.parts) == old['n_prefix'] + old['k_' + kind], (old['id'],kind)
        valid, error, collision = score_text(text)
        assert error is None and collision == [], (old['id'],kind,error,collision)
        assert valid <= 100
    row['prefix_sha256'] = hashlib.sha256(row['prefix'].encode()).hexdigest()
    row['prefix_collisions'] = []
    new_rows.append(row)
    if (index+1) % 25 == 0:
        print('VALIDATED',index+1,flush=True)
assert {r['id'] for r in new_rows} == {r['id'] for r in rows}
assert len({r['source_key'] for r in new_rows}) == 100
path = OUTPUT/'p1_inputs.jsonl'
path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in new_rows))
print('DATASET',len(new_rows),hashlib.sha256(path.read_bytes()).hexdigest(),flush=True)
