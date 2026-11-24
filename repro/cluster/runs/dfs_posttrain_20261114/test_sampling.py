import json
import numpy as np
from pathlib import Path
from bricknet import parse_sample,tree_to_graph,decode_graph
from dfs_sampling import masked_walk
from brickicl.posttrain.reference_last import connected_subsets

for version in ('v2','v3'):
    src=Path(f'runs/reference_last_06/{version}/dev.jsonl')
    for line in src.read_text().splitlines()[:20]:
        row=json.loads(line)
        old=parse_sample(row['prompt']+row['completion'])
        graph=tree_to_graph(old.tree)
        late=connected_subsets(row['evidence_positions'],graph.edges)[:1]
        smp=masked_walk(graph,row['target_positions'],seed=row['sampling_seed'],late=late)
        new=parse_sample(smp.prefix+smp.target)
        assert new.error is None
        expected=np.asarray(decode_graph(graph))[list(smp.walk.order)]
        actual=np.asarray(decode_graph(tree_to_graph(new.tree)))
        align=expected[0]@np.linalg.inv(actual[0])
        assert np.allclose(align[None]@actual,expected,atol=1e-4)
        assert all(new.tree.parts[i]==old.tree.parts[j] for i,j in enumerate(smp.walk.order))
        # Check the initial context stage is DFS.
        cut=smp.walk.n_prefix-len(late[0])
        stack=[0]
        for e in new.tree.edges[:cut-1]:
            assert e.parent in stack
            stack=stack[:stack.index(e.parent)+1]+[e.child]
    print(version,'20 geometry/order checks passed',flush=True)
