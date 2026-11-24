"""Depth-first traversal of context, then the missing subassemblies."""
import numpy as np
from bricknet import serialize_tree
from bricknet.graph import _adjacency, _build_tree
from brickicl.sampling.masked_walk import MaskedSample, Walk, split_text

def masked_walk(graph,M,*,exclude=(),seed=None,root=None,blocks=(),early=(),late=()):
    missing,excluded=set(M),set(exclude)
    nodes=set(range(len(graph.part_ids)))-excluded
    context=nodes-missing
    late_nodes={v for block in late for v in block}
    early_nodes={v for block in early for v in block}
    if not missing or not missing<=nodes or len(context)<2 or late_nodes-missing-context:
        raise ValueError('invalid completion partition')
    if missing & late_nodes or not late_nodes<=context:
        raise ValueError('late reference must be visible')
    rng=np.random.default_rng(seed)
    candidates=sorted(context-late_nodes)
    root=candidates[int(rng.integers(len(candidates)))] if root is None else root
    if root not in candidates:raise ValueError('root outside initial context')
    adj=_adjacency(graph.edges,nodes)
    order=[root];parent={};visited={root}
    block_ids={v:i for i,b in enumerate(blocks) for v in b}
    def visit(start,allowed):
        stack=[start]
        while stack:
            u=stack[-1]
            options=[edge for edge in adj.get(u,[]) if edge[0] in allowed and edge[0] not in visited]
            if not options:
                stack.pop()
                continue
            # Prioritize a designated reference, while always extending the current DFS stack.
            priority=[e for e in options if e[0] in early_nodes]
            same_block=[e for e in options if u in block_ids and block_ids.get(e[0])==block_ids[u]]
            options=same_block or priority or options
            v,ei,forward=options[int(rng.integers(len(options)))]
            parent[v]=(u,ei,forward);visited.add(v);order.append(v);stack.append(v)
    visit(root,context-late_nodes)
    if visited!=context-late_nodes:raise ValueError('disconnected context without reference')
    def release(allowed):
        while allowed-visited:
            boundary=[(u,v,ei,fwd) for u in order for v,ei,fwd in adj.get(u,[])
                      if v in allowed and v not in visited]
            if not boundary:raise ValueError('unreachable delayed component')
            u,v,ei,fwd=boundary[int(rng.integers(len(boundary)))]
            parent[v]=(u,ei,fwd);visited.add(v);order.append(v)
            visit(v,allowed)
    release(late_nodes)
    n=len(order)
    release(missing)
    if visited!=nodes or set(order[n:])!=missing:raise AssertionError('partition lost')
    prefix,target=split_text(serialize_tree(_build_tree(graph,order,parent)),n)
    return MaskedSample(Walk(tuple(order),parent,n),prefix,target)
