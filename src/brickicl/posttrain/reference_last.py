"""One-object completion: base/context, a complete reference, then missing copies."""
from brickicl.sampling.masked_walk import masked_walk


def connected_subsets(nodes, edges):
    nodes=set(nodes);adj={v:set() for v in nodes};result=[]
    for e in edges:
        a,b=int(e['a']),int(e['b'])
        if a in nodes and b in nodes:adj[a].add(b);adj[b].add(a)
    while nodes:
        seen={min(nodes)};todo=list(seen)
        while todo:
            for v in adj[todo.pop()]:
                if v not in seen:seen.add(v);todo.append(v)
        result.append(sorted(seen));nodes-=seen
    return result


def sample_reference_last(graph, target_copies, reference_copy, *, exclude=(), seed=0):
    target_copies=[tuple(c) for c in target_copies]
    missing={v for c in target_copies for v in c}
    reference=set(reference_copy)
    excluded=set(exclude)
    if not reference or not missing or reference & (missing|excluded):
        raise ValueError('reference must be a nonempty visible copy disjoint from targets and exclusions')
    if len(missing)!=sum(map(len,target_copies)):
        raise ValueError('target copies must be disjoint')
    base=set(range(len(graph.part_ids)))-missing-reference-excluded
    if not base:
        raise ValueError('need an existing base outside the reference and target')
    # masked_walk checks reachability of base, then late reference, then missing copies.
    sample=masked_walk(graph,missing,exclude=excluded,seed=seed,
                       blocks=[reference,*target_copies],late=[reference])
    n=sample.walk.n_prefix
    assert set(sample.walk.order[n-len(reference):n])==reference
    assert set(sample.walk.order[n:])==missing
    assert all(sample.walk.parent_of[v][0] in sample.walk.order[:i]
               for i,v in enumerate(sample.walk.order) if i)
    return sample
