from bricknet import parse_sample,tree_to_graph,serialize_tree,score_text
from bricknet.graph import sample_tree,_part_info
from brickicl.sampling.edits import decorate,free_connectors
import factory
factory.init()
graph=tree_to_graph(parse_sample('a brick 1x1 | black\n').tree)
conn=free_connectors(graph,0,'stud',('stud','open'))[0]
for pid,hole,stud in factory.PARTS:
    g,_=decorate(graph,[(0,conn,'stud')],[(pid,0,hole,stud)])
    text=serialize_tree(sample_tree(g,method='dfs',seed=0))
    print(factory.CAT.id_to_name[pid],score_text(text),repr(text),flush=True)
