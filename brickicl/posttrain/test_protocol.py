"""Small protocol regressions; no model loading, inference, or dataset exports."""
import copy
import unittest
from bricknet import parse_sample,tree_to_graph
from brickicl.sampling.missing import _signature
from brickicl.posttrain.prepare_first_brick_eval import paired_inputs,reusable

PREFIX='a brick 2x4 | red\n'
def tail(n):
    return ''.join(f'{chr(98+i)} brick 2x4 | blue\n{chr(97+i)} stud stud a hole a 0\n' for i in range(n))
def fixtures():
    return [{'source':1,'pair':'1-0-0','condition':c,'r':4,'prefix':PREFIX,
             'target_short':tail(2),'target_long':tail(6),'n_prefix':1,'k_short':2,'k_long':6}
            for c in ('original','stretched','ablated')]

class ProtocolTests(unittest.TestCase):
    def test_first_brick_moves_without_changing_complete_references(self):
        pairs=paired_inputs(fixtures(),sources=(1,))
        for a,b in zip(pairs[0],pairs[1]):
            self.assertEqual(b['n_prefix'],a['n_prefix']+1)
            self.assertEqual(len(b['prefix'].splitlines())-len(a['prefix'].splitlines()),2)
            self.assertEqual(b['k_short'],1)
            for kind in ('short','long'):
                self.assertEqual(a['prefix']+a['target_'+kind],b['prefix']+b['target_'+kind])

    def test_different_long_short_heads_are_rejected(self):
        rows=fixtures();rows[0]['target_long']=rows[0]['target_long'].replace('blue','white',1)
        with self.assertRaisesRegex(ValueError,'head differs'):paired_inputs(rows,sources=(1,))

    def test_cache_reuse_requires_input_weights_seed_cap_and_four_samples(self):
        rows=paired_inputs(fixtures(),sources=(1,))[1];adapters=['official-pt','test-checkpoint']
        plan=[{'ids':[r['id'] for r in rows],'seed':123,'max_new_tokens':608}]
        outputs=[{'id':r['id'],'prime':1,'adapters':adapters,'seed':123,'max_new_tokens':608,'samples':[{} for _ in range(4)]} for r in rows]
        self.assertTrue(reusable(rows,outputs,rows,plan,adapters))
        for field,value in [('seed',124),('max_new_tokens',609),('adapters',['another-checkpoint']),('samples',[{}])]:
            bad=copy.deepcopy(outputs);bad[0][field]=value
            self.assertFalse(reusable(rows,bad,rows,plan,adapters),field)
        changed=copy.deepcopy(rows);changed[0]['prefix']+=' '
        self.assertFalse(reusable(changed,outputs,rows,plan,adapters))
        duplicate=copy.deepcopy(outputs);duplicate[-1]=duplicate[0]
        self.assertFalse(reusable(rows,duplicate,rows,plan,adapters))

    def test_signature_initial_labels_are_sortable(self):
        graph=tree_to_graph(parse_sample(PREFIX+tail(2)).tree)
        signature=_signature(graph,{1,2},attach=1,anchor=0,iters=0)
        self.assertEqual(signature[0],2)
        self.assertEqual(len(signature[1]),2)

if __name__=='__main__':unittest.main()
