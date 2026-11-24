from pathlib import Path
p=Path('/home/shensx3017/brickicl_eval/frontier_testset/evaluate_dlo.py')
s=p.read_text()
s=s.replace("project = Path(__file__).resolve().parents[1]\nroot = project / 'inputs'", "project = Path(__file__).resolve().parent\nroot = project")
s=s.replace("arms = 'direct_g direct_m nzd_v2_masked nzd_v2_plain regularized_g regularized_g_wd010_do015 frozen_pt post_pt fullg_pt masked plain grpo_pt'.split()", "arms = ['plain', 'masked']")
s=s.replace("dest = project / 'results' / 'scores' / (arm + '.csv')", "dest = project / 'results' / arm / 'proposal_dlo.csv'")
s=s.replace("(project / 'weights' / 'official') if official", "Path('/home/shensx3017/brickicl_eval/proposal_dlo_100/weights/official') if official")
assert 'masked' in s and "project / 'inputs'" not in s
p.write_text(s)
