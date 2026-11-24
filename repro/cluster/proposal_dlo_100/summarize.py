"""Summarize the 12 per-model evaluate.py CSVs in ./scores, pairing by source."""
import csv
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1] / 'results'
arms = 'frozen_pt post_pt fullg_pt masked plain direct_m direct_g nzd_v2_masked nzd_v2_plain regularized_g regularized_g_wd010_do015 grpo_pt'.split()
weights = np.random.default_rng(20261123).multinomial(100, np.full(100, .01), size=100000)
def interval(x):
    samples = weights @ x / 100
    return [x.mean(), *np.quantile(samples, [.025, .975])]

vectors = {}
condition_means = {}
sources = None
for arm in arms:
    cases = {}
    with (root / 'scores' / (arm+'.csv')).open() as f:
        for row in csv.DictReader(f):
            conditions = cases.setdefault(row['source'], {})
            assert row['condition'] not in conditions
            lp = float(row['logp'])
            odds = float(row['log_odds'])
            assert np.isfinite(odds) and lp <= float(row['eos_logp']) < 0
            np.testing.assert_allclose(odds, lp - np.log(-np.expm1(lp)), atol=1e-12, rtol=0)
            conditions[row['condition']] = odds
    ids = sorted(cases)
    assert len(ids) == 100 and all(set(cases[k])=={'original','stretched'} for k in ids)
    if sources is None:
        sources = ids
    assert sources == ids
    condition_means[arm] = [np.mean([cases[k][c] for k in ids]) for c in ("original", "stretched")]
    vectors[arm] = np.array([cases[k]['stretched']-cases[k]['original'] for k in ids])

with (root/'summary.csv').open('w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['run','mean_dlo','lower95','upper95','positive_sources_of_100', 'original_mean_log_odds', 'stretched_mean_log_odds'])
    for arm,x in vectors.items():
        w.writerow([arm,*interval(x),sum(x>0),*condition_means[arm]])

pairs = [('post_pt','frozen_pt'),('post_pt','fullg_pt'),('direct_m','direct_g'),
         ('plain','frozen_pt'),('masked','plain'),('nzd_v2_masked','nzd_v2_plain'),
         ('regularized_g','direct_g'),('regularized_g_wd010_do015','regularized_g'),
         ('regularized_g_wd010_do015','direct_g'),('grpo_pt','masked')]
with (root/'contrasts.csv').open('w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['model','reference','dlo_difference','lower95','upper95'])
    for a,b in pairs:
        w.writerow([a,b,*interval(vectors[a]-vectors[b])])
