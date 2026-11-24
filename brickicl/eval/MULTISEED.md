# Multi-walk-seed matched ICL experiment

## Question

The original matched `G -> G/M` probe used only pair suffix `-0-0`: one held-out
copy and one serialization walk per source model. This experiment keeps the
held-out copy fixed and varies the walk seed, testing whether the measured ICL
effect survives different valid BrickNet serialization paths.

## Paired design

- Sources: the same 18 source models used by the matched probe.
- Target copy: index 0 by default, fixed across walks.
- Walk seeds: indexes 0, 1, and 2 by default.
- Conditions: `original`, `stretched`, and `ablated`.
- Prompt settings: matched complete-G demonstration and an identical no-demo
  query control for every model size.
- Base models: Qwen3 0.6B, 1.7B, and 4B with the corresponding frozen
  `kulits/BrickNet-*-PT` adapter.
- Sampling: eight continuations per row by default, temperature 1, top-k 20,
  top-p 0.95.

With all defaults, each prompt setting and model size has
`18 sources x 3 walks x 3 conditions = 162` rows and 1,296 sampled
continuations. Matched and no-demo prompts are evaluated on exactly the same
queries.

## Primary analysis

For every source/walk unit, compute the matched-demo minus no-demo difference.
The main interaction is:

`(matched_stretched - no_demo_stretched) - (matched_original - no_demo_original)`

The report also shows the demo gain in each condition. Confidence intervals
resample source models as clusters, so all walk seeds from one source move
together. Walk seeds increase within-source coverage without pretending that
they are independent source models. For each metric the report also includes:

- `walk+`: fraction of individual walks with positive difference-in-differences.
- `source+`: fraction of sources whose mean difference-in-differences is positive.
- `within-source SD`: average variation across walk seeds of the same source.

## Run

The input stretch file must contain the requested walk seeds. The existing
stretch builder creates three seeds by default. If the `r=4` input is absent:

```bash
python -m brickicl.sampling.build_stretch \
  --graphs baseline/bricknet/data/dataset/val.npz.xz \
  --out runs/stretch/stretch.contiguous.r4.jsonl \
  --r 4 --seeds 3 --layout contiguous
```

```bash
bash brickicl/eval/run_matched_multiseed.sh
```

Useful overrides:

```bash
SIZES="0.6B" WALK_SEEDS="0,1" N=2 \
  bash brickicl/eval/run_matched_multiseed.sh runs/twocopy-multiseed-smoke
```

Outputs are written under `runs/twocopy-multiseed/` by default:

- `twocopy.matched.target0.seeds0-1-2.jsonl`: matched prompts.
- `twocopy.nodemo.target0.seeds0-1-2.jsonl`: paired no-demo prompts.
- `matched.multiseed.<size>.jsonl`: matched model outputs.
- `nodemo.multiseed.<size>.jsonl`: no-demo model outputs.
- `matched.multiseed.summary.txt`: clustered summaries and ICL gains.
- `matched.multiseed.metrics.json`: path-free per-source/walk metrics.

This design tests robustness to serialization walks. It does not remove the
format-migration confound caused by placing a complete model and an internal
EOS before the query; delimiter-only and irrelevant-demo controls remain a
separate follow-up.
