# Matched two-model ICL probe (`G → G/M`)

## Location and protocol

- `brickicl/sampling/build_matched.py`: constructs one walk seed per source
  (`pair` suffix `-0-0`) from `runs/stretch/stretch.contiguous.r4.jsonl`.
- `brickicl/eval/eval_matched.py`: frozen BrickNet-PT likelihoods and eight
  free continuations for 0.6B, 1.7B, and 4B.
- `brickicl/eval/score_matched.py`: order-independent part/color and placed
  pose F1, exact rate, parse rate, and path-free per-source metric export.
- `brickicl/viz/pack_matched.py`: validates the demo/query match before it
  writes side-by-side Viewer models. `brickicl/viz/pack_probe.py` contains
  the shared `Packer` and the `single_model/` versus `two_copy/` folders.
- `brickicl/eval/run_matched.sh`: rebuilds inputs, runs all models and the
  4B no-demo control, and writes reports under `runs/twocopy/`.

For each old `original`, `stretched`, or `ablated` condition, let the old
single-model prefix be `P` and the correct target tail be `M`. The complete
demonstration is `G = P + M`. The second query starts at `P` plus the first
target part, then the model predicts the remainder. The builder and packer
both check that the complete left model equals the right input plus its
reference answer. The old one-model query is unchanged. There are 18 source
models × 3 conditions = 54 inputs per model, with 8 sampled completions per
input. Generation uses temperature 1, top-k 20, top-p 0.95, and EOS or a
length cap. No adapter parameters are updated.

Run from the repository root:

```bash
bash brickicl/eval/run_matched.sh
python -m brickicl.viz.pack_matched \
  --data runs/twocopy/twocopy.matched.r4.p1.jsonl \
  --results runs/twocopy/matched.0.6B.jsonl \
            runs/twocopy/matched.1.7B.jsonl \
            runs/twocopy/matched.4B.jsonl --samples 8
```

The second command requires the LDraw library and writes ignored Viewer
artifacts to `baseline/repro/viewer/models/`. Each source has 3 reference
answers plus 3 conditions × 3 model sizes × 8 continuations = 75 entries.
The 4B no-demo control is scored but is not a two-model Viewer
example. The earlier conflict-diagnostic inputs, where the complete left
model and the right query's known geometry disagreed, are excluded.

## Recorded results

Data: 2026-11-08 one-seed run, 18 source models, 8 generations per input.
`log-odds = log p(long target + EOS) − log p(short target + EOS)`.
`pose preference = pose F1(long) − pose F1(short)` for each generated sample.
The last two columns compare stretched with original, paired by source.

| Prompt / model | Log-odds original | Log-odds stretched | Δ log-odds (95% source CI) | Δ pose preference (95% source CI) | Stretched pose F1 | Stretched exact |
|---|---:|---:|---:|---:|---:|---:|
| Old single model, 0.6B | -15.233 | 3.852 | +19.085 [14.044, 24.382] | +0.219 [0.120, 0.337] | 0.369 | 0.076 |
| Matched `G → G/M`, 0.6B | -17.559 | 1.879 | +19.438 [14.143, 25.155] | +0.397 [0.312, 0.488] | 0.452 | 0.076 |
| Old single model, 1.7B | -16.073 | 3.399 | +19.472 [14.714, 24.488] | +0.280 [0.173, 0.405] | 0.431 | 0.111 |
| Matched `G → G/M`, 1.7B | -20.839 | 5.633 | +26.473 [20.869, 32.358] | +0.629 [0.475, 0.803] | 0.540 | 0.181 |
| Matched `G → G/M`, 4B | -17.700 | 3.300 | +21.001 [16.719, 25.435] | +0.647 [0.521, 0.765] | 0.623 | 0.229 |
| No demo, 4B | -15.096 | 1.103 | +16.199 [11.877, 20.837] | +0.212 [0.090, 0.359] | 0.325 | 0.139 |

Node and pose F1 use `2TP/(K+n)`, with one-to-one matching against the
target's parts and placements. The committed `matched_r4_p1.metrics.json`
contains the per-source, per-condition values; it has no paths, generations,
captions, or 3D assets.

These figures show a change in the model's length preference under the
matched two-model prompt. They do not establish a general ICL capability:
the sample has only one walk seed per source, exact assembly remains much
harder, and the internal `<|im_end|>` separator is outside BrickNet-PT's
single-path training format. The metric export is intended for reanalysis,
not as a substitute for the retained raw outputs on the cluster.

## Data boundary

`runs/` contains prompts and predictions derived from the gated dataset.
Viewer MPDs embed LDraw parts and dataset models; `models/` and captions
remain ignored. Only code and path-free aggregate metrics are committed.
