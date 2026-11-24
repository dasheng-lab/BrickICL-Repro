# PT single-copy vs PT double-copy vs BrickNet-SFT context

This experiment keeps the long/short-leg data, held-out copy, walk seeds,
sampling parameters, and scoring fixed. It changes only the model/context
setting:

1. Single-copy: BrickNet-PT receives the incomplete assembly path.
2. Double-copy: BrickNet-PT receives a complete matched assembly, EOS, and
   the same incomplete assembly path.
3. SFT caption+partial: BrickNet-PT and BrickNet-SFT are merged in that order;
   the prompt is the held-out validation caption, a newline, and the same
   incomplete assembly path.

The report includes each setting's stretched-minus-original effect and three
paired difference-in-differences contrasts. Bootstrap intervals cluster all
walk seeds from the same source model.

Smoke test for 1.7B:

```bash
SIZES="1.7B" WALK_SEEDS="0,1" N=2 TOKEN_BUDGET=180000 \
  bash brickicl/eval/run_context_comparison.sh runs/context-comparison-1.7B-smoke
```

Full 1.7B run:

```bash
SIZES="1.7B" WALK_SEEDS="0,1,2" N=8 TOKEN_BUDGET=180000 \
  bash brickicl/eval/run_context_comparison.sh runs/context-comparison-1.7B
```
