# Frontier testset for the 100 paired LEGO assemblies

This is a source-only release snapshot. The input JSONL, run plan, generated assembly text, model weights, and raw scored JSONL described below remain on the authorized cluster. The public geometry-free observations used to recreate the manuscript table are in ../../report/data/frontier_100.json.

This evaluation reserializes the same 100 works and the same original/stretched target geometry as `runs/eval100_official_dfs_20261122`. The old testset uses strict depth-first ordering. This one uses `brickicl.sampling.masked_walk` with the target held out until the end, no contiguous component blocks, a fixed root, and a reproducible per-work seed. All 200 prefixes differ from the DFS version; zero of the 200 complete expected sequences satisfy the strict DFS checker. The gold structures remain parseable, collision-free, and geometrically identical under part-and-pose matching. The two model checkpoints are unchanged from the original development-selected evaluation.

## Models and budget

- `plain`: BrickNet-0.6B-PT plus plain DFS Full-G, checkpoint 1750.
- `masked`: BrickNet-0.6B-PT plus component-last augmented Full-G, checkpoint 2000.
- Each model has 100 works, two reference conditions, and four sampled continuations per condition, giving 800 completions. Generation uses temperature 1, top-k 20, top-p 0.95, and the original length-cap rule. The 25 batches per model fit the H20 with a 35% GPU memory cap. A four-draw smoke check passed before the full run.

## Files

- `p1_inputs.jsonl` contains the paired frontier prompts and short/long gold continuations. `plan.json` records source hashes, model checkpoints, and generation settings.
- `build_dataset.py` recreates the frontier inputs from the strict-DFS evaluation file. `prepare.py` creates the model and batch plan.
- `run_eval.py` runs sampling and the unchanged scorer in `reference_scoring.py`. It writes raw and scored outputs to `results/{masked,plain}/` and supports completed-model and completed-batch resume.
- `evaluate_dlo.py` scores the complete long target continuation with teacher forcing, including EOS, and writes a per-condition CSV in each model directory.
- `summarize.py` produces `results/summary.csv` and `results/paired_differences.csv`. It compares frontier with DFS on matched works, compares aug with plain within each layout, and calculates the change in aug's advantage between layouts. All intervals are 100,000-repeat paired bootstrap percentile intervals over works.
- Independent generation has no reference prompt. Its four length metrics are linked to the original 64-sample evaluation in the summary, with the source path recorded. They are reused because changing this testset's traversal does not change the no-reference task.

This run uses the JY `bricknet` Python environment because the Sun environment is inaccessible to the current SSH account. The source code and checkpoints are shared from the existing brickICL tree. The environment uses PyTorch 2.6.0, Transformers 5.22.2, and PEFT 0.21.2.

The teacher-forced proposal DLO was run in the `cs183-hw5` environment with the same local PEFT dependency overlay as the previous strict-DFS proposal-DLO evaluation. The two layouts use the same 25 batch seeds and generation settings. One batch has its first two work IDs swapped because their reserialized prompt lengths reverse their sort order.

## Readout

`results/summary.csv` is the full four-row metric table. `results/paired_differences.csv` contains 95% paired bootstrap intervals. The frontier ordering substantially reduces both models' completion quality. Relative to the plain checkpoint, the augmented checkpoint has higher frontier Pose F1 in both conditions, but its proposal DLO advantage has a confidence interval crossing zero. This supports a traversal-order sensitivity finding; it does not establish a reliable ICL gain from augmentation. The checkpoints were selected at different training steps (1750 and 2000), so this comparison does not isolate augmentation from training duration. The independent generation metrics are reused unchanged from their original evaluation, because that task has no reference traversal.

The `checks/` directory contains the dataset audits and smoke-test helpers. The main runnable path is `build_dataset.py`, `prepare.py`, `run_eval.py`, `evaluate_dlo.py`, and `summarize.py`.
