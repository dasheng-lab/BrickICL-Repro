# BrickICL

BrickICL explores how small language models assemble LEGO structures from serialized parts and connections. Given a partly built object with one visible repeated module, the model completes a missing copy. We also ask whether improving this reference-guided task changes its ability to build an object without a reference. The project compares two starting models (released BrickNet-0.6B PT and Qwen3-0.6B), repeated-module and assembly-order data augmentation, M-only and Full-G supervision, regularization, and GRPO. It was developed by Sixian Shen, Zidong Ning, Shengqing Sun, and Junyang Wu for CS182.

## Start here

| Directory | What it contains |
| --- | --- |
| `src/brickicl/` | Reusable assembly, data-construction, model, and scoring code |
| `scripts/` | Data setup, supervised and GRPO training, generation, and result-building commands |
| `eval/` | Pose F1, Exact@N, paired reference sensitivity, and valid-prefix scoring |
| `data/` | An independently written five-brick example; restricted BrickNet data is downloaded here locally |
| `results/` | Public numeric observations and generated tables and plots |
| `third_party/bricknet/` | Upstream BrickNet and BrickNet-Render source, including their licenses and data instructions |

Run commands from the repository root. `runs/` and `logs/` are local output directories and are ignored by Git.

## 1. Get the data

BrickNet's full training and validation data require access from the authors. Apply using the [BrickNet dataset form](https://forms.gle/dm4eYSa5gh4DqzRT6), as described in [BrickNet's data guide](third_party/bricknet/BrickNet/DATA.md). Once access is granted, set `BRICKNET_SHARE_URL` to your authorized Keeper share URL and run:

```bash
export BRICKNET_SHARE_URL='YOUR_AUTHORIZED_SHARE_URL'
bash scripts/fetch_bricknet.sh --meshes
export BRICKNET_DATA="$PWD/data/bricknet"
python -m bricknet fetch-meshes
```

The fetcher places the dataset under `data/bricknet/dataset/` and additional mesh assets under `data/bricknet/`. BrickNet's public mesh fetch commands are documented in its data guide. Keep the share URL and downloaded files private. The 100-case evaluation prompts, target paths, and derived geometry are not redistributed.

The [small test example](data/smoke_example.jsonl) is independent of BrickNet validation data. A [Google Drive copy](https://drive.google.com/file/d/16NDCaDN4kLeSpejVeYvroTZIK34lPJxa/view?usp=drivesdk) is available to Berkeley accounts; the repository copy is public. It checks the input and parser flow but does not produce a reported result.

## 2. Set up the environment

Use Python 3.11 on Linux for training and evaluation. A CUDA GPU is needed for model training and practical model inference. The numeric results in `results/` only need NumPy and Matplotlib.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e third_party/bricknet/BrickNet
python -m pip install -e .
export BRICKNET_DATA="$PWD/data/bricknet"
python scripts/check_example.py
```

The last command checks that the example parses as five connected, collision-free bricks; it needs the collision meshes. Add `--tokenizer Qwen/Qwen3-0.6B` to check the completion loss mask. The reported GRPO run uses vLLM and its own pinned environment setup at `scripts/train/grpo/runner/install_vllm_env.sh`.

## 3. Prepare data and train

### Repeated-module augmentation and supervised objectives

`src/brickicl/posttrain/build_v2.py` builds larger repeated modules; `build_v3.py` builds small repeated modules inside more complex bodies. They consume the authorized BrickNet graph files and write training and development JSONL under `runs/`. Before training, create the token caches and fixed schedules:

```bash
python -m brickicl.posttrain.build_v2 --out runs/posttrain_icl_v2
python -m brickicl.posttrain.build_v3 --out runs/posttrain_icl_v3_local --pt-cache runs/posttrain_icl_v3_local/pt.cache.npz
python -m brickicl.posttrain.materialize_reference_last --out runs/reference_last_06
python -m brickicl.posttrain.submit_06_plan --data runs/reference_last_06 --out runs/bricknet_06_reference_last
python -m brickicl.posttrain.train_bricknet06 --config runs/bricknet_06_reference_last/PT.json
```

The last command trains an M-only LoRA adapter on the released BrickNet PT starting point. For Full-G, use the same prepared structures with a full-sequence objective:

```bash
python -m brickicl.posttrain.prepare_fullg --data runs/reference_last_06 --out runs/fullg_06
python -m brickicl.posttrain.submit_fullg_plan --data runs/fullg_06 --out runs/bricknet_06_fullg
python -m brickicl.posttrain.train_fullg06 --config runs/bricknet_06_fullg/PT.json
```

The `scripts/train/qwen/` directories contain the Qwen initialization, paired-order, and regularization trainers. The direct-Qwen M-only and Full-G configs are prepared from the RMA token caches under `runs/structural_20261114/`:

```bash
python scripts/train/qwen/baseline/prepare_base.py
python scripts/train/qwen/baseline/prepare.py
python scripts/train/qwen/baseline/train.py --config scripts/train/qwen/baseline/training/M_only/config.json
python scripts/train/qwen/baseline/train.py --config scripts/train/qwen/baseline/training/Full_G/config.json
```

The RMA cache recipe is in `scripts/train/rma/`: the DFS preparation scripts build candidate paths, and the structural scripts make the edited examples and token caches. This path needs a locally screened `runs/structural_20261114/screened_bases.json` mapping authorized SFT source IDs to checked captions; that curation file and the original run outputs cannot be distributed with the restricted dataset. Once the direct Full-G run is prepared, the two regularization variants use its config as a template:

```bash
python scripts/train/qwen/regularization_medium/prepare.py
python scripts/train/qwen/regularization_medium/train.py --config scripts/train/qwen/regularization_medium/training/Full_G_reg/config.json
python scripts/train/qwen/regularization_strong/prepare.py
python scripts/train/qwen/regularization_strong/train.py --config scripts/train/qwen/regularization_strong/training/Full_G_reg/config.json
```

Review generated data, model, and output paths before starting a GPU job.

### Assembly-order augmentation

The paired-order builder produces two views of each sampled tree: strict depth-first (`plain.jsonl`) and a sequence with one subtree moved to the end (`masked.jsonl`). The training and development pools must use disjoint source models.

```bash
python scripts/train/assembly_order/data_construction/build_data.py --split sft --seed 0 --models 10000 --offset 0 --out runs/assembly_order/sft_train_v2
python scripts/train/assembly_order/data_construction/build_data.py --split sft --seed 0 --models 300 --offset 10000 --out runs/assembly_order/sft_dev_v2
python scripts/train/assembly_order/data_construction/verify_data.py runs/assembly_order/sft_train_v2 runs/assembly_order/sft_dev_v2
python scripts/train/assembly_order/training/train.py --arm plain --output runs/assembly_order/train_v2/plain
python scripts/train/assembly_order/training/train.py --arm masked --output runs/assembly_order/train_v2/masked
```

The Qwen paired-order control uses those same pools. After the two BrickNet runs above have written their run configs, `scripts/train/qwen/assembly_order/build_data.py` prepares Qwen token caches and configs. Run its `train.py` with `training/plain/config.json` and `training/masked/config.json` under that directory.

### GRPO

`scripts/train/grpo/runner/run_grpo.sh` starts the GRPO variant from a supervised checkpoint. It needs the authorized paired-order pool, model weights, CUDA, and the vLLM environment. Create a development completion set from a disjoint validation pool, set `START` to the BrickNet PT adapter followed by your supervised adapter, and run:

```bash
python scripts/train/assembly_order/data_construction/build_data.py --split val --models 100 --out runs/assembly_order/val_pool_v2
python scripts/train/assembly_order/completion_evaluation/make_test.py runs/assembly_order/val_pool_v2 --out runs/grpo_dev/test.jsonl
START="kulits/BrickNet-0.6B-PT PATH_TO_SUPERVISED_ADAPTER" POOL=runs/assembly_order/sft_train_v2 DEV=runs/grpo_dev/test.jsonl bash scripts/train/grpo/runner/run_grpo.sh
```

The GRPO launcher uses a separate Conda environment. Install it with `bash scripts/train/grpo/runner/install_vllm_env.sh` after making `conda` available on `PATH`. It also uses a `bricknet` Conda environment for the CPU checkpoint merge; install the packages from Section 2 in that environment. Training configurations and checkpoints are not part of this repository.

## 4. Generate and evaluate

`scripts/generate.py` reads one JSON object per line with `id` and `prefix`. For a completion check, add `target` (or `completion`). For paired reference-sensitivity scoring, provide `source_key`, `condition` (`original` or `stretched`), the appropriate `target_short` and `target_long`, and keep `target_long` identical as the scored long target under both contexts. The [public example](data/smoke_example.jsonl) has a prefix and completion; it is suitable for a smoke run.

```bash
python scripts/generate.py --model Qwen/Qwen3-0.6B --adapter PATH_TO_ADAPTER --input data/smoke_example.jsonl --output runs/example_predictions.jsonl --samples 4
python eval/score.py --predictions runs/example_predictions.jsonl --output runs/example_scores.json
```

Omit `--adapter` for a raw base model; repeat it in merge order for a stack of adapters. The scorer reports part-and-pose F1 and Exact@N over the generated samples. On paired inputs, inference also scores the complete long target with teacher forcing, including EOS; evaluation reports the stretched-minus-original mean change in log-odds. To measure independent generation, create an input JSONL with `{"id":"independent_01","prefix":"a"}` and run:

```bash
python scripts/generate.py --model Qwen/Qwen3-0.6B --adapter PATH_TO_ADAPTER --input runs/independent_inputs.jsonl --output runs/independent_outputs.jsonl --samples 64 --max-new-tokens 6400 --no-eos
python eval/score_independent.py --predictions runs/independent_outputs.jsonl --output runs/independent_scores.json
```

The independent score is the mean valid brick prefix before a parser failure. For the paper's exact 100-case benchmark, authorized inputs are required. `eval/new100/source/build_dataset.py --out runs/new100` constructs a new 100-case set from gated validation graphs, and `eval/new100/source/validate_dataset.py --data runs/new100` checks it. Pass `--legacy-input` with authorized historical pairs only if you need the original source selection. `eval/frontier/source/build_dataset.py` reorders an authorized DFS input file for the alternate traversal. Results from newly sampled cases should be kept separate from the published observations.

## 5. Rebuild the published numerical results

The repository includes only pseudonymous numeric observations for the 100-case completion benchmark, paired frontier comparison, independent generations, and training curves. They are sufficient to regenerate the project's numerical figures and tables without a checkpoint or the gated dataset:

```bash
python scripts/results/build_results.py
```

Outputs appear in `results/tables/` and `results/figures/`. The reported metrics are teacher-forced paired change in log-odds, completion Pose F1, Exact@4, and the valid independent-generation prefix with EOS suppressed. The stopping-behavior appendix also reports collision-free prefix length. Qualitative assembly renders need the restricted input geometry and are not included.

The BrickNet authors retain their own software licenses and dataset access terms.
