"""Post-train BrickNet-PT on one arm of the paired data, with the official trainer.

This is a thin launcher around `baseline/bricknet/BrickNet/scripts/train.py` in `pt` mode: path +
EOS, loss on the whole sequence, a fresh LoRA on top of Qwen3 with the official PT adapter merged
in. The upstream file is not edited. Only the numbers of its PT recipe that assume a 250k-step run
from scratch are replaced:

  steps      one pass over the pool (rows / 32) instead of 250,000
  lr         1e-4 instead of 5e-4 (the start is already a trained model)
  warmup     5% of the steps, the upstream ratio
  eval/save  every --eval-every steps instead of 2,500 / 5,000, on all dev rows, keeping every
             checkpoint (upstream keeps the last three)

  batch      8 sequences per forward pass, gradients accumulated over 4 passes: the upstream
             batch of 32 at once needs more than the 96 GB of one H20 (upstream spreads it over GPUs)
  compile    off: upstream's torch.compile raises an internal Dynamo error on the cluster's
             torch 2.14.1 together with the Liger kernels. It only affects speed.

LoRA shape, optimizer, batch size, max length and loss are upstream's. The two arms must be run
with identical arguments except --arm.

Usage (from the repo root, in the bricknet env, on the GPU machine):
  python experiments/assembly_order/training/train.py --arm masked --output runs/assembly_order/train_v2/masked
  python experiments/assembly_order/training/train.py --arm plain  --output runs/assembly_order/train_v2/plain
"""

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
UPSTREAM = REPO / "baseline/bricknet/BrickNet/scripts/train.py"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "-uno"], capture_output=True, text=True)
        return out.stdout.strip() + (" (dirty)" if dirty.stdout.strip() else "")
    except OSError:
        return None


def dev_curve(output):
    """[(step, eval_loss)] from the newest checkpoint's trainer_state.json."""
    states = sorted(Path(output).glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[1]))
    if not states:
        return []
    log = json.loads(states[-1].read_text())["log_history"]
    return [(e["step"], e["eval_loss"]) for e in log if "eval_loss" in e]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["masked", "plain"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--data", default="runs/assembly_order/sft_train_v2", help="training pool (build_data.py output)")
    ap.add_argument("--dev", default="runs/assembly_order/sft_dev_v2", help="dev pool, models disjoint from --data")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--pt-adapter", default="kulits/BrickNet-0.6B-PT", help="official adapter merged in before training")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=250, help="evaluate on dev and save a checkpoint every N steps")
    ap.add_argument("--dev-samples", type=int, default=0, help="dev rows per evaluation (0: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="override the step count (smoke tests)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--micro-batch", type=int, default=8,
                    help="sequences per forward pass; gradients accumulate to the upstream batch of 32")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile as upstream does; fails on torch 2.14.1 with the Liger kernels (speed only)")
    args = ap.parse_args()

    train_file = Path(args.data) / f"{args.arm}.jsonl"
    dev_file = Path(args.dev) / f"{args.arm}.jsonl"
    output = Path(args.output)
    with open(train_file) as f:
        n_rows = sum(1 for _ in f)

    spec = importlib.util.spec_from_file_location("bricknet_train", UPSTREAM)
    up = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(up)

    recipe = up.RECIPES["pt"]
    batch = recipe["global_batch_size"]
    if batch % args.micro_batch:
        ap.error(f"--micro-batch must divide {batch}")
    steps = args.max_steps or math.ceil(n_rows * args.epochs / batch)
    recipe.update(
        train_iters=steps,
        lr=args.lr,
        warmup_iters=max(1, round(args.warmup_frac * steps)),
        eval_interval=args.eval_every,
        save_interval=args.eval_every,
        val_samples=args.dev_samples or 10**9,
    )

    upstream_arguments = up.TrainingArguments

    def training_arguments(**kw):
        kw.update(
            save_total_limit=None,
            seed=args.seed,
            data_seed=args.seed,
            logging_steps=25,
            torch_compile=args.compile,
            eval_on_start=True,  # step 0 is the frozen BrickNet-PT baseline on the dev pool
            # same effective batch as upstream, split so that it fits one GPU
            per_device_train_batch_size=args.micro_batch,
            per_device_eval_batch_size=args.micro_batch,
            gradient_accumulation_steps=batch // args.micro_batch,
        )
        return upstream_arguments(**kw)

    up.TrainingArguments = training_arguments
    # Upstream's compute_loss returns the token mean of one forward pass. The Trainer would assume
    # it is already normalized over the accumulated batch (Qwen3 accepts loss kwargs) and add the
    # four passes up, giving 4x the gradient of a batch of 32. Tell it to divide instead.
    up.LoraTrainer.loss_is_scaled_for_ga = False

    output.mkdir(parents=True, exist_ok=True)
    config = {
        "args": vars(args),
        "recipe": {k: v for k, v in recipe.items()},
        "train_file": str(train_file),
        "train_rows": n_rows,
        "train_sha256": sha256(train_file),
        "dev_file": str(dev_file),
        "dev_sha256": sha256(dev_file),
        "steps": steps,
        "upstream": str(UPSTREAM.relative_to(REPO)),
        "commit": git_commit(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"[nzd] {args.arm}: {n_rows} rows, {steps} steps of {batch}, lr {args.lr}, "
          f"warmup {recipe['warmup_iters']}, eval/save every {args.eval_every}", flush=True)

    # each run keeps its own tokenization scratch: upstream reuses a scratch dir keyed on file
    # paths only, so a regenerated pool at the same path would otherwise be served stale
    sys.argv = [
        str(UPSTREAM), "pt",
        "--model", args.model,
        "--pt_adapter", args.pt_adapter,
        "--paths", str(train_file),
        "--val_paths", str(dev_file),
        "--output", str(output),
        "--scratch", str(output / "scratch"),
    ] + (["--resume"] if args.resume else [])
    t0 = time.time()
    up.main()
    hours = (time.time() - t0) / 3600

    curve = dev_curve(output)
    best = min(curve, key=lambda x: x[1]) if curve else None
    import torch

    peak_gb = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    summary = {
        "hours": round(hours, 3),
        "peak_gpu_gb": peak_gb and round(peak_gb, 1),
        "dev_curve": curve,
        "best_step": best and best[0],
        "best_dev_loss": best and best[1],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[nzd] done in {hours:.2f} h, peak GPU memory {peak_gb and round(peak_gb, 1)} GB; dev loss by step: {curve}", flush=True)
    if best:
        print(f"[nzd] best checkpoint: {output}/checkpoint-{best[0]} (dev loss {best[1]:.4f})", flush=True)


if __name__ == "__main__":
    main()
