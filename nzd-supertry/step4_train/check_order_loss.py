"""Is a sequence order in the distribution BrickNet-PT was trained on? Judged by the frozen model's loss.

The shipped walks (paths_*.jsonl) are depth-first preorders of their spanning tree: every part
attaches to the previous part or to one of its ancestors. The released sampler
(sample_collision_free_tree) writes the same kind of tree in visit order instead, which the frozen
model finds far less likely.

Two modes, both on the GPU machine from the repo root:

  orders SPLIT   same models, different orders: the shipped walks, the released sampler's visit
                 order, and the same tree re-written depth-first with three sibling orders
  pool DIR       a pool written by step3_data/build_data.py: loss of both arms, overall, on the
                 block of M alone, and on the end-of-sequence token

  python nzd-supertry/step4_train/check_order_loss.py orders sft
  python nzd-supertry/step4_train/check_order_loss.py pool runs/nzd-supertry/sft_dev_v2
"""

import argparse
import io
import json
import lzma
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "step2_sampler"))
import bricknet.graph as bg  # noqa: E402
from bricknet import load_graphs, serialize_tree  # noqa: E402
from sampler import collision_free_walk, serializable  # noqa: E402

D = "baseline/bricknet/data/dataset/"


def preorder(order, parent_of, rng, kids_order):
    kids = defaultdict(list)
    for v in order[1:]:
        kids[parent_of[v][0]].append(v)  # placement order
    out, stack = [], [order[0]]
    while stack:
        u = stack.pop()
        out.append(u)
        ks = list(kids[u])
        if kids_order == "random":
            ks = [ks[i] for i in rng.permutation(len(ks))]
        elif kids_order == "reverse":
            ks = ks[::-1]
        stack.extend(reversed(ks))  # the first of ks is visited first
    return out


def shipped(split, sources, per=2, max_scan=4_000_000):
    """Up to `per` first-round shipped walks for each model in `sources`."""
    proc = subprocess.Popen(["xz", "-dcT0", D + f"paths_{split}.jsonl.xz"], stdout=subprocess.PIPE, text=True)
    got, first = {s: [] for s in sources}, None
    for i, line in enumerate(proc.stdout):
        r = json.loads(line)
        first = first or r["round"]
        if r["round"] == first and r["source"] in got and len(got[r["source"]]) < per:
            got[r["source"]].append(r["path"].strip(" "))
        if i >= max_scan:
            break
    proc.kill()
    return [p for v in got.values() for p in v]


@torch.no_grad()
def nll(model, tok, texts, spans=None, bs=8):
    """Mean nats/token over all tokens, over the tokens inside each text's character span, and of the EOS."""
    enc = tok(texts, add_special_tokens=False, return_offsets_mapping=True)
    ids = [x[:6400] + [tok.eos_token_id] for x in enc["input_ids"]]
    order = np.argsort([len(x) for x in ids])
    tot = cnt = eos = in_tot = in_cnt = 0.0
    for b in range(0, len(ids), bs):
        idx = order[b : b + bs]
        batch = [ids[j] for j in idx]
        w = max(map(len, batch))
        x = torch.full((len(batch), w), tok.eos_token_id)
        y = torch.full((len(batch), w), -100)
        for r, s in enumerate(batch):
            x[r, : len(s)] = torch.tensor(s)
            y[r, : len(s)] = torch.tensor(s)
        x, y = x.cuda(), y.cuda()
        logits = model(input_ids=x).logits.float()
        lp = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), y[:, 1:].reshape(-1), ignore_index=-100, reduction="none"
        ).view(len(batch), -1)  # lp[r, t] is the loss of token t + 1
        mask = y[:, 1:] != -100
        tot += lp[mask].sum().item()
        cnt += mask.sum().item()
        for r, j in enumerate(idx):
            n = len(batch[r])
            eos += lp[r, n - 2].item()
            if spans is not None:
                c0, c1 = spans[j]
                inside = [t for t, (a, _) in enumerate(enc["offset_mapping"][j][: n - 1]) if c0 <= a < c1 and t > 0]
                in_tot += lp[r, [t - 1 for t in inside]].sum().item()
                in_cnt += len(inside)
    return tot / cnt, eos / len(ids), (in_tot / in_cnt if in_cnt else float("nan"))


def block_span(row, start_part):
    """Character span of the 2k lines that write M, which starts at part index `start_part`."""
    lines = row["path"].splitlines(keepends=True)
    a = 1 + 2 * (start_part - 1)
    b = a + 2 * row["k"]
    return sum(map(len, lines[:a])), sum(map(len, lines[:b]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["orders", "pool"])
    ap.add_argument("target", help="orders: sft or val; pool: a build_data.py output directory")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--lora", action="append", default=None, help="adapters merged in order (default: the official PT)")
    ap.add_argument("-n", type=int, default=150, help="orders: models; pool: pairs = 4n")
    args = ap.parse_args()
    adapters = args.lora or ["kulits/BrickNet-0.6B-PT"]

    sets, spans = {}, {}
    if args.mode == "orders":
        split = args.target
        graphs = load_graphs(io.BytesIO(lzma.open(D + f"{split}.npz.xz").read()))
        if split == "sft" and Path("runs/nzd-supertry/sft_dev_v2/stats.json").exists():
            models = json.load(open("runs/nzd-supertry/sft_dev_v2/stats.json"))["sources"][: args.n]
        else:
            models = [i for i, g in enumerate(graphs) if 6 <= len(g.part_ids) <= 100 and serializable(g)][: args.n]
        sets["shipped walks"] = shipped(split, set(models))
        for name, ko in (
            ("frontier tree, visit order (released)", None),
            ("frontier tree, preorder, kids by placement", "placement"),
            ("frontier tree, preorder, kids reversed", "reverse"),
            ("frontier tree, preorder, kids random", "random"),
        ):
            out = []
            for s in models:
                for k in range(2):
                    order, parent_of, text = collision_free_walk(graphs[s], k)
                    if ko is not None:
                        new = preorder(order, parent_of, np.random.default_rng(k), ko)
                        text = serialize_tree(bg._build_tree(graphs[s], new, parent_of))
                    out.append(text)
            sets[name] = out
        title = f"{split}, {len(models)} models"
    else:
        n = 4 * args.n
        with open(Path(args.target) / "masked.jsonl") as f:
            masked = [json.loads(line) for _, line in zip(range(n), f)]
        with open(Path(args.target) / "plain.jsonl") as f:
            plain = [json.loads(line) for _, line in zip(range(n), f)]
        jump = [m["jump"] for m in masked]
        groups = {
            "plain (preorder)": (plain, [p["block_start"] for p in plain], [True] * len(plain)),
            "masked, all": (masked, [m["n_prefix"] for m in masked], [True] * len(masked)),
            "masked, off-path attachment": (masked, [m["n_prefix"] for m in masked], jump),
            "masked, still a preorder": (masked, [m["n_prefix"] for m in masked], [not j for j in jump]),
            "plain, pairs of the off-path rows": (plain, [p["block_start"] for p in plain], jump),
        }
        for name, (rows, starts, keep) in groups.items():
            sets[name] = [r["path"] for r, kp in zip(rows, keep) if kp]
            spans[name] = [block_span(r, s) for r, s, kp in zip(rows, starts, keep) if kp]
        title = f"{args.target}, first {len(masked)} pairs"

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    for a in adapters:
        model = PeftModel.from_pretrained(model, a).merge_and_unload()
    model = model.cuda().eval()
    print(f"{args.model} + {adapters}: {title}")
    print(f"  {'':44s} {'n':>4s}  nats/token   EOS loss" + ("   nats/token on M" if spans else ""))
    for name, texts in sets.items():
        loss, eos, inside = nll(model, tok, texts, spans.get(name))
        print(f"  {name:44s} {len(texts):4d}  {loss:10.4f} {eos:10.2f}" + (f" {inside:17.4f}" if spans else ""))


if __name__ == "__main__":
    main()
