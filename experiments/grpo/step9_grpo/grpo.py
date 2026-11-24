"""Step 9: GRPO on completion prompts, from the merged start model. Rollouts with vLLM, updates of a new LoRA in the
same process, scoring on CPU workers. (GPU, vllm env; ask before starting)

One step:
  1. draw --gen-prompts training prompts (|M| <= --max-k, the fixed seeded order of common.sampling.prompts) and
     sample --n completions each with vLLM and the current adapter, under the protocol of step 3 / step 11:
     temperature 1, top-k 20, top-p 0.95, stop only at the tokenizer's EOS (allowed from the first token),
     at most 64 (|M| + --extra-parts) + 32 new tokens;
  2. score them with common.rewards.completion (Pose F1, 0 if the text does not parse to its last part);
  3. keep only groups whose rewards differ (the rest give no gradient); while fewer than --min-groups are kept,
     sample another round (at most --max-draws); all kept groups go into the batch (about 27 of 64 in the pilot);
  4. advantage = reward - group mean (no division by the group std); one gradient step on
        sum over kept tokens of [ -A * logp + kl * k3(ref, logp) ] / number of kept tokens
     (one update per batch, so the PPO ratio is 1 and needs no clipping; ref = the start model, i.e. the adapter
     turned off; k3 = exp(ref - logp) - (ref - logp) - 1);
  5. save the adapter and load it into vLLM for the next step.

Every --save-every steps (and at the end): <out>/checkpoint-<step>/ (adapter, optimizer, state) and the dev test
sampled with --dev-n samples per prompt into <out>/dev/step-<step>.jsonl, in the format of step 11's eval_free.py,
so step7_eval/select_checkpoint.py reads it directly. <out>/log.jsonl has one line per step; <out>/summary.json is
written at the end (step 7 needs it). --resume continues from the last checkpoint.

The adapter is trained on top of the merged start model (merge_start.py), so for step 7 the arm is
"start adapters + checkpoint-<step>", like the RFT arms.

v2 options (all off by default, so the defaults are r1; run_grpo.sh PRESET=v2 turns them on; why: README):
  --beta B           reward F-beta (pose) instead of F1: B > 1 makes stopping short cost more than writing too much
  --fail-reward X    reward of a completion that does not parse or is empty (r1: 0, the same as a wrong one)
  --loss-norm        batch: divide the summed token loss by this batch's token count (r1);
                     const: by --norm-tokens, a fixed number, so the step size does not grow as outputs get shorter;
                     seq:   average over each sample's own tokens, then over samples (original GRPO)
  --stretch F        stretched-reference prompts (build_stretch_prompts.py), --stretch-frac of every draw, both
                     conditions of a pair in the same draw
  --stretch-dev F    the same kind of prompts from other sft models, sampled with the dev test at every checkpoint:
                     long-feature hit rate per condition and sunshq's sampled Δlog-odds, into dev.jsonl

Usage (from ~/RL, after `ENV_NAME=vllm source env.sh`): see run_grpo.sh.
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import rewards  # noqa: E402
from common.files import read_jsonl, write_jsonl, write_run_config  # noqa: E402

KEEP = ("source", "pair", "nodes", "k", "n_prefix", "jump", "lag", "kind", "copies",
        "condition", "r", "k_short", "k_long", "extra_part", "n_extra", "n_extra_long")


def cap_tokens(k, extra):
    return 64 * (k + extra) + 32  # as step 11's sample_free


def score_group(job):
    """CPU worker: rewards.completion for every sample of one prompt (and, for a stretched-reference prompt, how
    many of the extra part it writes and whether that reaches the long copy's count: sunshq's long-feature hit)."""
    r, comps, beta, fail = job
    ref = rewards.realize(r["prefix"] + r["target"])
    out = []
    for t, e in comps:
        s = {"text": t, "eos": e} | rewards.completion(r["prefix"], r["target"], t, r["n_prefix"], r["k"], ref,
                                                       beta=beta, fail=fail)
        if "extra_part" in r:
            s["n_extra"] = rewards.part_count(r["prefix"], t, r["n_prefix"], r["extra_part"])
            s["long_hit"] = s["n_extra"] >= r["n_extra_long"]
        out.append(s)
    return out


def jobs(rows, comps, args):
    return [(r, [(x[0], x[1]) for x in c], args.beta, args.fail_reward) for r, c in zip(rows, comps)]


def give_first_part(r):
    """sunshq's "first part given" (p1): M's first part (its two lines) moves from the target to the end of the
    prefix, so the model writes the other k - 1 parts. Rows with k = 1 have nothing left and are dropped (None)."""
    lines = r["target"].splitlines(keepends=True)
    assert len(lines) == 2 * r["k"], (r.get("pair"), len(lines), r["k"])
    if r["k"] < 2:
        return None
    return r | {"prefix": r["prefix"] + "".join(lines[:2]), "target": "".join(lines[2:]),
                "n_prefix": r["n_prefix"] + 1, "k": r["k"] - 1, "first_part_given": True}


def give_first_part_pairs(rows):
    """give_first_part for stretched-reference rows: the extra-part counts are recounted on the shorter targets
    (n_extra_long = the stretched row's count, as sunshq's p1 threshold), and a pair is kept only whole."""
    by = {}
    for r in rows:
        by.setdefault((r["source"], r["sample_index"]), {})[r["condition"]] = give_first_part(r)
    out = []
    for p in by.values():
        if len(p) != 2 or None in p.values():
            continue
        for r in p.values():
            r["n_extra"] = rewards.part_count(r["prefix"], r["target"], r["n_prefix"], r["extra_part"])
        for r in p.values():
            r["n_extra_long"] = p["stretched"]["n_extra"]
        out += [p["stretched"], p["original"]]
    return out


def first_part_all(rows):
    return [x for x in map(give_first_part, rows) if x is not None]


class Sampler:
    """vLLM engine holding the merged start model; the current adapter is passed per call."""

    def __init__(self, args, tok):
        from vllm import LLM

        self.tok, self.eos, self.extra = tok, tok.eos_token_id, args.extra_parts
        self.llm = LLM(model=str(args.start), tokenizer=str(args.start), dtype="bfloat16", seed=args.seed,
                       enable_lora=True, max_lora_rank=args.lora_r, max_loras=1, max_cpu_loras=2,
                       gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
                       generation_config="vllm", enable_prefix_caching=True)
        self.lora = None

    def set_adapter(self, path, uid):
        from vllm.lora.request import LoRARequest

        self.lora = LoRARequest(f"grpo-{uid}", uid, str(path))

    def sample(self, rows, n, seed):
        """[[(text, eos, completion token ids incl. EOS if it stopped)] * n per row]"""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = [SamplingParams(n=n, temperature=1.0, top_k=20, top_p=0.95, seed=seed + i,
                                 max_tokens=cap_tokens(r["k"], self.extra),
                                 ignore_eos=True, stop_token_ids=[self.eos],  # stop at tok.eos only, as HF did
                                 skip_special_tokens=True)
                  for i, r in enumerate(rows)]
        prompts = [TokensPrompt(prompt_token_ids=r["prompt_ids"]) for r in rows]
        outs = self.llm.generate(prompts, params, lora_request=self.lora, use_tqdm=False)
        result = []
        for o in outs:
            group = []
            for c in o.outputs:
                ids, stopped = list(c.token_ids), c.finish_reason == "stop"
                if stopped and (not ids or ids[-1] != self.eos):
                    ids.append(self.eos)
                text = self.tok.decode([t for t in ids if t != self.eos], skip_special_tokens=True)
                group.append((text, stopped, ids))
            result.append(group)
        return result


def token_logps(base, input_ids, attn, spans, chunk):
    """Log-probs (and entropies) of the completion tokens. spans: (row, first target position, length).

    Only completion positions go through lm_head, in checkpointed chunks, so the vocab-sized logits are never all
    in memory at once."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    h = base.model(input_ids=input_ids, attention_mask=attn).last_hidden_state
    rows = torch.cat([torch.full((n,), b, device=h.device) for b, _, n in spans])
    pos = torch.cat([torch.arange(s - 1, s - 1 + n, device=h.device) for _, s, n in spans])
    hs = h[rows, pos]
    targets = input_ids[rows, pos + 1]

    def piece(x, y):
        logits = base.lm_head(x).float()
        lp = F.log_softmax(logits, -1)
        ent = -(lp.exp() * lp).sum(-1)
        return lp.gather(-1, y[:, None])[:, 0], ent

    lps, ents = [], []
    for i in range(0, len(hs), chunk):
        if torch.is_grad_enabled():
            lp, ent = checkpoint(piece, hs[i : i + chunk], targets[i : i + chunk], use_reentrant=False)
        else:
            lp, ent = piece(hs[i : i + chunk], targets[i : i + chunk])
        lps.append(lp)
        ents.append(ent.detach())
    return torch.cat(lps), torch.cat(ents)


def micro_batches(seqs, budget):
    """Group sequence indices (sorted by length) so that rows x longest length <= budget tokens."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
    out, cur, longest = [], [], 0
    for i in order:
        n = len(seqs[i][0]) + len(seqs[i][1])
        if cur and max(longest, n) * (len(cur) + 1) > budget:
            out.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, n)
    if cur:
        out.append(cur)
    return out


def train_step(policy, opt, batch, args, pad_id, device):
    """batch: [(prompt ids, completion ids, advantage)]. One optimizer step; returns statistics."""
    import torch

    base = policy.get_base_model()
    total = sum(len(c) for _, c, _ in batch)
    if args.loss_norm == "batch":       # r1: summed token loss / this batch's tokens
        weight = lambda c: 1.0 / total  # noqa: E731
    elif args.loss_norm == "const":     # the same sum / a fixed number
        weight = lambda c: 1.0 / args.norm_tokens  # noqa: E731
    else:                               # seq: mean over each sample's tokens, then over samples
        weight = lambda c: 1.0 / (len(c) * len(batch))  # noqa: E731
    sums = {"loss": 0.0, "kl": 0.0, "entropy": 0.0, "logp": 0.0}
    policy.train()
    for mb in micro_batches(batch, args.micro_tokens):
        L = max(len(batch[i][0]) + len(batch[i][1]) for i in mb)
        ids = torch.full((len(mb), L), pad_id, dtype=torch.long)
        attn = torch.zeros((len(mb), L), dtype=torch.long)
        spans, adv, wt = [], [], []
        for b, i in enumerate(mb):
            p, c, a = batch[i]
            ids[b, : len(p) + len(c)] = torch.tensor(p + c)
            attn[b, : len(p) + len(c)] = 1
            spans.append((b, len(p), len(c)))
            adv.append(torch.full((len(c),), float(a)))
            wt.append(torch.full((len(c),), weight(c)))
        ids, attn = ids.to(device), attn.to(device)
        adv, wt = torch.cat(adv).to(device), torch.cat(wt).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp, ent = token_logps(base, ids, attn, spans, args.logit_chunk)
            loss_tok = -adv * torch.exp(lp - lp.detach())  # = -A * logp in value of the gradient
            if args.kl > 0:
                with torch.no_grad(), policy.disable_adapter():
                    ref, _ = token_logps(base, ids, attn, spans, args.logit_chunk)
                d = ref - lp
                k3 = torch.exp(d) - d - 1
                loss_tok = loss_tok + args.kl * k3
                sums["kl"] += k3.detach().sum().item()
        (loss_tok * wt).sum().backward()
        sums["loss"] += loss_tok.detach().sum().item()
        sums["entropy"] += ent.sum().item()
        sums["logp"] += lp.detach().sum().item()
    gnorm = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], args.max_grad_norm)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return {k: v / total for k, v in sums.items()} | {"grad_norm": float(gnorm), "tokens": total}


def group_stats(groups):
    """groups: [(row, scored samples)] -> sample-level means, as step 3's report."""
    s = [(r, x) for r, g in groups for x in g]
    if not s:
        return {}
    return {
        "reward": float(np.mean([x["reward"] for _, x in s])),
        "reward_f1": float(np.mean([x.get("reward_f1", x["reward"]) for _, x in s])),  # the r1 reward
        "f1": float(np.mean([x["f1"] for _, x in s])),
        "parses": float(np.mean([x["parses"] for _, x in s])),
        "exact": float(np.mean([x["exact"] for _, x in s])),
        "eos": float(np.mean([x["eos"] for _, x in s])),
        "np_over_nt": float(np.mean([x["n_generated"] / r["k"] for r, x in s])),
        "empty": float(np.mean([x["n_generated"] == 0 for _, x in s])),
        "short": float(np.mean([x["parses"] and x["n_generated"] < r["k"] for r, x in s])),  # stopped before n_t
        "varies": float(np.mean([len({x["reward"] for x in g}) > 1 for _, g in groups])),
    }


def logodds(k, n):
    return math.log((k + 0.5) / (n - k + 0.5))


def stretch_stats(groups):
    """Stretched-reference prompts: per condition the long-feature hit rate and the extra parts written, and
    sunshq's sampled Δlog-odds (mean over copies of logodds(hits | stretched) - logodds(hits | original))."""
    by = {}
    for r, g in groups:
        if "condition" in r:
            by.setdefault((r["source"], r["sample_index"]), {})[r["condition"]] = g
    st = {}
    for cond in ("stretched", "original"):
        gs = [p[cond] for p in by.values() if cond in p]
        if gs:
            st[f"hit, {cond}"] = float(np.mean([x["long_hit"] for g in gs for x in g]))
            st[f"extra parts, {cond}"] = float(np.mean([x["n_extra"] for g in gs for x in g]))
    pairs = [p for p in by.values() if len(p) == 2]
    if pairs:
        d = [logodds(sum(x["long_hit"] for x in p["stretched"]), len(p["stretched"]))
             - logodds(sum(x["long_hit"] for x in p["original"]), len(p["original"])) for p in pairs]
        st["dlogodds"] = float(np.mean(d))
        st["pairs"] = len(pairs)
    return st


def evaluate_dev(sampler, pool, dev, args, step, out, name="dev"):
    t = time.time()
    comps = sampler.sample(dev, args.dev_n, args.seed + 7)
    scored = pool.map(score_group, jobs(dev, comps, args), chunksize=4)
    proto = {"stop": "eos", "extra_parts": args.extra_parts, "n": args.dev_n, "seed": args.seed + 7, "sampler": "vllm",
             "beta": args.beta, "fail_reward": args.fail_reward}
    write_jsonl(out / name / f"step-{step}.jsonl",
                [{k: r[k] for k in KEEP if k in r} | {"samples": s, "name": f"grpo-{step}", "protocol": proto}
                 for r, s in zip(dev, scored)])
    st = group_stats(list(zip(dev, scored)))
    jump = [(r, s) for r, s in zip(dev, scored) if r.get("jump")]
    return st | stretch_stats(list(zip(dev, scored))) | {"reward, jump": group_stats(jump).get("reward"),
                                                          "seconds": time.time() - t}


def save_checkpoint(policy, opt, out, step, state):
    import torch

    d = out / f"checkpoint-{step}"
    policy.save_pretrained(d)
    torch.save(opt.state_dict(), d / "optimizer.pt")
    (d / "state.json").write_text(json.dumps(state | {"step": step}) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="merged start model (merge_start.py)")
    ap.add_argument("--pool", required=True, help="training pool (its masked.jsonl)")
    ap.add_argument("--dev", required=True, help="dev test (step 2)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--gen-prompts", type=int, default=64, help="prompts sampled per round")
    ap.add_argument("--min-groups", type=int, default=16, help="sample another round while fewer groups are kept")
    ap.add_argument("--max-draws", type=int, default=3, help="sampling rounds per step at most")
    ap.add_argument("--n", type=int, default=16, help="samples per prompt (group size)")
    ap.add_argument("--max-k", type=int, default=10)
    ap.add_argument("--extra-parts", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--kl", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--micro-tokens", type=int, default=32_768, help="rows x longest length per micro-batch")
    ap.add_argument("--logit-chunk", type=int, default=2048, help="completion tokens per lm_head chunk")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--dev-n", type=int, default=4)
    ap.add_argument("--dev-rows", type=int, default=None, help="first N dev prompts only (smoke tests)")
    ap.add_argument("--gpu-mem", type=float, default=0.3, help="fraction of GPU memory for vLLM")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=16, help="CPU processes for scoring")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--beta", type=float, default=1.0, help="reward F-beta; 1 = F1 (r1)")
    ap.add_argument("--fail-reward", type=float, default=0.0, help="reward when the text does not parse or is empty")
    ap.add_argument("--loss-norm", choices=["batch", "const", "seq"], default="batch")
    ap.add_argument("--norm-tokens", type=float, default=70_000, help="--loss-norm const: r1's median batch, steps 1-50")
    ap.add_argument("--stretch", help="stretched-reference training prompts (build_stretch_prompts.py)")
    ap.add_argument("--stretch-frac", type=float, default=0.25, help="share of every draw taken from --stretch")
    ap.add_argument("--stretch-dev", help="stretched-reference dev prompts, sampled at every checkpoint")
    ap.add_argument("--first-part", action="store_true",
                    help="give M's first part in the prompt (sunshq's p1) in training, dev and stretched prompts")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from common import sampling

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_run_config(out, args, rewards.EXTERNAL + sampling.EXTERNAL)
    pool = get_context("spawn").Pool(args.workers)  # before vLLM and CUDA

    tok = AutoTokenizer.from_pretrained(args.start)
    budget = args.max_model_len
    prompts = sampling.prompts(args.pool, max_k=args.max_k, seed=args.seed)
    dev = read_jsonl(args.dev)[: args.dev_rows]
    if args.first_part:
        n0, d0 = len(prompts), len(dev)
        prompts, dev = first_part_all(prompts), first_part_all(dev)
        print(f"first part given: {n0 - len(prompts)} training and {d0 - len(dev)} dev prompts with |M| = 1 dropped",
              flush=True)
    for r in prompts + dev:
        r["prompt_ids"] = tok.encode(r["prefix"], add_special_tokens=False)
    fits = lambda r: len(r["prompt_ids"]) + cap_tokens(r["k"], args.extra_parts) <= budget  # noqa: E731
    prompts = [r for r in prompts if fits(r)]
    assert all(map(fits, dev)), "a dev prompt is longer than --max-model-len"
    print(f"{len(prompts)} training prompts (|M| <= {args.max_k}), {len(dev)} dev prompts", flush=True)
    stretch, stretch_dev, n_st = [], [], 0
    if args.stretch:  # pairs (both conditions of one held-out copy) stay together, in a seeded order of pairs
        pairs = {}
        rows = read_jsonl(args.stretch)
        for r in give_first_part_pairs(rows) if args.first_part else rows:
            pairs.setdefault((r["source"], r["sample_index"]), []).append(r)
        keys = list(pairs)
        stretch = [r for i in np.random.default_rng(args.seed).permutation(len(keys)) for r in pairs[keys[i]]]
        for r in stretch:
            r["prompt_ids"] = tok.encode(r["prefix"], add_special_tokens=False)
        assert all(map(fits, stretch)), "a stretched prompt is longer than --max-model-len"
        n_st = 2 * round(args.stretch_frac * args.gen_prompts / 2)
        print(f"{len(stretch)} stretched-reference prompts ({len(keys)} pairs), {n_st} per draw", flush=True)
    if args.stretch_dev:
        stretch_dev = read_jsonl(args.stretch_dev)
        if args.first_part:
            stretch_dev = give_first_part_pairs(stretch_dev)
        for r in stretch_dev:
            r["prompt_ids"] = tok.encode(r["prefix"], add_special_tokens=False)
        assert all(map(fits, stretch_dev)), "a stretched dev prompt is longer than --max-model-len"
        print(f"{len(stretch_dev)} stretched-reference dev prompts", flush=True)

    sampler = Sampler(args, tok)
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(args.start, torch_dtype=torch.bfloat16).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    step, cursor, s_cursor, state = 0, 0, 0, {}
    ckpts = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if args.resume and ckpts:
        policy = PeftModel.from_pretrained(model, ckpts[-1], is_trainable=True)
        state = json.loads((ckpts[-1] / "state.json").read_text())
        step, cursor, s_cursor = state["step"], state["cursor"], state.get("stretch_cursor", 0)
    else:
        cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                         task_type="CAUSAL_LM")
        policy = get_peft_model(model, cfg)
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    if args.resume and ckpts:
        opt.load_state_dict(torch.load(ckpts[-1] / "optimizer.pt"))
        for g in opt.param_groups:  # the saved state carries the old lr; --lr on resume should win
            g["lr"] = g["initial_lr"] = args.lr
        sampler.set_adapter(ckpts[-1], step + 1)
        print(f"resumed at step {step} from {ckpts[-1]}", flush=True)
    first = step
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (first + s + 1) / max(1, args.warmup)))
    live = out / "live"

    def checkpoint_and_dev(s):
        save_checkpoint(policy, opt, out, s, {"cursor": cursor, "stretch_cursor": s_cursor})
        dv = evaluate_dev(sampler, pool, dev, args, s, out)
        if stretch_dev:
            dv |= {f"stretch {k}": v for k, v in evaluate_dev(sampler, pool, stretch_dev, args, s, out,
                                                              "dev_stretch").items()}
        print(f"[dev] step {s}: " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in dv.items()}),
              flush=True)
        with open(out / "dev.jsonl", "a") as f:
            f.write(json.dumps({"step": s} | dv) + "\n")

    if step == 0:
        checkpoint_and_dev(0)
    log = open(out / "log.jsonl", "a")
    while step < args.steps:
        t0 = time.time()
        kept, seen, draws, t_gen, t_score = [], [], 0, 0.0, 0.0
        while len(kept) < args.min_groups and draws < args.max_draws:
            rows = [prompts[(cursor + i) % len(prompts)] for i in range(args.gen_prompts - n_st)]
            rows += [stretch[(s_cursor + i) % len(stretch)] for i in range(n_st)]
            cursor += args.gen_prompts - n_st
            s_cursor += n_st
            draws += 1
            t = time.time()
            comps = sampler.sample(rows, args.n, args.seed + 1_000_003 * step + 10_007 * draws)
            t_gen += time.time() - t
            t = time.time()
            scored = pool.map(score_group, jobs(rows, comps, args), chunksize=2)
            t_score += time.time() - t
            for r, c, s in zip(rows, comps, scored):
                seen.append((r, s))
                rw = [x["reward"] for x in s]
                if max(rw) > min(rw):
                    kept.append((r, c, rw))
        st ={"step": step + 1, "draws": draws, "prompts_seen": len(seen), "groups_kept": len(kept)}
        st |= {f"rollout_{k}": v for k, v in group_stats(seen).items()}
        if n_st:
            both = group_stats([x for x in seen if "condition" in x[0]]) | stretch_stats(seen)
            st |= {f"rollout_stretch_{k}": v for k, v in both.items()}
        if kept:
            batch = []
            for r, c, rw in kept:
                mean = float(np.mean(rw))
                batch += [(r["prompt_ids"], x[2], w - mean) for x, w in zip(c, rw) if x[2]]
            t = time.time()
            ts = train_step(policy, opt, batch, args, tok.pad_token_id or tok.eos_token_id, device)
            sched.step()
            st |= ts | {"lr": sched.get_last_lr()[0], "train_s": time.time() - t}
        step += 1
        d = live / f"step-{step}"
        policy.save_pretrained(d)
        sampler.set_adapter(d, step + 1)
        for old in live.glob("step-*"):
            if int(old.name.split("-")[1]) < step - 1:
                shutil.rmtree(old, ignore_errors=True)
        st |= {"gen_s": t_gen, "score_s": t_score, "step_s": time.time() - t0, "cursor": cursor,
               "stretch_cursor": s_cursor}
        log.write(json.dumps(st) + "\n")
        log.flush()
        print(f"step {step}: reward {st.get('rollout_reward', 0):.3f} kept {len(kept)}/{len(seen)} "
              f"n_p/n_t {st.get('rollout_np_over_nt', 0):.2f} parses {st.get('rollout_parses', 0):.2f} "
              f"empty {st.get('rollout_empty', 0):.2f} short {st.get('rollout_short', 0):.2f} "
              f"kl {st.get('kl', 0):.4f} ent {st.get('entropy', 0):.3f} gen {t_gen:.0f}s train {st.get('train_s', 0):.0f}s",
              flush=True)
        if step % args.save_every == 0 or step == args.steps:
            checkpoint_and_dev(step)
    log.close()
    dev_log = read_jsonl(out / "dev.jsonl")
    best = max(dev_log, key=lambda r: (r["reward"], r["step"]))
    (out / "summary.json").write_text(json.dumps({"steps": step, "best_dev": best, "dev": dev_log}, indent=1) + "\n")
    pool.close()
    print(f"done: {out}; best dev reward {best['reward']:.4f} at step {best['step']}", flush=True)


if __name__ == "__main__":
    main()
