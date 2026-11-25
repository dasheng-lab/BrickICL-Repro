"""Train a LoRA adapter with GRPO on LEGO completion prompts.

Rollouts use vLLM. The reward is pose F2; advantages are centered within
each prompt group. A reference-policy KL term limits update size.
The launcher is runner/run_grpo.sh.
"""

import argparse
import json
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

KEEP = ("source", "pair", "nodes", "k", "n_prefix", "jump", "lag", "kind", "copies")


def cap_tokens(k, extra):
    return 64 * (k + extra) + 32  # sampling budget based on target length


def score_group(job):
    """CPU worker: score every completion sampled for one prompt."""
    r, comps, beta, fail = job
    ref = rewards.realize(r["prefix"] + r["target"])
    out = []
    for t, e in comps:
        s = {"text": t, "eos": e} | rewards.completion(r["prefix"], r["target"], t, r["n_prefix"], r["k"], ref,
                                                       beta=beta, fail=fail)
        out.append(s)
    return out


def jobs(rows, comps, args):
    return [(r, [(x[0], x[1]) for x in c], args.beta, args.fail_reward) for r, c in zip(rows, comps)]


def give_first_part(r):
    """Move the target's first part into the prefix; drop targets with no remaining parts."""
    lines = r["target"].splitlines(keepends=True)
    assert len(lines) == 2 * r["k"], (r.get("pair"), len(lines), r["k"])
    if r["k"] < 2:
        return None
    return r | {"prefix": r["prefix"] + "".join(lines[:2]), "target": "".join(lines[2:]),
                "n_prefix": r["n_prefix"] + 1, "k": r["k"] - 1, "first_part_given": True}


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
    """Sample-level means for completion groups."""
    s = [(r, x) for r, g in groups for x in g]
    if not s:
        return {}
    return {
        "reward": float(np.mean([x["reward"] for _, x in s])),
        "f1": float(np.mean([x["f1"] for _, x in s])),
        "parses": float(np.mean([x["parses"] for _, x in s])),
        "exact": float(np.mean([x["exact"] for _, x in s])),
        "eos": float(np.mean([x["eos"] for _, x in s])),
    }


def evaluate_dev(sampler, pool, dev, args, step, out, name="dev"):
    t = time.time()
    comps = sampler.sample(dev, args.dev_n, args.seed + 7)
    scored = pool.map(score_group, jobs(dev, comps, args), chunksize=4)
    proto = {"stop": "eos", "extra_parts": args.extra_parts, "n": args.dev_n, "seed": args.seed + 7, "sampler": "vllm",
             "beta": args.beta, "fail_reward": args.fail_reward}
    write_jsonl(out / name / f"step-{step}.jsonl",
                [{k: r[k] for k in KEEP if k in r} | {"samples": s, "name": f"grpo-{step}", "protocol": proto}
                 for r, s in zip(dev, scored)])
    return group_stats(list(zip(dev, scored))) | {"seconds": time.time() - t}


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
    ap.add_argument("--dev", required=True, help="development completion prompts")
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
    ap.add_argument("--beta", type=float, default=2.0, help="pose F2 reward")
    ap.add_argument("--fail-reward", type=float, default=0.0, help="reward when the text does not parse or is empty")
    ap.add_argument("--loss-norm", choices=["batch", "const", "seq"], default="batch")
    ap.add_argument("--norm-tokens", type=float, default=70_000, help="--loss-norm const: r1's median batch, steps 1-50")
    ap.add_argument("--first-part", action="store_true",
                    help="give the first target part in the prompt for training and evaluation")
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
    sampler = Sampler(args, tok)
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(args.start, torch_dtype=torch.bfloat16).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    step, cursor, state = 0, 0, {}
    ckpts = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if args.resume and ckpts:
        policy = PeftModel.from_pretrained(model, ckpts[-1], is_trainable=True)
        state = json.loads((ckpts[-1] / "state.json").read_text())
        step, cursor = state["step"], state["cursor"]
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
        save_checkpoint(policy, opt, out, s, {"cursor": cursor})
        dv = evaluate_dev(sampler, pool, dev, args, s, out)
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
            rows = [prompts[(cursor + i) % len(prompts)] for i in range(args.gen_prompts)]
            cursor += args.gen_prompts
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
        st |= {"gen_s": t_gen, "score_s": t_score, "step_s": time.time() - t0, "cursor": cursor}
        log.write(json.dumps(st) + "\n")
        log.flush()
        print(f"step {step}: reward {st.get('rollout_reward', 0):.3f} kept {len(kept)}/{len(seen)} "
              f"parses {st.get('rollout_parses', 0):.2f} exact {st.get('rollout_exact', 0):.2f} "
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
