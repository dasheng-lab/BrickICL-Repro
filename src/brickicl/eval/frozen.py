"""Shared model-loading and token-scoring helpers for assembly evaluation."""

import importlib.util
from pathlib import Path

import bricknet
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList



def upstream_stop_after_newlines():
    """StopAfterNewlines from the installed upstream checkout's scripts/generate.py."""
    path = Path(bricknet.__file__).resolve().parents[2] / "scripts" / "generate.py"
    spec = importlib.util.spec_from_file_location("bricknet_generate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.StopAfterNewlines


def load(model_name, loras, dtype, device):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": device}, dtype=dtype)
    if loras:
        from peft import PeftModel

        for path in loras:
            model = PeftModel.from_pretrained(model, path).merge_and_unload()
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tok.pad_token = tok.eos_token
    model.generation_config.pad_token_id = tok.pad_token_id
    return model, tok


@torch.inference_mode()
def target_logprobs(model, tok, rows, batch, device):
    """log p(target | prefix) summed over target tokens, and log p(EOS | prefix + target)."""
    out, mismatched = [], 0
    for i in range(0, len(rows), batch):
        chunk, seqs, spans = rows[i : i + batch], [], []
        for r in chunk:
            p = tok.encode(r["prefix"], add_special_tokens=False)
            t = tok.encode(r["target"], add_special_tokens=False)
            mismatched += tok.encode(r["prefix"] + r["target"], add_special_tokens=False) != p + t
            seqs.append(p + t + [tok.eos_token_id])
            spans.append((len(p), len(t) + 1))
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), tok.pad_token_id, device=device)
        attn = torch.zeros_like(ids)
        for j, s in enumerate(seqs):  # right padding: positions match the unpadded sequence
            ids[j, : len(s)] = torch.tensor(s, device=device)
            attn[j, : len(s)] = 1
        logits = model(input_ids=ids, attention_mask=attn).logits
        for j, (lp, lt) in enumerate(spans):
            pred = logits[j, lp - 1 : lp - 1 + lt].float().log_softmax(-1)
            tok_lp = pred.gather(-1, ids[j, lp : lp + lt, None])[:, 0]
            out.append({"lp_target": tok_lp[:-1].sum().item(), "lp_eos": tok_lp[-1].item(), "n_target_tokens": lt - 1})
    return out, mismatched


def _chunks(idx, lengths, n, new_tokens, budget):
    """Consecutive runs of rows (sorted by prompt length) whose n * (longest prompt + new tokens) fits the budget."""
    chunk = []
    for i in sorted(idx, key=lambda i: lengths[i]):
        if chunk and n * (len(chunk) + 1) * (lengths[i] + new_tokens) > budget:
            yield chunk
            chunk = []
        chunk.append(i)
    if chunk:
        yield chunk


@torch.inference_mode()
def sample(model, tok, rows, n, budget, seed, device, stop_cls=None, k_key="k", extra_parts=0):
    """n completions per row; rows sharing k are batched so one stop rule serves the batch.

    stop_cls: upstream StopAfterNewlines (exactly k parts), or None for free generation that ends at
    the model's own EOS, capped at k + extra_parts parts' worth of tokens.
    """
    nl_id = tok.encode("\n", add_special_tokens=False)[0]
    prompts = [tok.encode(r["prefix"], add_special_tokens=False) for r in rows]
    lengths = [len(p) for p in prompts]
    texts = [None] * len(rows)
    by_k = {}
    for i, r in enumerate(rows):
        by_k.setdefault(r[k_key], []).append(i)
    for k, idx in sorted(by_k.items()):
        done, new_tokens = 0, 64 * (k + extra_parts) + 32
        for chunk in _chunks(idx, lengths, n, new_tokens, budget):
            L = max(lengths[i] for i in chunk)
            padded = [[tok.pad_token_id] * (L - lengths[i]) + prompts[i] for i in chunk for _ in range(n)]
            inp = torch.tensor(padded, device=device)
            attn = torch.zeros_like(inp)
            for j, i in enumerate(chunk):
                attn[j * n : (j + 1) * n, L - lengths[i] :] = 1
            torch.manual_seed(seed + chunk[0])
            stop = LogitsProcessorList([stop_cls(nl_id, 2 * k, tok.eos_token_id, L)]) if stop_cls else None
            out = model.generate(
                inp,
                attention_mask=attn,
                logits_processor=stop,
                do_sample=True,
                temperature=1.0,
                top_k=20,
                top_p=0.95,
                max_new_tokens=new_tokens,
            )
            dec = tok.batch_decode(out[:, L:], skip_special_tokens=True)
            for j, i in enumerate(chunk):
                texts[i] = dec[j * n : (j + 1) * n]
            done += len(chunk)
            print(f"  sampled k={k}: {done}/{len(idx)} rows", flush=True)
    return texts
