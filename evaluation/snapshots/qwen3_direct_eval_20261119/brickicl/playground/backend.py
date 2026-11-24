"""Model registry, generation and self-contained LDraw previews.

No model is loaded at import time. All inference uses the existing HF cache.
"""
from __future__ import annotations

import gc
import html
import importlib.util
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
POSTTRAIN = Path(os.environ.get("BRICKICL_POSTTRAIN_ROOT", "/home/sunshq114514/brickICL"))
OUTPUT = Path(os.environ.get("BRICKICL_PLAYGROUND_OUTPUT", REPO / "runs/playground"))
os.environ.setdefault("BRICKNET_DATA", str(REPO / "baseline/bricknet/data"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    base: str
    adapters: tuple[str, ...]
    mode: str
    objective: str = "official"


def registry():
    cache = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"))
    specs = {}
    for size in ("0.6B", "1.7B", "4B", "8B"):
        base = f"Qwen/Qwen3-{size}"
        if not (cache / ("models--" + base.replace("/", "--")) / "snapshots").is_dir():
            continue
        for mode in ("PT", "SFT"):
            adapters = (f"kulits/BrickNet-{size}-PT",)
            if mode == "SFT":
                adapters += (f"kulits/BrickNet-{size}-SFT",)
            if all((cache / ("models--" + a.replace("/", "--")) / "snapshots").is_dir() for a in adapters):
                key = f"{size}-{mode}"
                specs[key] = ModelSpec(key, f"BrickNet {size} · {mode} 原版", base, adapters, mode)
        key = f"{size}-base"
        specs[key] = ModelSpec(key, f"Qwen3 {size} · 原始基座", base, (), "BASE")
    for objective,suffix in (("M-only","M"),("Full-G","G")):
        for mode in ("PT", "SFT"):
            root = POSTTRAIN / "runs/structural_20261114/training" / f"{mode}_{suffix}"
            best_file = root / "best.json"
            best = Path(json.loads(best_file.read_text())["path"]).name if best_file.exists() else ""
            for cp in sorted(root.glob("checkpoint-*")):
                if not (cp / "adapter_model.safetensors").exists() or not (cp / "provenance.json").exists():
                    continue
                config = json.loads((cp / "adapter_config.json").read_text())
                base = config["base_model_name_or_path"]
                initial = ("kulits/BrickNet-0.6B-PT",)
                if mode == "SFT":
                    initial += ("kulits/BrickNet-0.6B-SFT",)
                key = f"post-{mode}-{cp.name}" if objective=="M-only" else f"fullg-{mode}-{cp.name}"
                ongoing = not (root / "COMPLETED.json").exists()
                label = f"结构修改后训练 {objective} · 0.6B · {mode} · {cp.name.removeprefix('checkpoint-')}" + (" ★ dev best" if cp.name == best else "") + ("（训练中的阶段权重）" if ongoing else "")
                specs[key] = ModelSpec(key, label, base, initial + (str(cp),), mode, objective)
    nzd_root = Path('/home/nzd0420/brickICL/runs/nzd-supertry/train_v2')
    for arm, step, title in (('plain', 1750, '对照 PT'), ('masked', 2000, '增强 PT')):
        cp = nzd_root / arm / f'checkpoint-{step}'
        if (cp / 'adapter_config.json').is_file() and (cp / 'adapter_model.safetensors').is_file():
            key = f'nzd-{arm}-PT-{step}'
            specs[key] = ModelSpec(key, f'NZD {title} · 0.6B · step {step}',
                                   'Qwen/Qwen3-0.6B', ('kulits/BrickNet-0.6B-PT', str(cp)),
                                   'PT', 'order-augmentation' if arm == 'masked' else 'order-control')
    direct_root = POSTTRAIN / 'runs/qwen3_direct_pt_20261117'
    direct_base = direct_root / 'verified_base'
    for arm, title, objective in (('M_only', 'M-only', 'M-only'), ('Full_G', 'Full-G', 'Full-G'),
                                   ('BrickNet', 'BrickNet 数据', 'BrickNet-data')):
        run = direct_root / 'training' / arm
        for cp in sorted(run.glob('checkpoint-*')):
            if not (cp/'adapter_model.safetensors').is_file() or not (cp/'provenance.json').is_file():
                continue
            provenance = json.loads((cp/'provenance.json').read_text())
            assert provenance['initial_adapters'] == []
            assert provenance['base_provenance']['verified']
            key = f'direct-{arm}-{cp.name}'
            specs[key] = ModelSpec(key, f'Qwen3 直接 PT · {title} · step {int(cp.name.split("-")[-1])}',
                                   str(direct_base), (str(cp),), 'PT', objective)
    return specs


def resolve_spec(key, custom_path="", initialization="PT"):
    custom_path = custom_path or ""
    if not custom_path.strip():
        return registry()[key]
    path = Path(custom_path.strip()).expanduser().resolve()
    if not path.is_dir() or not (path / "adapter_config.json").exists() or not (path / "adapter_model.safetensors").exists():
        raise ValueError("自定义目录必须包含 adapter_config.json 和 adapter_model.safetensors。")
    config = json.loads((path / "adapter_config.json").read_text())
    base = config["base_model_name_or_path"]
    direct_root = POSTTRAIN / 'runs/qwen3_direct_pt_20261117'
    if path.is_relative_to(direct_root/'training') and (path/'provenance.json').is_file():
        provenance = json.loads((path/'provenance.json').read_text())
        assert provenance['initial_adapters'] == [] and provenance['base_provenance']['verified']
        objective = {'M_only':'M-only','Full_G':'Full-G','BrickNet':'BrickNet-data'}[path.parent.name]
        return ModelSpec(str(path), f'Qwen3 直接 PT · {path.parent.name} · {path.name}',
                         str(direct_root/'verified_base'), (str(path),), 'PT', objective)
    if not base.startswith("Qwen/Qwen3-"):
        raise ValueError("此界面目前支持 Qwen3 基座的 BrickNet LoRA。")
    size = base.removeprefix("Qwen/Qwen3-")
    adapters = () if initialization == "BASE" else (f"kulits/BrickNet-{size}-PT",)
    if initialization == "SFT":
        adapters += (f"kulits/BrickNet-{size}-SFT",)
    return ModelSpec(str(path), f"自定义 · {path.name}", base, adapters + (str(path),), initialization)


def normalized(text):
    text = text or ""
    return text.replace("\r\n", "\n").strip() + "\n" if text.strip() else ""


def encode_prompt(tok, mode, caption, prefix):
    """Match upstream generate.py: caption and separator are separately encoded."""
    if mode == "无条件生成":
        return tok.encode("a", add_special_tokens=False), "node_identifier_a"
    caption_ids = []
    if caption.strip():
        caption_ids = tok.encode(caption.strip(), add_special_tokens=False) + tok.encode("\n", add_special_tokens=False)
    if mode == "自然语言生成":
        return caption_ids, "official_separate_newline"
    return caption_ids + tok.encode(prefix, add_special_tokens=False), "caption_separate_newline_plus_prefix" if caption_ids else "path_prefix"


def validate_prefix(text):
    from bricknet import parse_sample
    text = normalized(text)
    if not text:
        raise ValueError("请先选择一个样本，或粘贴不完整搭建序列。")
    result = parse_sample(text)
    if result.error or not result.tree.parts:
        raise ValueError(f"输入序列不合法：{result.error or '没有零件'}。请保留完整的节点/连接行。")
    return text, len(result.tree.parts)


def truncate(text, count):
    text, total = validate_prefix(text)
    count = max(1, min(int(count), total))
    return "\n".join(text.splitlines()[: 2 * count - 1]) + "\n"


def row_text(row):
    if isinstance(row.get("path"),str) and not any(k in row for k in ("target_short","completion","target")):
        text,n=validate_prefix(row['path'])
        return text, text if row.get('use_full_input') else truncate(text,max(1,n//2))
    if row.get('use_full_input') and isinstance(row.get('text'),str):
        text,_=validate_prefix(row['text'])
        return text,text
    if row.get('query_prefix'):
        prefix=row['query_prefix']
        return normalized(prefix+row.get('target_'+row.get('expected','short'),'')),normalized(prefix)
    prefix = row.get("prefix", row.get("prompt", ""))
    if "target_short" in row:
        target = row.get("target_" + row.get("expected", "short"), row["target_short"])
    else:
        target = row.get("completion", row.get("target", ""))
    if prefix:
        return normalized(prefix + target), normalized(prefix)
    text = row.get("text", row.get("path", ""))
    if not isinstance(text, str):
        raise ValueError("JSON 样本缺少字符串格式的 path/text 或 prefix/prompt。")
    text, n = validate_prefix(text)
    return text, truncate(text, max(1, n // 2))


@lru_cache(maxsize=1)
def sample_library():
    sources = [
        ("统一评测 · 不给第一块", POSTTRAIN / "runs/structural_eval_20261115/p0_inputs.jsonl"),
        ("统一评测 · 给第一块", POSTTRAIN / "runs/structural_eval_20261115/p1_inputs.jsonl"),
    ]
    samples = {}
    for name, path in sources:
        if not path.exists():
            continue
        with path.open() as stream:
            for i, line in enumerate(stream):
                if i >= 400:
                    break
                row = json.loads(line)
                if row.get('condition') == 'ablated':
                    continue
                try:
                    full, prefix = row_text(row)
                    _, count = validate_prefix(prefix)
                    _, total = validate_prefix(full)
                except (ValueError, TypeError):
                    continue
                key = f"{name}:{i}"
                label = f"{name} · {row.get('id', row.get('pair', i))} · {row.get('condition', '')} · {count}/{total} 件"
                samples[key] = {"label": label, "full": full, "prefix": prefix, "caption": row.get("caption", ""), "count": count, "total": total}
    return samples


def read_upload(path):
    path = Path(path)
    if path.stat().st_size > 5_000_000:
        raise ValueError("请上传小于 5 MB 的 TXT、JSON、JSONL、LDR 或 MPD 文件。")
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in ('.ldr','.mpd'):
        from .ldraw_input import ldraw_to_path
        converted,info=ldraw_to_path(text)
        return [{'id':path.name,'path':converted,'use_full_input':True,'import_info':info}]
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        value = json.loads(text)
        rows = value if isinstance(value, list) else [value]
    else:
        rows = [{"text": text}]
    if not rows or len(rows) > 2000:
        raise ValueError("文件需要包含 1–2000 条样本。")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("JSON 中每条样本必须是一个对象。")
        if not any(k in row for k in ('prefix','query_prefix','completion','target','target_short')):
            row['use_full_input']=True
        if row.get('prompt_kind') and isinstance(row.get('path'),str):
            row['caption']=row.get('prompt','') if row['prompt_kind']!='node_identifier_only' else ''
    return rows


_render_lock = threading.RLock()


@lru_cache(maxsize=1)
def packer():
    spec = importlib.util.spec_from_file_location("playground_pack", REPO / "baseline/repro/viewer/pack.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Keep all newly generated resources inside runs/playground.
    module.HERE = OUTPUT
    OUTPUT.mkdir(parents=True, exist_ok=True)
    module.LDRAW = Path(os.environ["BRICKNET_DATA"]) / "ldraw"
    module.write_colors()
    return module, module.build_index()


def render(text, prefix_count=0, title="模型预览"):
    from bricknet import parse_sample, tree_to_graph, graph_to_ldr
    text = normalized(text)
    result = parse_sample(text)
    error = result.error
    info = {"parts": len(result.tree.parts), "parse_error": error, "missing_meshes": []}
    if not result.tree.parts:
        return "<div class='empty-view'>没有可显示的零件。请检查下方原始输出和解析信息。</div>", [], info
    with _render_lock:
        pack, index = packer()
        ldr = graph_to_ldr(tree_to_graph(result.tree))
        mpd, missing = pack.embed([line for line in ldr.splitlines() if line.startswith("1 ")], index)
        run = OUTPUT / uuid.uuid4().hex
        run.mkdir(parents=True)
        (run / "model.mpd").write_text(mpd)
        (run / "model.ldr").write_text(ldr)
        (run / "path.txt").write_text(text)
        info["missing_meshes"] = missing
        (run / "view.json").write_text(json.dumps({**info, "prefix_count": prefix_count, "title": title}, ensure_ascii=False))
    src = f"/playground-static/viewer.html?run={run.name}"
    iframe = f'<iframe src="{src}" title="{html.escape(title, quote=True)}" style="width:100%;height:475px;border:0;border-radius:14px" allow="fullscreen"></iframe>'
    return iframe, [str(run / name) for name in ("path.txt", "model.ldr", "model.mpd")], info


class Engine:
    def __init__(self):
        self.lock = threading.RLock()
        self.model = self.tok = self.loaded = None

    def unload(self):
        import torch
        with self.lock:
            self.model = self.tok = self.loaded = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load(self, spec):
        if self.loaded == spec:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.unload()
        if not torch.cuda.is_available():
            raise ValueError("未检测到 CUDA GPU，请在 GPU 节点启动服务。")
        fraction=float(os.environ.get('BRICKICL_GPU_MEMORY_FRACTION','.25'))
        if not 0 < fraction <= 1:raise ValueError('GPU memory fraction must be in (0,1]')
        torch.cuda.set_per_process_memory_fraction(fraction)
        model = None
        try:
            model = AutoModelForCausalLM.from_pretrained(spec.base, dtype=torch.bfloat16,
                device_map={"": "cuda"}, attn_implementation="sdpa", local_files_only=True)
            for adapter in spec.adapters:
                model = PeftModel.from_pretrained(model, adapter, local_files_only=True).merge_and_unload()
            model.eval()
            model.config.use_cache = True
            tok = AutoTokenizer.from_pretrained(spec.base, local_files_only=True, padding_side="left")
            tok.pad_token = tok.eos_token
            self.model, self.tok, self.loaded = model, tok, spec
        except Exception:
            del model
            self.unload()
            raise

    def generate(self, spec, mode, caption, prefix, max_parts, max_tokens, temperature, top_p, top_k, seed):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        caption, prefix = caption or "", prefix or ""
        if not 1<=int(max_parts)<=100 or not 64<=int(max_tokens)<=8192:raise ValueError('新增零件须1–100，tokens须64–8192。')
        if not 0<=float(temperature)<=2 or not .05<=float(top_p)<=1 or not 0<=int(top_k)<=100:raise ValueError('采样参数超出界面允许范围。')
        if not 0<=int(seed)<2**32:raise ValueError('seed须为0到2^32-1的整数。')
        if mode == "补全搭建序列":
            prefix, n_prefix = validate_prefix(prefix)
            prompt = (caption.strip() + "\n" if caption.strip() else "") + prefix
            path_prefix = prefix
        elif mode == "自然语言生成":
            if not caption.strip():
                raise ValueError("请先输入自然语言描述。")
            prompt, path_prefix, n_prefix = caption.strip() + "\n", "", 0
        else:
            prompt, path_prefix, n_prefix = "a", "a", 0
        with self.lock:
            self.load(spec)
            tok, model = self.tok, self.model
            prompt_ids, tokenization = encode_prompt(tok, mode, caption, prefix)
            input_ids = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)
            batch = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
            prompt_len = len(prompt_ids)
            if prompt_len > 6400:
                raise ValueError(f"输入有 {prompt_len} tokens，超过此界面的 6400 上限。请缩短前缀。")
            # Decode the entire continuation to handle tokens containing multiple newlines.
            target_lines = int(max_parts) * 2 - (1 if n_prefix == 0 else 0)

            class PartLimit(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    text = tok.decode(input_ids[0, prompt_len:], skip_special_tokens=True)
                    return text.count("\n") >= target_lines

            torch.manual_seed(int(seed))
            started = time.monotonic()
            kwargs = dict(max_new_tokens=int(max_tokens), do_sample=temperature > 0,
                pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
                stopping_criteria=StoppingCriteriaList([PartLimit()]))
            if temperature > 0:
                kwargs.update(temperature=float(temperature), top_p=float(top_p), top_k=int(top_k))
            with torch.inference_mode():
                ids = model.generate(**batch, **kwargs)[0, prompt_len:].tolist()
            continuation = tok.decode(ids, skip_special_tokens=True)
            record = {"model": spec.label, "base": spec.base, "adapters": list(spec.adapters),
                "mode": mode, "prompt": prompt, "input_tokenization": tokenization,
                "prompt_token_ids": prompt_ids, "prefix": path_prefix, "completion": continuation,
                "text": path_prefix + continuation, "n_prefix": n_prefix, "seed": int(seed),
                "max_new_parts": int(max_parts), "max_new_tokens": int(max_tokens),
                "temperature": temperature, "top_p": top_p, "top_k": top_k,
                "new_tokens": len(ids), "seconds": round(time.monotonic() - started, 2),
                "eos_emitted": tok.eos_token_id in ids,
                "hit_token_cap": len(ids) >= max_tokens and tok.eos_token_id not in ids}
            record['hit_part_cap']=not record['eos_emitted'] and continuation.count('\n')>=target_lines
            record['stop_reason']='eos' if record['eos_emitted'] else 'part_limit' if record['hit_part_cap'] else 'token_limit' if record['hit_token_cap'] else 'other'
            record['training_objective']=spec.objective
            del batch
            return record


ENGINE = Engine()
