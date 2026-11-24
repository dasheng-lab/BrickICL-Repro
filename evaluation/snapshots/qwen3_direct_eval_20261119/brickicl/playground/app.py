"""Run with python -m brickicl.playground.app --port 7860."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import gradio as gr
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

from .backend import (ENGINE, OUTPUT, REPO, registry, resolve_spec, sample_library,
                      read_upload, row_text, validate_prefix, truncate, render)

from .evaluated_models import evaluated_registry, evaluation_summary

EMPTY = '<div class="empty-view"><b>BRICK / ICL</b><br>选择样本或生成模型，开始探索搭建结构。</div>'
CSS = """
.gradio-container {max-width:1500px!important;margin:auto!important;font-family:Inter,'PingFang SC',system-ui,sans-serif!important}
#hero {padding:24px 4px 16px;border-bottom:1px solid #dce5eb;margin-bottom:14px}
#hero h1 {font-size:32px;letter-spacing:-1px;margin:0 0 8px;color:#15354c}
#hero p {color:#668093;margin:0;line-height:1.7}
#hero .eyebrow {font:600 11px system-ui;letter-spacing:3px;color:#0d8791;margin-bottom:8px}
.empty-view {height:475px;display:flex;flex-direction:column;align-items:center;justify-content:center;background:radial-gradient(ellipse at top,#fff,#e7eff5);border:1px solid #e1e9f0;border-radius:14px;color:#8498a7;text-align:center;line-height:2.3}
.empty-view b {font-size:24px;letter-spacing:5px;color:#bdccd5}
.viewer {padding:0!important;border:0!important} .viewer iframe{display:block}
#generate {min-height:49px;font-size:16px}
footer {display:none!important}
"""


def model_info(key, custom, initialization):
    try:
        spec = resolve_spec(key, custom, initialization)
        note = {"PT": "PT 以搭建序列训练；自然语言输入可用于探索，但不保证遵循描述。",
                "SFT": "SFT 支持自然语言描述。建议使用英文，与原训练数据一致。",
                "BASE": "原始 Qwen3 没有 BrickNet 结构训练，可能输出无法解析的文本。"}[spec.mode]
        if spec.objective == "M-only":
            note += " 此后训练阶段使用序列补全任务，未使用自然语言描述。"
        elif spec.objective == "Full-G":
            note += " 此后训练阶段预测完整 G。" + ("SFT 只给文字 prompt。" if spec.mode == "SFT" else "PT 从节点编号 a 开始，不给积木。")
        return f"**{spec.label}**  \n{note}\n\n加载顺序：`" + " → ".join((spec.base,) + spec.adapters) + "`"
    except Exception as exc:
        return f"模型配置错误：{exc}"


def preview(text, title="输入前缀", count=0):
    try:
        text, n = validate_prefix(text)
        view, _, info = render(text, count, title)
        return view, f"{n} 个零件 · 前缀语法有效" + (f" · 缺少 {len(info['missing_meshes'])} 个网格" if info['missing_meshes'] else "")
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def select_sample(key):
    sample = sample_library()[key]
    view, note = preview(sample["prefix"])
    return (sample["prefix"], sample["full"], gr.Slider(value=sample["count"], maximum=sample["total"]),
            view, note, "补全搭建序列", "")


def cut_sample(full, count):
    try:
        text = truncate(full, count)
        view, note = preview(text)
        return text, view, note
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def upload_file(path):
    try:
        if not path:
            return [], gr.Dropdown(choices=[], value=None)
        rows = read_upload(path)
        return rows, gr.Dropdown(choices=[(str(r.get("id", i)), str(i)) for i, r in enumerate(rows)], value="0")
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def select_uploaded(rows, index):
    try:
        if not rows or index is None:
            return (gr.skip(),) * 7
        row = rows[int(index)]
        full, prefix = row_text(row)
        _, total = validate_prefix(full)
        _, n = validate_prefix(prefix)
        view, note = preview(prefix)
        if row.get('import_info'):
            note += ' · 已导入全部零件；可用滑块主动截断，未自动删除一半模型。'
        return prefix, full, gr.Slider(value=n, maximum=total), view, note, "补全搭建序列", row.get("caption", "")
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def show_result(records, index):
    if not records or index is None:
        return EMPTY, "", "", {}, []
    record = records[int(index)]
    return record["view"], record["text"], record["completion"], record["metadata"], record["files"]


def generate(key, custom, initialization, mode, caption, prefix, max_parts, max_tokens,
             temperature, top_p, top_k, seed, samples, history, progress=gr.Progress()):
    try:
        caption, prefix = caption or "", prefix or ""
        if not 1 <= int(samples) <= 4:
            raise ValueError('每次请求可生成1–4个结果。')
        spec = resolve_spec(key, custom, initialization)
        # Validate before loading a GPU model.
        if mode == "补全搭建序列":
            validate_prefix(prefix)
        if mode == "自然语言生成" and not caption.strip():
            raise ValueError("请先输入自然语言描述。")
        progress(0, desc="正在加载 " + spec.label + "（切换模型需要一点时间）")
        records = list(history or [])
        for i in range(int(samples)):
            progress((i + .1) / int(samples), desc=f"生成 {i+1}/{int(samples)} · {spec.label}")
            record = ENGINE.generate(spec, mode, caption, prefix, max_parts, max_tokens,
                                     temperature, top_p, top_k, int(seed) + i)
            progress((i + .85) / int(samples), desc="正在构建 3D 零件与导出文件")
            view, files, parsed = render(record["text"], record["n_prefix"], spec.label)
            metadata = {**record, **parsed}
            # Persist raw generation even if there are zero parsable parts.
            if files:
                folder = Path(files[0]).parent
            else:
                import uuid
                folder = OUTPUT / uuid.uuid4().hex
                folder.mkdir(parents=True)
                (folder / "path.txt").write_text(record["text"])
                files = [str(folder / "path.txt")]
            saved = folder / "generation.json"
            saved.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
            files.append(str(saved))
            records.append({**record, "view": view, "files": files, "metadata": metadata})
        records = records[-40:]
        choices = [(f"{i+1} · {r['model']} · seed {r['seed']} · {r['metadata']['parts']} 件", str(i)) for i, r in enumerate(records)]
        last = records[-1]
        issues = []
        if last["metadata"]["parse_error"]:
            issues.append("输出存在解析错误；3D 仅展示有效部分，详情见生成记录")
        if last["metadata"]["missing_meshes"]:
            issues.append("部分零件网格缺失")
        if last["hit_token_cap"]:
            issues.append("已达到 token 上限，输出可能尚未完成")
        if last.get('hit_part_cap'):
            issues.append('已达到新增零件上限')
        summary = f"完成 · {last['seconds']} 秒 · {last['new_tokens']} tokens · {last['metadata']['parts']} 件"
        if issues:
            summary += "  \n⚠ " + "；".join(issues)
        return records, gr.Dropdown(choices=choices, value=str(len(records)-1)), *show_result(records, str(len(records)-1)), summary
    except Exception as exc:
        logging.exception("Generation failed")
        raise gr.Error(str(exc)) from exc
    finally:
        if os.environ.get('BRICKICL_AUTO_UNLOAD','1') == '1':
            ENGINE.unload()


def build_ui():
    specs = evaluated_registry()
    choices = [(s.label, k) for k, s in specs.items()]
    default = "direct-M_only-checkpoint-000750" if "direct-M_only-checkpoint-000750" in specs else next(iter(specs), None)
    library = sample_library()
    default_sample_key = next((k for k,v in library.items() if k.startswith("统一评测 · 不给第一块:") and "val:114:stretched" in v["label"]), next(iter(library), None))
    default_sample = library[default_sample_key] if default_sample_key else {"full":"", "prefix":"", "count":1, "total":100}
    theme = gr.themes.Soft(primary_hue="teal", secondary_hue="slate", neutral_hue="slate", radius_size="lg")
    with gr.Blocks(title="BrickICL · Model Playground") as demo:
        full = gr.State(default_sample["full"])
        uploaded = gr.State([])
        history = gr.State([])
        gr.HTML('<div id="hero"><div class="eyebrow">STRUCTURE LEARNING LAB</div><h1>BrickICL Playground</h1><p>从一句描述，到一块积木。选择模型，探索生成与结构补全。</p></div>')
        gr.Markdown('[Qwen3 直接 PT · 六模型 3D 对照](http://127.0.0.1:18765/qwen3_direct_compare.html?source=114&prime=1)')
        with gr.Accordion("新版评测：给第一块／不给第一块（0.6B）", open=False):
            gr.Markdown(evaluation_summary())
        with gr.Row():
            with gr.Column(scale=4, min_width=340):
                model = gr.Dropdown(choices, value=default, label="01 / 选择模型", filterable=True)
                show_all = gr.Checkbox(value=False, label='显示全部 checkpoint／其他尺寸')
                gr.Markdown('“Qwen3 直接 PT”从原始 Qwen3 训练；“结构修改后训练”从 BrickNet 初始化。默认展示各组开发集最佳且已完成评测的模型。')
                refresh = gr.Button('刷新模型列表', size='sm')
                with gr.Accordion("自定义后训练权重", open=False):
                    custom = gr.Textbox(value="", label="远端 LoRA checkpoint 目录", placeholder="留空使用上方模型；填写后将覆盖上方选择")
                    initialization = gr.Radio([("BrickNet PT", "PT"), ("BrickNet PT + SFT", "SFT"), ("仅 Qwen3", "BASE")], value="PT", label="训练时的初始化")
                note = gr.Markdown(model_info(default, "", "PT") if default else "没有找到已缓存的模型。")
                mode = gr.Radio(["自然语言生成", "补全搭建序列", "无条件生成"], value="补全搭建序列", label="02 / 生成方式")
                caption = gr.Textbox(label="自然语言描述（补全时可留空）", lines=3,
                    value="", placeholder="A small red car with four black wheels.")
                with gr.Group(visible=True) as sequence_controls:
                    with gr.Accordion("选择 / 上传搭建序列", open=True):
                        sample = gr.Dropdown([(v["label"], k) for k, v in library.items()], value=default_sample_key, label="统一测试案例 · 可搜索", filterable=True)
                        upload = gr.File(label="上传装配模型或路径：LDR / MPD / TXT / JSON / JSONL", file_types=[".ldr", ".mpd", ".txt", ".json", ".jsonl"], type="filepath")
                        uploaded_index = gr.Dropdown([], label="上传文件中的样本")
                        keep = gr.Slider(1, default_sample["total"], value=default_sample["count"], step=1, label="保留前 N 个零件", info="选择样本后拖动：删除末尾零件，并保留合法连接。")
                    prefix = gr.Textbox(value=default_sample["prefix"], label="不完整搭建序列 · 可直接编辑", lines=9, max_lines=16,
                        placeholder="a brick 2x4 | red\nb brick 2x4 | blue\na stud stud a hole a 0", elem_classes="path-input")
                    with gr.Row():
                        preview_btn = gr.Button("预览输入")
                        reference_btn = gr.Button("查看样本完整参考")
                    input_status = gr.Markdown("已载入统一测试案例。现场生成使用当前采样参数，可能不同于离线评测样本。")
                with gr.Accordion("03 / 采样参数", open=False):
                    with gr.Row():
                        max_parts = gr.Slider(1, 100, value=30, step=1, label="最多新增零件")
                        max_tokens = gr.Slider(64, 8192, value=2048, step=64, label="最多新增 tokens")
                    with gr.Row():
                        temperature = gr.Slider(0, 2, value=1, step=.05, label="Temperature（0 = greedy）")
                        top_p = gr.Slider(.05, 1, value=.95, step=.01, label="Top-p")
                    with gr.Row():
                        top_k = gr.Slider(0, 100, value=20, step=1, label="Top-k（0 = 不限制）")
                        seed = gr.Number(value=42, precision=0, label="随机种子")
                        samples = gr.Slider(1, 4, value=1, step=1, label="生成数量")
                run = gr.Button("生成乐高模型 →", variant="primary", elem_id="generate")
                unload = gr.Button("释放 GPU 模型", size="sm")
            with gr.Column(scale=6, min_width=420):
                with gr.Tabs():
                    with gr.Tab("生成结果"):
                        result_select = gr.Dropdown([], label="本次会话结果 · 可切换比较不同模型", interactive=True)
                        result_view = gr.HTML(EMPTY, elem_classes="viewer")
                        status = gr.Markdown("准备就绪。首次加载或切换模型需等待数秒。")
                    with gr.Tab("输入 / 参考预览"):
                        input_view = gr.HTML(EMPTY, elem_classes="viewer")
                        gr.Markdown("预览支持旋转、缩放和逐步搭建。参考答案仅用于查看，不会自动加入模型输入。")
                with gr.Tabs():
                    with gr.Tab("完整 path text"):
                        result_text = gr.Textbox(lines=12, label="输入前缀 + 生成序列", interactive=False, buttons=["copy"])
                    with gr.Tab("仅新增序列"):
                        completion = gr.Textbox(lines=12, label="模型的原始续写", interactive=False, buttons=["copy"])
                    with gr.Tab("生成记录"):
                        metadata = gr.JSON(label="权重、参数、耗时、EOS 与解析信息")
                downloads = gr.File(label="下载结果 · TXT / LDraw / 自包含 MPD / JSON", file_count="multiple", interactive=False)
                gr.Markdown("🟢 **新增高亮**区分补全部分与输入。3D 显示成功解析的几何结构，不代表模型一定满足描述或无碰撞。")
        for component in (model, custom, initialization):
            component.change(model_info, [model, custom, initialization], note, queue=False)
        def refresh_models(current, show_all=False):
            available=evaluated_registry(show_all)
            value=current if current in available else next(iter(available),None)
            return gr.Dropdown(choices=[(s.label,k) for k,s in available.items()],value=value)
        refresh.click(refresh_models, [model, show_all], model, queue=False, api_name='refresh_models')
        show_all.change(refresh_models, [model, show_all], model, queue=False, api_name='filter_models')
        mode.change(lambda value: gr.Group(visible=value == "补全搭建序列"), mode, sequence_controls, queue=False)
        mode.input(lambda value: "" if value == "补全搭建序列" else gr.skip(), mode, caption, queue=False)
        sample.change(select_sample, sample, [prefix, full, keep, input_view, input_status, mode, caption])
        keep.input(cut_sample, [full, keep], [prefix, input_view, input_status], trigger_mode="always_last")
        upload.change(upload_file, upload, [uploaded, uploaded_index]).then(select_uploaded, [uploaded, uploaded_index], [prefix, full, keep, input_view, input_status, mode, caption])
        uploaded_index.input(select_uploaded, [uploaded, uploaded_index], [prefix, full, keep, input_view, input_status, mode, caption])
        preview_btn.click(preview, prefix, [input_view, input_status], api_name="preview")
        reference_btn.click(lambda text: preview(text, "完整参考（未发送给模型）"), full, [input_view, input_status])
        outputs = [history, result_select, result_view, result_text, completion, metadata, downloads, status]
        run.click(generate, [model, custom, initialization, mode, caption, prefix, max_parts, max_tokens, temperature, top_p, top_k, seed, samples, history], outputs,
                  concurrency_limit=1, concurrency_id="gpu", api_name="generate")
        result_select.input(show_result, [history, result_select], [result_view, result_text, completion, metadata, downloads])
        unload.click(lambda: (ENGINE.unload(), "GPU 模型已释放。下次生成时自动加载。")[1], outputs=status, concurrency_id="gpu", concurrency_limit=1)
    demo.queue(max_size=16, default_concurrency_limit=1)
    return demo, theme


def create_app():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="BrickICL Playground")
    app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)
    app.mount("/playground-static", StaticFiles(directory=Path(__file__).parent / "static"), name="playground-static")
    app.mount("/playground-output", StaticFiles(directory=OUTPUT), name="playground-output")
    import os
    ldraw = Path(os.environ["BRICKNET_DATA"]) / "ldraw"
    app.mount("/playground-ldraw", StaticFiles(directory=ldraw), name="playground-ldraw")

    @app.get("/health")
    def health():
        return {"status": "ok", "service_id": os.environ.get('BRICKICL_SERVICE_ID','brickicl-playground'),
                "models": len(registry()), "samples": len(sample_library()),
                "auto_unload":os.environ.get('BRICKICL_AUTO_UNLOAD','1')=='1',
                "gpu_memory_fraction":float(os.environ.get('BRICKICL_GPU_MEMORY_FRACTION','.25'))}

    demo, theme = build_ui()
    return gr.mount_gradio_app(app, demo, path="/", allowed_paths=[str(OUTPUT)], theme=theme, css=CSS,
                               max_file_size="5mb", pwa=True)


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7860, type=int)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, timeout_graceful_shutdown=5)


if __name__ == "__main__":
    main()
