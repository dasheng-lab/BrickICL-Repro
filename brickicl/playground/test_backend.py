"""CPU checks: python -m unittest brickicl.playground.test_backend -v."""
import json
import tempfile
import unittest
from pathlib import Path

from .backend import registry, row_text, truncate, validate_prefix, read_upload, render, encode_prompt, resolve_spec

PATH = "a brick 2x4 | red\nb brick 2x4 | blue\na stud stud a hole a 0\n"


class PlaygroundTests(unittest.TestCase):
    def test_plain_upload_keeps_all_given_bricks(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'input.txt';p.write_text(PATH)
            full,prefix=row_text(read_upload(p)[0])
            self.assertEqual(full,PATH);self.assertEqual(prefix,PATH)

    def test_ldr_and_embedded_mpd_import(self):
        from bricknet import parse_sample,tree_to_graph,graph_to_ldr
        from .ldraw_input import ldraw_to_path
        ldr=graph_to_ldr(tree_to_graph(parse_sample(PATH).tree))
        converted,info=ldraw_to_path(ldr)
        self.assertEqual(validate_prefix(converted)[1],2)
        self.assertEqual(info['imported_parts'],2)
        mpd='0 FILE main.ldr\n1 16 40 0 0 1 0 0 0 1 0 0 0 1 child.ldr\n0 FILE child.ldr\n'+ldr+'0 NOFILE\n'
        converted,info=ldraw_to_path(mpd)
        self.assertEqual(validate_prefix(converted)[1],2)

    def test_mpd_cycle_is_rejected(self):
        from .ldraw_input import ldraw_to_path
        with self.assertRaisesRegex(ValueError,'循环'):
            ldraw_to_path('0 FILE loop.ldr\n1 16 0 0 0 1 0 0 0 1 0 0 0 1 loop.ldr\n')

    def test_disconnected_ldr_is_not_silently_truncated(self):
        from .ldraw_input import ldraw_to_path
        with self.assertRaisesRegex(ValueError,'未连接'):
            ldraw_to_path('1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat\n1 1 1000 0 0 1 0 0 0 1 0 0 0 1 3001.dat\n')

    def test_fullg_json_caption_is_not_parsed_as_path(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'fullg.json';p.write_text(json.dumps({'path':PATH,'prompt':'A red and blue LEGO model.','prompt_kind':'author_caption'}))
            row=read_upload(p)[0]
            self.assertEqual(row['caption'],'A red and blue LEGO model.')
            self.assertEqual(row_text(row),(PATH,PATH))

    def test_unset_browser_fields(self):
        self.assertEqual(resolve_spec("0.6B-SFT", None).key, "0.6B-SFT")
        with self.assertRaises(ValueError):
            validate_prefix(None)

    def test_caption_separator_matches_upstream_tokenization(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
        caption = "A small red car with four black wheels."
        ids, _ = encode_prompt(tok, "自然语言生成", caption, "")
        self.assertEqual(ids, tok.encode(caption, add_special_tokens=False) + tok.encode("\n", add_special_tokens=False))
        self.assertNotEqual(ids, tok.encode(caption + "\n", add_special_tokens=False))
        ids, _ = encode_prompt(tok, "补全搭建序列", "", PATH)
        self.assertEqual(ids, tok.encode(PATH, add_special_tokens=False))

    def test_truncate_preserves_valid_parent_references(self):
        self.assertEqual(validate_prefix(PATH)[1], 2)
        self.assertEqual(truncate(PATH, 1), "a brick 2x4 | red\n")
        self.assertEqual(validate_prefix(truncate(PATH, 2))[1], 2)

    def test_incomplete_edge_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            validate_prefix("a brick 2x4 | red\nb brick 2x4 | blue\n")

    def test_target_is_separate_from_prompt(self):
        prefix = truncate(PATH, 1)
        full, actual_prefix = row_text({"prompt": prefix, "completion": PATH[len(prefix):]})
        self.assertEqual(full, PATH)
        self.assertEqual(actual_prefix, prefix)

    def test_sft_merge_order_and_best_checkpoint_present(self):
        models = registry()
        for spec in models.values():
            if spec.mode == "SFT":
                self.assertTrue(spec.adapters[0].endswith("-PT"))
                self.assertTrue(spec.adapters[1].endswith("-SFT"))
            if spec.key.startswith("post-"):
                self.assertTrue(Path(spec.adapters[-1], "adapter_model.safetensors").is_file())
        self.assertTrue(any("dev best" in s.label for s in models.values()))

    def test_upload_multiple_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "examples.jsonl"
            p.write_text(json.dumps({"text": PATH}) + "\n" + json.dumps({"text": PATH}) + "\n")
            rows = read_upload(p)
            self.assertEqual(len(rows), 2)
            self.assertEqual(row_text(rows[1])[0], PATH)

    def test_render_reports_partial_parse_without_faking_completion(self):
        invalid = PATH + "c nonexistent part | red\n"
        view, files, info = render(invalid, 1)
        self.assertEqual(info["parts"], 2)
        self.assertTrue(info["parse_error"])
        self.assertIn("iframe", view)
        self.assertEqual(info["missing_meshes"], [])
        self.assertEqual(Path(files[0]).read_text(), invalid)


if __name__ == "__main__":
    unittest.main()
