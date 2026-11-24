"""nzd's completion comparison viewer, with a GRPO column. (CPU, bricknet env)

Reuses nzd-supertry/viewer/pack.py and index.html unchanged (read only): pack.py is loaded as a module, a GRPO
column is added to its column list, and it is run on a folder of links (nzd's step 5 test and result files for
the start models, plus our step 7 forced-length file for GRPO, same eval_completion.py protocol). The page is
copied next to the data with one CSS colour added for the new column.

The output holds gated dataset content: keep it on the server and look at it through an ssh tunnel.

Usage (from ~/RL):
  python step9_grpo/viewer.py --round runs/r1 --arm grpo
  python -m http.server 8767 --bind 127.0.0.1 --directory runs/r1/viewer     # then, on the laptop:
  ssh -p 2222 -N -L 8767:127.0.0.1:8767 <server>                             # and open http://localhost:8767
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

SRC = Path(os.environ.get("SRC", "/home/nzd0420/brickICL"))  # nzd's repo root, from env.sh
NZD_RUNS = Path(os.environ.get("NZD_RUNS", SRC / "runs/nzd-supertry"))
VIEWER = SRC / "nzd-supertry/viewer"
COLOUR = {"light": "#ff9f0a", "dark": "#ffb340"}  # the page's light and dark column colours


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", type=Path, default=Path("runs/r1"))
    ap.add_argument("--arm", default="grpo")
    ap.add_argument("--start", nargs="+", default=["frozen", "plain", "masked"], help="nzd's step 5 models to show")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = (args.round / "viewer").resolve()
    links = out / "eval"
    links.mkdir(parents=True, exist_ok=True)
    for name in ["test"] + args.start:
        dst = links / f"{name}.jsonl"
        dst.unlink(missing_ok=True)
        dst.symlink_to(NZD_RUNS / "eval_v2" / f"{name}.jsonl")
    dst = links / f"{args.arm}.jsonl"
    dst.unlink(missing_ok=True)
    dst.symlink_to((args.round / "eval/val/forced" / f"{args.arm}.jsonl").resolve())

    spec = importlib.util.spec_from_file_location("nzd_viewer_pack", VIEWER / "pack.py")
    pack = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pack)
    pack.COLUMNS.append({"id": args.arm, "name": "GRPO", "sub": "增强组起点 + GRPO（第 9 步 checkpoint-300）",
                         "kind": "model"})
    # the page shows the first 4 columns by default: reference, the start model, GRPO, then the rest
    order = ["ref", "masked", args.arm, "frozen", "plain"]
    pack.COLUMNS.sort(key=lambda c: order.index(c["id"]) if c["id"] in order else len(order))

    page = (VIEWER / "index.html").read_text(encoding="utf-8")
    for old in ("--masked: #af52de;", "--masked: #bf5af2;"):
        assert old in page, f"index.html changed: {old!r} not found"
    page = page.replace("--masked: #af52de;", f"--masked: #af52de; --{args.arm}: {COLOUR['light']};")
    page = page.replace("--masked: #bf5af2;", f"--masked: #bf5af2; --{args.arm}: {COLOUR['dark']};")
    (out / "index.html").write_text(page, encoding="utf-8")

    sys.argv = [str(VIEWER / "pack.py"), str(links), "--out", str(out / "data"),
                "--models", *args.start, args.arm, "--title", "GRPO · val 补全对比"]
    if args.limit:
        sys.argv += ["--limit", str(args.limit)]
    pack.main()


if __name__ == "__main__":
    main()
