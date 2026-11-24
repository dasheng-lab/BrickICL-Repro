"""User requested loss artifacts in chat/cluster only, not in the Viewer UI."""
import shutil
from pathlib import Path
from common import ROOT,atomic

viewer=Path('/home/sunshq114514/brickicl-viewer/baseline/repro/viewer').resolve()
source=(viewer/'training_qwen3_direct').resolve()
destination=ROOT/'archived_web_loss'
assert source.parent==viewer and destination.is_relative_to(ROOT)
if source.exists():
    assert not destination.exists()
    shutil.move(str(source),str(destination))
atomic(ROOT/'LOSS_WEB_WITHDRAWN.json',{'reason':'User requested loss images directly in chat; viewer only shows rendered outputs.',
       'archived_to':str(destination),'artifacts':str(ROOT/'delivery')})
print('Loss webpage archived outside the viewer; cluster artifacts retained.')
