"""Portable locations and provenance-derived labels; no filesystem writes on import."""
import json,os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARMS = ('frozen_pt','post_pt','fullg_pt','frozen_sft','post_sft','fullg_sft')

def experiment_root():
    return Path(os.environ.get('BRICKICL_POSTTRAIN_ROOT', str(REPO))).expanduser().resolve()

def best_checkpoint(mode, objective):
    folder='bricknet_06_fullg' if objective=='Full-G' else 'bricknet_06_reference_last'
    run=experiment_root()/'runs'/folder/mode
    path=Path(json.loads((run/'best.json').read_text())['path']).expanduser()
    return path.resolve() if path.is_absolute() else (run/path).resolve()

def expected_adapters(arm):
    mode='PT' if arm.endswith('_pt') else 'SFT'
    values=['kulits/BrickNet-0.6B-PT']
    if mode=='SFT':values.append('kulits/BrickNet-0.6B-SFT')
    if arm.startswith('post_'):values.append(str(best_checkpoint(mode,'M-only')))
    if arm.startswith('fullg_'):values.append(str(best_checkpoint(mode,'Full-G')))
    return values

def model_label(arm, adapters):
    mode='PT' if arm.endswith('_pt') else 'SFT'
    if arm.startswith('frozen_'):return '官方 '+mode
    objective='Full-G' if arm.startswith('fullg_') else 'M-only'
    checkpoint=Path(adapters[-1]).name.removeprefix('checkpoint-')
    step=str(int(checkpoint)) if checkpoint.isdigit() else checkpoint
    return f'{objective} {mode} · {step}'
