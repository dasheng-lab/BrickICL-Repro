"""Create ignored compatibility links for the preserved GRPO v2.1 source."""
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
source = repo / "experiments" / "assembly_order"
alias = repo / "nzd-supertry"
mapping = {
    "step1_checks": "validation",
    "step2_sampler": "sampling",
    "step3_data": "data_construction",
    "step4_train": "training",
    "step5_eval": "completion_evaluation",
    "step6_render_score": "rendering",
    "step8_shared_test": "robustness_evaluation",
    "step11_report_metrics": "metrics",
    "viewer": "viewer",
}

assert source.is_dir() and source.resolve().is_relative_to(repo.resolve())
if alias.is_symlink() or (alias.exists() and not alias.is_dir()):
    raise RuntimeError(f"Unexpected historical alias path: {alias}")
alias.mkdir(exist_ok=True)
for old, new in mapping.items():
    target = source / new
    link = alias / old
    if not target.is_dir():
        raise FileNotFoundError(target)
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"Historical alias points elsewhere: {link}")
    elif link.exists():
        raise RuntimeError(f"Historical alias conflicts with existing directory: {link}")
    else:
        link.symlink_to(target, target_is_directory=True)
print("Historical GRPO dependency links ready:", alias)
