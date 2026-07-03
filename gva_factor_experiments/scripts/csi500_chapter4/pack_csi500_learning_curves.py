#!/usr/bin/env python3
"""Pack CSI500 learning-curve metrics dirs for local RankIC plot sync."""
from pathlib import Path
import shutil

PACK = Path("/tmp/csi500_lc_pack")
PACK.mkdir(parents=True, exist_ok=True)

COPY_PLAN = [
    (
        "/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_main_compare_20260630_220231",
        ["full_gva25_s0", "full_gva25_s1", "full_gva25_s2", "ppo_s0", "ppo_s1", "ppo_s2"],
    ),
    (
        "/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_ppo_history_filter_20260701_004456",
        ["ppo_filter_s0", "ppo_filter_s1", "ppo_filter_s2"],
    ),
    (
        "/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_rl_advanced_20260630_220231",
        ["a2c_baseline_s0", "a2c_baseline_s1", "a2c_baseline_s2"],
    ),
    (
        "/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_reinforce_qfr_20260630_220932",
        ["qfr_s0"],
    ),
    (
        "/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_reinforce_qfr_resume_20260701_003217",
        ["qfr_s1", "qfr_s2"],
    ),
]

for src_root, names in COPY_PLAN:
    src_root = Path(src_root)
    for name in names:
        src = src_root / name
        if not src.exists():
            print("MISS", src)
            continue
        dst = PACK / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print("copied", dst)

# PPO-filter learning curve expects ppo_filter_weak_s* glob; add symlinks.
for seed in range(3):
    src = PACK / f"ppo_filter_s{seed}"
    dst = PACK / f"ppo_filter_weak_s{seed}"
    if src.exists() and not dst.exists():
        dst.symlink_to(src.resolve())
        print("link", dst, "->", src)

print("PACK", PACK)
