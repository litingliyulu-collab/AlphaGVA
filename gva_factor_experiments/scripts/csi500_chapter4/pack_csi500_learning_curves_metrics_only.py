#!/usr/bin/env python3
"""Pack only metrics.csv for CSI500 RankIC learning curves (avoids 8GB pool.json sync)."""
from pathlib import Path
import shutil
import tarfile

PACK = Path("/tmp/csi500_lc_metrics_only")
TAR = Path("/tmp/csi500_lc_metrics_only.tar.gz")
if PACK.exists():
    shutil.rmtree(PACK)
PACK.mkdir(parents=True, exist_ok=True)

COPY_PLAN = [
    (
        "/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_main_compare_20260630_220231",
        ["full_gva25_s0", "full_gva25_s1", "full_gva25_s2", "ppo_s0", "ppo_s1", "ppo_s2"],
    ),
    (
        "/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_filter_xgb_20260701_003217",
        ["ppo_filter_weak_s0", "ppo_filter_weak_s1", "ppo_filter_weak_s2"],
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


def copy_metrics(src_run: Path, dst_run: Path) -> None:
    metrics = next(src_run.glob("results/*/metrics.csv"), None)
    if metrics is None:
        print("MISS metrics", src_run)
        return
    rel = metrics.relative_to(src_run)
    out = dst_run / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metrics, out)
    print("ok", dst_run.name, rel)


for src_root, names in COPY_PLAN:
    src_root = Path(src_root)
    for name in names:
        copy_metrics(src_root / name, PACK / name)

if TAR.exists():
    TAR.unlink()
with tarfile.open(TAR, "w:gz") as tf:
    tf.add(PACK, arcname=PACK.name)
print("TAR", TAR, "size_mb", round(TAR.stat().st_size / 1024 / 1024, 2))
