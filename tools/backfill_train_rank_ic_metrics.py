#!/usr/bin/env python3
"""Backfill train/rank_ic_mean into metrics.csv from colocated *_steps_pool.json files."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphagen.data.expression import Feature, Ref
from alphagen_generic.features import FeatureType
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import StockData, initialize_qlib
from alphagen_qlib.utils import load_alpha_pool_by_path


def build_target():
    close = Feature(FeatureType.CLOSE)
    return Ref(close, -20) / close - 1


def pool_path_for_timestep(metrics_dir: Path, timestep: int) -> Path | None:
    exact = metrics_dir / f"{timestep}_steps_pool.json"
    if exact.exists():
        return exact
    candidates = sorted(
        metrics_dir.glob("*_steps_pool.json"),
        key=lambda p: int(re.search(r"(\d+)_steps_pool", p.name).group(1)),  # type: ignore[union-attr]
    )
    best = None
    best_ts = -1
    for path in candidates:
        ts = int(re.search(r"(\d+)_steps_pool", path.name).group(1))  # type: ignore[union-attr]
        if ts <= timestep and ts > best_ts:
            best_ts = ts
            best = path
    return best


def evaluate_train_rank_ic(
    pool_path: Path,
    train_calc: QLibStockDataCalculator,
    capacity: int = 10,
) -> tuple[float, float]:
    import torch

    exprs, weights = load_alpha_pool_by_path(str(pool_path))
    if not exprs:
        return 0.0, 0.0
    pool = MseAlphaPool(capacity, train_calc, device=torch.device("cpu"))
    pool.size = len(exprs)
    for i, expr in enumerate(exprs):
        pool.exprs[i] = expr
    if pool.size:
        pool._weights[: pool.size] = np.asarray(weights[: pool.size], dtype=float)
    return pool.test_ensemble(train_calc)


def backfill_metrics(
    metrics_path: Path,
    qlib_data_path: str,
    instruments: str,
    train_start: str,
    train_end: str,
    in_place: bool = False,
) -> Path:
    import torch

    initialize_qlib(qlib_data_path)
    device = torch.device("cpu")
    train = StockData(instruments, train_start, train_end, device=device)
    train_calc = QLibStockDataCalculator(train, build_target())

    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8-sig")))
    if not rows:
        raise RuntimeError(f"Empty metrics: {metrics_path}")

    fieldnames = list(rows[0].keys())
    if "train/rank_ic_mean" not in fieldnames:
        fieldnames.append("train/rank_ic_mean")
    if "train/ic_mean" not in fieldnames:
        fieldnames.append("train/ic_mean")

    metrics_dir = metrics_path.parent
    filled = 0
    for row in rows:
        ts = int(float(row["timestep"]))
        pool_path = pool_path_for_timestep(metrics_dir, ts)
        if pool_path is None:
            continue
        ic, ric = evaluate_train_rank_ic(pool_path, train_calc)
        row["train/ic_mean"] = f"{ic:.8f}"
        row["train/rank_ic_mean"] = f"{ric:.8f}"
        filled += 1

    out_path = metrics_path if in_place else metrics_path.with_name(metrics_path.stem + "_train_rank_ic.csv")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Filled {filled}/{len(rows)} rows -> {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("metrics", type=Path, nargs="+")
    p.add_argument("--qlib-data-path", default=str(Path.home() / ".qlib/qlib_data/cn_data"))
    p.add_argument("--instruments", default="csi300")
    p.add_argument("--train-start", default="2010-01-04")
    p.add_argument("--train-end", default="2021-12-31")
    p.add_argument("--in-place", action="store_true")
    args = p.parse_args()
    for path in args.metrics:
        backfill_metrics(
            path,
            args.qlib_data_path,
            args.instruments,
            args.train_start,
            args.train_end,
            in_place=args.in_place,
        )


if __name__ == "__main__":
    main()
