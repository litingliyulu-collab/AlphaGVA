import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    xr = pd.Series(x[mask]).rank(method="average").to_numpy()
    yr = pd.Series(y[mask]).rank(method="average").to_numpy()
    return float(np.corrcoef(xr, yr)[0, 1])


def pearson_ic(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def topk_backtest(pred: np.ndarray, label: np.ndarray, bench_ret: np.ndarray, dates, topk: int, cost: float):
    rows = []
    prev_sel = None
    cum = 1.0
    bench_cum = 1.0
    excess_cum = 1.0
    daily_excess = []
    daily_ret = []
    turnovers = []

    for i, dt in enumerate(dates):
        p = pred[i]
        r = label[i]
        mask = np.isfinite(p) & np.isfinite(r)
        if mask.sum() < topk:
            continue
        valid_idx = np.where(mask)[0]
        order = valid_idx[np.argsort(p[mask])[::-1]]
        sel = order[:topk]

        gross = float(np.nanmean(r[sel]))
        if prev_sel is None:
            turnover = 1.0
        else:
            turnover = 1.0 - len(set(sel.tolist()) & prev_sel) / float(topk)
        net = gross - cost * turnover
        bench = float(bench_ret[i])
        excess = net - bench

        cum *= 1.0 + net
        bench_cum *= 1.0 + bench
        excess_cum *= 1.0 + excess

        daily_ret.append(net)
        daily_excess.append(excess)
        turnovers.append(turnover)
        prev_sel = set(sel.tolist())
        rows.append(
            {
                "date": str(dt)[:10],
                "return": net,
                "benchmark_return": bench,
                "excess_return": excess,
                "cum_return": cum - 1.0,
                "bench_cum_return": bench_cum - 1.0,
                "excess_cum_return": excess_cum - 1.0,
                "turnover": turnover,
                "ic": pearson_ic(p, r),
                "rank_ic": rank_ic(p, r),
            }
        )

    daily_ret = np.asarray(daily_ret, dtype=float)
    daily_excess = np.asarray(daily_excess, dtype=float)
    cum_curve = np.asarray([r["cum_return"] for r in rows], dtype=float)
    if len(daily_excess) > 1 and np.nanstd(daily_excess) > 0:
        ir = float(np.nanmean(daily_excess) / np.nanstd(daily_excess) * np.sqrt(252))
    else:
        ir = np.nan
    if len(cum_curve):
        wealth = 1.0 + cum_curve
        peak = np.maximum.accumulate(wealth)
        max_dd = float(np.nanmin(wealth / peak - 1.0))
    else:
        max_dd = np.nan

    metrics = {
        "days": len(rows),
        "stocks": pred.shape[1],
        "test/ic_mean": float(np.nanmean([r["ic"] for r in rows])),
        "test/rank_ic_mean": float(np.nanmean([r["rank_ic"] for r in rows])),
        "cum_return": float(cum - 1.0),
        "bench_cum_return": float(bench_cum - 1.0),
        "excess_cum_return": float(excess_cum - 1.0),
        "annual_return": float((cum ** (252.0 / max(len(rows), 1))) - 1.0),
        "information_ratio": ir,
        "max_drawdown": max_dd,
        "avg_turnover": float(np.nanmean(turnovers)) if turnovers else np.nan,
    }
    return pd.DataFrame(rows), metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--save-name", default="affofficial")
    parser.add_argument("--instruments", default="csi300")
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-factors", type=int, default=10)
    parser.add_argument("--window", default="inf")
    parser.add_argument("--method", default="AFF-official")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--cost", type=float, default=0.0015)
    args = parser.parse_args()

    from gan.utils.data import get_data_by_year

    out_dir = (
        Path(args.workspace)
        / "out"
        / f"{args.save_name}_{args.instruments}_{args.train_end_year}_{args.seed}"
    )
    pred_path = out_dir / f"pred_{args.train_end_year}_{args.n_factors}_{args.window}_{args.seed}.pt"
    pred = torch.load(pred_path, map_location="cpu").detach().cpu().numpy()

    returned = get_data_by_year(
        train_start=2011,
        train_end=args.train_end_year,
        valid_year=args.train_end_year + 1,
        test_year=args.train_end_year + 2,
        instruments=args.instruments,
        target="normal",
        freq="day",
    )
    data_all, data_train, data_valid, data_valid_withhead, data_test, data_test_withhead, _ = returned
    close_idx = list(data_test._features).index(1) if 1 in list(data_test._features) else 1
    close = data_test.data[:, close_idx, :].detach().cpu().numpy()
    bt = data_test.max_backtrack_days
    n_days = data_test.n_days
    label = close[bt + 1 : bt + n_days + 1] / close[bt : bt + n_days] - 1.0
    dates = pd.to_datetime(data_test._dates[bt : bt + n_days])

    if pred.shape != label.shape:
        raise RuntimeError(f"pred shape {pred.shape} != label shape {label.shape}")

    # Equal-weight CSI300 universe benchmark using the same next-period label matrix.
    bench = np.nanmean(label, axis=1)
    daily, metrics = topk_backtest(pred, label, bench, dates, args.topk, args.cost)
    metrics = {
        "method": args.method,
        "seed": args.seed,
        "test_start": str(dates[0])[:10],
        "test_end": str(dates[-1])[:10],
        **metrics,
    }

    np.savez_compressed(out_dir / "pred_test.npz", pred=pred, label=label, dates=dates.astype(str).to_numpy())
    daily.to_csv(out_dir / "aff_official_daily_report.csv", index=False)
    with open(out_dir / "aff_official_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(out_dir)


if __name__ == "__main__":
    main()



