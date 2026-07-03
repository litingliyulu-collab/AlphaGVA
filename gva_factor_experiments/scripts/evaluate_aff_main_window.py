#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from alphagen.data.expression import *
from alphagen_generic.features import *
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr, batch_ret
from gan.utils import load_pickle, get_blds_list_df
from gan.utils.builder import exprs2tensor
from gan.utils.data import StockData


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


def chunk_batch_spearmanr(x, y, chunk_size=400):
    vals = []
    for i in range(0, len(x), chunk_size):
        vals.append(batch_spearmanr(x[i : i + chunk_size], y[i : i + chunk_size]))
    return torch.cat(vals, dim=0)


def get_tensor_metrics_raw(x, y):
    ic_s = torch.nan_to_num(batch_pearsonr(x, y), nan=0)
    ric_s = torch.nan_to_num(chunk_batch_spearmanr(x, y, chunk_size=400), nan=0)
    ret_s = torch.nan_to_num(batch_ret(x, y), nan=0)
    return ic_s, ric_s, ret_s


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
        rows.append({
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
        })

    daily_ret = np.asarray(daily_ret, dtype=float)
    daily_excess = np.asarray(daily_excess, dtype=float)
    cum_curve = np.asarray([r["cum_return"] for r in rows], dtype=float)
    ir = float(np.nanmean(daily_excess) / np.nanstd(daily_excess) * np.sqrt(252)) if len(daily_excess) > 1 and np.nanstd(daily_excess) > 0 else np.nan
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
    parser.add_argument("--workspace", default="/root/autodl-tmp/aff_official_workspace_20260629")
    parser.add_argument("--save-name", default="affofficial")
    parser.add_argument("--instruments", default="csi300")
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-factors", type=int, default=10)
    parser.add_argument("--window", default="inf")
    parser.add_argument("--freq", default="day")
    parser.add_argument("--test-start", default="2024-01-02")
    parser.add_argument("--test-end", default="2026-05-28")
    parser.add_argument("--history-start", default="2011-01-01")
    parser.add_argument("--qlib-path", default="/root/autodl-tmp/cn_data_akshare_2010_2026")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--cost", type=float, default=0.0015)
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()

    window: Union[float, int]
    if isinstance(args.window, str) and args.window == "inf":
        window = float("inf")
    else:
        window = int(args.window)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.workspace) / "out" / f"{args.save_name}_{args.instruments}_{args.train_end_year}_{args.seed}_mainwindow"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.workspace) / "out" / f"{args.save_name}_{args.instruments}_{args.train_end_year}_{args.seed}"
    zoo_path = source_dir / "z_bld_zoo_final.pkl"
    zoo = load_pickle(str(zoo_path))

    print(f"[AFF_MAIN] source_zoo={zoo_path}")
    print(f"[AFF_MAIN] test={args.test_start}..{args.test_end} device={device}")
    data_all = StockData(args.instruments, args.history_start, args.test_end, raw=True, qlib_path=args.qlib_path, freq=args.freq, device=device)
    df = get_blds_list_df([zoo]).sort_values("score", ascending=False, key=lambda x: abs(x))
    fct_tensor = exprs2tensor(df["exprs"], data_all, normalize=True)
    tgt_tensor = exprs2tensor([target], data_all, normalize=False)
    if fct_tensor.shape[0] != data_all.n_days:
        raise RuntimeError(f"fct days {fct_tensor.shape[0]} != data_all.n_days {data_all.n_days}")

    ic_list, ric_list, ret_list = [], [], []
    for cur in tqdm(range(fct_tensor.shape[-1]), desc="factor metrics"):
        ic_s, ric_s, ret_s = get_tensor_metrics_raw(fct_tensor[..., cur], tgt_tensor[..., 0])
        ic_list.append(ic_s)
        ric_list.append(ric_s)
        ret_list.append(ret_s)
    ic_s = torch.stack(ic_list, dim=-1)
    ric_s = torch.stack(ric_list, dim=-1)
    ret_s = torch.stack(ret_list, dim=-1)

    bt = data_all.max_backtrack_days
    n_days = data_all.n_days
    all_dates = pd.to_datetime(data_all._dates[bt : bt + n_days])
    close_idx = list(data_all._features).index(1) if 1 in list(data_all._features) else 1
    close = data_all.data[:, close_idx, :].detach().cpu().numpy()
    all_label = close[bt + 1 : bt + n_days + 1] / close[bt : bt + n_days] - 1.0
    test_start = pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end)
    selected = np.where((all_dates >= test_start) & (all_dates <= test_end))[0]
    if len(selected) == 0:
        raise RuntimeError("No selected dates in requested main test window")

    shift = 21
    pred_list = []
    good_idx_list = []
    weight_list = []
    for cur in tqdm(selected.tolist(), desc="main-window inference"):
        begin = int(cur - window - shift) if np.isfinite(window) else 0
        begin = max(begin, 0)
        hist_end = cur - shift
        if hist_end <= begin:
            raise RuntimeError(f"Not enough history for cur={cur}, begin={begin}, hist_end={hist_end}")
        cur_ic = ic_s[begin:hist_end]
        cur_ric = ric_s[begin:hist_end]
        cur_ret = ret_s[begin:hist_end]
        metrics = {
            "ic": cur_ic.mean(dim=0).detach().cpu().numpy(),
            "ic_std": cur_ic.std(dim=0).detach().cpu().numpy(),
            "ric": cur_ric.mean(dim=0).detach().cpu().numpy(),
            "ric_std": cur_ric.std(dim=0).detach().cpu().numpy(),
            "ret": cur_ret.mean(dim=0).detach().cpu().numpy(),
            "ret_std": cur_ret.std(dim=0).detach().cpu().numpy(),
        }
        metrics["icir"] = metrics["ic"] / metrics["ic_std"]
        metrics["ricir"] = metrics["ric"] / metrics["ric_std"]
        metrics["retir"] = metrics["ret"] / metrics["ret_std"]
        tmp = pd.DataFrame(metrics).replace([np.inf, -np.inf], np.nan).fillna(0)
        tmp = tmp.sort_values("ricir", ascending=False, key=lambda x: abs(x))
        chosen = tmp[(tmp["ric"] > 0.02) & (tmp["ricir"] > 0.2)]
        if len(chosen) < 1:
            chosen = tmp.iloc[:1]
        good_idx = chosen.iloc[: args.n_factors].index.to_list()
        good_idx_list.append(good_idx)

        x = fct_tensor[begin:hist_end, :, good_idx]
        y = tgt_tensor[begin:hist_end]
        to_pred = torch.nan_to_num(fct_tensor[cur, :, good_idx], nan=0)
        y = y.reshape(-1, y.shape[-1])
        x = x.reshape(-1, x.shape[-1])
        valid = torch.isfinite(y)[:, 0]
        y = y[valid]
        x = x[valid]
        ones = torch.ones_like(x[..., 0:1])
        x = torch.cat([x, ones], dim=-1)
        ones = torch.ones_like(to_pred[..., 0:1])
        to_pred = torch.cat([to_pred, ones], dim=-1)
        coef = torch.linalg.lstsq(x, y).solution
        pred = to_pred @ coef
        pred_list.append(pred[:, 0].detach().cpu())
        weight_list.append(coef.detach().cpu().numpy())

    pred = torch.stack(pred_list, dim=0).numpy()
    label = all_label[selected]
    dates = all_dates[selected]
    if pred.shape != label.shape:
        raise RuntimeError(f"pred shape {pred.shape} != label shape {label.shape}")
    bench = np.nanmean(label, axis=1)
    daily, metrics = topk_backtest(pred, label, bench, dates, args.topk, args.cost)
    metrics = {
        "method": "AFF-official-adapted-mainwindow",
        "seed": args.seed,
        "test_start": str(dates[0])[:10],
        "test_end": str(dates[-1])[:10],
        **metrics,
    }

    tensor_name = f"{args.train_end_year}_{args.n_factors}_{args.window}_{args.seed}_mainwindow"
    torch.save(torch.as_tensor(pred), out_dir / f"pred_{tensor_name}.pt")
    np.savez_compressed(out_dir / "pred_test_mainwindow.npz", pred=pred, label=label, dates=dates.astype(str).to_numpy())
    daily.to_csv(out_dir / "aff_official_daily_report.csv", index=False)
    with open(out_dir / "aff_official_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    with open(out_dir / "selection_trace.json", "w", encoding="utf-8") as f:
        json.dump({"dates": [str(d)[:10] for d in dates], "good_idx": good_idx_list}, f)
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(out_dir)


if __name__ == "__main__":
    main()
