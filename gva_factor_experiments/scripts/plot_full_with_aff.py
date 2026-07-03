#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_METHODS = {
    'AlphaGen-PPO': ['alphagen_ppo_s0','alphagen_ppo_s1','alphagen_ppo_s2'],
    'Critic-GVA': ['critic_gva_s0','critic_gva_s1','critic_gva_s2'],
    'Full-GVA': ['full_gva_s0','full_gva_s1','full_gva_s2'],
    'Locked-WarmPool-GVA': ['locked_warmpool_gva_s0','locked_warmpool_gva_s1','locked_warmpool_gva_s2'],
    'PPO_filter_weak': ['ppo_filter_weak_s0','ppo_filter_weak_s1','ppo_filter_weak_s2'],
    'GP_filter_weak': ['gp_filter_weak_s0','gp_filter_weak_s1','gp_filter_weak_s2'],
    'GP_filter_strong': ['gp_filter_strong_s0','gp_filter_strong_s1','gp_filter_strong_s2'],
    'XGBoost': ['xgboost_s0','xgboost_s1','xgboost_s2'],
    'LightGBM': ['lightgbm_s0','lightgbm_s1','lightgbm_s2'],
    'MLP': ['mlp_s0','mlp_s1','mlp_s2'],
    'AFF-official-adapted': ['aff_official_adapted_s0'],
}


def read_method(base: Path, keys: list[str], col: str):
    xs = []
    for key in keys:
        path = base / key / 'daily_report.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        date_col = 'datetime' if 'datetime' in df.columns else 'date'
        df[date_col] = pd.to_datetime(df[date_col])
        xs.append(df.set_index(date_col)[col].rename(key))
    if not xs:
        return None
    return pd.concat(xs, axis=1).sort_index().fillna(0).mean(axis=1)


def plot_one(base: Path, col: str, fname: str, title: str, ylabel: str) -> None:
    plt.figure(figsize=(13, 6.5))
    for name, keys in DEFAULT_METHODS.items():
        series = read_method(base, keys, col)
        if series is None or series.empty:
            continue
        curve = (1 + series).cumprod() - 1
        lw = 3.0 if name in {'Full-GVA', 'Locked-WarmPool-GVA'} else 1.8
        linestyle = '--' if name == 'AFF-official-adapted' else '-'
        plt.plot(curve.index, curve.values, label=name, linewidth=lw, linestyle=linestyle)
    if col == 'return':
        bench = read_method(base, ['aff_official_adapted_s0'], 'benchmark_return')
        if bench is not None and not bench.empty:
            plt.plot(bench.index, (1 + bench).cumprod() - 1, label='CSI300 universe EW', color='black', linestyle=':', linewidth=2)
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.legend(fontsize=8, ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(base / fname, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--backtest-dir', required=True)
    args = parser.parse_args()
    base = Path(args.backtest_dir)
    plot_one(base, 'return', 'cumulative_return_2024_2026_with_aff.png', 'Cumulative Return on Main Test Window', 'Cumulative Return')
    plot_one(base, 'excess_return', 'cumulative_excess_return_2024_2026_with_aff.png', 'Cumulative Excess Return on Main Test Window', 'Cumulative Excess Return')
    print(base / 'cumulative_return_2024_2026_with_aff.png')
    print(base / 'cumulative_excess_return_2024_2026_with_aff.png')


if __name__ == '__main__':
    main()
