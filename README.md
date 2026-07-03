# GVA-AlphaGen Thesis Code

This repository contains the code used for a graduation thesis project on
formulaic alpha mining with reinforcement learning.  The project is based on
AlphaGen and adds GVA-style value augmentation, actor-side advantage shaping,
filter-based factor pool selection, reinforcement-learning baselines, machine
learning baselines, and Qlib-style stock-selection backtests.

The code is organized for reproduction rather than for storing experiment
outputs.  Large datasets, checkpoints, logs, and backtest results are ignored by
Git and should be generated locally or on a server.

## Main Components

- `alphagen/`, `alphagen_qlib/`, `alphagen_generic/`: Alpha expression
  grammar, Qlib data adapter, factor pool evaluation, and core data structures.
- `scripts/rl_v1.py`: main AlphaGen/GVA training entry point.
- `scripts/reinforce_qfr_newdata.py`: REINFORCE and QFR-style RL baselines.
- `scripts/ml_baselines_newdata.py`, `scripts/ml_baselines_xgb_newdata.py`:
  MLP, LightGBM, and XGBoost baselines.
- `gva_factor_experiments/scripts/`: experiment orchestration, filtering,
  backtesting, metric compilation, and thesis figure scripts.
- `data_collection/`: data download/dump helpers inherited from AlphaGen.

## Environment

The experiments were run on Ubuntu 22.04 with Python 3.12 and CUDA-enabled
PyTorch.  CPU mode is possible for data checks and small backtests, but training
RL models is much faster with GPU.

Create an environment:

```bash
conda create -n gva-alpha python=3.12 -y
conda activate gva-alpha
pip install -r requirements.txt
pip install qlib akshare pandas numpy matplotlib scikit-learn lightgbm xgboost stable-baselines3 sb3-contrib
```

If you use a different Python version, keep the PyTorch, Qlib, and
stable-baselines3 versions mutually compatible.

## Data Preparation

The thesis experiments use Chinese A-share daily data converted to Qlib binary
format.  The server-side default path used in scripts is:

```text
/root/autodl-tmp/cn_data_akshare_2010_2026
```

Download and dump data with AKShare:

```bash
python gva_factor_experiments/scripts/download_akshare_cn_data.py \
  --raw_dir /root/autodl-tmp/cn_data_akshare_raw \
  --qlib_dir /root/autodl-tmp/cn_data_akshare_2010_2026 \
  --start 20100101 \
  --end 20260528 \
  --universes csi300,csi500,csi1000 \
  --workers 8
```

The standard split used in the thesis is:

```text
train: 2010-01-04 .. 2021-12-31
valid: 2022-01-04 .. 2023-12-29
test : 2024-01-02 .. 2026-05-28
```

To inspect the dumped data:

```bash
python gva_factor_experiments/scripts/audit_new_qlib_data.py \
  --qlib_dir /root/autodl-tmp/cn_data_akshare_2010_2026
```

## Mining Factors

Run the main AlphaGen/GVA comparison on CSI300, CSI500, or CSI1000 with the
provided shell scripts.  Example for CSI1000:

```bash
bash gva_factor_experiments/scripts/run_csi1000_main_compare_parallel.sh
```

This runs:

- `AlphaGen-PPO`: original AlphaGen-style PPO training.
- `Critic-GVA`: GVA-style greedy value baseline on the critic side.
- `Full-GVA`: critic-side GVA plus actor-side advantage shaping.

Actor-only ablation:

```bash
bash gva_factor_experiments/scripts/run_csi1000_actor_gva_parallel.sh
```

Additional RL baselines:

```bash
bash gva_factor_experiments/scripts/run_csi1000_a2c.sh
bash gva_factor_experiments/scripts/run_csi1000_reinforce_qfr.sh
bash gva_factor_experiments/scripts/run_csi1000_qfr_only.sh
```

Symbolic and machine-learning baselines:

```bash
bash gva_factor_experiments/scripts/run_csi1000_gp_xgb.sh
bash gva_factor_experiments/scripts/run_csi1000_ml_baselines.sh
```

The scripts write factor pools, metrics, and logs under
`/root/autodl-tmp/gva_factor_experiments/runs_newdata/` by default.

## Filtering Factor Pools

GVA-filter and PPO-filter reuse the factor history produced by GVA or AlphaGen
and select a compact pool by validation-set performance and redundancy control:

```bash
python gva_factor_experiments/scripts/gva_filter_from_history.py --help
python gva_factor_experiments/scripts/ppo_filter_variants.py --help
```

The CSI1000 scripts already point to the corresponding history pools used in
the thesis experiments.

## Backtesting Stock Selection

Backtesting uses a simple top-k daily rebalanced long-only strategy:

1. Evaluate the mined factor pool on the test set.
2. Rank CSI constituent stocks by daily factor score.
3. Buy the top-k stocks with equal weights.
4. Subtract turnover cost.
5. Compare against equal-weight CSI universe benchmark.

For the final CSI1000 top100 figures:

```bash
python gva_factor_experiments/scripts/backtest_csi1000_chapter4_top100.py
```

This produces:

```text
/root/autodl-tmp/gva_factor_experiments/backtests/csi1000_chapter4_filter_top100_20260701
/root/autodl-tmp/gva_factor_experiments/backtests/csi1000_extra_rl_top100_20260701
```

Each method/seed directory contains:

- `daily_report.csv`: daily strategy return, benchmark return, cost, turnover.
- `positions.csv`: daily selected stocks and weights.

## Compiling Tables and Figures

Compile pure-test IC/RankIC tables:

```bash
python gva_factor_experiments/scripts/compile_csi1000_metric_tables.py
```

Plot the two CSI1000 top100 cumulative-return figures:

```bash
python tools/csi1000/plot_csi1000_top100_cum_return.py
```

The generated figures are:

```text
paper/figures/chapter4/fig_csi1000_top100_backtest_comparison_cum_return.png
paper/figures/chapter4/fig_csi1000_top100_backtest_ablation_cum_return.png
```

## Reproduction Notes

- Do not commit `data/`, `runs_newdata/`, `backtests/`, checkpoints, logs, or
  Qlib binary data to Git.
- Use three random seeds for thesis-level results unless a script explicitly
  states otherwise.
- In RL metrics generated by `scripts/rl_v1.py`, the pure test columns are
  `test/ic_2` and `test/rank_ic_2`.
- For standalone baselines and filtering scripts, use `test/ic_mean` and
  `test/rank_ic_mean`.
- Use cumulative return plots for final thesis backtest comparison.  Relative
  excess return plots are diagnostic only unless explicitly stated.

## Acknowledgement

This project builds on AlphaGen:

> Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning,
> KDD 2023.

The original AlphaGen README is kept as `README_ALPHAGEN_ORIGINAL.md`.
