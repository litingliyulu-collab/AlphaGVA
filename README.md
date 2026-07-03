# AlphaGVA

AlphaGVA is a graduation-thesis research codebase for formulaic alpha mining with reinforcement learning.  It is based on AlphaGen and adds GVA-style value augmentation, actor-side advantage shaping, factor-pool filtering, multiple RL/ML baselines, and Qlib-style stock-selection backtests.

If you want to find the GVA implementation first, read:

- [`GVA_CODE_MAP.md`](GVA_CODE_MAP.md)
- `alphagen/rl/custom_ppo_trainer.py`
- `scripts/rl_v1.py`

## Clean Project Map

```text
alphagen/                         Core AlphaGen package with GVA extensions
alphagen_qlib/                    Qlib data adapter and StockData wrapper
alphagen_generic/                 Generic operators/features for baselines
gva_factor_experiments/scripts/   Thesis experiment, filtering, backtest scripts
scripts/                          Main training entry scripts, especially rl_v1.py
tools/csi1000/                    Figure helpers for CSI1000 top100 backtests
data_collection/                  Data download/conversion helpers
dso/, gplearn/                    Baseline dependencies kept for reproduction
docs/                             Original AlphaGen notes and structure docs
legacy/root_entrypoints/          Old root-level demos kept for reference
dev_checks/                       Smoke tests and inspection scripts
```

See [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) for a more detailed explanation.

## Environment

The experiments were run on Ubuntu 22.04 with Python 3.12 and CUDA-enabled PyTorch.  CPU mode is enough for data checks and small backtests, but RL training is much faster on GPU.

```bash
conda create -n gva-alpha python=3.12 -y
conda activate gva-alpha
pip install -r requirements.txt
pip install qlib akshare pandas numpy matplotlib scikit-learn lightgbm xgboost stable-baselines3 sb3-contrib
```

## Data Preparation

The thesis experiments use Chinese A-share daily data converted to Qlib binary format.  The server-side default path used in scripts is:

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

Standard split:

```text
train: 2010-01-04 .. 2021-12-31
valid: 2022-01-04 .. 2023-12-29
test : 2024-01-02 .. 2026-05-28
```

Audit the dumped data:

```bash
python gva_factor_experiments/scripts/audit_new_qlib_data.py \
  --qlib_dir /root/autodl-tmp/cn_data_akshare_2010_2026
```

## Mine Factors

Main comparison on CSI1000:

```bash
bash gva_factor_experiments/scripts/run_csi1000_main_compare_parallel.sh
```

This runs:

- `AlphaGen-PPO`: original AlphaGen-style PPO baseline.
- `Critic-GVA`: critic-side greedy value augmentation.
- `Full-GVA`: critic-side GVA plus actor-side advantage shaping.

Actor-only ablation:

```bash
bash gva_factor_experiments/scripts/run_csi1000_actor_gva_parallel.sh
```

Other baselines:

```bash
bash gva_factor_experiments/scripts/run_csi1000_a2c.sh
bash gva_factor_experiments/scripts/run_csi1000_reinforce_qfr.sh
bash gva_factor_experiments/scripts/run_csi1000_gp_xgb.sh
bash gva_factor_experiments/scripts/run_csi1000_ml_baselines.sh
```

## Filter Factor Pools

GVA-filter and PPO-filter reuse mined factor histories and select compact factor pools:

```bash
python gva_factor_experiments/scripts/gva_filter_from_history.py --help
python gva_factor_experiments/scripts/ppo_filter_variants.py --help
```

## Backtest Stock Selection

The thesis backtest uses a simple top-k daily rebalanced long-only strategy:

1. Evaluate the mined factor pool on the test set.
2. Rank stocks by factor score.
3. Buy the top-k stocks with equal weights.
4. Subtract turnover cost.
5. Compare against equal-weight CSI universe benchmark.

For the final CSI1000 top100 figures:

```bash
python gva_factor_experiments/scripts/backtest_csi1000_chapter4_top100.py
python tools/csi1000/plot_csi1000_top100_cum_return.py
```

The two final figures are:

```text
paper/figures/chapter4/fig_csi1000_top100_backtest_comparison_cum_return.png
paper/figures/chapter4/fig_csi1000_top100_backtest_ablation_cum_return.png
```

## Notes

- Do not commit Qlib binary data, AKShare raw data, checkpoints, logs, or backtest result folders.
- For `scripts/rl_v1.py`, pure test metrics are `test/ic_2` and `test/rank_ic_2`.
- For standalone baselines and filtering scripts, use `test/ic_mean` and `test/rank_ic_mean`.
- Use cumulative-return figures for final thesis backtest comparison.

## Acknowledgement

This project builds on AlphaGen: *Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning*, KDD 2023.  The original AlphaGen README is kept at [`docs/ALPHAGEN_ORIGINAL.md`](docs/ALPHAGEN_ORIGINAL.md).
