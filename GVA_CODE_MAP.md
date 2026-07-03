# Where is the GVA code?

This repository keeps the original Python package name `alphagen` for compatibility.  The GVA method is not a separate package named `alphagen-GVA`; it is implemented as extensions inside the AlphaGen training stack.

## Core GVA implementation

| Purpose | File |
|---|---|
| Custom PPO/GVA trainer, critic-side GVA loss, actor-side advantage shaping | `alphagen/rl/custom_ppo_trainer.py` |
| Main training entry, command-line flags for PPO / Critic-GVA / Full-GVA / Actor-GVA | `scripts/rl_v1.py` |
| Alpha expression environment and state transition | `alphagen/rl/env/core.py`, `alphagen/rl/env/wrapper.py` |
| Alpha pool reward and linear pool evaluation | `alphagen/models/linear_alpha_pool.py`, `alphagen/models/alpha_pool.py` |
| Reward shaping modules used by later thesis experiments | `alphagen/reward/` |

## Experiment scripts

| Experiment | Entry scripts |
|---|---|
| CSI300/CSI500/CSI1000 main comparison: AlphaGen, Critic-GVA, Full-GVA | `gva_factor_experiments/scripts/run_*_main_compare_parallel.sh` |
| Actor-only ablation | `gva_factor_experiments/scripts/run_*_actor_gva_parallel.sh` |
| GVA-filter and PPO-filter | `gva_factor_experiments/scripts/gva_filter_from_history.py`, `gva_factor_experiments/scripts/ppo_filter_variants.py` |
| GP/XGBoost/LightGBM/MLP baselines | `gva_factor_experiments/scripts/run_*_gp_xgb.sh`, `gva_factor_experiments/scripts/run_*_ml_baselines.sh` |
| A2C / REINFORCE / QFR baselines | `gva_factor_experiments/scripts/run_*_a2c.sh`, `gva_factor_experiments/scripts/run_*_reinforce_qfr.sh`, `gva_factor_experiments/scripts/run_csi1000_qfr_only.sh` |
| CSI1000 top100 backtest figures | `gva_factor_experiments/scripts/backtest_csi1000_chapter4_top100.py`, `tools/csi1000/plot_csi1000_top100_cum_return.py` |

## Why not rename `alphagen/` to `alphagen_gva/`?

The original project and most scripts import modules such as `alphagen.rl.env`, `alphagen.models`, and `alphagen_qlib`.  Renaming the package would require changing many imports and would make it harder to compare against the original AlphaGen implementation.  Therefore the package name stays `alphagen`, while the repository name and documentation identify this thesis version as AlphaGVA.
