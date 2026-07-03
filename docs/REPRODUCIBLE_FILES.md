# What is included in the GitHub release

The release repository is intentionally code-first.

Included:

- Core AlphaGen/GVA Python packages.
- Main RL training script `scripts/rl_v1.py`.
- Baseline scripts for GP, ML, REINFORCE, QFR, A2C, and filtering.
- Backtest and metric compilation scripts used by the thesis.
- README and reproduction instructions.

Excluded:

- Qlib binary data.
- AKShare raw downloaded data.
- Training checkpoints and tensorboard logs.
- Full backtest output directories.
- Large thesis Word/PDF drafts.
- Temporary server archives.

Regenerate excluded artifacts by following `README.md`.
