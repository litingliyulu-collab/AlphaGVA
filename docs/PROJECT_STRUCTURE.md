# Project Structure

```text
AlphaGVA/
??? README.md                         # Reproduction guide
??? GVA_CODE_MAP.md                   # Exact location of GVA implementation
??? alphagen/                         # Core AlphaGen package with GVA extensions
?   ??? rl/custom_ppo_trainer.py      # PPO / Critic-GVA / Actor-GVA / Full-GVA trainer
?   ??? rl/env/                       # Formula generation environment
?   ??? models/                       # Alpha pool and pool reward evaluation
?   ??? reward/                       # Reward-shaping modules
??? alphagen_qlib/                    # Qlib data adapter and StockData wrapper
??? alphagen_generic/                 # Generic operators/features for baselines
??? gva_factor_experiments/scripts/   # Thesis experiment, filtering, backtest scripts
??? scripts/                          # Original and main RL entry scripts, especially rl_v1.py
??? tools/csi1000/                    # Thesis figure helpers
??? data_collection/                  # Data download/conversion helpers
??? dso/, gplearn/                    # Baseline dependencies kept for reproduction
??? docs/                             # Original AlphaGen README and release notes
??? legacy/root_entrypoints/          # Old root-level demos kept for reference
??? dev_checks/                       # Smoke tests and inspection scripts
```

The most important files for thesis reproduction are `GVA_CODE_MAP.md`, `scripts/rl_v1.py`, `alphagen/rl/custom_ppo_trainer.py`, and `gva_factor_experiments/scripts/`.
