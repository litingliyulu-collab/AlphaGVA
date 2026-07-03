# GVA Factor Experiments

This directory is the clean workspace for follow-up thesis experiments.

Old frozen results:
- /root/alpha_1203/AlphaForge-master/alphagen-master/out_PPO
- /root/alpha_1203/AlphaForge-master/alphagen-master/out_MSE
- /root/alpha_1203/AlphaForge-master/alphagen-master/out_GVA
- /root/alpha_1203/GVA

Local essential archive:
- D:/all_python_project/alphagen-master/_remote_archive/2026-06-27_old_results/old_results_essential_20260627.full.tar.gz

New experiment outputs:
- runs/E1_baseline
- runs/E2_critic_gva_sweep
- runs/E3_actor_gva
- runs/E4_full_gva
- runs/R_reward_ablation

Rule: sync new runs back to local after each batch, then remove non-final checkpoints from remote.
