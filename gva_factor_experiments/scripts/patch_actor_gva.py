from pathlib import Path
path = Path('/root/alpha_1203/AlphaForge-master/alphagen-master/alphagen/rl/custom_ppo_trainer.py')
backup = Path('/root/alpha_1203/gva_factor_experiments/archive/code_backups/custom_ppo_trainer_before_actor_gva_20260630.py')
backup.parent.mkdir(parents=True, exist_ok=True)
backup.write_text(path.read_text(), encoding='utf-8')
text = path.read_text(encoding='utf-8')
old = '        if use_history_best and baseline_weight > 0 and critic_loss_type == "hybrid" and len(state_keys) > 0:\n'
new = '        gva_needs_baseline_bank = (baseline_weight > 0 and critic_loss_type == "hybrid") or (actor_gap_weight != 0)\n        if use_history_best and gva_needs_baseline_bank and len(state_keys) > 0:\n'
if old not in text:
    raise SystemExit('target condition not found')
text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')
print('patched', path)
print('backup', backup)
