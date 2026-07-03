from pathlib import Path
import shutil
import time


repo = Path("/root/alpha_1203/AlphaForge-master/alphagen-master")
target = repo / "alphagen/rl/custom_ppo_trainer.py"
backup_dir = Path("/root/alpha_1203/gva_factor_experiments/archive/code_backups")
backup_dir.mkdir(parents=True, exist_ok=True)
backup = backup_dir / f"custom_ppo_trainer_before_old_full_gva_weight_{time.strftime('%Y%m%d%H%M%S')}.py"
shutil.copy2(target, backup)

text = target.read_text()
old = """                        value_gap = baseline_tensor.detach() - b_values.detach()
                        value_gap = torch.clamp(value_gap, min=-actor_gap_clip, max=actor_gap_clip)
                        agreement = torch.sign(b_advantages.detach() * value_gap)
                        actor_delta = actor_gap_weight * agreement * torch.abs(value_gap)
                        actor_weights = torch.clamp(1.0 + actor_delta, min=0.1, max=3.0)
"""
new = """                        value_gap = baseline_tensor.detach() - b_values.detach()
                        value_gap = torch.clamp(value_gap, min=-actor_gap_clip, max=actor_gap_clip)
                        actor_weights = torch.clamp(1.0 + actor_gap_weight * value_gap, min=0.1, max=3.0)
"""
if old not in text:
    raise SystemExit("target block not found; file may already be changed")
target.write_text(text.replace(old, new))
print(f"patched {target}")
print(f"backup {backup}")
