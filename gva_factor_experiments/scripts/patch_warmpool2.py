from pathlib import Path
p = Path('/root/alpha_1203/AlphaForge-master/alphagen-master/scripts/rl_v1.py')
text = p.read_text()
text = text.replace('    reward_mode: str = "original",\n):', '    reward_mode: str = "original",\n    warm_pool_json: Optional[str] = None,\n):', 1)
text = text.replace('    reward_mode: str = "original",\n    \n):', '    reward_mode: str = "original",\n    warm_pool_json: Optional[str] = None,\n    \n):', 1)
text = text.replace('    GVA Min State Length: {gva_min_state_len}\n    AlphaGPT-Like Init-Only LLM Usage', '    GVA Min State Length: {gva_min_state_len}\n    Warm Pool JSON: {warm_pool_json}\n    AlphaGPT-Like Init-Only LLM Usage')
text = text.replace('            reward_mode=reward_mode,\n        )', '            reward_mode=reward_mode,\n            warm_pool_json=warm_pool_json,\n        )', 1)
p.write_text(text)
print('done')
