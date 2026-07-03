from pathlib import Path
p = Path('/root/alpha_1203/AlphaForge-master/alphagen-master/scripts/rl_v1.py')
text = p.read_text()
backup = p.with_suffix('.py.bak_warmpool_20260629')
if not backup.exists():
    backup.write_text(text)
insert = '''\n\ndef read_warm_pool(path_template: Optional[str], seed: int) -> List[Expression]:\n    if not path_template:\n        return []\n    path = path_template.format(seed=seed)\n    parser = build_parser()\n    with open(path) as f:\n        data = json.load(f)\n    if isinstance(data, dict):\n        expr_strings = data.get("exprs", [])\n    else:\n        expr_strings = data\n    exprs = []\n    seen = set()\n    for expr_str in expr_strings:\n        if isinstance(expr_str, (list, tuple)):\n            expr_str = expr_str[0]\n        if expr_str in seen:\n            continue\n        seen.add(expr_str)\n        exprs.append(parser.parse(expr_str))\n    print(f"[WarmPool] Loaded {len(exprs)} expressions from {path}")\n    return exprs\n\n'''
if 'def read_warm_pool' not in text:
    text = text.replace('\n\ndef build_parser()', insert + '\ndef build_parser()')
text = text.replace('    output_root: str = None,\n):', '    output_root: str = None,\n    warm_pool_json: Optional[str] = None,\n):', 1)
text = text.replace('    GVA Min State Length: {gva_min_state_len}\n    AlphaGPT-Like Init-Only LLM Usage', '    GVA Min State Length: {gva_min_state_len}\n    Warm Pool JSON: {warm_pool_json}\n    AlphaGPT-Like Init-Only LLM Usage')
text = text.replace('    chat, inter, pool = None, None, build_pool([])', '    warm_exprs = read_warm_pool(warm_pool_json, seed)\n    chat, inter, pool = None, None, build_pool(warm_exprs)')
text = text.replace('    output_root: str = None,\n    \n):', '    output_root: str = None,\n    warm_pool_json: Optional[str] = None,\n    \n):', 1)
text = text.replace('            output_root=output_root,\n        )', '            output_root=output_root,\n            warm_pool_json=warm_pool_json,\n        )', 1)
p.write_text(text)
print('patched', p)
