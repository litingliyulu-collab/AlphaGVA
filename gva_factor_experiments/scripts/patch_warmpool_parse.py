from pathlib import Path
p = Path('/root/alpha_1203/AlphaForge-master/alphagen-master/scripts/rl_v1.py')
text = p.read_text()
old = '''        if expr_str in seen:\n            continue\n        seen.add(expr_str)\n        exprs.append(parser.parse(expr_str))\n    print(f"[WarmPool] Loaded {len(exprs)} expressions from {path}")\n'''
new = '''        if expr_str in seen:\n            continue\n        seen.add(expr_str)\n        try:\n            exprs.append(parser.parse(expr_str))\n        except Exception as exc:\n            print(f"[WarmPool] Skip unparsable expression: {expr_str} ({type(exc).__name__}: {exc})")\n    print(f"[WarmPool] Loaded {len(exprs)} expressions from {path}")\n'''
if old not in text:
    raise SystemExit('target block not found')
text = text.replace(old, new, 1)
p.write_text(text)
print('done')
