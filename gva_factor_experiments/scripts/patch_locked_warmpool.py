from pathlib import Path

# Patch AlphaPool to support locked warm factors.
p = Path('/root/alpha_1203/AlphaForge-master/alphagen-master/alphagen/models/linear_alpha_pool.py')
text = p.read_text()
backup = p.with_suffix('.py.bak_locked_warm_20260629')
if not backup.exists():
    backup.write_text(text)
if 'self._locked_expr_keys' not in text:
    text = text.replace('        self._failure_cache: Set[str] = set()\n', '        self._failure_cache: Set[str] = set()\n        self._locked_expr_keys: Set[str] = set()\n', 1)
method = '''\n    def lock_current_exprs(self, n: int) -> None:\n        locked = []\n        for expr in self.exprs[:min(n, self.size)]:\n            if expr is not None:\n                key = str(expr)\n                self._locked_expr_keys.add(key)\n                locked.append(key)\n        print(f"[AlphaPool] Locked {len(locked)} warm expressions")\n\n    def _is_locked_index(self, index: int) -> bool:\n        expr = self.exprs[index]\n        return expr is not None and str(expr) in self._locked_expr_keys\n\n    def _select_pop_index(self, index_hint: Optional[int] = None) -> int:\n        if index_hint is not None and not self._is_locked_index(index_hint):\n            return index_hint\n        candidates = [i for i in range(self.size) if not self._is_locked_index(i)]\n        if not candidates:\n            candidates = list(range(self.size))\n        return int(min(candidates, key=lambda i: abs(self.weights[i])))\n'''
if 'def lock_current_exprs' not in text:
    text = text.replace('    def _calc_ics(\n', method + '\n    def _calc_ics(\n', 1)
old = '''    def _pop(self, index_hint: Optional[int] = None) -> None:\n        if self.size <= self.capacity:\n            return\n        index = int(np.argmin(np.abs(self.weights))) if index_hint is None else index_hint\n        self._swap_idx(index, self.capacity)\n        self.size = self.capacity\n'''
new = '''    def _pop(self, index_hint: Optional[int] = None) -> None:\n        if self.size <= self.capacity:\n            return\n        index = self._select_pop_index(index_hint)\n        self._swap_idx(index, self.capacity)\n        self.size = self.capacity\n'''
if old in text:
    text = text.replace(old, new, 1)
elif 'self._select_pop_index' not in text:
    raise SystemExit('Could not patch _pop')
p.write_text(text)

# Patch rl_v1 warm loader and args.
p = Path('/root/alpha_1203/AlphaForge-master/alphagen-master/scripts/rl_v1.py')
text = p.read_text()
backup = p.with_suffix('.py.bak_locked_warm_20260629')
if not backup.exists():
    backup.write_text(text)
text = text.replace('def read_warm_pool(path_template: Optional[str], seed: int) -> List[Expression]:', 'def read_warm_pool(path_template: Optional[str], seed: int, sort_by_abs_weight: bool = False) -> List[Expression]:', 1)
old = '''    if isinstance(data, dict):\n        expr_strings = data.get("exprs", [])\n    else:\n        expr_strings = data\n    exprs = []\n    seen = set()\n    for expr_str in expr_strings:\n        if isinstance(expr_str, (list, tuple)):\n            expr_str = expr_str[0]\n'''
new = '''    if isinstance(data, dict):\n        expr_strings = data.get("exprs", [])\n        weights = data.get("weights", [0.0] * len(expr_strings))\n        pairs = list(zip(expr_strings, weights))\n        if sort_by_abs_weight:\n            pairs.sort(key=lambda item: abs(float(item[1])), reverse=True)\n    else:\n        pairs = [(expr_str, 0.0) for expr_str in data]\n    exprs = []\n    seen = set()\n    for expr_str, _weight in pairs:\n        if isinstance(expr_str, (list, tuple)):\n            expr_str = expr_str[0]\n'''
if old in text:
    text = text.replace(old, new, 1)
# Add parameters after warm_pool_json in both signatures.
text = text.replace('    warm_pool_json: Optional[str] = None,\n):', '    warm_pool_json: Optional[str] = None,\n    warm_lock_n: int = 0,\n    warm_sort_by_abs_weight: bool = False,\n):', 1)
text = text.replace('    warm_pool_json: Optional[str] = None,\n    \n):', '    warm_pool_json: Optional[str] = None,\n    warm_lock_n: int = 0,\n    warm_sort_by_abs_weight: bool = False,\n    \n):', 1)
text = text.replace('    Warm Pool JSON: {warm_pool_json}\n    AlphaGPT-Like Init-Only LLM Usage', '    Warm Pool JSON: {warm_pool_json}\n    Warm Lock N: {warm_lock_n}\n    Warm Sort By Abs Weight: {warm_sort_by_abs_weight}\n    AlphaGPT-Like Init-Only LLM Usage')
text = text.replace('    warm_exprs = read_warm_pool(warm_pool_json, seed)\n    chat, inter, pool = None, None, build_pool(warm_exprs)', '    warm_exprs = read_warm_pool(warm_pool_json, seed, warm_sort_by_abs_weight)\n    chat, inter, pool = None, None, build_pool(warm_exprs)\n    if warm_lock_n > 0 and hasattr(pool, "lock_current_exprs"):\n        pool.lock_current_exprs(warm_lock_n)', 1)
text = text.replace('            warm_pool_json=warm_pool_json,\n        )', '            warm_pool_json=warm_pool_json,\n            warm_lock_n=warm_lock_n,\n            warm_sort_by_abs_weight=warm_sort_by_abs_weight,\n        )', 1)
p.write_text(text)
print('patched locked warm pool support')
