from pathlib import Path
from bisect import bisect_left, bisect_right
cal = [x.strip() for x in Path('/root/autodl-tmp/cn_data_akshare_2010_2026/calendars/day.txt').read_text().splitlines() if x.strip()]
print('n_days', len(cal), 'first', cal[0], 'last', cal[-1])
for h in [1, 5, 10, 20]:
    print('last_usable_for_h', h, cal[-h-1])
periods = [
    ('train', '2010-01-04', '2021-12-31'),
    ('valid', '2022-01-01', '2023-12-31'),
    ('test', '2024-01-01', '2026-05-28'),
    ('test_raw', '2024-01-01', '2026-06-26'),
    ('focus_2025', '2025-01-01', '2026-05-28'),
]
for name, s, e in periods:
    l = bisect_left(cal, s)
    r = bisect_right(cal, e)
    print(name, s, e, 'days', r-l, 'first', cal[l] if l < r else None, 'last', cal[r-1] if l < r else None)
