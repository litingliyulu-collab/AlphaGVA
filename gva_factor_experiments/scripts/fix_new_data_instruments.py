from pathlib import Path
root = Path('/root/autodl-tmp/cn_data_akshare_2010_2026/instruments')
for name in ['csi300.txt', 'csi500.txt', 'csi1000.txt']:
    path = root / name
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split('\t')
        rows.append(f'{parts[0]}\t2010-01-04\t2026-06-26')
    path.write_text('\n'.join(rows) + '\n')
print('fixed csi instruments date bounds')
