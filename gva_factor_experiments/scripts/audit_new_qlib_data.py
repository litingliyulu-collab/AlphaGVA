from pathlib import Path
import struct
import json

qlib = Path('/root/autodl-tmp/cn_data_akshare_2010_2026')
print('__DATASET__')
print('path', qlib)
print('exists', qlib.exists())

cal_path = qlib / 'calendars' / 'day.txt'
cal = [x.strip() for x in cal_path.read_text().splitlines() if x.strip()]
print('__CALENDAR__')
print('count', len(cal))
print('first', cal[0] if cal else None)
print('last', cal[-1] if cal else None)
print('sample_head', cal[:5])
print('sample_tail', cal[-5:])

inst_dir = qlib / 'instruments'
print('__INSTRUMENTS__')
inst = {}
for name in ['all.txt', 'csi300.txt', 'csi500.txt', 'csi1000.txt']:
    p = inst_dir / name
    rows = [x.strip().split('\t') for x in p.read_text().splitlines() if x.strip()]
    inst[name] = rows
    starts = sorted({r[1] for r in rows if len(r) >= 3})
    ends = sorted({r[2] for r in rows if len(r) >= 3})
    print(name, 'rows', len(rows), 'start_min', starts[0] if starts else None, 'start_max', starts[-1] if starts else None, 'end_min', ends[0] if ends else None, 'end_max', ends[-1] if ends else None)
    print(name, 'head', rows[:3])

all_codes = {r[0].lower() for r in inst['all.txt'] if r}
feature_dirs = sorted([p.name for p in (qlib / 'features').iterdir() if p.is_dir()])
feature_codes = set(feature_dirs)
print('__FEATURES__')
print('feature_dirs', len(feature_dirs))
print('in_all_not_features', len(all_codes - feature_codes), sorted(list(all_codes - feature_codes))[:10])
print('features_not_in_all', len(feature_codes - all_codes), sorted(list(feature_codes - all_codes))[:20])

field_counts = {}
missing_fields = {}
fields_by_code = {}
for code in feature_dirs:
    files = sorted([p.name for p in (qlib / 'features' / code).glob('*.day.bin')])
    fields = [f.replace('.day.bin', '') for f in files]
    fields_by_code[code] = fields
    for f in fields:
        field_counts[f] = field_counts.get(f, 0) + 1
expected = sorted(field_counts.keys())
print('fields', expected)
print('field_counts', json.dumps(field_counts, ensure_ascii=False, sort_keys=True))
for code, fields in fields_by_code.items():
    miss = sorted(set(expected) - set(fields))
    if miss:
        missing_fields[code] = miss
print('codes_with_missing_fields', len(missing_fields))
print('missing_sample', list(missing_fields.items())[:10])

print('__BIN_SANITY__')
for code in feature_dirs[:5]:
    f = qlib / 'features' / code / 'close.day.bin'
    if not f.exists():
        print(code, 'no close')
        continue
    size = f.stat().st_size
    n_float = size // 4
    with f.open('rb') as fp:
        first_float = struct.unpack('<f', fp.read(4))[0]
    print(code, 'close_bin_bytes', size, 'float_count', n_float, 'calendar_start_index', first_float)

print('__INDEX_OVERLAP__')
sets = {k: {r[0].lower() for r in v if r} for k, v in inst.items()}
print('csi300_in_all', len(sets['csi300.txt'] - sets['all.txt']) == 0)
print('csi500_in_all', len(sets['csi500.txt'] - sets['all.txt']) == 0)
print('csi1000_in_all', len(sets['csi1000.txt'] - sets['all.txt']) == 0)
print('overlap_300_500', len(sets['csi300.txt'] & sets['csi500.txt']))
print('overlap_300_1000', len(sets['csi300.txt'] & sets['csi1000.txt']))
print('overlap_500_1000', len(sets['csi500.txt'] & sets['csi1000.txt']))
