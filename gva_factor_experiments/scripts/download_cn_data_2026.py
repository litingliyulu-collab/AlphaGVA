import datetime
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path('/root/alpha_1203/AlphaForge-master/alphagen-master/data_collection')
sys.path.insert(0, str(SCRIPT_DIR))
from fetch_baostock_data import DataManager

raw_dir = '/root/autodl-tmp/baostock_2026_raw'
qlib_dir = '/root/autodl-tmp/cn_data_2026'
base_dir = '/root/autodl-tmp/cn_data'
adjust_date = str(datetime.date.today())

print('download_cn_data_2026')
print('raw_dir =', raw_dir)
print('qlib_dir =', qlib_dir)
print('base_dir =', base_dir)
print('adjust_date =', adjust_date)
print('max_workers = 4')

Path(raw_dir).mkdir(parents=True, exist_ok=True)
Path(qlib_dir).mkdir(parents=True, exist_ok=True)

dm = DataManager(
    save_path=raw_dir,
    qlib_export_path=qlib_dir,
    qlib_base_data_path=base_dir,
    use_forward_adjust=True,
    adjust_date=adjust_date,
    max_workers=4,
    max_retries=20,
    retry_wait_seconds=5.0,
)
dm.fetch_and_save_data(
    use_cached_basic_info=Path(raw_dir, 'basic_info.csv').exists(),
    use_cached_adjust_factor=Path(raw_dir, 'adjust_factors.csv').exists(),
)

cal = Path(qlib_dir) / 'calendars' / 'day.txt'
if cal.exists():
    lines = [x.strip() for x in cal.read_text().splitlines() if x.strip()]
    print('DONE')
    print('calendar_first =', lines[0] if lines else None)
    print('calendar_last =', lines[-1] if lines else None)
    print('calendar_days =', len(lines))
else:
    print('DONE but calendar not found:', cal)
