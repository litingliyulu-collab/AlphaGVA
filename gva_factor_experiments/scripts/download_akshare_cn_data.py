import argparse
import datetime as dt
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import akshare as ak
import pandas as pd
from tqdm import tqdm

ALPHAGEN_DATA_COLLECTION = "/root/alpha_1203/AlphaForge-master/alphagen-master/data_collection"
if ALPHAGEN_DATA_COLLECTION not in sys.path:
    sys.path.insert(0, ALPHAGEN_DATA_COLLECTION)
from qlib_dump_bin import DumpDataAll  # noqa: E402

INDEX_MAP = {
    "csi300": "000300",
    "csi500": "000905",
    "csi1000": "000852",
}
INSTRUMENT_FILE_MAP = {
    "csi300": "csi300.txt",
    "csi500": "csi500.txt",
    "csi1000": "csi1000.txt",
}


def market_prefix(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith(("6", "9")):
        return "SH"
    return "SZ"


def qlib_code(code: str) -> str:
    code = str(code).zfill(6)
    return f"{market_prefix(code)}{code}"


def normalize_hist(df: pd.DataFrame, code: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d")
    out["open"] = pd.to_numeric(df["开盘"], errors="coerce")
    out["high"] = pd.to_numeric(df["最高"], errors="coerce")
    out["low"] = pd.to_numeric(df["最低"], errors="coerce")
    out["close"] = pd.to_numeric(df["收盘"], errors="coerce")
    out["volume"] = pd.to_numeric(df["成交量"], errors="coerce")
    out["amount"] = pd.to_numeric(df["成交额"], errors="coerce")
    out["change"] = pd.to_numeric(df["涨跌幅"], errors="coerce") / 100.0
    out["turn"] = pd.to_numeric(df.get("换手率", 0.0), errors="coerce")
    out["factor"] = 1.0
    out["vwap"] = out["amount"] / out["volume"].replace(0, pd.NA)
    out["code"] = qlib_code(code)
    out = out.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    out = out.sort_values("date").drop_duplicates("date")
    return out


def fetch_index_members(index_names):
    members = {}
    for name in index_names:
        symbol = INDEX_MAP[name]
        print(f"Fetching index constituents: {name} {symbol}", flush=True)
        df = ak.index_stock_cons_csindex(symbol=symbol)
        code_col = "成分券代码"
        codes = sorted({str(x).zfill(6) for x in df[code_col].astype(str)})
        members[name] = codes
        print(f"  {name}: {len(codes)} stocks", flush=True)
    return members


def fetch_all_a_codes():
    print("Fetching all A-share spot list", flush=True)
    df = ak.stock_zh_a_spot_em()
    codes = []
    for code in df["代码"].astype(str):
        code = code.zfill(6)
        if code.startswith(("0", "3", "6")):
            codes.append(code)
    codes = sorted(set(codes))
    print(f"  all_a: {len(codes)} stocks", flush=True)
    return codes


def download_one(code, start_date, end_date, export_dir, force=False, retries=3, sleep_seconds=0.6):
    out_path = Path(export_dir) / f"{qlib_code(code)}.csv"
    if out_path.exists() and not force:
        try:
            if out_path.stat().st_size > 100:
                return code, "skip", str(out_path)
        except OSError:
            pass
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            raw = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )
            df = normalize_hist(raw, code)
            if df.empty:
                return code, "empty", str(out_path)
            df.to_csv(out_path, index=False)
            time.sleep(sleep_seconds)
            return code, "ok", str(out_path)
        except Exception as exc:  # keep downloader alive on flaky symbols
            last_err = exc
            time.sleep(sleep_seconds * attempt)
    return code, f"error: {type(last_err).__name__}: {last_err}", str(out_path)


def write_instrument_file(path, codes, start_date_text, end_date_text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for code in sorted(codes):
            f.write(f"{qlib_code(code)}\t{start_date_text}\t{end_date_text}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="/root/autodl-tmp/akshare_cn_2010_2026")
    parser.add_argument("--qlib_dir", default="/root/autodl-tmp/cn_data_akshare_2010_2026")
    parser.add_argument("--start", default="20100101")
    parser.add_argument("--end", default=dt.date.today().strftime("%Y%m%d"))
    parser.add_argument("--universes", default="csi300,csi500,csi1000")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_dump", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    export_dir = raw_dir / "export"
    qlib_dir = Path(args.qlib_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    qlib_dir.mkdir(parents=True, exist_ok=True)

    requested = [x.strip().lower() for x in args.universes.split(",") if x.strip()]
    index_names = [x for x in requested if x in INDEX_MAP]
    members = fetch_index_members(index_names)

    universe_codes = set()
    for codes in members.values():
        universe_codes.update(codes)
    if "all" in requested or "all_a" in requested:
        universe_codes.update(fetch_all_a_codes())

    codes = sorted(universe_codes)
    if args.limit and args.limit > 0:
        codes = codes[: args.limit]
    print(f"Download universe size: {len(codes)}", flush=True)
    print(f"Date range: {args.start} -> {args.end}", flush=True)
    print(f"CSV export dir: {export_dir}", flush=True)
    print(f"Qlib dir: {qlib_dir}", flush=True)

    status_counts = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download_one, code, args.start, args.end, export_dir, args.force)
            for code in codes
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="download"):
            code, status, path = future.result()
            status_counts[status] = status_counts.get(status, 0) + 1
            if status.startswith("error") or status == "empty":
                print(f"{code}\t{status}\t{path}", flush=True)
    print(f"Download summary: {status_counts}", flush=True)

    if args.skip_dump:
        print("Skip qlib dump by request", flush=True)
        return

    print("Dump CSV to qlib bin", flush=True)
    DumpDataAll(
        csv_path=str(export_dir),
        qlib_dir=str(qlib_dir),
        max_workers=max(1, min(args.workers, 8)),
        exclude_fields="date,code",
        symbol_field_name="code",
    ).dump()

    cal_path = qlib_dir / "calendars" / "day.txt"
    future_path = qlib_dir / "calendars" / "day_future.txt"
    if cal_path.exists():
        future_path.write_text(cal_path.read_text(encoding="utf-8"), encoding="utf-8")

    instrument_dir = qlib_dir / "instruments"
    start_text = f"{args.start[:4]}-{args.start[4:6]}-{args.start[6:8]}"
    end_text = f"{args.end[:4]}-{args.end[4:6]}-{args.end[6:8]}"
    for name, codes_for_index in members.items():
        if args.limit and args.limit > 0:
            codes_for_index = [c for c in codes_for_index if c in set(codes)]
        write_instrument_file(instrument_dir / INSTRUMENT_FILE_MAP[name], codes_for_index, start_text, end_text)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
