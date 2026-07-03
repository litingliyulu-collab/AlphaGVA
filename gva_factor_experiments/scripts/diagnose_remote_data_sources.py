import importlib.util
import time


def pkg_status():
    mods = ["baostock", "akshare", "qlib", "pandas"]
    print("__PKG__")
    for name in mods:
        print(f"{name}: {bool(importlib.util.find_spec(name))}")


def test_baostock():
    print("__BAOSTOCK__")
    if not importlib.util.find_spec("baostock"):
        print("baostock not installed")
        return
    import baostock as bs

    t0 = time.time()
    lg = bs.login()
    print("login", lg.error_code, lg.error_msg, "sec", round(time.time() - t0, 2))
    if lg.error_code != "0":
        return

    try:
        t0 = time.time()
        rs = bs.query_all_stock(day="2026-06-26")
        print("all_stock", rs.error_code, rs.error_msg, "sec", round(time.time() - t0, 2))
        rows = []
        while rs.error_code == "0" and rs.next() and len(rows) < 5:
            rows.append(rs.get_row_data())
        print("all_stock_rows", len(rows), rows[:2])

        t0 = time.time()
        rs2 = bs.query_history_k_data_plus(
            "sh.600000",
            "date,code,open,high,low,close,volume,amount,adjustflag",
            start_date="2025-01-01",
            end_date="2025-01-10",
            frequency="d",
            adjustflag="2",
        )
        print("hist", rs2.error_code, rs2.error_msg, "sec", round(time.time() - t0, 2))
        count = 0
        last = None
        while rs2.error_code == "0" and rs2.next():
            count += 1
            last = rs2.get_row_data()
        print("hist_rows", count, last)
    finally:
        bs.logout()


def test_akshare():
    print("__AKSHARE__")
    if not importlib.util.find_spec("akshare"):
        print("akshare not installed")
        return
    import akshare as ak

    t0 = time.time()
    spot = ak.stock_zh_a_spot_em()
    print("spot_shape", getattr(spot, "shape", None), "sec", round(time.time() - t0, 2))
    print(spot.head(2).to_string(index=False))

    t0 = time.time()
    hist = ak.stock_zh_a_hist(
        symbol="600000",
        period="daily",
        start_date="20250101",
        end_date="20250110",
        adjust="qfq",
    )
    print("hist_shape", getattr(hist, "shape", None), "sec", round(time.time() - t0, 2))
    print(hist.tail(2).to_string(index=False))


if __name__ == "__main__":
    print("__PY__")
    import sys

    print(sys.version)
    pkg_status()
    test_baostock()
    test_akshare()
