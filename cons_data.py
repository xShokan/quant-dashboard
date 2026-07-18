# -*- coding: utf-8 -*-
"""科创50成分股数据: 拉取 + 缓存 + 个股统计"""
import time
from pathlib import Path

import akshare as ak
import pandas as pd

DATA_DIR = Path("data/cons")

import sys
sys.path.insert(0, str(Path(__file__).parent))
from kc50_analysis import perf_stats


def get_constituents() -> pd.DataFrame:
    """当前科创50成分股 (代码/名称), 缓存到 data/constituents.csv"""
    cache = Path("data/constituents.csv")
    if cache.exists():
        return pd.read_csv(cache, dtype={"成分券代码": str})
    df = ak.index_stock_cons_csindex(symbol="000688")
    out = df[["成分券代码", "成分券名称"]].rename(
        columns={"成分券代码": "code", "成分券名称": "name"})
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def _fetch_sina(code: str, **kw) -> pd.DataFrame:
    df = ak.stock_zh_a_daily(symbol=f"sh{code}", adjust="qfq", **kw)
    time.sleep(0.5)  # 限速, 避免被新浪封
    return _normalize(df)


def fetch_stock_daily(code: str) -> pd.DataFrame:
    """新浪日线(前复权), CSV 缓存 + 增量更新. code 形如 688008."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    f = DATA_DIR / f"{code}.csv"
    if not f.exists():
        df = _fetch_sina(code)
        df.to_csv(f, index=False)
        return df.set_index("date")

    old = _normalize(pd.read_csv(f, parse_dates=["date"]))
    last = old["date"].max()
    if last >= pd.Timestamp.today().normalize():
        return old.set_index("date")

    start = (last - pd.Timedelta(days=7)).strftime("%Y%m%d")
    end = pd.Timestamp.today().strftime("%Y%m%d")
    new = _fetch_sina(code, start_date=start, end_date=end)
    if new is None or len(new) == 0:
        return old.set_index("date")

    # 除权除息后前复权因子会变, 重叠日收盘价偏差 >1% 说明缓存已失效, 全量重拉
    overlap = old.merge(new, on="date", suffixes=("_o", "_n"))
    if len(overlap) and ((overlap["close_o"] - overlap["close_n"]).abs()
                         / overlap["close_o"] > 0.01).any():
        df = _fetch_sina(code)
    else:
        df = (pd.concat([old, new]).drop_duplicates("date", keep="last")
              .sort_values("date"))
    df.to_csv(f, index=False)
    return df.set_index("date")


def fetch_all(refresh: bool = False) -> dict:
    """返回 {code: {"name": ..., "df": ...}}"""
    cons = get_constituents()
    if refresh:
        for f in DATA_DIR.glob("*.csv"):
            f.unlink()
    out = {}
    for _, row in cons.iterrows():
        code, name = str(row["code"]), row["name"]
        try:
            out[code] = {"name": name, "df": fetch_stock_daily(code)}
        except Exception as e:
            print(f"  {code} {name} 获取失败: {type(e).__name__}")
    return out


def stock_stats(df: pd.DataFrame) -> dict:
    ret = df["close"].pct_change()
    s = perf_stats(ret, "")
    s.pop("策略", None)
    s["起始日期"] = str(df.index[0].date())
    return s


VAL_DIR = Path("data/val")


def load_valuation(code: str) -> dict:
    """读取缓存的估值数据, 返回 {'YYYY-MM-DD': [pe_ttm, pb]}"""
    out = {}
    for tag in ("pe", "pb"):
        f = VAL_DIR / f"{code}_{tag}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            d = str(row["date"])[:10]
            out.setdefault(d, [None, None])[0 if tag == "pe" else 1] = round(float(row["value"]), 2)
    return out


if __name__ == "__main__":
    stocks = fetch_all(refresh="--refresh" in sys.argv)
    print(f"成功获取 {len(stocks)} 只成分股")
    for code, v in list(stocks.items())[:5]:
        df = v["df"]
        print(f"  {code} {v['name']}: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)} 条")
