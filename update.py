# -*- coding: utf-8 -*-
"""每日更新流水线: 行情增量 -> 估值增量 -> 新闻情绪 -> 预测 -> 重建网页

用法: .venv/bin/python update.py
说明: 行情/估值/新闻都是增量更新 (只拉新数据, 已打分的新闻不重复计费);
     预测为全量重算 (基于缓存行情, 无需网络, 约1-2分钟);
     成分股名单缓存于 data/constituents.csv, 指数季度调样后删除该文件即可刷新。
"""
import time
from pathlib import Path

import akshare as ak
import pandas as pd

VAL_DIR = Path("data/val")


def update_valuations():
    """估值(PE/PB)增量: 已有数据只补最近一年并合并"""
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    cons = pd.read_csv("data/constituents.csv", dtype={"code": str})
    today = pd.Timestamp.today().normalize()
    n = 0
    for _, r in cons.iterrows():
        for ind, tag in [("市盈率(TTM)", "pe"), ("市净率", "pb")]:
            f = VAL_DIR / f"{r['code']}_{tag}.csv"
            old = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["date", "value"])
            if len(old) and pd.Timestamp(str(old["date"].iloc[-1])) >= today:
                continue
            try:
                df = ak.stock_zh_valuation_baidu(symbol=str(r["code"]),
                                                 indicator=ind, period="近一年")
                old["date"] = old["date"].astype(str)
                df["date"] = df["date"].astype(str)
                merged = (pd.concat([old, df]).drop_duplicates("date", keep="last")
                          .sort_values("date"))
                merged.to_csv(f, index=False)
                n += 1
            except Exception as e:
                print(f"  {r['code']} {tag} 失败: {type(e).__name__}")
            time.sleep(0.3)
    print(f"  更新 {n} 个估值文件")


def main():
    print("1/5 成分股行情增量更新...")
    from cons_data import fetch_all
    stocks = fetch_all()
    print(f"  {len(stocks)} 只股票就绪")

    print("2/5 估值增量更新...")
    update_valuations()

    print("3/5 新闻情绪更新 (DeepSeek 只对新新闻打分)...")
    from sentiment import main as s
    s()

    print("4/5 重算预测 (walk-forward, 本地计算)...")
    from predict import main as p
    p()

    print("5/5 重建网页...")
    from build_site import main as b
    b()

    print("\n完成, 刷新 http://localhost:8000 查看")


if __name__ == "__main__":
    main()
