# -*- coding: utf-8 -*-
"""成分股方向预测: walk-forward LogisticRegression

- 目标: 未来5日收益 > 0 的概率
- 特征: 滞后收益/均线偏离/波动率/量比/区间位置 (全部为 T 日收盘可得, 无前视)
- 验证: walk-forward, 初始训练250日, 每21日重训, 全部预测都是样本外
- 输出: data/predict/{code}.csv (date, prob) + data/predict/_summary.json
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from cons_data import fetch_all

warnings.filterwarnings("ignore")

OUT_DIR = Path("data/predict")
HORIZON = 5          # 预测未来5日方向
MIN_TRAIN = 250      # 最少训练样本
REFIT = 21           # 每隔多少交易日重训


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    v = df["volume"]
    feat = pd.DataFrame(index=df.index)
    for n in (1, 3, 5, 10):
        feat[f"ret{n}"] = c.pct_change(n)
    for n in (5, 10, 20, 60):
        feat[f"ma{n}_dev"] = c / c.rolling(n).mean() - 1
    feat["vol20"] = c.pct_change().rolling(20).std()
    feat["vol_ratio"] = v.rolling(5).mean() / v.rolling(60).mean()
    lo, hi = df["low"].rolling(20).min(), df["high"].rolling(20).max()
    feat["range_pos"] = (c - lo) / (hi - lo)
    feat["target"] = (c.shift(-HORIZON) / c - 1 > 0).astype(float)
    return feat.dropna()


def walk_forward_predict(feat: pd.DataFrame) -> pd.Series:
    X = feat.drop(columns=["target"]).values
    y = feat["target"].values
    dates = feat.index
    probs = pd.Series(np.nan, index=dates)
    i = MIN_TRAIN
    while i < len(feat):
        j = min(i + REFIT, len(feat))
        scaler = StandardScaler().fit(X[:i])
        model = LogisticRegression(C=1.0, max_iter=1000)
        model.fit(scaler.transform(X[:i]), y[:i])
        probs.iloc[i:j] = model.predict_proba(scaler.transform(X[i:j]))[:, 1]
        i = j
    return probs


def evaluate(probs: pd.Series, feat: pd.DataFrame) -> dict:
    df = pd.DataFrame({"prob": probs, "target": feat["target"]}).dropna()
    if len(df) < 50:
        return {}
    hit = ((df["prob"] > 0.5).astype(float) == df["target"]).mean()
    base = max(df["target"].mean(), 1 - df["target"].mean())  # 永远猜多数类
    return {"样本数": len(df), "模型胜率": round(float(hit), 4),
            "基准胜率": round(float(base), 4)}


def run_stock(code: str, df: pd.DataFrame) -> dict:
    feat = make_features(df)
    if len(feat) < MIN_TRAIN + 50:
        return {"error": "数据不足"}
    probs = walk_forward_predict(feat)
    out = pd.DataFrame({"date": probs.index.strftime("%Y-%m-%d"),
                        "prob": probs.round(3)}).dropna()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / f"{code}.csv", index=False)
    summary = evaluate(probs, feat)
    if len(out):
        summary["最新日期"] = out["date"].iloc[-1]
        summary["最新概率"] = float(out["prob"].iloc[-1])
    return summary


def main():
    stocks = fetch_all()
    summaries = {}
    for code, v in stocks.items():
        try:
            summaries[code] = run_stock(code, v["df"])
            summaries[code]["name"] = v["name"]
            print(f"{code} {v['name']}: {summaries[code]}")
        except Exception as e:
            summaries[code] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{code} {v['name']} 失败: {e}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=1)
    print(f"\n完成 {len(summaries)} 只, 结果在 {OUT_DIR}/")


if __name__ == "__main__":
    main()
