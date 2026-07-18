# -*- coding: utf-8 -*-
"""科创50(000688) 基础分析 + 均线择时回测

- 数据: akshare 东方财富接口, 日线
- 分析: 收益/波动/回撤/月度胜率 等统计特征
- 回测: 简单的双均线/单均线择时 vs 买入持有, 含交易成本, 信号次日执行(防前视偏差)
"""
import warnings
warnings.filterwarnings("ignore")

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

COST = 0.0005  # 单边交易成本 0.05% (ETF 佣金量级)


def fetch_kc50() -> pd.DataFrame:
    # 东财接口不稳定时用新浪: sh000688 = 科创50
    df = ak.stock_zh_index_daily(symbol="sh000688")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df[["open", "high", "low", "close", "volume"]]


def perf_stats(ret: pd.Series, name: str) -> dict:
    ret = ret.dropna()
    if len(ret) == 0:
        return {}
    nav = (1 + ret).cumprod()
    n = len(ret)
    total = nav.iloc[-1] - 1
    ann_ret = nav.iloc[-1] ** (252 / n) - 1
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    dd = (nav / nav.cummax() - 1).min()
    win = (ret > 0).mean()
    return {
        "策略": name,
        "累计收益": f"{total:+.1%}",
        "年化收益": f"{ann_ret:+.1%}",
        "年化波动": f"{ann_vol:.1%}",
        "夏普": f"{sharpe:.2f}",
        "最大回撤": f"{dd:.1%}",
        "日胜率": f"{win:.1%}",
    }


def ma_timing(close: pd.Series, window: int) -> pd.Series:
    """收盘价 > MA(window) 则次日持有, 否则空仓. 信号 T 日收盘产生, T+1 生效."""
    ma = close.rolling(window).mean()
    signal = (close > ma).astype(int)
    position = signal.shift(1).fillna(0)          # 次日执行, 防前视
    trade_cost = position.diff().abs().fillna(0) * COST
    strat_ret = position * close.pct_change().fillna(0) - trade_cost
    return strat_ret


def main():
    df = fetch_kc50()
    close = df["close"]
    print(f"数据区间: {close.index[0].date()} ~ {close.index[-1].date()}, 共 {len(close)} 个交易日\n")

    # ---------- 基础统计 ----------
    ret = close.pct_change()
    rows = [perf_stats(ret, "买入持有")]
    base = rows[0]

    print("== 基础特征 ==")
    for k, v in base.items():
        if k != "策略":
            print(f"  {k}: {v}")

    # 年度收益
    yearly = (1 + ret).resample("Y").prod() - 1
    print("\n== 分年度收益 ==")
    for d, r in yearly.items():
        print(f"  {d.year}: {r:+.1%}")

    # 滚动60日年化波动
    roll_vol = ret.rolling(60).std() * np.sqrt(252)
    print(f"\n滚动60日年化波动: 中位数 {roll_vol.median():.1%}, "
          f"最高 {roll_vol.max():.1%}, 最低 {roll_vol.min():.1%}")

    # 大跌频率: 单日跌超3%的天数占比
    print(f"单日跌幅>3% 的交易日占比: {(ret < -0.03).mean():.1%}")

    # ---------- 均线择时回测 ----------
    print("\n== 均线择时回测 (含万5单边成本, 信号次日执行) ==")
    strats = [rows[0]]
    curves = {"买入持有": (1 + ret.fillna(0)).cumprod()}
    for w in (20, 60, 120):
        s = ma_timing(close, w)
        strats.append(perf_stats(s, f"MA{w}择时"))
        curves[f"MA{w}择时"] = (1 + s).cumprod()
    print(pd.DataFrame(strats).to_string(index=False))

    # ---------- 画图 ----------
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    for name, nav in curves.items():
        axes[0].plot(nav.index, nav.values, label=name, lw=1.2)
    axes[0].set_title("科创50: 买入持有 vs 均线择时 (净值)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    nav = curves["买入持有"]
    dd = nav / nav.cummax() - 1
    axes[1].fill_between(dd.index, dd.values, 0, color="steelblue", alpha=0.5)
    axes[1].set_title("买入持有回撤")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = "kc50_analysis.png"
    plt.savefig(out, dpi=130)
    print(f"\n图表已保存: {out}")


if __name__ == "__main__":
    main()
